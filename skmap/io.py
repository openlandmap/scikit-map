'''
Raster data input and output
'''
from osgeo import gdal
from pathlib import Path
from hashlib import sha256
from pandas import DataFrame, to_datetime
import numpy
import numpy as np
import requests
import tempfile
import traceback
import math
import re

from typing import List, Union
from .misc import ttprint, _eval
from . import parallel
from . import SKMapBase

from datetime import datetime

from pathlib import Path
import rasterio
from rasterio.windows import Window
from dateutil.relativedelta import relativedelta

_INT_DTYPE = (
  'uint8', 'uint8',
  'int16', 'uint16',
  'int32', 'uint32',
  'int64', 'uint64',
  'int', 'uint'
)

def _nodata_replacement(dtype):
  if dtype in _INT_DTYPE:
    return np.iinfo(dtype).max
  else:
    return np.nan

def _fit_in_dtype(data, dtype, nodata):

  if dtype in _INT_DTYPE:
    data = np.rint(data)

    min_val = np.iinfo(dtype).min
    max_val = np.iinfo(dtype).max

    data = np.where((data < min_val), min_val, data)
    data = np.where((data > max_val), max_val, data)

    if nodata == min_val:
      data = np.where((data == nodata), (min_val + 1), data)
    elif nodata == max_val:
      data = np.where((data == nodata), (max_val - 1), data)

  return data

def _read_raster(raster_idx, raster_files, window, dtype, data_mask, expected_shape, 
  try_without_window, overview):

  raster_file = raster_files[raster_idx]
  ds, band_data = None, None
  nodata = None

  try:
    ds = rasterio.open(raster_file)
    overviews = ds.overviews(1)

    if overview is not None and overview in overviews:
      band_data = ds.read(1, out_shape=(1, math.ceil(ds.height // overview), math.ceil(ds.width // overview)), window=window)
    else:
      band_data = ds.read(1, window=window)

    if band_data.size == 0 and try_without_window:
      band_data = ds.read(1)

    band_data = band_data.astype(dtype)
    nodata = ds.nodatavals[0]
  except:
    if window is not None:
      if verbose:
        ttprint(f'ERROR: Failed to read {raster_file} window {window}.')
      band_data = np.empty((int(window.width), int(window.height)))
      band_data[:] = _nodata_replacement(dtype)

    if expected_shape is not None:
      if verbose:
        ttprint(f'Full nan image for {raster_file}')
      band_data = np.empty(expected_shape)
      band_data[:] = _nodata_replacement(dtype)

  if data_mask is not None:

    if (data_mask.shape == band_data.shape):
      band_data[np.logical_not(data_mask)] = np.nan
    else:
      ttprint(f'WARNING: incompatible data_mask shape {data_mask.shape} != {band_data.shape}')

  return raster_idx, band_data, nodata

def _read_auth_raster(raster_files, url_pos, bands, username, password, dtype, nodata):

  url = raster_files[url_pos]

  data = None
  ds_params = None

  try:
    data = requests.get(url, auth=(username, password), stream=True)

    with rasterio.io.MemoryFile(data.content) as memfile:

      if verbose:
        ttprint(f'Reading {url} to {memfile.name}')

      with memfile.open() as ds:
        if bands is None:
          bands = range(1, ds.count+1)

        if nodata is None:
          nodata = ds.nodatavals[0]

        data = ds.read(bands)
        if (isinstance(data, np.ndarray)):
          data = data.astype(dtype)
          data[data == nodata] = _nodata_replacement(dtype)

        nbands, x_size, y_size = data.shape

        ds_params = {
          'driver': ds.driver,
          'width': x_size,
          'height': y_size,
          'count': nbands,
          'dtype': ds.dtypes[0],
          'crs': ds.crs,
          'transform': ds.transform
        }

  except:
    ttprint(f'Invalid raster file {url}')
    #traceback.print_exc()
    pass

  return url_pos, data, ds_params

def _new_raster(base_raster, raster_file, data, window = None,
  dtype = None, nodata = None):

  if (not isinstance(raster_file, Path)):
    raster_file = Path(raster_file)

  raster_file.parent.mkdir(parents=True, exist_ok=True)

  if len(data.shape) < 3:
    data = np.stack([data], axis=2)

  x_size, y_size, nbands = data.shape

  with rasterio.open(base_raster, 'r') as base_raster:
    if dtype is None:
      dtype = base_raster.dtypes[0]

    if nodata is None:
      nodata = base_raster.nodata

    transform = base_raster.transform

    if window is not None:
      transform = rasterio.windows.transform(window, transform)

    return rasterio.open(raster_file, 'w',
            driver='GTiff',
            height=x_size,
            width=y_size,
            count=nbands,
            dtype=dtype,
            crs=base_raster.crs,
            compress='LZW',
            transform=transform,
            nodata=nodata)

def _save_raster(
  fn_base_raster:str,
  fn_new_raster:str,
  data:numpy.array,
  spatial_win:Window = None,
  dtype:str = None,
  nodata = None,
  fit_in_dtype = False
):

  if len(data.shape) < 3:
    data = np.stack([data], axis=2)

  _, _, nbands = data.shape

  with _new_raster(fn_base_raster, fn_new_raster, data, spatial_win, dtype, nodata) as new_raster:

    for band in range(0, nbands):

      band_data = data[:,:,band]
      band_dtype = new_raster.dtypes[band]

      if fit_in_dtype:
        band_data = _fit_in_dtype(band_data, band_dtype, new_raster.nodata)

      band_data[np.isnan(band_data)] = new_raster.nodata
      new_raster.write(band_data.astype(band_dtype), indexes=(band+1))

  return fn_new_raster

def read_rasters(
  raster_files:Union[List,str] = [],
  window:Window = None,
  dtype:str = 'float32',
  n_jobs:int = 4,
  data_mask:numpy.array = None,
  expected_shape = None,
  try_without_window = False,
  overview = None,
  verbose = False
):
  """
  Read raster files aggregating them into a single array.
  Only the first band of each raster is read.

  The ``nodata`` value is replaced by ``np.nan`` in case of ``dtype=float*``,
  and for ``dtype=*int*`` it's replaced by the the lowest possible value
  inside the range (for ``int16`` this value is ``-32768``).

  :param raster_files: A list with the raster paths. Provide it and the ``raster_dirs``
    is ignored.
  :param window: Read the data according to the spatial window. By default is ``None``,
    reading all the raster data.
  :param dtype: Convert the read data to specific ``dtype``. By default it reads in
    ``float16`` to save memory, however pay attention in the precision limitations for
    this ``dtype`` [1].
  :param n_jobs: Number of parallel jobs used to read the raster files.
  :param data_mask: A array with the same space dimensions of the read data, where
    all the values equal ``0`` are converted to ``np.nan``.
  :param expected_shape: The expected size (space dimension) of the read data.
    In case of error in reading any of the raster files, this is used to create a
    empty 2D array. By default is ``None``, throwing a exception if the raster
    doesn't exists.
  :param try_without_window: First, try to read using ``window``, if fails
    try to read without it.
  :param overview: Overview level to be read. In COG files are usually `[2, 4, 8, 16, 32, 64, 128, 256]`.
  :param verbose: Use ``True`` to print the reading progress.

  :returns: A 3D array, where the last dimension refers to the read files, and a list
    containing the read paths.
  :rtype: Tuple[Numpy.array, List[Path]]

  Examples
  ========

  >>> import rasterio
  >>> from skmap.raster import read_rasters
  >>>
  >>> # skmap COG layers - NDVI seasons for 2000
  >>> raster_files = [
  >>>     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_200003_skmap_epsg3035_v1.0.tif', # winter
  >>>     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_200006_skmap_epsg3035_v1.0.tif', # spring
  >>>     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_200009_skmap_epsg3035_v1.0.tif', # summer
  >>>     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_200012_skmap_epsg3035_v1.0.tif'  # fall
  >>> ]
  >>>
  >>> # Transform for the EPSG:3035
  >>> eu_transform = rasterio.open(raster_files[0]).transform
  >>> # Bounding box window over Wageningen, NL
  >>> window = rasterio.windows.from_bounds(left=4020659, bottom=3213544, right=4023659, top=3216544, transform=eu_transform)
  >>>
  >>> data, _ = read_rasters(raster_files=raster_files, window=window, verbose=True)
  >>> print(f'Data shape: {data.shape}')

  References
  ==========

  [1] `Float16 Precision <https://github.com/numpy/numpy/issues/8063>`_

  """
  if data_mask is not None and dtype not in ('float16', 'float32'):
    raise Exception('The data_mask requires dtype as float')

  if isinstance(raster_files, str):
    raster_files = [ raster_files ]

  if len(raster_files) < n_jobs:
    n_jobs = len(raster_files)

  if verbose:
    ttprint(f'Reading {len(raster_files)} raster file(s) using {n_jobs} workers')

  raster_data = {}
  args = [ 
    (raster_idx, raster_files, window, dtype, 
    data_mask, expected_shape, try_without_window, overview) 
    for raster_idx in range(0,len(raster_files)) 
  ]

  for raster_idx, band_data, nodata in parallel.job(_read_raster, args, n_jobs=n_jobs):
    raster_file = raster_files[raster_idx]

    if (isinstance(band_data, np.ndarray)):

      if nodata is not None:
        band_data[band_data == nodata] = _nodata_replacement(dtype)

    else:
      raise Exception(f'The raster {raster_file} not exists')
    raster_data[raster_idx] = band_data

  raster_data = [raster_data[i] for i in range(0,len(raster_files))]
  raster_data = np.ascontiguousarray(np.stack(raster_data, axis=2))
  return raster_data

def read_auth_rasters(
  raster_files:List,
  username:str,
  password:str,
  bands = None,
  dtype:str = 'float16',
  n_jobs:int = 4,
  return_base_raster:bool = False,
  nodata = None,
  verbose:bool = False
):
  """
  Read raster files trough a authenticate HTTP service, aggregating them into
  a single array. For raster files without authentication it's better
  to use read_rasters.

  The ``nodata`` value is replaced by ``np.nan`` in case of ``dtype=float*``,
  and for ``dtype=*int*`` it's replaced by the the lowest possible value
  inside the range (for ``int16`` this value is ``-32768``).

  :param raster_files: A list with the raster urls.
  :param username: Username to provide to the basic access authentication.
  :param password: Password to provide to the basic access authentication.
  :param bands: Which bands needs to be read. By default is ``None`` reading all
    the bands.
  :param dtype: Convert the read data to specific ``dtype``. By default it reads in
    ``float16`` to save memory, however pay attention in the precision limitations for
    this ``dtype`` [1].
  :param n_jobs: Number of parallel jobs used to read the raster files.
  :param return_base_raster: Return an empty raster with the same properties
    of the read rasters ``(height, width, n_bands, crs, dtype, transform)``.
  :param nodata: Use this value if the nodata property is not defined in the
    read rasters.
  :param verbose: Use ``True`` to print the reading progress.

  :returns: A 4D array, where the first dimension refers to the bands and the last
    dimension to read files. If ``return_base_raster=True`` the second value
    will be a base raster path.
  :rtype: Numpy.array or Tuple[Numpy.array, Path]

  Examples
  ========

  >>> from skmap.raster import read_auth_rasters
  >>>
  >>> # Do the registration in
  >>> # https://glad.umd.edu/ard/user-registration
  >>> username = '<YOUR_USERNAME>'
  >>> password = '<YOUR_PASSWORD>'
  >>> raster_files = [
  >>>     'https://glad.umd.edu/dataset/landsat_v1.1/47N/092W_47N/850.tif',
  >>>     'https://glad.umd.edu/dataset/landsat_v1.1/47N/092W_47N/851.tif',
  >>>     'https://glad.umd.edu/dataset/landsat_v1.1/47N/092W_47N/852.tif',
  >>>     'https://glad.umd.edu/dataset/landsat_v1.1/47N/092W_47N/853.tif'
  >>> ]
  >>>
  >>> data, base_raster = read_auth_rasters(raster_files, username, password,
  >>>                         return_base_raster=True, verbose=True)
  >>> print(f'Data: shape={data.shape}, dtype={data.dtype} and base_raster={base_raster}')

  References
  ==========

  [1] `Float16 Precision <https://github.com/numpy/numpy/issues/8063>`_
  """

  if verbose:
    ttprint(f'Reading {len(raster_files)} remote raster files using {n_jobs} workers')

  args = [ (raster_files, url_pos, bands, username, password, dtype, nodata) for url_pos in range(0,len(raster_files)) ]

  raster_data = {}
  fn_base_raster = None

  for url_pos, data, ds_params in parallel.job(_read_auth_raster, args, n_jobs=n_jobs):
    url = raster_files[url_pos]

    if data is not None:
      raster_data[url_pos] = data

      if return_base_raster and fn_base_raster is None:
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as base_raster:
          with rasterio.open(base_raster.name, 'w',
            driver = ds_params['driver'],
            width = ds_params['width'],
            height = ds_params['height'],
            count = ds_params['count'],
            crs = ds_params['crs'],
            dtype = ds_params['dtype'],
            transform = ds_params['transform']
          ) as ds:
            fn_base_raster = ds.name

  raster_data_arr = []
  for i in range(0,len(raster_files)):
    if i in raster_data:
      raster_data_arr.append(raster_data[i])

  raster_data = np.stack(raster_data_arr, axis=-1)
  del raster_data_arr

  if return_base_raster:
    if verbose:
      ttprint(f'The base raster is {fn_base_raster}')
    return raster_data, fn_base_raster
  else:
    return raster_data

def save_rasters(
  base_raster:str,
  raster_files:List,
  data:numpy.array,
  window:Window = None,
  dtype:str = None,
  nodata = None,
  fit_in_dtype:bool = False,
  n_jobs:int = 4,
  verbose:bool = False
):
  """
  Save a 3D array in multiple raster files using as reference one base raster.
  The last dimension is used to split the data in different rasters. GeoTIFF is
  the only output format supported. It always replaces the ``np.nan`` value
  by the specified ``nodata``.

  :param base_raster: The base raster path used to retrieve the
    parameters ``(height, width, n_bands, crs, dtype, transform)`` for the
    new rasters.
  :param raster_files: A list containing the paths for the new raster. It creates
    the folder hierarchy if not exists.
  :param data: 3D data array.
  :param window: Save the data considering a spatial window, even if the ``base_rasters``
    refers to a bigger area. For example, it's possible to have a base raster covering the whole
    Europe and save the data using a window that cover just part of Wageningen. By default is
    ``None`` saving the raster data in position ``0, 0`` of the raster grid.
  :param dtype: Convert the data to a specific ``dtype`` before save it. By default is ``None``
    using the same ``dtype`` from the base raster.
  :param nodata: Use the specified value as ``nodata`` for the new rasters. By default is ``None``
    using the same ``nodata`` from the base raster.
  :param fit_in_dtype: If ``True`` the values outside of ``dtype`` range are truncated to the minimum
    and maximum representation. It's also change the minimum and maximum data values, if they exist,
    to avoid overlap with ``nodata`` (see the ``_fit_in_dtype`` function). For example, if
    ``dtype='uint8'`` and ``nodata=0``, all data values equal to ``0`` are re-scaled to ``1`` in the
    new rasters.
  :param n_jobs: Number of parallel jobs used to save the raster files.
  :param verbose: Use ``True`` to print the saving progress.

  :returns: A list containing the path for new rasters.
  :rtype: List[Path]

  Examples
  ========

  >>> import rasterio
  >>> from skmap.raster import read_rasters, save_rasters
  >>>
  >>> # skmap COG layers - NDVI seasons for 2019
  >>> raster_files = [
  >>>     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201903_skmap_epsg3035_v1.0.tif', # winter
  >>>     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201906_skmap_epsg3035_v1.0.tif', # spring
  >>>     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201909_skmap_epsg3035_v1.0.tif', # summer
  >>>     'http://s3.eu-central-1.wasabisys.com/skmap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201912_skmap_epsg3035_v1.0.tif'  # fall
  >>> ]
  >>>
  >>> # Transform for the EPSG:3035
  >>> eu_transform = rasterio.open(raster_files[0]).transform
  >>> # Bounding box window over Wageningen, NL
  >>> window = rasterio.windows.from_bounds(left=4020659, bottom=3213544, right=4023659, top=3216544, transform=eu_transform)
  >>>
  >>> data, _ = read_rasters(raster_files=raster_files, window=window, verbose=True)
  >>>
  >>> # Save in the current execution folder
  >>> raster_files = [
  >>>     './lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201903_wageningen_epsg3035_v1.0.tif',
  >>>     './lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201906_wageningen_epsg3035_v1.0.tif',
  >>>     './lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201909_wageningen_epsg3035_v1.0.tif',
  >>>     './lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201912_wageningen_epsg3035_v1.0.tif'
  >>> ]
  >>>
  >>> save_rasters(raster_files[0], raster_files, data, window=window, verbose=True)

  """

  if type(raster_files) == str: 
    raster_files = [ raster_files ]

  if len(data.shape) < 3:
    data = np.stack([data], axis=2)
  
  n_files = data.shape[-1]
  if n_files != len(raster_files):
    raise Exception(f'The data shape {data.shape} is incompatible with the raster_files size {len(raster_files)}.')

  if verbose:
    ttprint(f'Saving {len(raster_files)} raster files using {n_jobs} workers')

  args = [ \
    (base_raster, raster_files[i], data[:,:,i], window, dtype,
      nodata, fit_in_dtype) \
    for i in range(0,len(raster_files))
  ]

  out_files = []
  for out_raster in parallel.job(_save_raster, args, n_jobs=n_jobs):
    out_files.append(out_raster)
    continue

  return out_files

class RasterData(SKMapBase):
  
  NAME_COL = 'name'
  PATH_COL = 'path'
  
  def __init__(self,
    array:numpy.array,
    label:DataFrame
  ):
    self.array = array
    self.label = label

  def filter_contains(self, text, return_label=False):
    return self.filter(f'{self.NAME_COL}.str.contains("{text}")', return_label=return_label)

  def filter(self, expr, return_label=False):
    label_filtered = self.label.query(expr)
    
    if return_label:
      return self.array[:,:,label_filtered.index], label_filtered
    else:
      return self.array[:,:,label_filtered.index]

class RasterDataT(RasterData):
  
  DT_COL = 'date'
  START_DT_COL = 'start_date'
  END_DT_COL = 'end_date'

  def __init__(self,
    array:numpy.array,
    label:DataFrame,
    date_format:str,
  ):
    super().__init__(array, label)
    
    self.date_format = date_format

  def filter_date(self, 
    dt1, 
    dt2 = None, 
    date_format = None,
    return_label=False
  ):

    if date_format is None:
      date_format = self.date_format

    dt1_col, dt2_col = (RasterDataT.START_DT_COL, RasterDataT.END_DT_COL)
    
    if RasterDataT.DT_COL in self.label.columns:
      dt1_col, dt2_col = (RasterDataT.DT_COL, None)

    dt_mask = self.label[dt1_col] >= to_datetime(dt1, format=date_format)
    if dt2 is not None and dt2_col is not None:
      dt_mask = np.logical_and(
        dt_mask,
        self.label[dt2_col] <= to_datetime(dt2, format=date_format),
      )

    label_filtered = self.label[dt_mask]

    if return_label:
      return self.array[:,:,label_filtered.index], label_filtered
    else:
      return self.array[:,:,label_filtered.index]

class RasterCube(SKMapBase):
    
  def __init__(self,
    raster_files:List = [],
    dtype:str = 'float32',
    expected_shape = None,
    n_jobs:int = 4,
    verbose = False
  ):

    self.dtype = dtype
    self.n_jobs = n_jobs
    self.verbose = verbose

    self.raster_files = raster_files
    self.expected_shape = expected_shape
    self.raster_data = {}

  def _label(self, raster_files):
    rows = []
    
    for raster_file in raster_files:
      row_data = {}
      row_data[RasterData.PATH_COL] = raster_file
      row_data[RasterData.NAME_COL] = Path(raster_file).stem
      rows.append(row_data)
    
    return DataFrame(rows)

  def _key(self, obj):
    return sha256(str(obj).encode('UTF-8'))

  def _new_raster_data(self, 
    array, 
    raster_files
  ):
    return RasterData(
      array, self._label(raster_files)
    )

  def _read(self,
    raster_files,
    window:Window = None,
    data_mask:numpy.array = None
  ):

    key = self._key(window)         

    if key not in self.raster_data:
      self._verbose("Reading data")
      array = read_rasters(
        raster_files,
        window=window, data_mask=data_mask,
        dtype=self.dtype, expected_shape=self.expected_shape,
        n_jobs=self.n_jobs, verbose=self.verbose
      )

      self.raster_data[key] = self._new_raster_data(array, raster_files)

    return self.raster_data[key]

  def read(self,
    window:Window = None,
    data_mask:numpy.array = None
  ):
    return self._read(self.raster_files)

class RasterCubeT(RasterCube):

  def __init__(self,
    raster_files:List = [],
    date_style:str = 'interval',
    date_format:str = '%Y%m%d',
    date_path_regex:str = None,
    interval_sep:str = '_',
    dtype:str = 'float32',
    expected_shape = None,
    n_jobs:int = 4,
    verbose = False
  ):
    super().__init__(
      raster_files, dtype, 
      expected_shape, n_jobs,
      verbose
    )

    self.date_style = date_style
    self.date_format = date_format
    self.interval_sep = interval_sep

    self._regex_dt = {
      '%Y%m%d': r'\d{4}(\/|-)?\d{2}(\/|-)?\d{2}',
      '%Y-%m-%d': r'\d{4}(\/|-)?\d{2}(\/|-)?\d{2}',
      '%Y/%m/%d': r'\d{4}(\/|-)?\d{2}(\/|-)?\d{2}',
      '%y%m%d': r'\d{2}(\/|-)?\d{2}(\/|-)?\d{2}',
      '%y-%m-%d': r'\d{2}(\/|-)?\d{2}(\/|-)?\d{2}',
      '%y-%m-%d': r'\d{2}(\/|-)?\d{2}(\/|-)?\d{2}',
      '%d%m%Y': r'\d{2}(\/|-)?\d{2}(\/|-)?\d{4}',
      '%d-%m-%Y': r'\d{2}(\/|-)?\d{2}(\/|-)?\d{4}',
      '%d/%m/%Y': r'\d{2}(\/|-)?\d{2}(\/|-)?\d{4}',
      '%d%m%y': r'\d{2}(\/|-)?\d{2}(\/|-)?\d{2}',
      '%d-%m-%y': r'\d{2}(\/|-)?\d{2}(\/|-)?\d{2}',
      '%d/%m/%y': r'\d{2}(\/|-)?\d{2}(\/|-)?\d{2}'
    }

  def _new_raster_data(self, 
    array, 
    raster_files
  ):
    return RasterDataT(
      array, self._label(raster_files),
      date_format=self.date_format
    )

  def _label(self, 
    raster_files
  ):
    rows = []
    
    regex_dt = self._regex_dt[self.date_format]

    for raster_file in raster_files:
      dates = []
      for r in re.finditer(regex_dt, raster_file):
        dates.append(r.group(0))
      
      row_data = {}
      row_data[RasterData.PATH_COL] = raster_file
      row_data[RasterData.NAME_COL] = Path(raster_file).stem

      if len(dates) == 1:
        row_data[RasterDataT.DT_COL] = datetime.strptime(dates[0], self.date_format)
      elif len(dates) > 1:
        row_data[RasterDataT.START_DT_COL] = datetime.strptime(dates[0], self.date_format)
        row_data[RasterDataT.END_DT_COL] = datetime.strptime(dates[1], self.date_format)

      rows.append(row_data)
    
    return DataFrame(rows)

  def _fill_date(self, 
    raster_file, 
    dt1, 
    dt2
  ):
      
    if (self.date_style == 'start_date'):
      dt = f'{dt1.strftime(self.date_format)}'
    elif (self.date_style == 'end_date'):
      dt = f'{dt2.strftime(self.date_format)}'
    else:
      dt = f'{dt1.strftime(self.date_format)}'
      dt += f'{self.interval_sep}'
      dt += f'{dt2.strftime(self.date_format)}'

    return _eval(str(raster_file), locals())

  def _date_step(self, 
    date_step, 
    i = 0
  ):
    if isinstance(date_step, list):
      if i >= len(date_step):
        i = 0
      date_step_cur = int(date_step[i])
      i += 1
      return i, date_step_cur
    else:
      int(date_step)

  def _gen_dates(self, 
    start_date, 
    end_date, 
    date_unit, 
    date_step, 
    ignore_29feb
  ):

    result = []

    dt1 = start_date
    date_step_i = 0

    while(dt1 <= end_date):
      delta_args = {}
      date_step_i, date_step_cur = self._date_step(date_step, date_step_i)
      delta_args[date_unit] = date_step_cur # TODO: Threat the value "month"
      
      dt1n = dt1 + relativedelta(**delta_args)
      dt_feb = (datetime.strptime(f'{dt1n.year}0228', '%Y%m%d'))

      if (dt_feb > dt1 and dt_feb <= dt1n):
        dt1n = dt1n + relativedelta(leapdays=+1)

      dt2 = dt1n + relativedelta(days=-1)
    
      if ignore_29feb:
        if dt2.month == 2 and dt2.day == 29:
          dt2 = dt2 + relativedelta(days=-1)
          
      result.append((dt1, dt2))       
      dt1 = dt1n
    
    return result

  def read(self,
    start_date,
    end_date,
    date_unit,
    date_step,
    ignore_29feb = True,
    window:Window = None,
    data_mask:numpy.array = None
  ):

    start_date = datetime.strptime(start_date, self.date_format)
    end_date = datetime.strptime(end_date, self.date_format)
    
    dates = self._gen_dates(start_date, end_date, date_unit, date_step, ignore_29feb)

    rasters_filled = []

    for raster_file in self.raster_files:
      for dt1, dt2 in dates:
        raster_filled = self._fill_date(raster_file, dt1, dt2)
        self._verbose(f"Preparing {raster_filled}")
        rasters_filled.append(raster_filled)

    return self._read(
      rasters_filled,
      window=window, 
      data_mask=data_mask
    )