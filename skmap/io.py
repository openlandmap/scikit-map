'''
Raster data input and output
'''
from osgeo import gdal
from pathlib import Path
from hashlib import sha256
from pandas import DataFrame, to_datetime
import copy
import pandas as pd
import numpy
import numpy as np
import requests
import tempfile
import traceback
import math
import re
import os
import time
import tempfile

from typing import List, Union, Callable
from .misc import ttprint, _eval, update_by_separator
from . import parallel
from skmap.transform import SKMapTransformer
from skmap.misc import gen_dates
from . import SKMapBase

from datetime import datetime
from minio import Minio

from pathlib import Path
import rasterio
from rasterio.windows import Window

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
  try_without_window, overview, verbose):

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
      band_data = np.empty((int(window.height), int(window.width)))
      band_data[:] = _nodata_replacement(dtype)

    if expected_shape is not None:
      if verbose:
        ttprint(f'Full nan image for {raster_file}')
      band_data = np.empty(expected_shape)
      band_data[:] = _nodata_replacement(dtype)

  if data_mask is not None:

    if len(data_mask.shape) == 3:
      data_mask = data_mask[:,:,0]

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
  fit_in_dtype = False,
  on_each_outfile:Callable = None
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

  if on_each_outfile is not None:
    on_each_outfile(fn_new_raster)

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
    (raster_idx, raster_files, window, dtype, data_mask, 
      expected_shape, try_without_window, overview, verbose) 
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
  on_each_outfile:Callable = None,
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
      nodata, fit_in_dtype, on_each_outfile) \
    for i in range(0,len(raster_files))
  ]

  out_files = []
  for out_raster in parallel.job(_save_raster, args, n_jobs=n_jobs):
    out_files.append(out_raster)
    continue

  return out_files

class RasterData(SKMapBase):
  
  PLACEHOLDER_DT = '{dt}'
  INTERVAL_DT_SEP = '_'

  NAME_COL = 'name'
  PATH_COL = 'input_path'
  DT_COL = 'date'
  START_DT_COL = 'start_date'
  END_DT_COL = 'end_date'
  MAIN_TS_COL = 'main_ts'
  
  TRANSFORM_SEP = '.'

  def __init__(self,
    raster_files:Union[List,str],
    raster_mask:str = None,
    raster_mask_val = np.nan,
    verbose = False
  ):

    if not isinstance(raster_files, List):
      raster_files = [ raster_files ]

    self.verbose = verbose

    self.raster_mask = raster_mask
    self.raster_mask_val = raster_mask_val
    self.raster_data = {}

    self.rasters_static = []
    self.rasters_temporal = []

    for r in raster_files:
      if RasterData.PLACEHOLDER_DT in str(r):
        self.rasters_temporal.append(r)
      else:
        self.rasters_static.append(r)

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
      '%d/%m/%y': r'\d{2}(\/|-)?\d{2}(\/|-)?\d{2}',
      '%Y%j': r'\d{4}\d{3}',
      '%Y-%j': r'\d{4}(\/|-)?\d{3}',
      '%Y/%j': r'\d{4}(\/|-)?\d{3}'
    }

    self.info = self._info_static()

  def _info_temporal(self, date_format, date_style):
    rows = []
    
    regex_dt = self._regex_dt[date_format]

    for raster_file in self.rasters_temporal:
      name = Path(raster_file).stem

      dates = []
      for r in re.finditer(regex_dt, name):
        dates.append(r.group(0))
      
      row_data = {}
      row_data[RasterData.PATH_COL] = raster_file
      row_data[RasterData.NAME_COL] = name
      row_data[RasterData.MAIN_TS_COL] = True

      if date_style == 'interval':
        row_data[RasterData.START_DT_COL] = datetime.strptime(dates[0], date_format)
        row_data[RasterData.END_DT_COL] = datetime.strptime(dates[1], date_format)
      else:
        row_data[RasterData.DT_COL] = datetime.strptime(dates[0], date_format)

      rows.append(row_data)
    
    return DataFrame(rows)

  def _info_static(self):
    rows = []
    
    for raster_file in self.rasters_static:
      row_data = {}
      row_data[RasterData.PATH_COL] = raster_file
      row_data[RasterData.NAME_COL] = Path(raster_file).stem
      rows.append(row_data)
    
    return DataFrame(rows)

  def _set_date(self, 
    text, 
    dt1, 
    dt2,
    date_format,
    date_style,
    **kwargs
  ):
    
    if (date_style == 'start_date'):
      dt = f'{dt1.strftime(date_format)}'
    elif (date_style == 'end_date'):
      dt = f'{dt2.strftime(date_format)}'
    else:
      dt = f'{dt1.strftime(date_format)}'
      dt += f'{RasterData.INTERVAL_DT_SEP}'
      dt += f'{dt2.strftime(date_format)}'

    return _eval(str(text), {**kwargs,**locals()})

  def timespan(self,
    start_date,
    end_date,
    date_unit,
    date_step,
    date_style:str = 'interval',
    date_format:str = '%Y%m%d',
    ignore_29feb = True
  ):
    
    self.date_style = date_style
    self.date_format = date_format

    dates = gen_dates(start_date, end_date, 
      date_unit=date_unit, date_step=date_step, 
      date_format=date_format, ignore_29feb=ignore_29feb)
    
    for raster_file in self.rasters_temporal.copy():
      self.rasters_temporal.remove(raster_file)
      
      count = 0
      for dt1, dt2 in dates:
        raster_temporal = self._set_date(raster_file, dt1, dt2, date_format, date_style)
        
        if count == 0:
          self._verbose(f"First temporal raster: {raster_temporal}")
        
        self.rasters_temporal.append(raster_temporal)
        count += 1
      
      self._verbose(f"Last temporal raster: {raster_temporal}")
      self._verbose(f"{count} temporal rasters added ")
    
    self.info = pd.concat([self.info, self._info_temporal(date_format, date_style)])

    return self

  def read(self,
    window:Window = None,
    dtype:str = 'float32',
    expected_shape = None,
    n_jobs:int = 4,
  ):

    self.window = window
    
    data_mask = None
    if self.raster_mask is not None:
      self._verbose(f"Masking {self.raster_mask_val} values considering {Path(self.raster_mask).name}")
      data_mask = read_rasters([self.raster_mask], window=window)
      if self.raster_mask_val is np.nan:
        data_mask = np.logical_not(np.isnan(data_mask))
      else:
        data_mask = (data_mask != self.raster_mask_val)

    raster_files = self.rasters_temporal + self.rasters_static
    
    self._verbose(
        f"RasterData with {len(self.rasters_temporal)} temporal" 
      + f" and {len(self.rasters_static)} static rasters" 
    )
    self.array = read_rasters(
      raster_files,
      window=self.window, data_mask=data_mask,
      dtype=dtype, expected_shape=expected_shape,
      n_jobs=n_jobs, verbose=self.verbose
    )

    self._verbose(f"Read array shape: {self.array.shape}")

    return self

  def _tranform_info(self, 
    transformer:SKMapTransformer, 
    inplace = False,
    outname = None,
    suffix = ''
  ):
    tr = transformer.__class__.__name__
    tr = re.sub(r'(?<!^)(?=[A-Z])', '.', tr).lower()

    if suffix != '':
      tr += RasterData.TRANSFORM_SEP + suffix

    for index, row in self.info.iterrows():
      if row[RasterData.MAIN_TS_COL]:
        if outname is not None:
          if RasterData.START_DT_COL in self.info.columns:
            row[RasterData.NAME_COL] = self._set_date(outname, 
              row[RasterData.START_DT_COL], row[RasterData.END_DT_COL], 
              self.date_format, self.date_style, tr=tr
            )
          else:
            row[RasterData.NAME_COL] = _eval(outname, locals())
        else:
          row[RasterData.NAME_COL] = row[RasterData.NAME_COL] +  RasterData.TRANSFORM_SEP + tr

        i = len(self.info)
        if inplace:
          i = index
        else:
          row[RasterData.MAIN_TS_COL] = False
        self.info.loc[i] = row

  def transform(self, 
    transformer:SKMapTransformer, 
    inplace = False,
    outname:str = None
  ):
    info_main = self.info.query(f'{RasterData.MAIN_TS_COL} == True')
    transformer_name = transformer.__class__.__name__
    
    start = time.time()
    self._verbose(f"Executing tranformer {transformer_name}"
      + f" on {self.array[:,:,info_main.index].shape}")

    array_t = transformer.run(self.array[:,:,info_main.index])
    
    self._verbose(f"Tranformer {transformer_name} execution"
      + f" time: {(time.time() - start):.2f} segs")

    if isinstance(array_t, tuple) and len(array_t) >= 2:
      self.array = np.concatenate([self.array, array_t[1]], axis=-1)
      self._tranform_info(transformer, inplace=False, 
        outname=outname, suffix='qa')
      array_t = array_t[0]

    if inplace:
      self.array[:,:,info_main.index] = array_t
    else:
      self.array = np.concatenate([self.array, array_t], axis=-1)

    self._tranform_info(transformer, inplace, outname=outname)

    return self

  def filter_date(self, 
    start_date, 
    end_date = None, 
    date_format = '%Y-%m-%d',
    return_array=False, 
    return_copy=True
  ):

    start_dt_col, end_dt_col = (RasterData.START_DT_COL, RasterData.END_DT_COL)
    info_main = self.info.query(f'{RasterData.MAIN_TS_COL} == True')

    if RasterData.DT_COL in info_main.columns:
      start_dt_col, end_dt_col = (RasterData.DT_COL, None)

    dt_mask = info_main[start_dt_col] >= to_datetime(start_date, format=date_format)
    if end_date is not None and end_dt_col is not None:
      dt_mask = np.logical_and(
        dt_mask,
        info_main[end_dt_col] <= to_datetime(end_date, format=date_format),
      )

    return self._filter(info_main[dt_mask],
      return_array=return_array, return_copy=return_copy
    )

  def filter_contains(self, 
    text, 
    return_array=False, 
    return_copy=True
  ):
    return self.filter(f'{self.NAME_COL}.str.contains("{text}")', 
      return_array=return_array, return_copy=return_copy
    )

  def filter(self, 
    expr, 
    return_array=False, 
    return_copy=True
  ):
    info_main = self.info.query(f'{RasterData.MAIN_TS_COL} == True')
    return self._filter(info_main.query(expr),
      return_array=return_array, return_copy=return_copy
    )

  def _filter(self, 
    info, 
    return_array=False, 
    return_copy=True
  ):
    if return_array:
      return self.array[:,:,info.index]
    elif return_copy:
      rdata = copy.copy(self)
      rdata.array = self.array[:,:,info.index]
      rdata.info = info
      return rdata
    else:
      self.array = self.array[:,:,info.index]
      self.info = info
      return self

  def to_dir(self,
    out_dir:Union[Path,str],
    dtype:str = None,
    nodata = None,
    fit_in_dtype:bool = False,
    n_jobs:int = 4,
    return_outfiles = False,
    on_each_outfile:Callable = None,
  ):

    if isinstance(out_dir,str):
      out_dir = Path(out_dir)

    base_raster = self.info.iloc[-1][RasterData.PATH_COL]
    outfiles = [
      out_dir.joinpath(f'{name}.tif')
      for name in list(self.info[RasterData.NAME_COL])
    ]
    
    self._verbose(f"Saving rasters in {out_dir}")

    save_rasters(
      base_raster, outfiles,
      self.array, self.window,
      dtype=dtype, nodata=nodata,
      fit_in_dtype=fit_in_dtype, n_jobs=n_jobs,
      on_each_outfile = on_each_outfile, verbose = self.verbose
    )

    if return_outfiles:
      return outfiles
    else:
      return self

  def to_s3(self,
    host:str,
    access_key:str,
    secret_key:str,
    path:str,
    secure:bool = True,
    tmp_dir:str = None,
    dtype:str = None,
    nodata = None,
    fit_in_dtype:bool = False,
    n_jobs:int = 4,
    verbose_cp = False,
  ):

    bucket = path.split('/')[0]
    prefix = '/'.join(path.split('/')[1:])

    if tmp_dir is None:
      tmp_dir = Path(tempfile.TemporaryDirectory().name)
      tmp_dir = tmp_dir.joinpath(prefix)

    def _to_s3(outfile):
      
      client = Minio(host, access_key, secret_key, secure=secure)
      name = f'{outfile.name}'
      
      if verbose_cp:
        ttprint(f"Copying {outfile} to http://{host}/{bucket}/{prefix}/{name}")
      
      client.fput_object(bucket, f'{prefix}/{name}', outfile)
      os.remove(outfile)
    
    outfiles = self.to_dir(
      tmp_dir, dtype=dtype, nodata=nodata, 
      fit_in_dtype=fit_in_dtype, n_jobs=n_jobs, 
      return_outfiles=True, on_each_outfile = _to_s3
    )

    name = outfiles[len(outfiles)-1].name
    last_url = f'http://{host}/{bucket}/{prefix}/{name}'
    
    self._verbose(f"{len(outfiles)} rasters copied to s3")
    self._verbose(f"Last raster in s3: {last_url}")

    return self