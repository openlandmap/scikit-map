'''
Raster data input and output
'''
from osgeo import gdal
from pathlib import Path
from hashlib import sha256
from pandas import DataFrame, Series, to_datetime
from types import MappingProxyType

import shutil
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
import gc
import matplotlib as mpl
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
import traceback

from typing import List, Union, Callable
from skmap.misc import ttprint, _eval, update_by_separator, date_range, new_memmap, del_memmap
from skmap import SKMapRunner, SKMapBase, parallel

from dateutil.relativedelta import relativedelta
from datetime import datetime
from minio import Minio

from pathlib import Path
import rasterio
from rasterio.windows import Window

import bottleneck as bn

from IPython.display import HTML
from joblib import Parallel, delayed
from tempfile import TemporaryDirectory
from io import BytesIO
from base64 import encodebytes, b64decode
from uuid import uuid4
from contextlib import ExitStack
from matplotlib._animation_data import JS_INCLUDE, STYLE_INCLUDE, DISPLAY_TEMPLATE
from PIL import Image

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

def _read_raster(raster_idx, raster_files, array_mm, band, window, dtype, data_mask, expected_shape, 
  try_without_window, overview, verbose):

  raster_file = raster_files[raster_idx]
  ds, band_data = None, None
  nodata = None

  try:
    ds = rasterio.open(raster_file)
    
    if overview is not None:
      overviews = ds.overviews(band)
      if overview in overviews:
          band_data = ds.read(band, out_shape=(1, math.ceil(ds.height // overview), math.ceil(ds.width // overview)), window=window)
      else:
        band_data = ds.read(band, window=window)
    else:
      band_data = ds.read(band, window=window)

    if band_data.size == 0 and try_without_window:
      band_data = ds.read(band)

    band_data = band_data.astype(dtype)
    nodata = ds.nodatavals[0]
  except Exception as ex:
    ttprint(f"Exception: {ex}")
    #traceback.iprint_exc()
    
    if window is not None:
      if verbose:
        ttprint(f'ERROR: Failed to read {raster_file} window {window}')
      band_data = np.empty((int(window.height), int(window.width)))
      band_data[:,:] = _nodata_replacement(dtype)

    if expected_shape is not None:
      if verbose:
        ttprint(f'Full nan image for {raster_file}')
      band_data = np.empty(expected_shape)
      band_data[expected_shape] = _nodata_replacement(dtype)

  data_exists = (band_data is not None)

  if data_exists:
    if data_mask is not None:

      if len(data_mask.shape) == 3:
        data_mask = data_mask[:,:,0]

      if (data_mask.shape == band_data.shape):
        band_data[np.logical_not(data_mask)] = np.nan
      else:
        ttprint(f'WARNING: incompatible data_mask shape {data_mask.shape} != {band_data.shape}')
    
    if nodata is not None:
      band_data[band_data == nodata] = _nodata_replacement(dtype)
    
    array_mm[:,:,raster_idx] = band_data

  return raster_idx, data_exists

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
  raster_files:list,
  data:numpy.array,
  i:int,
  spatial_win:Window = None,
  dtype:str = None,
  nodata = None,
  fit_in_dtype = False,
  on_each_outfile:Callable = None
):

  if len(data.shape) < 3:
    data = np.stack([data], axis=2)

  #_, _, nbands = data.shape
  
  with _new_raster(fn_base_raster, raster_files[i], data[:,:,i], spatial_win, dtype, nodata) as new_raster:

    band_data = np.array(data[:,:,i])
    band_dtype = new_raster.dtypes[0]

    if fit_in_dtype:
      band_data = _fit_in_dtype(band_data, band_dtype, new_raster.nodata)

    band_data[np.isnan(band_data)] = new_raster.nodata
    new_raster.write(band_data.astype(band_dtype), indexes=1)

  if on_each_outfile is not None:
    on_each_outfile(raster_files[i])

  return raster_files[i]

def read_rasters(
  raster_files:Union[List,str] = [],
  band = 1,
  window:Window = None,
  dtype:str = 'float32',
  n_jobs:int = 4,
  data_mask:numpy.array = None,
  expected_shape = None,
  try_without_window = False,
  overview = None,
  keep_memmap = False,
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

  ds = rasterio.open(raster_files[-1])
  if overview is not None:
    overviews = ds.overviews(band)
    if overview in overviews:
        n_bands, height, width  = ds.count, math.ceil(ds.height // overview), math.ceil(ds.width // overview)
    else:
      raise Exception(f"Overview {overviews} is invalid for {raster_files[-1]}.\n"
        f"Use one of overviews: {ds.overviews(band)}")
  elif window is not None:
    n_bands, height, width,  = ds.count, window.height, window.width
  else:
    n_bands, height, width,  = ds.count, ds.height, ds.width

  #temp_folder = tempfile.mkdtemp()
  #filename = os.path.join(temp_folder, 'joblib_readraster.mmap')
  #array_mm = np.memmap(filename, dtype=dtype, shape=(height, width, len(raster_files)), mode='w+')
  array_mm = new_memmap(dtype, shape=(height, width, len(raster_files)))

  args = [ 
    (raster_idx, raster_files, array_mm, band, window, dtype, data_mask, 
      expected_shape, try_without_window, overview, verbose) 
    for raster_idx in range(0,len(raster_files)) 
  ]
  
  for raster_idx, data_exists in parallel.job(_read_raster, args, n_jobs=n_jobs, joblib_args={'backend': 'multiprocessing'}):
    
    if (not data_exists):
      raster_file = raster_files[raster_idx]
      raise Exception(f'The raster {raster_file} not exists')
  
  if not keep_memmap:
    return del_memmap(array_mm, True)
  else:
    return array_mm

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
    (base_raster, raster_files, data, i, window, dtype,
      nodata, fit_in_dtype, on_each_outfile) \
    for i in range(0,len(raster_files))
  ]

  out_files = []
  for out_raster in parallel.job(_save_raster, args, n_jobs=n_jobs, joblib_args={'backend': 'multiprocessing'}):
    out_files.append(out_raster)
    continue

  return out_files

class RasterData(SKMapBase):
  
  PLACEHOLDER_DT = '{dt}'
  INTERVAL_DT_SEP = '_'

  GROUP_COL = 'group'
  NAME_COL = 'name'
  PATH_COL = 'input_path'
  BAND_COL = 'input_band'
  TEMPORAL_COL = 'temporal'
  DT_COL = 'date'
  START_DT_COL = 'start_date'
  END_DT_COL = 'end_date'
  
  TRANSFORM_SEP = '.'

  def __init__(self,
    raster_files:Union[List,str,dict],
    raster_mask:str = None,
    raster_mask_val = np.nan,
    verbose = False
  ):

    if isinstance(raster_files, str):
      raster_files = { 'default': [raster_files] } 
    elif isinstance(raster_files, list):
      raster_files = { 'default': raster_files }

    self.raster_files = raster_files

    self.verbose = verbose

    self.raster_mask = raster_mask
    self.raster_mask_val = raster_mask_val

    rows = []
    for group in raster_files.keys():
      if isinstance(raster_files[group], str):
        rows.append([group, raster_files[group], 1])
      else:
        for r in raster_files[group]:
          if isinstance(r, tuple):
            rows.append([group, r[0], r[1]])
          else:
            rows.append([group, r, 1])

    self.info = DataFrame(rows, columns=[RasterData.GROUP_COL, RasterData.PATH_COL, RasterData.BAND_COL])
    self.info[RasterData.TEMPORAL_COL] = self.info.apply(lambda r: RasterData.PLACEHOLDER_DT in str(r[RasterData.PATH_COL]), axis=1)
    self.info[RasterData.NAME_COL] = self.info.apply(lambda r: Path(r[RasterData.PATH_COL]).stem if not r[RasterData.TEMPORAL_COL] else None, axis=1)
    self.info.reset_index(drop=True, inplace=True)

    self._active_group = None
    self.date_args = {}

  def _new_info_row(self, 
    raster_file:str,
    name:str,
    group:str = None,
    dates:list = [],
  ):

    row = {}

    if group is None or 'default' in group:
      group = 'default'

    row[RasterData.PATH_COL] = raster_file
    row[RasterData.NAME_COL] = name
    row[RasterData.GROUP_COL] = group
    row[RasterData.BAND_COL] = 1

    date_style = self.date_args[self._active_group]['date_style']
    date_format = self.date_args[self._active_group]['date_format']
    self.date_args[group] = self.date_args[self._active_group]

    if len(dates) > 0 and date_style is not None:
      row[RasterData.TEMPORAL_COL] = True

      dt1, dt2 = (dates[0], dates[1] )
      
      if isinstance(dt1, str):
        dt1 = datetime.strptime(dt1, date_format)
      if isinstance(dt2, str):
        dt2 = datetime.strptime(dt2, date_format)
      row[RasterData.START_DT_COL] = dt1
      row[RasterData.END_DT_COL] = dt2

    else:
      row[RasterData.PATH_COL] = raster_file
      row[RasterData.NAME_COL] = name

    return row

  def _set_date(self, 
    text, 
    dt1, 
    dt2,
    date_format = None,
    date_style = None,
    ignore_29feb = None,
    **kwargs
  ):
    
    if 'gr' in kwargs and 'default' in kwargs.get('gr'):
      gr = ''

    if date_format is None:
      date_format = self.date_args[self._active_group]['date_format']

    if date_style is None:
      date_style = self.date_args[self._active_group]['date_style']

    if ignore_29feb and '%j' in date_format:
      dt1 = dt1 + relativedelta(leapdays=-1)
      dt2 = dt2 + relativedelta(leapdays=-1)

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
    ignore_29feb = True,
    group:[list,str] = []
  ):
    
    if isinstance(group, str):
      group = [ group ]

    new_info = [] 

    for _group, ginfo in self.info.groupby(RasterData.GROUP_COL):

      self.date_args[_group] = {
        'date_style': date_style,
        'date_format': date_format,
        'ignore_29feb': ignore_29feb
      }

      dates = date_range(start_date, end_date, 
        date_unit=date_unit, date_step=date_step, 
        date_format=date_format, ignore_29feb=ignore_29feb)
    
      def fun(r):
        if r[RasterData.TEMPORAL_COL]:
          names, start, end = [], [], []
          for dt1, dt2 in dates:
            names.append(self._set_date(
              r[RasterData.PATH_COL], dt1, dt2, 
              date_format=date_format, date_style=date_style, ignore_29feb=ignore_29feb)
            )
            start.append(dt1)
            end.append(dt2)
          return Series([names, start, end])
        else:
          return Series([[r[RasterData.PATH_COL]],[None],[None]])

      temporal_cols = [RasterData.PATH_COL, RasterData.START_DT_COL, RasterData.END_DT_COL]

      ginfo[temporal_cols] = ginfo.apply(fun, axis=1)
      ginfo = ginfo.explode(temporal_cols)
      ginfo[RasterData.NAME_COL] = ginfo.apply(lambda r: Path(r[RasterData.PATH_COL]).stem, axis=1)
      print(ginfo.shape)
      new_info.append(ginfo)

    self.info = pd.concat(new_info).reset_index(drop=True)

    return self

  def read(self,
    window:Window = None,
    dtype:str = 'float32',
    expected_shape = None,
    overview:int = None,
    n_jobs:int = 4,
  ):

    self.window = window
    
    data_mask = None
    if self.raster_mask is not None:
      self._verbose(f"Masking {self.raster_mask_val} values considering {Path(self.raster_mask).name}")
      data_mask = read_rasters([self.raster_mask], window=window, overview=overview)
      if self.raster_mask_val is np.nan:
        data_mask = np.logical_not(np.isnan(data_mask))
      else:
        data_mask = (data_mask != self.raster_mask_val)

    #for band in list(self.info[RasterData.PATH_COL].unique()):
    #if expected_shape is None:
    #  base_raster = self._base_raster()
    #  ds = rasterio.open(base_raster)
    #  width, height = ds.width, ds.height
    #else:
    #  width, height = expected_shape[0], expected_shape[1]
    
    #n_rows = self.info.shape[0]
    #self.array = np.empty((width, height, n_rows), dtype=dtype)
    self.array = []

    for band, rows in self.info.groupby(RasterData.BAND_COL):

      rows[RasterData.PATH_COL]
      raster_files = [ Path(r) for r in rows[RasterData.PATH_COL] ]
      
      self._verbose(
          f"RasterData with {len(raster_files)} rasters (band: {band})" 
        + f" and {len(self.info[RasterData.GROUP_COL].unique())} group(s)" 
      )

      array = read_rasters(
        raster_files, band=band,
        window=self.window, data_mask=data_mask,
        dtype=dtype, expected_shape=expected_shape,
        n_jobs=n_jobs, overview=overview, verbose=self.verbose
      )
      
      self.array.append(array)
      self._verbose(f"Read array shape: {array.shape}")

    self.array = np.concatenate(self.array, axis=-1)

    return self

  def run(self, 
    process:SKMapRunner,
    group:[list,str] = [],
    outname:str = None,
    drop_input:bool = False
  ):
    
    if isinstance(group, str):
      group = [ group ]

    for _group, ginfo in self.info.groupby(RasterData.GROUP_COL):

      if ginfo[RasterData.TEMPORAL_COL].iloc[0] != process.temporal:
        self._verbose(f"Skipping {process.__class__.__name__} for {_group} group")
        continue

      if len(group) > 0 and _group not in group:
        continue

      self._active_group = _group

      expr_group = f'{RasterData.GROUP_COL} == "{self._active_group}"'
      ginfo = self.info.query(expr_group)

      process_name = process.__class__.__name__
      
      start = time.time()
      self._verbose(f"Running {process_name}"
        + f" on {self.array[:,:,ginfo.index].shape}"
        + f" for {_group} group")

      new_array, new_info = process.run(self, _group, outname)
      
      if drop_input:
        self._verbose(f"Dropping data and info for {_group} group")
        self.array = np.delete(self.array, ginfo.index, axis=-1) 
        self.info = self.info.drop(ginfo.index)

      self.array = np.concatenate([self.array, new_array], axis=-1)
      self.info = pd.concat([self.info, new_info])
      self.info.reset_index(drop=True, inplace=True)
      
      self._verbose(f"Execution"
        + f" time for {process_name}: {(time.time() - start):.2f} segs")

      self._active_group = None

    return self

  def rename(self, groups:dict):
    self.info[RasterData.GROUP_COL] = self.info[RasterData.GROUP_COL].replace(groups)
    self.info[RasterData.NAME_COL] = self.info[RasterData.NAME_COL].replace(groups)
    for old_group in groups.keys():
      new_group = groups[old_group]
      self.date_args[new_group] = self.date_args[old_group]
      del self.date_args[old_group]
    return self

  def filter_date(self, 
    start_date, 
    end_date = None, 
    date_format = '%Y-%m-%d',
    date_overlap = False,
    return_array=False, 
    return_copy=True
  ):

    start_dt_col, end_dt_col = (RasterData.START_DT_COL, RasterData.END_DT_COL)
    info_main = self.info

    if RasterData.DT_COL in info_main.columns:
      start_dt_col, end_dt_col = (RasterData.DT_COL, None)

    if date_overlap:
      dt_mask = np.logical_or(
        info_main[start_dt_col] >= to_datetime(start_date, format=date_format),
        info_main[end_dt_col] >= to_datetime(start_date, format=date_format)
      )
    else:
      dt_mask = info_main[start_dt_col] >= to_datetime(start_date, format=date_format)

    if end_date is not None and end_dt_col is not None:
      
      if date_overlap:
        dt_mask_end = np.logical_or(
          info_main[end_dt_col] <= to_datetime(end_date, format=date_format),
          info_main[start_dt_col] <= to_datetime(end_date, format=date_format)
        )
      else:
        dt_mask_end = info_main[end_dt_col] <= to_datetime(end_date, format=date_format)

      dt_mask = np.logical_and(
        dt_mask,
        dt_mask_end
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
    return self._filter(self.info.query(expr),
      return_array=return_array, return_copy=return_copy
    )

  def _filter(self, 
    info, 
    return_info=False,
    return_array=False, 
    return_copy=True
  ):

    # Active filters 
    if self._active_group is not None:
      info = info.query(f'{RasterData.GROUP_COL} == "{self._active_group}"')

    if return_array:
      return self.array[:,:,info.index]
    elif return_info:
      return info
    elif return_copy:
      rdata = copy.copy(self)
      rdata.array = self.array[:,:,info.index]
      rdata.info = info
      return rdata
    else:
      self.array = self.array[:,:,info.index]
      self.info = info
      return self

  def _array(self):
    return self._filter(self.info, return_array=True)

  def _info(self):
    return self._filter(self.info, return_info=True)

  def _base_raster(self):
    for _, row  in self.info.iterrows():
      path = row[RasterData.PATH_COL]
      if 'http:' in str(path):
        res = requests.head(path)
        if (res.status_code == 200):
          return path
      elif os.path.isfile(path):
        return path

    raise Exception(f'No base raster is available.')

  def to_dir(self,
    out_dir:Union[Path,str],
    group_expr:str = None,
    dtype:str = None,
    nodata = None,
    fit_in_dtype:bool = False,
    n_jobs:int = 4,
    return_outfiles = False,
    on_each_outfile:Callable = None,
  ):

    if isinstance(out_dir,str):
      out_dir = Path(out_dir)

    info = self.info
    if group_expr is not None:
      info = self.info.query(group_expr)

    if info.size == 0:
      ttprint("No rasters to save. Double check group_expr arg.")
      return self

    base_raster = self._base_raster()
    outfiles = [
      out_dir.joinpath(f'{name}.tif')
      for name in list(info[RasterData.NAME_COL])
    ]
    
    self._verbose(f"Saving rasters in {out_dir}")

    save_rasters(
      base_raster, outfiles,
      self.array[:,:,info.index], self.window,
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
    group_expr:str = None,
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
      tmp_dir, group_expr=group_expr, dtype=dtype, nodata=nodata, 
      fit_in_dtype=fit_in_dtype, n_jobs=n_jobs, 
      return_outfiles=True, on_each_outfile = _to_s3
    )

    name = outfiles[len(outfiles)-1].name
    last_url = f'http://{host}/{bucket}/{prefix}/{name}'
    
    self._verbose(f"{len(outfiles)} rasters copied to s3")
    self._verbose(f"Last raster in s3: {last_url}")

    return self

  def _get_titles(self, img_title):
    if img_title == 'date':
      titles = list(self.info['start_date'].astype(str) + ' - ' + self.info['end_date'].astype(str))
    elif img_title == 'index':
      titles = [i for i in range(self.info.shape[0])]
    elif img_title == 'name':
      titles = []
      n = 20
      for name in list(self.info['name']):
        titles.append('\n'.join(name[i:i+n] for i in range(0, len(name), n)))
    else:
      titles = [''] * self.info.shape[0]
    return titles

  def animate(self, 
    cmap:str = 'Spectral_r', 
    legend_title:str = "", 
    img_title:str ="index", 
    interval:int = 250,
    figsize:tuple = None,
    v_minmax:tuple = None,
    to_gif:str = None
  ):

    """
    Generates an animation to view and save.

    :param cmap: colormap name one of the `matplotlib.colormaps()`
    :param legend_title: title of the colorbar that will be used within the animation
      default is an empty string
    :param img_title: this could be `name`,`date`, `index` or None. Default value 
      is `index`
    :param interval: delay-time in between two images in miliseconds. Default is 250 ms
    :param figsize: figure size that will be generated. Default value is `(8,8)`
    :param v_minmax: minimum and maximum boundaries of the colorscale. Default is None and 
      it will be derived from the dataset if not defined.
    :param to_gif: this should be directory that indicating the location where user want to
      save the animation. Default is None
    
    Examples
    ========
    from skmap.data import toy

    data = toy.ndvi_data(gappy=True, verbose=True)
    data.animate(cmap='Spectral_r', legend_title="NDVI", img_title='date')

    """

    titles = self._get_titles(img_title)
    fontsize = 10 if img_title == 'name' else 14

    if figsize is None:
      width, height = self.array.shape[:2]
      ratio_width = width/height if width>height else height/width
      ratio_height = 1 if ratio_width < 2 else 2
      figsize = (6*ratio_width,6*ratio_height)

    if v_minmax is None:
      vmin , vmax = np.nanquantile(self.array.flatten(), [.1, .9])
      #vmin, vmax = (bn.nanmin(self.array), bn.nanmax(self.array))
    else:
      vmin, vmax = v_minmax
    
    def _populateImages(array, tile):
      fig, ax = pyplot.subplots(figsize=figsize)
      fig.colorbar(
        ax.imshow(array, vmin=vmin, vmax=vmax, cmap=cmap), 
        aspect=12, shrink=0.8, 
        label=legend_title, 
        orientation='vertical',
        location='right'
      )
      ax.axis('off')
      pyplot.suptitle(tile, fontsize=fontsize, y=0.9)
      f = BytesIO()
      pyplot.tight_layout()
      fig.savefig(f, format='png')
      imgdata64 = encodebytes(f.getvalue()).decode('ascii')
      return imgdata64
    
    # find a way to optimize processor count
    args = [ (self.array[:,:,i],titles[i]) for i in range(self.array.shape[2]) ]
    images = [ img for img in parallel.job(_populateImages, args) ]
    
    if to_gif is not None:
      with ExitStack() as stack:
        imgs = (
          stack.enter_context(Image.open(BytesIO(b64decode(f))))
          for f in images
        )
        img = next(imgs)
        img.save(to_gif, format="GIF", append_images=imgs, save_all=True, duration= interval, loop=0)

    template = '  frames[{0}] = "data:image/{1};base64,{2}"\n'

    embedded_frames = "\n" + "".join(
      template.format(i, 'png', imgdata.replace("\n","\\\n"))
      for i, imgdata in enumerate(images)
    )
    mode_dict = dict(
      once_checked="",
      loop_checked="checked",
      reflect_checked=""
    )
    
    with TemporaryDirectory() as tmpdir:
      path = Path(tmpdir, 'temp.html')
      with open(path, 'w') as of:
        of.write(JS_INCLUDE + STYLE_INCLUDE)
        of.write(DISPLAY_TEMPLATE.format(
          id=uuid4().hex,
          Nframes=self.array.shape[2],
          fill_frames = embedded_frames,
          interval = interval,
          **mode_dict
        ))
      html_rep = path.read_text()
    return HTML(html_rep)

  def plot(self, 
    cmap:str = 'Spectral_r',
    legend_title:str = "", 
    img_title:str = 'index',
    figsize:tuple = None,
    v_minmax:tuple = None,
    to_img:str = None,
    dpi:int = 300
  ):
    """
    Generates a square grid plot to view and save with colorscale.

    :param cmap: Default is Spectral_r
      colormap name one of the `matplotlib.colormaps()`
    :param legend_title: Default is an empty string
      title of the colorbar that will be used within the animation
    :param img_title: Default value is None
      this could be `name`,`date`, `index` or None. 
    :param figsize: Default is None. 
      Default size for figure will be calculated dynamically respect to the
      data volume.
    :param v_minmax: minimum and maximum boundaries of corethe colorscale. Default is None and 
      it will be derived from the dataset if not defined.
    :param to_img:  this should be directory that indicating the location where user want to
      save the image. Default is None
    :param dpi: Dot per inch. This params used for to save the image with a required DPI.
      Default is 300. 

    Examples
    ========
    from skmap.data import toy
    %matplotlib # to stop pouring out when calling the function. 

    data = toy.ndvi(gappy=True, verbose=True)
    figure = rdata.grid_plot(cmap='Spectral_r', legend_title="NDVI", img_title='date', save=True)
    
    # to view in jupyter notebook  
    figure.show()   
    """
    gridsize = math.ceil(math.sqrt(self.array.shape[2]))
    
    if v_minmax is None:
      vmin , vmax = np.nanquantile(self.array.flatten(), [.1, .9])
    else:
      vmin, vmax = v_minmax
    figsize = (5*gridsize, 5*gridsize) if figsize is None else figsize
    titles = self._get_titles(img_title)
    fig, axs = pyplot.subplots(ncols=gridsize, nrows=gridsize, figsize= figsize)
    
    for i, ax in enumerate(axs.flatten()):
      try:
        ax.imshow(self.array[:,:,i], vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(titles[i], fontsize= 3.5 * gridsize)
        ax.axis('off')
      except IndexError:
        ax.axis('off')

    pyplot.tight_layout()
    fig.subplots_adjust(bottom=0, top=1, left=0, right=.82)
    cbar_ax = fig.add_axes([0.83, 0.2, 0.02, 0.6])
    cbar_ax.tick_params(labelsize=3.5 * gridsize)
    cbar = fig.colorbar(
      pyplot.imshow(self.array[:,:,0], vmin=vmin, vmax=vmax, cmap=cmap),
      cax=cbar_ax
    ).set_label(label=legend_title, size = 4 * gridsize, weight = 'bold')
    
    if to_img is not None:
      fig.savefig(to_img, dpi=dpi, bbox_inches='tight')
      return to_img
    else:
      pyplot.close()
      return fig