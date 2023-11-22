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

from shapely.geometry import box,shape

from typing import List, Union, Callable
from skmap.misc import ttprint, _eval, update_by_separator, date_range, new_memmap, del_memmap, ref_memmap, load_memmap, concat_memmap
from skmap.misc import vrt_warp
from skmap import SKMapGroupRunner, SKMapRunner, SKMapBase, parallel

from dateutil.relativedelta import relativedelta
from datetime import datetime
from minio import Minio

from pathlib import Path
import rasterio
from rasterio.windows import Window, from_bounds
from pystac.item import Item

import bottleneck as bn

from IPython.display import HTML
from tempfile import TemporaryDirectory
from io import BytesIO
from base64 import encodebytes, b64decode
from uuid import uuid4
from PIL import Image
from contextlib import ExitStack
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib._animation_data import JS_INCLUDE, STYLE_INCLUDE, DISPLAY_TEMPLATE
from copy import deepcopy

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
  try_without_window, scale, gdal_opts, overview, verbose):

  for key in gdal_opts.keys():
    gdal.SetConfigOption(key,gdal_opts[key])

  raster_file = raster_files[raster_idx]
  ds, band_data = None, None
  nodata = None

  try:
    ds = rasterio.open(raster_file)
    
    if overview is not None:
      overviews = ds.overviews(band)
      if overview in overviews:
          ds.read(band, out=array_mm[:,:,raster_idx], out_dtype=dtype, out_shape=(1, math.ceil(ds.height // overview), math.ceil(ds.width // overview)), window=window)
      else:
        ds.read(band, out=array_mm[:,:,raster_idx],  out_dtype=dtype, window=window)
    else:
      ds.read(band, out=array_mm[:,:,raster_idx],  out_dtype=dtype, window=window)

    #if band_data.size == 0 and try_without_window:
    #  band_data = ds.read(band, out=array_mm[:,:,raster_idx])

    #band_data = band_data.astype(dtype)
    nodata = ds.nodatavals[0]
    data_exists = True
    #print(f"Data was read: {raster_file}")
  except Exception as ex:
    ttprint(f"Exception: {ex}")
    #traceback.i(print)_exc()
    
    if window is not None:
      if verbose:
        ttprint(f'ERROR: Failed to read {raster_file} window {window}')
      array_mm[:,:,raster_idx] = np.empty((int(window.height), int(window.width)))
      array_mm[:,:,raster_idx] = _nodata_replacement(dtype)

    if expected_shape is not None:
      if verbose:
        ttprint(f'Full nan image for {raster_file}')
      array_mm[:,:,raster_idx] = np.empty(expected_shape)
      array_mm[:,:,raster_idx] = _nodata_replacement(dtype)

  if data_exists:
    if data_mask is not None:

      if len(data_mask.shape) == 3:
        data_mask = data_mask[:,:,0]

      if (data_mask.shape == band_data.shape):
        array_mm[:,:,raster_idx][np.logical_not(data_mask)] = np.nan
      else:
        ttprint(f'WARNING: incompatible data_mask shape {data_mask.shape} != {band_data.shape}')
    
    if nodata is not None:
      array_mm[:,:,raster_idx][array_mm[:,:,raster_idx] == nodata] = _nodata_replacement(dtype)

  if scale != 1.0:
    array_mm[:,:,raster_idx] = array_mm[:,:,raster_idx] * scale

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
  raster_file:str,
  ref_array,
  i:int,
  spatial_win:Window = None,
  dtype:str = None,
  nodata = None,
  fit_in_dtype = False,
  on_each_outfile:Callable = None
):

  #if len(data.shape) < 3:
  #  data = np.stack([data], axis=2)

  #_, _, nbands = data.shape

  array = load_memmap(**ref_array)

  with _new_raster(fn_base_raster, raster_file, array[:,:,i], spatial_win, dtype, nodata) as new_raster:

    band_dtype = new_raster.dtypes[0]

    if fit_in_dtype:
      array[:,:,i] = _fit_in_dtype(array[:,:,i], band_dtype, new_raster.nodata)

    array[:,:,i][np.isnan(array[:,:,i])] = new_raster.nodata
    new_raster.write(array[:,:,i].astype(band_dtype), indexes=1)

  if on_each_outfile is not None:
    on_each_outfile(raster_file)

  return raster_file

def read_rasters(
  raster_files:Union[List,str] = [],
  band = 1,
  window:Window = None,
  bounds:[] = None,
  dtype:str = 'float32',
  n_jobs:int = 8,
  data_mask:numpy.array = None,
  scale:float = 1.0,
  expected_shape = None,
  try_without_window = False,
  gdal_opts:dict = {},
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

  ds = rasterio.open(raster_files[-1])
  if bounds is not None and len(bounds) == 4:
    bounds = shape(rasterio.warp.transform_geom(
        src_crs='EPSG:4326',
        dst_crs=ds.crs,
        geom=box(*bounds),
    )).bounds
    window = from_bounds(*bounds, ds.transform).round_lengths()
    if verbose:
      ttprint(f'Transform {bounds} into {window}')

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

  array_mm = new_memmap(dtype, shape=(height, width, len(raster_files)))

  args = [ 
    (raster_idx, raster_files, array_mm, band, window, dtype, data_mask, 
      expected_shape, try_without_window, scale, gdal_opts, overview, verbose) 
    for raster_idx in range(0,len(raster_files)) 
  ]
  
  for raster_idx, data_exists in parallel.job(_read_raster, args, n_jobs=n_jobs, 
    joblib_args={
      'backend': 'threading', 
      'pre_dispatch': math.ceil(n_jobs / 3), 
      'batch_size': math.floor(len(args) / n_jobs),
      'return_as': 'generator'
    }):
    
      if (not data_exists):
        raster_file = raster_files[raster_idx]
        raise Exception(f'The raster {raster_file} not exists')
  
  return array_mm

  #if not keep_memmap:
  #  return del_memmap(array_mm, True)
  #else:
  #  return array_mm

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
  array,
  window:Window = None,
  bounds:[] = None,
  dtype:str = None,
  nodata = None,
  array_idx:List = [],
  fit_in_dtype:bool = False,
  n_jobs:int = 8,
  on_each_outfile:Callable = None,
  verbose:bool = False
):
  """
  Save a 3D array in multiple raster files using as reference one base raster.
  The last dimension is used to split the array in different rasters. GeoTIFF is
  the only output format supported. It always replaces the ``np.nan`` value
  by the specified ``nodata``.

  :param base_raster: The base raster path used to retrieve the
    parameters ``(height, width, n_bands, crs, dtype, transform)`` for the
    new rasters.
  :param raster_files: A list containing the paths for the new raster. It creates
    the folder hierarchy if not exists.
  :param array: 3D data array.
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

  #if len(data.shape) < 3:
  #  data = np.stack([data], axis=2)
  
  if (len(array_idx) == 0):
    array_idx = list(range(0, array.shape[-1]))

  if len(array_idx) != len(raster_files):
    raise Exception(f'The array shape {array.shape} is incompatible with the raster_files size {len(raster_files)}.')

  ds = rasterio.open(base_raster)
  if bounds is not None and len(bounds) == 4:
    bounds = shape(rasterio.warp.transform_geom(
        src_crs='EPSG:4326',
        dst_crs=ds.crs,
        geom=box(*bounds),
    )).bounds
    window = from_bounds(*bounds, ds.transform).round_lengths()
    if verbose:
      ttprint(f'Transform {bounds} into {window}')

  if verbose:
    ttprint(f'Saving {len(raster_files)} raster files using {n_jobs} workers')

  ref_array = ref_memmap(array)

  args = [ \
    (base_raster, raster_file, ref_array, i, window, dtype,
      nodata, fit_in_dtype, on_each_outfile) \
    for raster_file, i in zip(raster_files, array_idx)
  ]

  batch_size = math.floor(len(args) / n_jobs)
  if batch_size <= 0:
    batch_size = 'auto'

  out_files = []
  for out_raster in parallel.job(_save_raster, args, n_jobs=n_jobs, 
    joblib_args={
      'pre_dispatch': n_jobs, 
      'batch_size': batch_size
    }):
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
        rows.append([group, raster_files[group], 1, None, None])
      else:
        for r in raster_files[group]:
          if isinstance(r, tuple):
            if (len(r) == 2):
              rows.append([group, r[0], r[1], None, None])
            elif (len(r) == 4):
              rows.append([group, r[0], r[1], r[2], r[3]])
            else:
              raise Exception(f'Wrong tuple size {len(r)}. Please provide 2 or 4 size tuple.')
          else:
            rows.append([group, r, 1, None, None])

    self.info = DataFrame(rows, columns=[RasterData.GROUP_COL, RasterData.PATH_COL, RasterData.BAND_COL, 
      RasterData.START_DT_COL, RasterData.END_DT_COL])

    self.info[RasterData.TEMPORAL_COL] = self.info.apply(lambda r: RasterData.PLACEHOLDER_DT in str(r[RasterData.PATH_COL]), axis=1)
    self.info[RasterData.NAME_COL] = self.info.apply(lambda r: Path(str(r[RasterData.PATH_COL]).split('?')[0]).stem if not r[RasterData.TEMPORAL_COL] else None, axis=1)
    
    self.date_args = {}
    self._active_group = None

    has_date = ~self.info[RasterData.START_DT_COL].isnull().any()

    if has_date:
      self.info[RasterData.TEMPORAL_COL] = True
      for g in self.info[RasterData.GROUP_COL].unique():
        self.date_args[g] = {
          'date_style': 'interval',
          'date_format': '%Y%m%d',
          'ignore_29feb': True
        }

    self.info.reset_index(drop=True, inplace=True)

  def _new_info_row(self, 
    raster_file:str,
    name:str,
    group:str = None,
    dates:list = [],
    date_format = None,
    date_style = None,
    ignore_29feb = True
  ):

    row = {}

    if group is None or 'default' in group:
      group = 'default'

    row[RasterData.PATH_COL] = raster_file
    row[RasterData.NAME_COL] = name
    row[RasterData.GROUP_COL] = group
    row[RasterData.BAND_COL] = 1

    if self._active_group is not None:
      if date_style is None:
        date_style = self.date_args[self._active_group]['date_style']
      if date_format is None:
        date_format = self.date_args[self._active_group]['date_format']

      self.date_args[group] = self.date_args[self._active_group]
    else:
      self.date_args[group] = {
        'date_style': date_style,
        'date_format': date_format,
        'ignore_29feb': ignore_29feb
      }

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

  def from_stac_items(
    stac_items:List[Item], 
    bands:List[str] = None,
    to_crs = rasterio.crs.CRS.from_epsg(4326),
    spatial_res = None,
    resamp_method = 'near',
    n_jobs:int = 10,
    verbose = False
  ):

    all_bands = list(stac_items[0].assets.keys())
    if bands is None:
      if verbose: 
        ttprint(f'Reading band {all_bands[0]} from {all_bands}')
      bands = [ all_bands[0] ]

    stac_info = []
    stac_href = {}

    for i in stac_items:
      for band in  i.assets.keys():
        if band in bands:
          href = i.assets[band].href
          if href not in stac_href:
            stac_href[href] = False
            stac_info.append({
              'href': i.assets[band].href,
              'band': band,
              'date': i.datetime.replace(tzinfo=None)
            })

    stac_info = pd.DataFrame(stac_info)

    raster_file, vrt_files = vrt_warp(stac_info['href'], dst_crs=to_crs.to_wkt(), 
      tr=spatial_res, r_method=resamp_method, return_input_files=True
    )
    vrt_info = pd.DataFrame({ 'href':raster_file, 'vrt':vrt_files })
    stac_info = stac_info.merge(vrt_info, on='href', how='inner')

    groups = {}
    for g, row in stac_info.groupby('band'):
      if g not in groups:
        groups[g] = []

      groups[g] += [ (v, 1, d, d) for v, d in zip(row['vrt'], row['date']) ]

    return RasterData(groups, verbose=verbose)

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

    to_drop = []
    to_add = []

    for _group, ginfo in self.info.groupby(RasterData.GROUP_COL):

      if len(group) > 0 and _group not in group:
        continue

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
      
      to_drop.append(ginfo.index)
      to_add.append(ginfo)

    for idx in to_drop:
      self.info = self.info.drop(index=idx)
    
    self.info = pd.concat([ self.info ] + to_add).reset_index(drop=True)

    return self

  def _base_raster(self):

    for filepath in list(self.info[RasterData.PATH_COL]):
      if 'http' in filepath:
        res = requests.head(filepath)
        if (res.status_code == 200):
          return True
      else:
        if Path(filepath).exists():
          return True
        
  def read(self,
    window:Window = None,
    bounds:list = None,
    dtype:str = 'float32',
    expected_shape = None,
    overview:int = None,
    n_jobs:int = 4,
    scale:float = 1,
    gdal_opts:dict = {}
  ):

    self.window = window
    self.bounds = bounds
    
    data_mask = None
    if self.raster_mask is not None:
      self._verbose(f"Masking {self.raster_mask_val} values considering {Path(self.raster_mask).name}")
      data_mask = read_rasters([self.raster_mask], window=window, overview=overview, gdal_opts=gdal_opts)
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
    self.base_raster = self._base_raster()
    raster_files = []

    for band, rows in self.info.groupby(RasterData.BAND_COL):
      raster_files += [ Path(r) for r in rows[RasterData.PATH_COL] ]
      
    self._verbose(
        f"RasterData with {len(raster_files)} rasters" 
      + f" and {len(self.info[RasterData.GROUP_COL].unique())} group(s)" 
    )

    self.array = read_rasters(
      raster_files, band=band, window=self.window, 
      bounds=bounds, data_mask=data_mask,
      dtype=dtype, expected_shape=expected_shape,
      n_jobs=n_jobs, overview=overview, scale=scale,
      gdal_opts=gdal_opts, verbose=self.verbose
    )
    self._verbose(f"Read array shape: {self.array.shape}")

    return self

  def run(self, 
    process:SKMapRunner,
    group:[list,str] = [],
    outname:str = None,
    drop_input:bool = False
  ):

    if isinstance(process, SKMapGroupRunner):
      return self._group_run(process, group, outname, drop_input)
    else:
      
      process_name = process.__class__.__name__
      
      start = time.time()
      self._verbose(f"Running {process_name}"
        + f" on {self.array.shape}")

      kwargs = {'rdata': self}
      if outname is not None:
        kwargs['outname'] = outname
      
      new_array, new_info = process.run(**kwargs)

      if new_info.shape[0] > 0:
        new_array.insert(0, self.array)
        self.array = concat_memmap(new_array, axis=2)
        self.info = pd.concat([self.info, new_info])
        self.info.reset_index(drop=True, inplace=True)
      
      self._verbose(f"Execution"
        + f" time for {process_name}: {(time.time() - start):.2f} segs")

      return self

  def _group_run(self, 
    process:SKMapGroupRunner,
    group:[list,str] = [],
    outname:str = None,
    drop_input:bool = False
  ):
    
    if isinstance(group, str):
      group = [ group ]

    to_drop = []
    to_add_arr = []
    to_add_info = []

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
        to_drop.append(ginfo.index)
        
      to_add_arr.append(new_array)
      to_add_info.append(new_info)
      
      self._verbose(f"Execution"
        + f" time for {process_name}: {(time.time() - start):.2f} segs")

      self._active_group = None

    idx_list = []
    for idx in to_drop:
      self.info = self.info.drop(idx)
      idx_list += list(idx)
    
    self.array = np.delete(self.array, idx_list, axis=-1) 

    self.array = np.concatenate( [self.array] + to_add_arr, axis=-1)
    self.info = pd.concat( [self.info] + to_add_info)
    self.info.reset_index(drop=True, inplace=True)

    return self

  def drop(self, group):

    if isinstance(group, str):
      group = [ group ]

    self._verbose(f"Dropping data and info for groups: {group}")
    idx = self.info[self.info[RasterData.GROUP_COL].isin(group)].index
    self.array = np.delete(self.array, idx, axis=-1) 
    self.info = self.info.drop(idx)
    self.info.reset_index(drop=True, inplace=True)

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
      if 'http' in str(path):
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
      self.array, array_idx = info.index,
      window=self.window, bounds=self.bounds,
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

  def __del__(self):
    print("Deleting")
    del_memmap(self.array)
  
  def __exit__(self):
    self.__del__()
  
  def _get_titles(self, img_title, bands):
    f_arr = self.filter(f"group=={bands}")
    
    if img_title == 'date':
      titles = list(f_arr.info['start_date'].astype(str) + ' - ' + f_arr.info['end_date'].astype(str))
    elif img_title == 'index':
      titles = [str(i) for i in range(f_arr.info.shape[0])]
    elif img_title == 'name':
      titles = f_arr.info['name'].to_list()
      #titles = []
      # n = 20
      # for name in list(f_arr.info['name']):
      #   titles.append('\n'.join(name[i:i+n] for i in range(0, len(name), n)))
    else:
      titles = [''] * f_arr.info.shape[0]
    return titles

  def point_query(self, 
                  x:list, 
                  y:list, 
                  cols:int=3,
                  titles:list = None, 
                  label_xaxis:str='index', 
                  return_data:bool=False,
                  ):
    """
    Makes point queries on dataset and provide plots and data

    :param x: longitude value(s) of the given point(s)
    :param y: latitude value(s) of the given point(s)
    :param cols: column count of the desired layout. Default is 3.
    :param titles: list of the titles that will be placed on top of the each graph
    :param label_xaxis: labels of the x axes. it could be `index`, `name`,`date` or None.
    :param return_data: If the user wants to access the data sampled from rasters, this
      needs to be set to True. Default is False 

    Examples
    ========
    import geopandas as gpd
    from skmap.data import toy
    rasterdata = toy.ndvi_rdata(gappy=False)
    points = gpd.read_file('./skmap/data/toy/samples/samples.gpkg')
    rdata.point_query(x=points.geometry.x.to_list(), y=points.geometry.y.to_list() , label_xaxis='index', cols=3, titles=points.label)
    """
    df = pd.DataFrame()
    df['x'], df['y'], df['title'] = x, y, titles
    bbox = rasterio.open(self._base_raster()).bounds
    # filtering points based on the bounds of the base raster
    df = df[(bbox.left <= df['x']) & (df['x'] <= bbox.right) & (bbox.bottom <= df['y']) & (df['y'] <= bbox.top)]

    with rasterio.open(self._base_raster()) as src:
      row_id, col_id = rasterio.transform.rowcol(src.transform, df.x, df.y)
    df['data']= np.array(self.array[row_id,col_id]).tolist()
    # if data is required no need to create figures
    if return_data:
      return df.data.to_numpy()
    
    labels_x = self._get_titles(label_xaxis)
    fig, axs = pyplot.subplots(ncols=cols, nrows=math.ceil(len(x)/cols), figsize=(6 * cols, 2 * math.ceil(len(x)/cols)), sharex=True, sharey=True)
    mgc = df.shape[0] # maximum graph count
    for i, ax in enumerate(axs.flatten()):
        if i < mgc:
          ax.plot(labels_x, df.data[i], '-o', markersize=4, color='blue', lw=1)
          ax.set_title(titles[i], fontsize=10)
          ax.tick_params(axis='x',rotation=90)
        else:
          ax.axis('off')
    pyplot.tight_layout()
    pyplot.close()
    return fig
  
  def _vminmax(self, vmm, arr):
    """
    To check and calculate the boundaries of the data. If the bounds are supplied
    it will return it, If not function will return the 1 and 99% of the data as bounds.
    :param vmm: supplied min/max bounds of data
    :param arr: the data will be used to generate a image  
    """
    if vmm[0]:
      return vmm
    return np.nanquantile(arr.flatten(), [.02, .98])
  
  def _op_io(self, figure):
    """
    converts figure to image and ascii representation of it to use with
    HTML embeded animation.
    :param figure: matplotlib figure object
    """
    buffer = BytesIO()
    figure.savefig(buffer, format='png', bbox_inches='tight')
    img64 = encodebytes(buffer.getvalue()).decode('ascii')
    return img64

  def _percent_clip(self, arr):
    """
    To calculate and scale the band upper and lower limits to generate a composite
    image from 3 bands. returns the scaled data
    :param arr: the data usually single band data in np.array format.
    """
    return (arr - np.nanpercentile(arr, 1)) / (np.nanpercentile(arr,99)-np.nanpercentile(arr,1))
  
  def _mutate_baseshot(self, img, arr, titletext, textfontsize):
    """
    takes imshow generated mock image copies it and replaces the nested image data.
    :param img: the mock image, generated with pyplot.imshow
    :param arr: the scaled data for the frame
    :param title_params: it is a dict. It will be used for the title generation on the frame.
    """
    c_img = deepcopy(img)
    c_img.set_data(arr)
    if titletext:
      c_img._axes.set_title(label=titletext,fontdict=dict(fontsize=textfontsize), pad=1)
    return c_img.get_figure()
   
  def _gen_baseshot(self, arr, scaling:int=1, img_style:dict=None, cbar_props:dict=None, composite=False):
    #base figure with predefined style
    
    # no axis labels
    tick_params=dict(left=False, labelleft=False, labelbottom=False, bottom=False)
    
    # scaling the figsize based on the passed array shape
    # the base figsize is 3.15 inc = 8cm almost half short side of a A4 page 
    fig, ax = pyplot.subplots()
    rc,cc = arr.shape[:2]
    fig.set_size_inches(scaling * 3.15, scaling * 3.15*rc/cc)
    
    # generation of basedata based on the array shape
    basedata = np.zeros(rc*cc).reshape(rc,cc)
    if composite:
      basedata = np.zeros(rc*cc*3).reshape(rc,cc,3)

    # crafting the base image
    ax.tick_params(**tick_params)
    ax.margins(x=0)
    
    if img_style:
      baseimg = ax.imshow(basedata, **img_style) # img_style = dict(vmin=, vmax=, cmap=)
    else:
      baseimg = ax.imshow(basedata)
    #if there will be a colorbar there will be a colorbar
    if cbar_props: # cbar_props is dict(label='text')
      div = make_axes_locatable(ax)
      pyplot.colorbar(
        baseimg, 
        orientation='horizontal', label=cbar_props['label'],
        cax = div.append_axes('bottom', size='3%', pad = 0.05)
      )
    pyplot.close()
    return baseimg
  
  def _band_manage(self, groups):
    """
    to structure the band(s) data based on the provided band information.
    either single or multiple band data.
    :params groups: list of band names.
    """

    if len(groups) == 1: # single band raster
      arr = self.filter(f"group=={groups}").array
    elif len(groups) == 3: # composite
      arr = []
      band1 = self.filter(f"group=={groups}[0]", return_array=True)
      band2 = self.filter(f"group=={groups}[1]", return_array=True)
      band3 = self.filter(f"group=={groups}[2]", return_array=True)

      alpha = np.ones(band3.shape)
      mask = np.any(np.isnan(np.stack([band1, band2, band3], axis=-1)), axis=-1)
      alpha[mask] = 0

      for i in range(band1.shape[2]):
        arr.append(
          np.stack([
            np.clip(self._percent_clip(band1[:,:,i]),0,1),
            np.clip(self._percent_clip(band2[:,:,i]),0,1),
            np.clip(self._percent_clip(band3[:,:,i]),0,1),
            alpha[:,:,i],
          ], axis=2)
        )
    else:
      raise Exception("""The band count should either be one or three. 
                      Current plotting capabilites are limited to single 
                      or composite image generation.""")
    return arr
      
  def plot(
      self,
      groups:list = None,
      cmap:str = 'Spectral_r',
      cbar_title:str = None, 
      img_title_text:str or list = "index",
      img_title_fontsize:int = 10,
      vminmax:tuple = (None,None),
      to_img:str = None,
      dpi:int = 100,
      layout_col: int = 4
  ):
    """
      Generates a grid plot to view and save with a colorscale with a desired layout.
      :param cmap                 : This sets the colorscale with given matplotlib.colormap. Default is Spectral_r
      :param cbar_title           : This sets the colorbar title if the cbar exists in the plot. Default is None.
      :param img_title_text       : This sets the image titles that will be display on top of the each image. Default is `index`.
      :param img_ltitle_fontsize  : This sets the fontsize of the image label which will be on top of the image. Default is 10.
      :param v_minmax             : This sets the loower and upper limits of the data that will be plot and the colorbar. Default is None and will be calculated on he fly.
      :param groups                : This used for to generate composite plot. Pass one or tree group names (groups) which will be used to generate. Default is None.
      :param to_img               : This sets the directory adn the format of the file where the generated image will be saved. Default is None.
      :param dpi                  : dot per inch value to save the figure. If the `to_img` param provided
      :param layout_col           : This controls the column count that will be used in the grid plot. Default is 3.
    """
    if not groups:
      groups = [self.info.group.to_list()[0]]

    arr = self._band_manage(groups=groups)

    if isinstance(img_title_text, str):
      img_title_text = self._get_titles(img_title_text, groups)
    
    if len(groups) == 3:
      img_cnt = len(arr)
      composite=True
      baseimg = self._gen_baseshot(arr[0][:,:,0])
    elif len(groups) == 1:
      img_cnt = arr.shape[2]
      composite=False
      vminmax = self._vminmax(vminmax, arr)
      baseimg = self._gen_baseshot(arr[:,:,0])

    if img_cnt < layout_col:
      layout_col = img_cnt

    layout_row = math.ceil(img_cnt/layout_col)

    set_h = baseimg.get_size()[0] / baseimg.get_figure()._dpi
    set_w = baseimg.get_size()[1] / baseimg.get_figure()._dpi
    if set_w > set_h:
      set_w = set_w *3.15 / set_h
      set_h = 3.15
    else:
      set_h = set_h *3.15 / set_w
      set_w = 3.15
    grd_fig, grd_axs = pyplot.subplots(
      nrows=layout_row, ncols=layout_col,
      gridspec_kw=dict(wspace=0.1, hspace=0.1),
      figsize=(
        set_w * layout_col + (layout_col-1) * 0.1,
        set_h * layout_row + (layout_row-1) * 0.1 #+ 1
      ),
    )
    
    def _preprocess(arr_, ind, composite):
      if composite:
        return np.flipud(arr_[ind])
      else:
        return np.flipud(arr_[:,:,ind])
      
    matrix_params = dict(vmin=vminmax[0], vmax=vminmax[1], cmap=cmap)

    def gen_pane(ind, arr, ax, composite, matrix_params, img_title_text, img_title_fontsize):
      try:
        ax.pcolorfast(_preprocess(arr, ind, composite=composite), **matrix_params)
        ax.set_title(img_title_text[ind], fontsize=img_title_fontsize, pad=1)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
      except IndexError:
        ax.axis('off')
    try:
      for i, ax in enumerate(grd_axs.flatten()):
        gen_pane(i, arr, ax, composite, matrix_params, img_title_text, img_title_fontsize)
      
    except AttributeError:
      gen_pane(0, arr, grd_axs, composite, matrix_params, img_title_text, img_title_fontsize)
    pyplot.close()

    if not composite:
      grd_fig.subplots_adjust(
        left=0, right=1,
        bottom=0,top=1
      )
      w,h = grd_fig.get_size_inches()
      cbar_ax= grd_fig.add_axes([0.1, 1+(3.15/h) * 0.1, 0.8, 0.16/h])
      cbar_ax = grd_fig.colorbar(
        pyplot.imshow(arr[:,:,0], vmin= vminmax[0], vmax=vminmax[1], cmap=cmap),
        orientation = 'horizontal',
        cax = cbar_ax,
        ticklocation='top'
      ).set_label(label = cbar_title)
      pyplot.tight_layout()
      pyplot.close()

    if to_img:
      grd_fig.savefig(to_img, format=f"{to_img.split('.')[-1]}", dpi=dpi, bbox_inches='tight')
    return grd_fig
  
  def animate(
      self,
      cmap:str = 'Spectral_r',
      groups:list = None,
      scaling:float = 2,
      cbar_title:str = None,
      img_title_text:str or list = 'index',
      img_title_fontsize:int = 10,
      vminmax:tuple = (None, None),
      interval:int = 250,
      to_gif:str = None,
      n_jobs:int = 4
  ):
    """
    Generates an animation with the given band(s) and saves it.
    :param cmap: colormap name that will derived from `matplotlib.colormaps()`
    :param groups: this is used for to select the band(s) or to generate a composite images, 
      that will be used as animation frame. Default is None but it will select the first band on RasterData.
    :param scaling: scaling can be used to increase/decrease the frame size. Default is 2.
    :param cbar_title: 
    """
    
    if not groups:
      groups = [self.info.group.to_list()[0]]
    arr = self._band_manage(groups=groups)
    
    if isinstance(img_title_text, str):
      img_title_text = self._get_titles(img_title_text, groups)
    
    if len(groups) == 3:
      img_cnt = len(arr)
      baseimg = self._gen_baseshot(arr=arr[0][:,:,0], scaling=scaling, composite=True)
      args = [(baseimg,arr[i], img_title_text[i], img_title_fontsize) for i in range(img_cnt)]
    elif len(groups) == 1:
      img_cnt = arr.shape[2]
      vminmax = self._vminmax(vminmax, arr)
      baseimg = self._gen_baseshot(arr=arr[:,:,0],
                                   scaling=scaling,
                                   img_style=dict(vmin=vminmax[0], vmax=vminmax[1], cmap=cmap),
                                   cbar_props=dict(label=cbar_title),
                                   composite=False
                                   )
      args = [(baseimg, arr[:,:,i], img_title_text[i], img_title_fontsize) for i in range(img_cnt)]

    int_fig = [f for f in parallel.job(self._mutate_baseshot, args, n_jobs=n_jobs)]
    int_img = [j for j in parallel.job(self._op_io, [(fig,) for fig in int_fig], n_jobs=n_jobs)]
    
    
    if to_gif is not None:
      with ExitStack() as stack:
        imgs = (
          stack.enter_context(Image.open(BytesIO(b64decode(img))))
          for img in int_img
        )
        img = next(imgs)
        img.save(to_gif, format='GIF', append_images=imgs, save_all=True, duration=interval, loop=0)
    
    template = '  frames[{0}] = "data:image/{1};base64,{2}"\n'
    embedded_frames = "\n" + "".join(
      template.format(i, 'png', imgdata.replace("\n","\\\n"))
      for i, imgdata in enumerate(int_img)
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
          Nframes=img_cnt,
          fill_frames = embedded_frames,
          interval = interval,
          **mode_dict
        ))
      html_rep = path.read_text()
    return HTML(html_rep)
