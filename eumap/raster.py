'''
Functions to read/write raster data
'''
import gdal
import numpy
import numpy as np
import requests
import tempfile
import traceback
from uuid import uuid4

from typing import List, Union
from .misc import ttprint, find_files
from . import parallel

from pathlib import Path
from affine import Affine
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
    return np.iinfo(dtype).min
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

def _new_raster(fn_base_raster, fn_raster, data, spatial_win = None, 
  dtype = None, nodata = None):
  
  if (not isinstance(fn_raster, Path)):
    fn_raster = Path(fn_raster)
  
  fn_raster.parent.mkdir(parents=True, exist_ok=True)

  if len(data.shape) < 3:
    data = np.stack([data], axis=2)

  x_size, y_size, nbands = data.shape

  with rasterio.open(fn_base_raster, 'r') as base_raster:
    if dtype is None:
      dtype = base_raster.dtypes[0]

    if nodata is None:
      nodata = base_raster.nodata

    transform = base_raster.transform

    if spatial_win is not None:
      transform = rasterio.windows.transform(spatial_win, transform)
    
    return rasterio.open(fn_raster, 'w',
            driver='GTiff',
            height=x_size,
            width=y_size,
            count=nbands,
            dtype=dtype,
            crs=base_raster.crs,
            compress='LZW',
            transform=transform,
            nodata=nodata)

def read_rasters(
  raster_dirs:List = [],
  raster_files:List = [],
  raster_ext:str = 'tif',
  spatial_win:Window = None,
  dtype:str = 'float16',
  n_jobs:int = 4,
  data_mask:numpy.array = None,
  expected_img_size = None,
  try_without_window = False,
  verbose = False
):
  """ 
  Read raster files aggregating them into a single array. 
  Only the first band of each raster is read.

  The ``nodata`` value is replaced by ``np.nan`` in case of ``dtype=float*``, 
  and for ``dtype=*int*`` it's replaced by the the lowest possible value 
  inside the range (for ``int16`` this value is ``-32768``).

  :param raster_dirs: A list of folders where the raster files are located. The raster 
    are selected according to the ``raster_ext``.
  :param raster_files: A list with the raster paths. Provide it and the ``raster_dirs``
    is ignored.
  :param raster_ext: The raster file extension.
  :param spatial_win: Read the data according to the spatial window. By default is ``None``,
    reading all the raster data.
  :param dtype: Convert the read data to specific ``dtype``. By default it reads in 
    ``float16`` to save memory, however pay attention in the precision limitations for
    this ``dtype`` [1].
  :param n_jobs: Number of parallel jobs used to read the raster files.
  :param data_mask: A array with the same space dimensions of the read data, where
    all the values equal ``0`` are converted to ``np.nan``.
  :param expected_img_size: The expected size (space dimension) of the read data. 
    In case of error in reading any of the raster files, this is used to create a 
    empty 2D array. By default is ``None``, throwing a exception if the raster
    doesn't exists.
  :param try_without_window: First, try to read using ``spatial_win``, if fails
    try to read without it.
  :param verbose: Use ``True`` to print the reading progress.

  :returns: A 3D array, where the last dimension refers to the read files, and a list
    containing the read paths.
  :rtype: Tuple[Numpy.array, List[Path]]

  Examples
  ========

  >>> import rasterio
  >>> from eumap.raster import read_rasters
  >>> 
  >>> # EUMAP COG layers - NDVI seasons for 2000
  >>> raster_files = [
  >>>     'http://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_200003_eumap_epsg3035_v1.0.tif', # winter
  >>>     'http://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_200006_eumap_epsg3035_v1.0.tif', # spring
  >>>     'http://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_200009_eumap_epsg3035_v1.0.tif', # summer
  >>>     'http://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_200012_eumap_epsg3035_v1.0.tif'  # fall
  >>> ]
  >>> 
  >>> # Transform for the EPSG:3035
  >>> eu_transform = rasterio.open(raster_files[0]).transform
  >>> # Bounding box window over Wageningen, NL
  >>> window = rasterio.windows.from_bounds(left=4020659, bottom=3213544, right=4023659, top=3216544, transform=eu_transform)
  >>> 
  >>> data, _ = read_rasters(raster_files=raster_files, spatial_win=window, verbose=True)
  >>> print(f'Data shape: {data.shape}')

  References
  ==========

  [1] `Float16 Precision <https://github.com/numpy/numpy/issues/8063>`_

  """
  if len(raster_dirs) == 0 and len(raster_files) == 0:
    raise Exception('The raster_dirs and raster_files params can not be empty at same time.')

  if len(raster_files) == 0:
    raster_files = find_files(raster_dirs, f'*.{raster_ext}')

  if verbose:
    ttprint(f'Reading {len(raster_files)} raster files using {n_jobs} workers')

  def _read_raster(raster_pos):
    raster_file = raster_files[raster_pos]
    raster_ds, band_data = None, None
    nodata = None

    try:
      raster_ds = rasterio.open(raster_file)
      
      band_data = raster_ds.read(1, window=spatial_win)
      if band_data.size == 0 and try_without_window:
        band_data = raster_ds.read(1)

      band_data = band_data.astype(dtype)
      nodata = raster_ds.nodatavals[0]
    except:
      if spatial_win is not None:
        ttprint(f'ERROR: Failed to read {raster_file} window {spatial_win}.')
        band_data = np.empty((int(spatial_win.width), int(spatial_win.height)))
        band_data[:] = _nodata_replacement(dtype)

      if expected_img_size is not None:
        ttprint(f'Full nan image for {raster_file}')
        band_data = np.empty(expected_img_size)
        band_data[:] = _nodata_replacement(dtype)

    if data_mask is not None:
      if (data_mask.shape == band_data.shape):
        band_data[~data_mask] = np.nan
      else:
        ttprint(f'WARNING: incompatible data_mask shape {data_mask.shape} != {band_data.shape}')
    
    return raster_pos, band_data, nodata

  raster_data = {}
  args = [ (raster_pos,) for raster_pos in range(0,len(raster_files)) ]

  for raster_pos, band_data, nodata in parallel.job(_read_raster, args, n_jobs=n_jobs):
    raster_file = raster_files[raster_pos]

    if (isinstance(band_data, np.ndarray)):

      if nodata is not None:
        band_data[band_data == nodata] = _nodata_replacement(dtype)

    else:
      raise Exception(f'The raster {raster_file} was not found.')
    raster_data[raster_pos] = band_data

  raster_data = [raster_data[i] for i in range(0,len(raster_files))]
  raster_data = np.ascontiguousarray(np.stack(raster_data, axis=2))
  return raster_data, raster_files

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

  >>> from eumap.raster import read_auth_rasters
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
  
  def _read_raster(raster_files, url_pos, bands, username, password, dtype, nodata):
    
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

  args = [ (raster_files, url_pos, bands, username, password, dtype, nodata) for url_pos in range(0,len(raster_files)) ]
  
  raster_data = {}
  fn_base_raster = None
    
  for url_pos, data, ds_params in parallel.job(_read_raster, args, n_jobs=n_jobs):
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
  fn_base_raster:str,
  fn_raster_list:List,
  data:numpy.array,
  spatial_win:Window = None,
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

  :param fn_base_raster: The base raster path used to retrieve the 
    parameters ``(height, width, n_bands, crs, dtype, transform)`` for the 
    new rasters.
  :param fn_raster_list: A list containing the paths for the new raster. It creates 
    the folder hierarchy if not exists.
  :param data: 3D data array.
  :param spatial_win: Save the data considering a spatial window, even if the ``fn_base_rasters``
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
  >>> from eumap.raster import read_rasters, save_rasters
  >>> 
  >>> # EUMAP COG layers - NDVI seasons for 2019
  >>> raster_files = [
  >>>     'http://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201903_eumap_epsg3035_v1.0.tif', # winter
  >>>     'http://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201906_eumap_epsg3035_v1.0.tif', # spring
  >>>     'http://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201909_eumap_epsg3035_v1.0.tif', # summer
  >>>     'http://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201912_eumap_epsg3035_v1.0.tif'  # fall
  >>> ]
  >>> 
  >>> # Transform for the EPSG:3035
  >>> eu_transform = rasterio.open(raster_files[0]).transform
  >>> # Bounding box window over Wageningen, NL
  >>> window = rasterio.windows.from_bounds(left=4020659, bottom=3213544, right=4023659, top=3216544, transform=eu_transform)
  >>> 
  >>> data, _ = read_rasters(raster_files=raster_files, spatial_win=window, verbose=True)
  >>> 
  >>> # Save in the current execution folder
  >>> fn_raster_list = [
  >>>     './lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201903_wageningen_epsg3035_v1.0.tif',
  >>>     './lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201906_wageningen_epsg3035_v1.0.tif',
  >>>     './lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201909_wageningen_epsg3035_v1.0.tif',
  >>>     './lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201912_wageningen_epsg3035_v1.0.tif' 
  >>> ]
  >>> 
  >>> save_rasters(raster_files[0], fn_raster_list, data, spatial_win=window, verbose=True)

  """
  
  if len(data.shape) < 3:
    data = np.stack([data], axis=2)
  else:
    n_files = data.shape[2]
    if n_files != len(fn_raster_list):
      raise Exception(f'The data dimension {data.shape} is incompatible with the fn_raster_list size {len(fn_raster_list)}.')

  if verbose:
    ttprint(f'Writing {len(fn_raster_list)} raster files using {n_jobs} workers')

  args = [ \
    (fn_base_raster, fn_raster_list[i], data[:,:,i], spatial_win, dtype, 
      nodata, fit_in_dtype) \
    for i in range(0,len(fn_raster_list)) 
  ]

  result = []
  for fn_new_raster in parallel.job(write_new_raster, args, n_jobs=n_jobs):
    result.append(fn_new_raster)
    continue

  return result

def write_new_raster(
  fn_base_raster:str,
  fn_new_raster:str,
  data:numpy.array,
  spatial_win:Window = None,
  dtype:str = None,
  nodata = None,
  fit_in_dtype = False
):
  """ 
  Save an array to a raster file using as reference a base raster. GeoTIFF is
  the only output format supported. It always replaces the ``np.nan`` value 
  by the specified ``nodata``.

  :param fn_base_raster: The base raster path used to retrieve the 
    parameters ``(height, width, n_bands, crs, dtype, transform)`` for the 
    new raster.
  :param fn_new_raster: The path for the new raster. It creates the 
    folder hierarchy if not exists.
  :param data: 3D data array where the last dimension is the number of bands for the new raster.
    For 2D array it saves only one band.
  :param spatial_win: Save the data considering a spatial window, even if the ``fn_base_rasters``
    refers to a bigger area. For example, it's possible to have a base raster covering the whole
    Europe and save the data using a window that cover just part of Wageningen. By default is 
    ``None`` saving the raster data in position ``0, 0`` of the raster grid.
  :param dtype: Convert the data to a specific ``dtype`` before save it. By default is ``None`` 
    using the same ``dtype`` from the base raster.
  :param nodata: Use the specified value as ``nodata`` for the new raster. By default is ``None`` 
    using the same ``nodata`` from the base raster.
  :param fit_in_dtype: If ``True`` the values outside of ``dtype`` range are truncated to the minimum
    and maximum representation. It's also change the minimum and maximum data values, if they exist, 
    to avoid overlap with ``nodata`` (see the ``_fit_in_dtype`` function). For example, if 
    ``dtype='uint8'`` and ``nodata=0``, all data values equal to ``0`` are re-scaled to ``1`` in the
    new rasters.

  :returns: The path of the new raster.
  :rtype: Path

  Examples
  ========

  >>> import rasterio
  >>> from eumap.raster import read_rasters, save_rasters
  >>> 
  >>> # EUMAP COG layers - NDVI seasons for 2019
  >>> raster_files = [
  >>>     'http://s3.eu-central-1.wasabisys.com/eumap/lcv/lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201903_eumap_epsg3035_v1.0.tif', # winter
  >>> ]
  >>> 
  >>> # Transform for the EPSG:3035
  >>> eu_transform = rasterio.open(raster_files[0]).transform
  >>> # Bounding box window over Wageningen, NL
  >>> window = rasterio.windows.from_bounds(left=4020659, bottom=3213544, right=4023659, top=3216544, transform=eu_transform)
  >>> 
  >>> data, _ = read_rasters(raster_files=raster_files, spatial_win=window, verbose=True)
  >>> 
  >>> # Save in the current execution folder
  >>> fn_new_raster = './lcv_ndvi_landsat.glad.ard_p50_30m_0..0cm_201903_wageningen_epsg3035_v1.0.tif',
  >>> 
  >>> write_new_raster(raster_files[0], fn_new_raster, data, spatial_win=window)

  """

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