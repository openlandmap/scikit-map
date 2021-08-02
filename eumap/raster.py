'''
Functions to read/write raster data
'''
import gdal
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

def _fit_in_dtype(data, dtype, nodata):
  
  if dtype in ('uint8', 'uint8', 'int', 'int'):
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

def save_rasters(
                    fn_base_raster,
                    fn_raster_list,
                    data,
                    spatial_win = None,
                    data_type = None,
                    raster_format = 'GTiff',
                    nodata = None, 
                    fit_in_data_type = False,
                    n_jobs = 4,
                    verbose=False
                  ):
  
  if len(data.shape) < 3:
    data = np.stack([data], axis=2)
  else:
    n_files = data.shape[2]
    if n_files != len(fn_raster_list):
      raise Exception(f'The data dimension {data.shape} is incompatible with the fn_raster_list size {len(fn_raster_list)}.')

  if verbose:
    ttprint(f'Writing {len(fn_raster_list)} raster files using {n_jobs} workers')

  args = [ \
    (fn_base_raster, fn_raster_list[i], data[:,:,i], spatial_win, data_type, 
      raster_format, nodata, fit_in_data_type) \
    for i in range(0,len(fn_raster_list)) 
  ]

  result = []
  for fn_new_raster in parallel.job(write_new_raster, args, n_jobs=n_jobs):
    result.append(fn_new_raster)
    continue

  return result

def create_raster(
                    fn_base_raster,
                    fn_raster,
                    data,
                    spatial_win = None,
                    data_type = None,
                    raster_format = 'GTiff',
                    nodata = None
                  ):
  
  fn_raster.parent.mkdir(parents=True, exist_ok=True)

  if len(data.shape) < 3:
    data = np.stack([data], axis=2)

  x_size, y_size, nbands = data.shape

  with rasterio.open(fn_base_raster, 'r') as base_raster:
    if data_type is None:
      data_type = base_raster.dtypes[0]

    if nodata is None:
      nodata = base_raster.nodata

    transform = base_raster.transform

    if spatial_win is not None:
      transform = rasterio.windows.transform(spatial_win, transform)
    
    return rasterio.open(fn_raster, 'w',
            driver=raster_format,
            height=x_size,
            width=y_size,
            count=nbands,
            dtype=data_type,
            crs=base_raster.crs,
            compress='LZW',
            transform=transform,
            nodata=nodata)

def write_new_raster(
                      fn_base_raster,
                      fn_new_raster,
                      data,
                      spatial_win = None,
                      data_type = None,
                      raster_format = 'GTiff',
                      nodata = None,
                      fit_in_data_type = False
                    ):

  if len(data.shape) < 3:
    data = np.stack([data], axis=2)

  _, _, nbands = data.shape

  with create_raster(fn_base_raster, fn_new_raster, data, spatial_win, data_type, raster_format, nodata) as new_raster:

    for band in range(0, nbands):
      
      band_data = data[:,:,band]
      band_dtype = new_raster.dtypes[band]
      
      if fit_in_data_type:
        band_data = _fit_in_dtype(band_data, band_dtype, new_raster.nodata)

      band_data[np.isnan(band_data)] = new_raster.nodata
      new_raster.write(band_data.astype(band_dtype), indexes=(band+1))

  return fn_new_raster

def read_rasters_remote(
                  url_list:List = [],
                  bands = None,
                  dtype = 'float16',
                  username = None,
                  password = None,
                  n_jobs = 4,
                  verbose = False,
                  return_base_raster = False,
                  driver = 'GTiff',
                  nodata = None
                ):
  
  if verbose:
    ttprint(f'Reading {len(url_list)} remote raster files using {n_jobs} workers')

  def _read_raster(url_list, url_pos, bands, username, password, dtype, nodata):
    url = url_list[url_pos]
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
            data[data == nodata] = np.nan

          nbands, x_size, y_size = data.shape

          ds_params = {
            'driver': driver,
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

  args = [ (url_list, url_pos, bands, username, password, dtype, nodata) for url_pos in range(0,len(url_list)) ]
  
  raster_data = {}
  fn_base_raster = None
    
  for url_pos, data, ds_params in parallel.job(_read_raster, args, n_jobs=n_jobs):
    url = url_list[url_pos]
  
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
  for i in range(0,len(url_list)):
    if i in raster_data:
      raster_data_arr.append(raster_data[i])
  
  raster_data = np.stack(raster_data_arr, axis=-1)
  del raster_data_arr

  if return_base_raster:
    if verbose:
      ttprint(f'The base raster is {fn_base_raster}')
    return raster_data, url_list, fn_base_raster
  else:
    return raster_data, url_list

    
def read_rasters(
                  raster_dirs:List = [],
                  raster_files:List = [],
                  raster_ext = 'tif',
                  spatial_win = None,
                  dtype = 'float16',
                  n_jobs = 4,
                  verbose = False,
                  data_mask = None,
                  expected_img_size = None,
                  try_without_window = False
                ):
  if len(raster_dirs) == 0 and len(raster_files) == 0:
    raise Exception('The raster_dirs and raster_files params can not be empty at same time.')

  if len(raster_files) == 0:
    raster_files = find_files(raster_dirs, f'*.{raster_ext}')

  if verbose:
    ttprint(f'Reading {len(raster_files)} raster files using {n_jobs} workers')

  def _read_raster(raster_pos):
    raster_file = raster_files[raster_pos]
    raster_ds, band_data = None, None
    nodata = 0

    try:
      raster_ds = rasterio.open(raster_file)
      
      band_data = raster_ds.read(1, window=spatial_win)
      if band_data.size == 0 and try_without_window:
        band_data = raster_ds.read(1)

      nodata = raster_ds.nodatavals[0]
    except:
      if spatial_win is not None:
        ttprint(f'ERROR: Failed to read {raster_file} window {spatial_win}.')
        band_data = np.empty((int(spatial_win.width), int(spatial_win.height)))
        band_data[:] = np.nan

      if expected_img_size is not None:
        ttprint(f'Full nan image for {raster_file}')
        band_data = np.empty(expected_img_size)
        band_data[:] = np.nan

    if data_mask is not None:
      if (data_mask.shape == band_data.shape):
        band_data[data_mask] = np.nan
      else:
        ttprint(f'WARNING: incompatible data_mask shape {data_mask.shape} != {band_data.shape}')
    
    return raster_pos, band_data, nodata

  raster_data = {}
  args = [ (raster_pos,) for raster_pos in range(0,len(raster_files)) ]

  for raster_pos, band_data, nodata in parallel.job(_read_raster, args, n_jobs=n_jobs):
    raster_file = raster_files[raster_pos]

    if (isinstance(band_data, np.ndarray)):

      band_data = band_data.astype(dtype)
      band_data[band_data == nodata] = np.nan

    else:
      raise Exception(f'The raster {raster_file} was not found.')
    raster_data[raster_pos] = band_data

  raster_data = [raster_data[i] for i in range(0,len(raster_files))]
  raster_data = np.ascontiguousarray(np.stack(raster_data, axis=2))
  return raster_data, raster_files