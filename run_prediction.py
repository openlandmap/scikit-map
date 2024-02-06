from typing import Callable, Iterator, List,  Union
from osgeo import gdal, gdal_array
import numpy as np
import SharedArray as sa

gdal_opts = {
 #'GDAL_HTTP_MULTIRANGE': 'SINGLE_GET',
 #'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES': 'NO',
 'GDAL_HTTP_VERSION': '1.0',
 #'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
 #'VSI_CACHE': 'FALSE',
 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
 #'GDAL_HTTP_CONNECTTIMEOUT': '320',
 #'CPL_VSIL_CURL_USE_HEAD': 'NO',
 #'GDAL_HTTP_TIMEOUT': '320',
 #'CPL_CURL_GZIP': 'NO'
}

def ProcessGeneratorLazy(
  worker:Callable,
  args:Iterator[tuple],
  max_workers:int = None
):
  import concurrent.futures
  import multiprocessing
  from itertools import islice
  
  if max_workers is None:
    max_workers = multiprocessing.cpu_count()

  with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = { executor.submit(worker, **arg) for arg in args }

    done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
    for task in done:
      err = task.exception()
      if err is not None:
        raise err
      else:
          yield task.result()

def read_raster(raster_file, array_fn, i, band=1, minx = None, maxy = None):
    
  for key in gdal_opts.keys():
      gdal.SetConfigOption(key,gdal_opts[key])
  
  array = sa.attach(array_fn, False)
  
  ds = gdal.Open(raster_file)

  xoff, yoff = 0, 0
  win_xsize, win_ysize = None, None
  if minx is not None and maxy is not None:
    gt = ds.GetGeoTransform()
    gti = gdal.InvGeoTransform(gt)

    xoff, yoff = gdal.ApplyGeoTransform(gti, minx, maxy)
    xoff, yoff = int(xoff), int(yoff)
    win_xsize, win_ysize = array.shape[0], array.shape[1]

  band = ds.GetRasterBand(1)
  nodata = band.GetNoDataValue()
  
  gdal_array.BandReadAsArray(band, buf_obj=array[:,:,i],
          xoff=xoff, yoff=yoff, win_xsize=win_xsize, win_ysize=win_ysize)

  array[:,:,i][array[:,:,i] == nodata] = np.nan

def _model_input(tile, start_year = 2000, end_year = 2022, bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal'], base_url='http://192.168.49.30:8333'):
  prediction_layers = []
  
  for year in range(start_year, end_year + 1):
    for band in bands:
      prediction_layers += [
        f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}0101_{year}0228_go_epsg.4326_v20230908.tif',
        f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}0301_{year}0430_go_epsg.4326_v20230908.tif',
        f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}0501_{year}0630_go_epsg.4326_v20230908.tif',
        f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}0701_{year}0831_go_epsg.4326_v20230908.tif',
        f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}0901_{year}1031_go_epsg.4326_v20230908.tif',
        f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}1101_{year}1231_go_epsg.4326_v20230908.tif'
      ]
  
  raster_files = []
  dict_layers_newnames = {}
  for l in prediction_layers:
  
    key = Path(l).stem.replace('{year}', '')
    value = Path(l).stem.replace('{year}', str(year))
    dict_layers_newnames[key] = value
    raster_files.append('/vsicurl/' + l.replace('{year}', str(year)))
  
  hosts = [ f'192.168.49.{i}:8333' for i in range(30,43) ]
  raster_files = [str(r).replace("192.168.49.30", f"192.168.49.{30 + int.from_bytes(Path(r).stem.encode(), 'little') % len(hosts)}") for r in raster_files]
  
  return raster_files, dict_layers_newnames

def make_tempdir(basedir='skmap', make_subdir = True):
  tempdir = Path(TMP_DIR).joinpath(basedir)
  if make_subdir: 
    name = Path(tempfile.NamedTemporaryFile().name).name
    tempdir = tempdir.joinpath(name)
  tempdir.mkdir(parents=True, exist_ok=True)
  return tempdir

def make_tempfile(basedir='skmap', prefix='', suffix='', make_subdir = False):
  tempdir = make_tempdir(basedir, make_subdir=make_subdir)
  return tempdir.joinpath(
    Path(tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix).name).name
  )

def read_rasters(raster_files, array_fn = None, array_i = 0, minx = None, maxy = None, gdal_opts = []):
  n_files = len(raster_files)

  if array_fn is None:
    ds = gdal.Open(raster_files[0])
    cols, rows = ds.RasterXSize, ds.RasterYSize
    
    shape = (cols, rows, n_files * 10)
    array_fn = 'file://' + str(make_tempfile(prefix='shm_array'))
    print(array_fn)
    
    sa.create(array_fn, shape, dtype=float)
  
  print(f"Reading {len(raster_files)} raster files.")
  
  args = []
  for i in range(0, n_files):
    args.append({
      'raster_file': raster_files[i], 
      'array_fn': array_fn, 
      'i': array_i + i,
      'minx': minx,
      'maxy': maxy
    })

  for result in ProcessGeneratorLazy(read_raster, args, len(args)):
    continue

  return sa.attach(array_fn)

def eval_calc(array_fn, index, expr, local_dict, gi):
  array = sa.attach(array_fn, False)

  local_dict = { b: array[:,:,local_dict[b]:local_dict[b]+1]  for b in local_dict.keys() }

  #print(f'Calculating {index}')
  array[:,:,gi:gi+1] = ne.evaluate(expr, local_dict=local_dict).round()
  array[:,:,gi:gi+1][array[:,:,gi:gi+1] == -np.inf] = 0
  array[:,:,gi:gi+1][array[:,:,gi:gi+1] == +np.inf] = 255

  return index

def in_mem_calc(lockup, array):
  pref = 'glad.SeasConv.ard2_m_30m_s'
  suff = 'go_epsg.4326_v20230908'
  dates = ['0101_0228','0301_0430','0501_0630','0701_0831','0901_1031','1101_1231']
  bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']

  indices = {}
  gi = len(lockup)

  for dt in dates:
    #local_dict = { b: array[:, :, lockup[f'{b}_{pref}_{dt}_{suff}']:lockup[f'{b}_{pref}_{dt}_{suff}']+1 ] for b in bands}
    local_dict = { b: lockup[f'{b}_{pref}_{dt}_{suff}'] for b in bands}
    indices[f'ndvi_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) - (red * 0.004) ) / ( (nir * 0.004) + (red * 0.004) ) ) * 125 + 125', local_dict
    indices[f'ndwi_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) - (swir1 * 0.004) ) / ( (nir * 0.004) + (swir1 * 0.004) ) ) * 125 + 125', local_dict
    #indices[f'savi_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) - (red * 0.004) )*1.5 / ( (nir * 0.004) + (red * 0.004)  + 0.5) ) * 125 + 125', local_dict
    #indices[f'msavi_{pref}_{dt}_{suff}'] = f'( (2 *  (nir * 0.004) + 1 - sqrt((2 *  (nir * 0.004) + 1)**2 - 8 * ( (nir * 0.004) - (red * 0.004) ))) / 2 ) * 125 + 125', local_dict
    #indices[f'nbr_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) - ( swir2 * 0.004) ) / ( (nir * 0.004) + ( swir2 * 0.004) ) ) * 125 + 125', local_dict
    #indices[f'ndmi_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) -  (swir1 * 0.004)) / ( (nir * 0.004) +  (swir1 * 0.004)) ) * 125 + 125', local_dict
    #indices[f'nbr2_{pref}_{dt}_{suff}'] = f'( ( (swir1 * 0.004) - ( thermal * 0.004) ) / ( (swir1 * 0.004) + ( thermal * 0.004) ) ) * 125 + 125', local_dict
    #indices[f'rei_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) - blue)/( (nir * 0.004) + blue *  (nir * 0.004)) ) * 125 + 125', local_dict
    indices[f'bsi_{pref}_{dt}_{suff}'] = f'( ( ( (swir1 * 0.004) + (red * 0.004) ) - ( (nir * 0.004) + blue) ) / ( ( (swir1 * 0.004) + (red * 0.004) ) + ( (nir * 0.004) + blue) ) ) * 125 + 125', local_dict
    indices[f'ndti_{pref}_{dt}_{suff}'] = f'( ( (swir1 * 0.004) - (swir2 * 0.004) )  / ( (swir1 * 0.004) + (swir2 * 0.004) )  ) * 125 + 125', local_dict
    #indices[f'ndsi_{pref}_{dt}_{suff}'] = f'( ( (green * 0.004) -  (swir1 * 0.004) ) / ( (green * 0.004) +  (swir1 * 0.004) ) ) * 125 + 125', local_dict
    #indices[f'ndsmi_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) - (swir2 * 0.004) )  / ( (nir * 0.004) + (swir2 * 0.004) )  ) * 125 + 125', local_dict
    indices[f'nirv_{pref}_{dt}_{suff}'] = f'( ( ( ( (nir * 0.004) - (red * 0.004) ) / ( (nir * 0.004) + (red * 0.004) ) ) - 0.08) *  (nir * 0.004) ) * 125 + 125', local_dict
    indices[f'evi_{pref}_{dt}_{suff}'] = f'( 2.5 * ( (nir * 0.004) - (red * 0.004) ) / ( (nir * 0.004) + 6 * (red * 0.004) - 7.5 * ( blue * 0.004) + 1) ) * 125 + 125', local_dict
    indices[f'fapar_{pref}_{dt}_{suff}'] = f'( ((( (( (nir * 0.004) - (red * 0.004) ) / ( (nir * 0.004) + (red * 0.004) )) - 0.03) * (0.95 - 0.001)) / (0.96 - 0.03)) + 0.001 ) * 125 + 125', local_dict

  new_lockup = []
  bcf_local_dict = {}
  args = []
  for index, (expr, local_dict) in  indices.items():
    args.append({
      'array_fn': array.base.name,
      'index': index,
      'expr': expr,
      'local_dict': local_dict,
      'gi': gi
    })
    gi += 1
    
    if 'ndvi_' in index:
      _index = index.replace(f'_{pref}','').replace(f'_{suff}','')
      bcf_local_dict[_index] = array[:,:,gi:gi+1]

  print(f"Calculating {len(args) + 1} in mem features.")
  for index in ProcessGeneratorLazy(eval_calc, args, len(args)):
    new_lockup.append(index)
    #print(index)
    continue

  expr = f'( where( ndvi_0101_0228 <= 169, 100, 0) + where( ndvi_0301_0430 <= 169, 100, 0) + ' + \
         f'  where( ndvi_0501_0630 <= 169, 100, 0) + where( ndvi_0701_0831 <= 169, 100, 0) + ' + \
         f'  where( ndvi_0501_0630 <= 169, 100, 0) + where( ndvi_0701_0831 <= 169, 100, 0) ) / 6'

  index = f'bsf_{pref}_{suff}'
  #print(f'Calculating {index}')
  newdata = ne.evaluate(expr, local_dict=bcf_local_dict).round()
  array[:,:,gi:gi+1] = newdata
  new_lockup.append(index)

  new_lockup = {new_lockup[i]: len(lockup) + i for i in range(0, len(new_lockup))}
  lockup = {**lockup, **new_lockup}

  return lockup, array

def run_model(model_type, model, model_fn, array_fn, out_fn, i0, i1):

  array = sa.attach(array_fn, False)
  array = array[:,:,0:169]
  out = sa.attach(out_fn, False)
  n_features = array.shape[-1]

  if model_type == 'tl2cgen':
    print(f"Running a {model_type} model ({model_fn})")
    import tl2cgen
    model = tl2cgen.Predictor(libpath=model_fn)
    out[:,:,i0:i1] = model.predict(tl2cgen.DMatrix(array.reshape(-1, n_features), dtype="float32")).reshape((array.shape[0],array.shape[0],-1))

  elif model_type == 'xgboost':
    print(f"Running a {model_type} model ({model_fn})")
    #import xgboost
    #model = xgboost.XGBClassifier() 
    #model.load_model("model/xgb_model.bin")
    out[:,:,i0:i1] = model.predict_proba(array.reshape(-1, n_features)).reshape((array.shape[0],array.shape[0],-1))

  elif model_type == 'hummingbird':
    print(f"Running a {model_type} model ({model_fn})")
    #from hummingbird.ml import load
    #model = load(model_fn)
    out[:,:,i0:i1] = model.predict_proba(array.reshape(-1, n_features)).reshape((array.shape[0],array.shape[0],-1))

  else:
    raise Exception(f'Invalid model type {model_type}')

if __name__ == '__main__':

  import rasterio
  from osgeo import gdal, gdal_array

  import numpy as np
  import multiprocessing
  import ctypes
  from multiprocessing import RawArray
  import time
  import concurrent
  from concurrent.futures.process import ProcessPoolExecutor
  from concurrent.futures.thread import ThreadPoolExecutor
  from multiprocessing import shared_memory
  import math
  import tempfile
  from pathlib import Path
  import SharedArray as sa

  import multiprocessing as mp

  TMP_DIR = tempfile.gettempdir()

  from skmap.mapper import LandMapper
  from skmap.misc import find_files
  from skmap.misc import vrt_warp, ttprint
  from skmap.mapper import LandMapper
  from skmap.misc import find_files
  import geopandas as gpd
  import numexpr as ne
  import numpy as np
  import traceback
  import os

  from pathlib import Path
  import time

  print("Reading tiles gpkg")
  tiles = gpd.read_file('ard2_final_status.gpkg')

  tile = '047W_11S'
  year = 2020
  minx, miny, maxx, maxy = tiles[tiles['TILE'] == tile].iloc[0].geometry.bounds

  landsat_files, dict_layers_newnames = _model_input(tile, year, year)
  static_files = find_files('./static', '*.vrt')
  static_files = [ str(f) for f in static_files ]
  n_features = 172

  shape = (4000, 4000, n_features)
  array_fn = 'file://' + str(make_tempfile(prefix='shm_array'))
  array = sa.create(array_fn, shape, dtype=float)

  start = time.time()
  array = read_rasters(static_files, array_fn=array_fn, minx=minx, maxy=maxy)
  print(f"Reading static: {(time.time() - start):.2f} segs")

  start = time.time()
  i = len(static_files) 
  array = read_rasters(landsat_files, array_i=i, array_fn=array_fn, minx=minx, maxy=maxy)
  print(f"Reading landsat: {(time.time() - start):.2f} segs")

  lockup = [ Path(l).stem for l in static_files ] + \
           [ Path(l).stem.replace(f'{year}', '') for l in landsat_files ]
  lockup = { lockup[i]: i for i in range(0, len(lockup)) }

  start = time.time()
  lockup, array = in_mem_calc(lockup, array)
  print(f"In memory calc: {(time.time() - start):.2f} segs")
  print(f"Number of feature: {len(lockup)}")
  print(f"Array shape: {array.shape}")

  start = time.time()
  #import tl2cgen
  #model_rf = tl2cgen.Predictor(libpath='model/skl_rf_intel.so')

  import xgboost
  model_xgb = xgboost.XGBClassifier() 
  model_xgb.load_model('model/xgb_model.bin')

  from hummingbird.ml import load
  model_ann = load('model/ann.torch')
  print(f"Loading models: {(time.time() - start):.2f} segs")

  n_classes = 3
  shape = (4000, 4000, n_classes * 5 + 1)
  out_fn = 'file://' + str(make_tempfile(prefix='shm_output'))
  out = sa.create(out_fn, shape, dtype=float)

  args = [
    {
      'model_type': 'tl2cgen',
      'model_fn': 'model/skl_rf_intel.so',
      'model': None,
      'array_fn': array_fn,
      'out_fn': out_fn,
      'i0': 0,
      'i1': 3
    },
    {
      'model_type': 'xgboost',
      'model_fn': 'model/xgb_model.bin',
      'model': model_xgb,
      'array_fn': array_fn,
      'out_fn': out_fn,
      'i0': 3,
      'i1': 6
    },
    {
      'model_type': 'hummingbird',
      'model_fn': 'model/ann.torch',
      'model': model_ann,
      'array_fn': array_fn,
      'out_fn': out_fn,
      'i0': 6,
      'i1': 9
    }
  ]

  start = time.time()
  #for index in ProcessGeneratorLazy(run_model, args, len(args)):
  #  continue
  for arg in args:
    run_model(**arg)
  print(f"Running models: {(time.time() - start):.2f} segs")