from typing import Callable, Iterator, List,  Union
from osgeo import gdal, gdal_array
import numpy as np
import SharedArray as sa
import skmap_bindings

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

co = ['TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE', 'BLOCKXSIZE=1024', 'BLOCKYSIZE=1024']

executor = None

def ttprint(*args, **kwargs):
  from datetime import datetime
  import sys

  print(f'[{datetime.now():%H:%M:%S}] ', end='')
  print(*args, **kwargs, flush=True)

def ProcessGeneratorLazy(
  worker:Callable,
  args:Iterator[tuple],
  max_workers:int = None
):
  import concurrent.futures
  import multiprocessing
  from itertools import islice

  global executor

  if executor is None:
    max_workers = multiprocessing.cpu_count()
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

  #with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
  futures = { executor.submit(worker, **arg) for arg in args }

  done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
  for task in done:
    err = task.exception()
    if err is not None:
      raise err
    else:
        yield task.result()

def save_raster(base_raster, out_file, array_fn, i, minx = None, maxy = None, co = []):
    
  #for key in gdal_opts.keys():
  #  gdal.SetConfigOption(key,gdal_opts[key])
  
  array = sa.attach(array_fn, False)
  base_ds = gdal.Open(base_raster)

  driver = gdal.GetDriverByName('GTiff')
  #cols, rows = ds.RasterXSize, ds.RasterYSize

  out_ds = driver.CreateCopy(out_file, base_ds, options=co)
  #out_ds = driver.Create(out_file, cols, rows, 1, dtype)
  #out_ds.SetGeoTransform(*base_ds.GetGeoTransform())
  out_band = out_ds.GetRasterBand(1)

  xoff, yoff = 0, 0
  if minx is not None and maxy is not None:
    gt = base_ds.GetGeoTransform()
    gti = gdal.InvGeoTransform(gt)

    xoff, yoff = gdal.ApplyGeoTransform(gti, minx, maxy)
    xoff, yoff = int(xoff), int(yoff)

  gdal_array.BandWriteArray(out_band, array[:,:,i], xoff=xoff, yoff=yoff)

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

def save_rasters(base_raster, out_files, idx_list, array_fn, minx = None, maxy = None):

  ttprint(f"Saving {len(out_files)} raster files.")
  
  args = []
  for out_file, i in zip(out_files, idx_list):
    args.append({
      'base_raster': base_raster, 
      'out_file': out_file, 
      'array_fn': array_fn, 
      'i': i,
      'minx': minx,
      'maxy': maxy,
      'co': co
    })

  for result in ProcessGeneratorLazy(save_raster, args, len(args)):
    continue

def run_model(model_type, model, model_fn, array_fn, out_fn, i0, i1):

  array = sa.attach(array_fn, False)
  #array = array[:,:,0:169]
  out = sa.attach(out_fn, False)
  n_features = array.shape[-1]

  start = time.time()
  if model_type == 'tl2cgen':
    import tl2cgen
    model = tl2cgen.Predictor(libpath=model_fn)
    out[:,:,i0:i1] = model.predict(tl2cgen.DMatrix(array.transpose(1,0))).reshape((out.shape[0],out.shape[1],-1))

  elif model_type == 'xgboost':
    #ttprint(f"Running a {model_type} model ({model_fn})")
    #import xgboost
    #model = xgboost.XGBClassifier() 
    #model.load_model("model/xgb_model.bin")
    out[:,:,i0:i1] = model.predict_proba(array.transpose(1,0)).reshape((out.shape[0],out.shape[1],-1))

  elif model_type == 'hummingbird':
    #ttprint(f"Running a {model_type} model ({model_fn})")
    from hummingbird.ml import load
    model = load(model_fn)
    out[:,:,i0:i1] = model.predict_proba(array.transpose(1,0)).reshape((out.shape[0],out.shape[1],-1))

  else:
    raise Exception(f'Invalid model type {model_type}')
  ttprint(f"Model {model_type} ({model_fn}): {(time.time() - start):.2f} segs")

def _raster_paths(df_features, ftype, tile = None, year = None):

  mask = (df_features['type'] == ftype)

  path_col = 'path'
  if ftype == 'landsat':
    df_features['ypath'] = df_features[mask]['path'].apply(lambda p: p.replace('{tile}', tile).replace('{year}', str(year)))
    path_col = 'ypath'

  ids_list = list(df_features[mask]['idx'])
  raster_files = list(df_features[mask][path_col])

  return raster_files, ids_list

def in_mem_calc(data, df_feat, n_threads):

  band_scaling = 0.004
  result_scaling = 125.
  result_offset = 125.

  blue_idx = list(df_feat[df_feat['name'].str.contains('blue_glad')].index)
  red_idx = list(df_feat[df_feat['name'].str.contains('red_glad')].index)
  nir_idx = list(df_feat[df_feat['name'].str.contains('nir_glad')].index)

  swir1_idx = list(df_feat[df_feat['name'].str.contains('swir1_glad')].index)
  swir2_idx = list(df_feat[df_feat['name'].str.contains('swir2_glad')].index)

  # NDVI
  ndvi_idx = list(df_feat[df_feat['name'].str.contains('ndvi_glad')].index)
  skmap_bindings.computeNormalizedDifference(data, n_threads,
                              nir_idx, red_idx, ndvi_idx,
                              band_scaling, band_scaling, result_scaling, result_offset)
  # NDWI
  ndwi_idx = list(df_feat[df_feat['name'].str.contains('ndwi_glad')].index)
  skmap_bindings.computeNormalizedDifference(data, n_threads,
                              nir_idx, swir1_idx, ndwi_idx,
                              band_scaling, band_scaling, result_scaling, result_offset)
  # BSI
  bsi_idx = list(df_feat[df_feat['name'].str.contains('bsi_glad')].index)
  skmap_bindings.computeBsi(data, n_threads,
                              swir1_idx, red_idx, nir_idx, blue_idx, bsi_idx,
                              band_scaling, band_scaling, band_scaling, band_scaling, result_scaling, result_offset)
  # NDTI
  ndti_idx = list(df_feat[df_feat['name'].str.contains('ndti_glad')].index)
  skmap_bindings.computeNormalizedDifference(data, n_threads,
                              swir1_idx, swir2_idx, ndti_idx,
                              band_scaling, band_scaling, result_scaling, result_offset)
  # NIRV
  nirv_idx = list(df_feat[df_feat['name'].str.contains('nirv_glad')].index)
  skmap_bindings.computeNirv(data, n_threads,
                              red_idx, nir_idx, nirv_idx,
                              band_scaling, band_scaling, result_scaling, result_offset)
  # EVI
  evi_idx = list(df_feat[df_feat['name'].str.contains('evi_glad')].index)
  skmap_bindings.computeEvi(data, n_threads,
                              red_idx, nir_idx, blue_idx, evi_idx,
                              band_scaling, band_scaling, band_scaling, result_scaling, result_offset)
  # FAPAR
  fapar_idx = list(df_feat[df_feat['name'].str.contains('fapar_glad')].index)
  skmap_bindings.computeFapar(data, n_threads,
                              red_idx, nir_idx, fapar_idx,
                              band_scaling, band_scaling, result_scaling, result_offset)

  data_ndvi = data[ndvi_idx,:]
  expr = f'( where( ndvi_0101_0228 <= 169, 100, 0) + where( ndvi_0301_0430 <= 169, 100, 0) + ' + \
         f'  where( ndvi_0501_0630 <= 169, 100, 0) + where( ndvi_0701_0831 <= 169, 100, 0) + ' + \
         f'  where( ndvi_0501_0630 <= 169, 100, 0) + where( ndvi_0701_0831 <= 169, 100, 0) ) / 6'
  data[-1,:] = ne.evaluate(expr, local_dict={
    'ndvi_0101_0228': data_ndvi[0,:], 'ndvi_0301_0430': data_ndvi[1,:],
    'ndvi_0501_0630': data_ndvi[2,:], 'ndvi_0701_0831': data_ndvi[3,:],
    'ndvi_0501_0630': data_ndvi[4,:], 'ndvi_0701_0831': data_ndvi[5,:],
  }).round()

def _model_input(tile, start_year = 2000, end_year = 2022, bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal'], base_url='http://192.168.49.30:8333'):
  prediction_layers = []

  for year in range(start_year, end_year + 1):
    for band in bands:
      prediction_layers += [
      f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + f'{year}0101_{year}0228_go_epsg.4326_v20230908.tif',
      f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + f'{year}0301_{year}0430_go_epsg.4326_v20230908.tif',
      f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + f'{year}0501_{year}0630_go_epsg.4326_v20230908.tif',
      f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + f'{year}0701_{year}0831_go_epsg.4326_v20230908.tif',
      f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + f'{year}0901_{year}1031_go_epsg.4326_v20230908.tif',
      f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + f'{year}1101_{year}1231_go_epsg.4326_v20230908.tif'
      ]

  return prediction_layers

def _get_static_layers_info(tiles, df_features, tile):
  
  min_x, _, _, max_y = tiles[tiles['TILE'] == tile].iloc[0].geometry.bounds
  static_files, _ = _raster_paths(df_features, 'static')
  
  gidal_ds = gdal.Open(static_files[0]) # It is assumed to be the same for all static layers
  gt = gidal_ds.GetGeoTransform()
  gti = gdal.InvGeoTransform(gt)
  x_off_s, y_off_s = gdal.ApplyGeoTransform(gti, min_x, max_y)
  x_off_s, y_off_s = int(x_off_s), int(y_off_s)
  
  return static_files, x_off_s, y_off_s

if __name__ == '__main__':

  from osgeo import gdal, gdal_array

  from pathlib import Path
  import SharedArray as sa
  import geopandas as gpd
  import bottleneck as bn
  import numexpr as ne
  import pandas as pd
  import numpy as np
  import tempfile
  import time
  import os
  
  TMP_DIR = tempfile.gettempdir()

  ttprint("Reading tiles gpkg")
  tiles = gpd.read_file('../ard2_final_status.gpkg')
  df_features = pd.read_csv('../model/features.csv')

  n_threads = 96
  tile = '029E_51N'
  minx, miny, maxx, maxy = tiles[tiles['TILE'] == tile].iloc[0].geometry.bounds

  static_files, static_idx = _raster_paths(df_features, 'static')
  n_features = df_features.shape[0]

  bands_list = [1,]
  x_size, y_size = (4000, 4000)
  shape = (n_features, x_size * y_size)
  array_fn = 'file://' + str(make_tempfile(prefix='shm_array'))
  array = sa.create(array_fn, shape, dtype=np.float32)
  ttprint(f"Data in: {array_fn}")

  start = time.time()
  static_files, x_off_s, y_off_s = _get_static_layers_info(tiles, df_features, tile)
  n_static = len(static_files)
  skmap_bindings.readData(array, n_threads, static_files, static_idx, x_off_s, y_off_s, x_size, y_size, bands_list, gdal_opts)
  ttprint(f"Reading static: {(time.time() - start):.2f} segs")

  start = time.time()
  import xgboost
  model_xgb = xgboost.XGBClassifier() 
  model_xgb.load_model('../model/xgb_model.bin')
  ttprint(f"Loading xgb model: {(time.time() - start):.2f} segs")

  start = time.time()
  from hummingbird.ml import load
  model_ann = load('../model/ann.torch')
  ttprint(f"Loading ann model: {(time.time() - start):.2f} segs")

  start = time.time()
  from hummingbird.ml import load
  model_meta = load('../model/log-reg.torch.zip')
  ttprint(f"Loading log-reg models: {(time.time() - start):.2f} segs")

  for year in [2020]: #range(2000, 2023, 2):
    
    landsat_files, landsat_idx = _raster_paths(df_features, 'landsat', tile, year)

    x_off_d, y_off_d = (2, 2)
    subnet = '192.168.49'
    hosts = [ f'{subnet}.{i}:8333' for i in range(30,43) ]
    landsat_files = [str(r).replace(f"{subnet}.30", f"{subnet}.{30 + int.from_bytes(Path(r).stem.encode(), 'little') % len(hosts)}") for r in landsat_files]

    start = time.time()
    i = len(static_files) 
    skmap_bindings.readData(array, n_threads, landsat_files, landsat_idx, x_off_d, y_off_d, x_size, y_size, bands_list, gdal_opts, 255., np.nan)
    ttprint(f"Reading landsat: {(time.time() - start):.2f} segs")

    lockup = { row['name']: row['idx'] for _, row in df_features[['idx','name']].iterrows() }

    start = time.time()
    in_mem_calc(array, df_features, n_threads)
    ttprint(f"In memory calc: {(time.time() - start):.2f} segs")
    ttprint(f"Number of feature: {n_features}")
    ttprint(f"Array shape: {array.shape}")

    start = time.time()
    #import tl2cgen
    #model_rf = tl2cgen.Predictor(libpath='model/skl_rf_intel.so')

    n_classes = 3
    shape = (4000, 4000, n_classes * 5 + 1)
    out_fn = 'file://' + str(make_tempfile(prefix='shm_output'))
    out = sa.create(out_fn, shape, dtype=np.float32)

    args = [
      {
        'model_type': 'tl2cgen',
        'model_fn': '../model/skl_rf_intel.so',
        'model': None,
        'array_fn': array_fn,
        'out_fn': out_fn,
        'i0': 0,
        'i1': 3
      },
      {
        'model_type': 'xgboost',
        'model_fn': '../model/xgb_model.bin',
        'model': model_xgb,
        'array_fn': array_fn,
        'out_fn': out_fn,
        'i0': 3,
        'i1': 6
      },
      {
        'model_type': 'hummingbird',
        'model_fn': '../model/ann.torch',
        'model': model_ann,
        'array_fn': array_fn,
        'out_fn': out_fn,
        'i0': 6,
        'i1': 9
      }
    ]

    n_models = len(args)

    start = time.time()
    for index in ProcessGeneratorLazy(run_model, args, len(args)):
      continue
    #for arg in args:
    #  run_model(**arg)
    ttprint(f"Running models: {(time.time() - start):.2f} segs")

    mi = n_classes * n_models

    start = time.time()
    # EML probabilities
    out[:,:,9:12] = model_meta.predict_proba(out[:,:,0:mi].reshape(-1,mi)).reshape(out.shape[0],out.shape[1],-1).round(2) * 100
    
    # Model deviance
    out[:,:,12:15] = bn.nanstd(out[:,:,0:mi].reshape(out.shape[0],out.shape[1], n_models, n_classes), axis=-2).round(2) * 100

    # Final map
    out[:,:,15] = np.argmax(out[:,:,9:12], axis=-1) + 1

    ttprint(f"Running meta-learner: {(time.time() - start):.2f} segs")
    ttprint(f'Probabilities are in {out_fn}')

    start = time.time()
    out_files = [
      f'./gpw_eml.seeded.grass_30m_m_{year}0101_{year}1231_go_epsg.4326_v20240206.tif',
      f'./gpw_eml.semi.nat.grass_30m_m_{year}0101_{year}1231_go_epsg.4326_v20240206.tif',
      f'./gpw_eml.seeded.grass_30m_md_{year}0101_{year}1231_go_epsg.4326_v20240206.tif',
      f'./gpw_eml.semi.nat.grass_30m_md_{year}0101_{year}1231_go_epsg.4326_v20240206.tif',
      f'./gpw_eml.grass.type_30m_c_{year}0101_{year}1231_go_epsg.4326_v20240206.tif'
    ]

    out_idx = [ 9, 10, 12, 13, 15 ]

    save_rasters(landsat_files[0], out_files, out_idx, out_fn, minx = minx, maxy = maxy)
    ttprint(f"Writing files: {(time.time() - start):.2f} segs")
