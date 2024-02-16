import os
os.environ['USE_PYGEOS'] = '0'
os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'

from datetime import datetime
from osgeo import gdal, gdal_array
from pathlib import Path
from typing import Callable, Iterator, List,  Union
import bottleneck as bn
import geopandas as gpd
import numexpr as ne
import numpy as np
import pandas as pd
import SharedArray as sa
import skmap_bindings
import tempfile
import time
import sys
import requests
from hummingbird.ml import load

import concurrent.futures
import multiprocessing
from itertools import islice

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

  global executor

  #with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
  futures = { executor.submit(worker, **arg) for arg in args }

  done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
  for task in done:
    err = task.exception()
    if err is not None:
      raise err
    else:
        yield task.result()

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
        
def _features(csv_file, years, tile_id):
  
  df_features = pd.read_csv(csv_file,index_col=0).reset_index(drop=True)
  #df_features.loc[df_features['type'] == 'static',['path']] = df_features[df_features['type'] == 'static']['path'].apply(lambda f: f'../{f}')

  df_list = []

  df_list += [ df_features[np.logical_and(df_features['type'] == 'static', df_features['selected'] > -1)] ]

  for year in years:
    mask = (df_features['type'] == 'landsat')
    df = df_features[mask].copy()
    df['path'] = df['path'].apply(lambda p: p.replace('{tile}', tile_id).replace('{year}', str(year)))
    df_list += [ df ]

  otf_mask = df_features['type'] == 'on-the-fly'
  otf_sel = df_features[np.logical_and(otf_mask,df_features['selected'] > -1)]['name'].apply(lambda f: '_'.join(f.split('_')[0:2])).unique()
  for year in years:
    df_list += [ df_features[np.logical_and(otf_mask, df_features['name'].str.contains('|'.join(otf_sel)))] ]

  df_features = pd.concat(df_list)
  df_features = df_features.reset_index(drop=True)
  df_features['idx'] = df_features.index

  matrix_idx = []

  for i in range(0, df_features['selected'].max()+1):
      sel_mask = df_features['selected'] == i
      idx = list(df_features[sel_mask]['idx'])
      if len(idx) == 1:
          idx = [ idx[0] for i in range(0,len(years)) ]
      matrix_idx.append(idx)

  matrix_idx = np.array(matrix_idx)
  
  return df_features, matrix_idx

def _raster_paths(df_features, ftype):

  mask = (df_features['type'] == ftype)
  ids_list = list(df_features[mask]['idx'])
  raster_files = list(df_features[mask]['path'])

  return raster_files, ids_list

def _get_static_layers_info(df_features, tiles, tile):
  
  min_x, _, _, max_y = tiles[tiles['TILE'] == tile].iloc[0].geometry.bounds
  static_files, static_idx = _raster_paths(df_features, 'static')
  
  gidal_ds = gdal.Open(static_files[0]) # It is assumed to be the same for all static layers
  gt = gidal_ds.GetGeoTransform()
  gti = gdal.InvGeoTransform(gt)
  x_off_s, y_off_s = gdal.ApplyGeoTransform(gti, min_x, max_y)
  x_off_s, y_off_s = int(x_off_s), int(y_off_s)
  
  return static_files, static_idx, x_off_s, y_off_s


def _geom_temperature(df_features, array, n_threads):

  elev_idx = list(df_features[df_features['name'].str.contains('dtm.bareearth_ensemble')].index)
  lst_min_geo_idx = list(df_features[df_features['name'].str.contains('clm_lst_min.geom.temp')].index)
  lst_max_geo_idx = list(df_features[df_features['name'].str.contains('clm_lst_max.geom.temp')].index)

  x_off, y_off = (2, 2)
  base_landsat = landsat_files[-1]
  lon_lat = np.zeros((2, array.shape[1]), dtype=np.float32)
  skmap_bindings.getLatLonArray(lon_lat, n_threads, gdal_opts, base_landsat, x_off, y_off, x_size, y_size)
  latitude = lon_lat[1,:].copy()

  doys = [ datetime.strptime(f'2000-{m}-15', '%Y-%m-%d').timetuple().tm_yday for m in range(1,13) ]
  doys_all = sum([ doys for i in range(0, len(years)) ],[])

  elevation = array[elev_idx[0],:]

  skmap_bindings.computeGeometricTemperature(array, n_threads, latitude, elevation, 0.1, 24.16453, -15.71751, 100., lst_min_geo_idx, doys_all)
  skmap_bindings.computeGeometricTemperature(array, n_threads, latitude, elevation, 0.1, 37.03043, -15.43029, 100., lst_max_geo_idx, doys_all)

def in_mem_calc(data, df_features, n_threads):
    
  band_scaling = 0.004
  result_scaling = 125.
  result_offset = 125.

  blue_idx = list(df_features[df_features['name'].str.contains('blue_glad')].index)
  red_idx = list(df_features[df_features['name'].str.contains('red_glad')].index)
  nir_idx = list(df_features[df_features['name'].str.contains('nir_glad')].index)

  swir1_idx = list(df_features[df_features['name'].str.contains('swir1_glad')].index)
  swir2_idx = list(df_features[df_features['name'].str.contains('swir2_glad')].index)
  bsf_idx = list(df_features[df_features['name'].str.contains('bsf')].index)

  ndvi_idx = list(df_features[df_features['name'].str.contains('ndvi_glad')].index)
  ndwi_idx = list(df_features[df_features['name'].str.contains('ndwi_glad')].index)
  bsi_idx = list(df_features[df_features['name'].str.contains('bsi_glad')].index)
  ndti_idx = list(df_features[df_features['name'].str.contains('ndti_glad')].index)
  fapar_idx = list(df_features[df_features['name'].str.contains('fapar_glad')].index)

  # NDVI
  skmap_bindings.computeNormalizedDifference(data, n_threads,
                            nir_idx, red_idx, ndvi_idx,
                            band_scaling, band_scaling, result_scaling, result_offset, [0., 250.])
  # NDWI
  skmap_bindings.computeNormalizedDifference(data, n_threads,
                            nir_idx, swir1_idx, ndwi_idx,
                            band_scaling, band_scaling, result_scaling, result_offset, [0., 250.])
  # BSI
  skmap_bindings.computeBsi(data, n_threads,
                            swir1_idx, red_idx, nir_idx, blue_idx, bsi_idx,
                            band_scaling, band_scaling, band_scaling, band_scaling, result_scaling, result_offset, [0., 250.])
  # NDTI
  skmap_bindings.computeNormalizedDifference(data, n_threads,
                            swir1_idx, swir2_idx, ndti_idx,
                            band_scaling, band_scaling, result_scaling, result_offset, [0., 250.])
  # NIRV
  #nirv_idx = list(df_feat[df_feat['name'].str.contains('nirv_glad')].index)
  #skmap_bindings.computeNirv(data, n_threads,
  #                          red_idx, nir_idx, nirv_idx,
  #                          band_scaling, band_scaling, result_scaling, result_offset, [0., 250.])
  # EVI
  #evi_idx = list(df_feat[df_feat['name'].str.contains('evi_glad')].index)
  #skmap_bindings.computeEvi(data, n_threads,
  #                          red_idx, nir_idx, blue_idx, evi_idx,
  #                          band_scaling, band_scaling, band_scaling, result_scaling, result_offset, [0., 250.])

  # FAPAR
  
  skmap_bindings.computeFapar(data, n_threads,
                            red_idx, nir_idx, fapar_idx,
                            band_scaling, band_scaling, result_scaling, result_offset, [0., 250.])
  
  _geom_temperature(df_features, array, n_threads)

def run_model(model_type, model_fn, array_fn, out_fn, i0, i1):

  import SharedArray as sa

  array = sa.attach(array_fn, False)
  #array = array
  out = sa.attach(out_fn, False)
  n_features = array.shape[-1]

  start = time.time()
  if model_type == 'tl2cgen':
    import tl2cgen
    model = tl2cgen.Predictor(libpath=model_fn)
    out[:,i0:i1] = model.predict(tl2cgen.DMatrix(array))

  elif model_type == 'xgboost':
    import xgboost
    model = xgboost.XGBClassifier() 
    model.load_model(model_fn)
    out[:,i0:i1] = model.predict_proba(array)

  elif model_type == 'hummingbird':
    from hummingbird.ml import load
    model = load(model_fn)
    out[:,i0:i1] = model.predict_proba(array)

  else:
    raise Exception(f'Invalid model type {model_type}')
  ttprint(f"Model {model_type} ({model_fn}): {(time.time() - start):.2f} segs")

def run_meta(model_type, n_classes, n_predictions, log_fn, out_fn, i0, i1):

  import SharedArray as sa
  out = sa.attach(out_fn, False)
  mi = n_classes * n_predictions

  start = time.time()
  if model_type == 'proba':
    model_meta = load(log_fn)
    # EML probabilities
    out[:,i0:i1] = model_meta.predict_proba(out[:,0:mi]).round(2) * 100
    ttprint(f"Running meta-learner: {(time.time() - start):.2f} segs")

  elif model_type == 'md':
    out[:,i0:i1] = bn.nanstd(out[:,0:mi].reshape(-1, n_predictions, n_classes), axis=-2).round(2) * 100
    ttprint(f"Model deviance: {(time.time() - start):.2f} segs")

  elif model_type == 'class':
    start = time.time()
    out[:,i0] = np.argmax(out[:,9:12], axis=-1) + 1
    out[:,i0][out[:,i0] == 3] = 255
    ttprint(f"Argmax (hard_class): {(time.time() - start):.2f} segs")

def _processed(tile):
  url = f'http://192.168.49.30:8333/gpw/tmp-prod/{tile}/gpw_eml.grass.type_30m_c_20220101_20221231_go_epsg.4326_v20240206.tif'
  r = requests.head(url)
  return (r.status_code == 200)

TMP_DIR = tempfile.gettempdir()

base_dir = Path('/mnt/slurm/jobs/wri_pasture_class')
model_dir = Path('/mnt/gaia/tmp/WRI_GPW/models/compiled')
rf_fn = str(model_dir.joinpath('landmapper_100_rf.so'))
xgb_fn = str(model_dir.joinpath('landmapper_100_xgb.bin'))
ann_fn = str(model_dir.joinpath('landmapper_100_ann.zip'))
log_fn = str(model_dir.joinpath('landmapper_100_logreg.zip'))

mask_prefix = 'http://192.168.1.30:8333/gpw/landmask'
tiles_fn = str(base_dir.joinpath('gpw_tiles.gpkg'))
ids_fn = str(base_dir.joinpath('gpw_pasture.class_ids.csv'))
features_fn = str(base_dir.joinpath('models/features.csv'))

years = range(2000,2022 + 1, 2)
x_size, y_size = (4000, 4000)
#tiles_id = [ '055W_17S' ]
# '055W_17S', '091W_17N', '003W_57N','006E_45N',
#tiles_id = ['016E_12S','029E_51N','047W_11S','055W_28S','056W_10S','061W_28S','075E_25N','081E_60N','081E_60N','101E_28N','102W_43N','103E_46N','115E_28S','121E_04S','145E_18S','146W_62N','147E_25S'] #_tiles = ['003W_57N','006E_45N','016E_12S','029E_51N','047W_11S','055W_28S','056W_10S','061W_28S','075E_25N','081E_60N','081E_60N','101E_28N','102W_43N','103E_46N','115E_28S','121E_04S','145E_18S','146W_62N','147E_25S'
n_threads = 96
n_classes = 3
s3_prefix = 'gpw/tmp-prod'

subnet = '192.168.49'
hosts = [ f'{subnet}.{i}:8333' for i in range(30,43) ]

start_tile=int(sys.argv[1])
end_tile=int(sys.argv[2])
#server_name=sys.argv[3]

tiles_id = pd.read_csv(ids_fn)['TILE'][start_tile:end_tile]

ttprint(f"Processing {len(tiles_id)} tiles")

ttprint("Reading tiling system")
tiles = gpd.read_file(tiles_fn)

#start = time.time()
executor = concurrent.futures.ProcessPoolExecutor(max_workers=3)
#ttprint(f"Creating python workers pool: {(time.time() - start):.2f} segs")

for tile_id in tiles_id:

  if _processed(tile_id):
    ttprint(f"Tile {tile_id} is processed. Ignoring it.")
    continue

  minx, miny, maxx, maxy = tiles[tiles['TILE'] == tile_id].iloc[0].geometry.bounds

  df_features, matrix_idx = _features(features_fn, years, tile_id)
  #ttprint(f"df_features={df_features.shape} & matrix_idx={matrix_idx.shape}")

  bands_list = [1,]
  n_rasters = df_features.shape[0]
  
  shape = (n_rasters, x_size * y_size)
  array = np.empty(shape, dtype=np.float32)

  landsat_files, landsat_idx = _raster_paths(df_features, 'landsat')
  static_files, static_idx, x_off_s, y_off_s = _get_static_layers_info(df_features, tiles, tile_id)

  start = time.time()
  skmap_bindings.readData(array, n_threads, static_files, static_idx, x_off_s, y_off_s, x_size, y_size, bands_list, gdal_opts)
  ttprint(f"Tile {tile_id} - Reading static: {(time.time() - start):.2f} segs")

  start = time.time()
  x_off_d, y_off_d = (2, 2)
  landsat_files = [str(r).replace(f"{subnet}.30", f"{subnet}.{30 + int.from_bytes(Path(r).stem.encode(), 'little') % len(hosts)}") for r in landsat_files]
  skmap_bindings.readData(array, n_threads, landsat_files, landsat_idx, x_off_d, y_off_d, x_size, y_size, bands_list, gdal_opts, 255., np.nan)
  ttprint(f"Tile {tile_id} - Reading landsat: {(time.time() - start):.2f} segs")

  start = time.time()
  in_mem_calc(array, df_features, n_threads)
  ttprint(f"Tile {tile_id} - In memory calc: {(time.time() - start):.2f} segs")

  #mask_file = f'http://192.168.1.30:8333/gpw/landmask/{tile}.tif'
  #mask = np.zeros((1,x_size * y_size), dtype=np.float32)
  #skmap_bindings.readData(mask, n_threads, [mask_file,], [0,], x_off_d, x_off_d, x_size, y_size, [1,], gdal_opts)
  #n_data = int(np.sum(mask))

  start = time.time()
  n_features = df_features['selected'].max() + 1
  n_pix = len(years) * x_size * y_size
  array_mem_t = np.empty((n_pix, n_features), dtype=np.float32)
  array_mem = np.empty((n_features, n_pix), dtype=np.float32)

  skmap_bindings.reorderArray(array, n_threads, array_mem, matrix_idx)
  skmap_bindings.transposeArray(array_mem, n_threads, array_mem_t)
  ttprint(f"Tile {tile_id} - Transposing data: {(time.time() - start):.2f} segs")

  start = time.time()
  mask_file = f'{mask_prefix}/{tile_id}.tif'
  mask = np.zeros((1,x_size * y_size), dtype=np.float32)

  skmap_bindings.readData(mask, n_threads, [mask_file,], [0,], x_off_d, x_off_d, x_size, y_size, [1,], gdal_opts)
  n_data = int(np.sum(mask)) * len(years)
  selected_pix = np.arange(0, x_size * y_size)[mask[0,:]==1]
  selected_rows = np.concatenate([ selected_pix + (x_size * y_size) * i for i in range(0,len(years)) ]).tolist()

  array_fn = 'file://' + str(make_tempfile(prefix='shm_array'))
  array_t = sa.create(array_fn, (n_data, n_features), dtype=np.float32)
  skmap_bindings.selArrayRows(array_mem_t, n_threads, array_t, selected_rows)

  shape = (array_t.shape[0], n_classes * 5 + 1)
  out_fn = 'file://' + str(make_tempfile(prefix='shm_output'))
  out = sa.create(out_fn, shape, dtype=np.float32)
  ttprint(f"Tile {tile_id} - Masking data: {(time.time() - start):.2f} segs")

  args = [
    {
      'model_type': 'tl2cgen',
      'model_fn': rf_fn,
      'array_fn': array_fn,
      'out_fn': out_fn,
      'i0': 0,
      'i1': 3
    },
    {
      'model_type': 'xgboost',
      'model_fn': xgb_fn,
      'array_fn': array_fn,
      'out_fn': out_fn,
      'i0': 3,
      'i1': 6
    },
    {
      'model_type': 'hummingbird',
      'model_fn': ann_fn,
      'array_fn': array_fn,
      'out_fn': out_fn,
      'i0': 6,
      'i1': 9
    }
  ]

  #  args_meta = [
  #    {
  #      'model_type': 'proba',
  #      'n_classes': n_classes,
  #      'n_predictions': len(args),
  #      'log_fn': log_fn,
  #      'out_fn': out_fn,
  #      'i0': 9,
  #      'i1': 12
  #    },
  #    {
  #      'model_type': 'md',
  #      'n_classes': n_classes,
  #      'n_predictions': len(args),
  #      'log_fn': None,
  #      'out_fn': out_fn,
  #      'i0': 12,
  #      'i1': 15
  #    },
  #    {
  #      'model_type': 'class',
  #      'n_classes': n_classes,
  #      'n_predictions': len(args),
  #      'log_fn': None,
  #      'out_fn': out_fn,
  #      'i0': 15,
  #      'i1': None
  #    }
  #  ]

  start = time.time()
  for index in ProcessGeneratorLazy(run_model, args, len(args)):
    continue
  ttprint(f"Tile {tile_id} - Running models: {(time.time() - start):.2f} segs")

  #start = time.time()
  #for index in ProcessGeneratorLazy(run_meta, args_meta, len(args_meta)):
  #  continue
  #ttprint(f"Tile {tile_id} - Running meta models: {(time.time() - start):.2f} segs")

  model_meta = load(log_fn)
  n_predictions = len(args)
  mi = n_classes * n_predictions

  start = time.time()
  # EML probabilities
  out[:,9:12] = model_meta.predict_proba(out[:,0:mi]).round(2) * 100
  ttprint(f"Tile {tile_id} - Running meta-learner: {(time.time() - start):.2f} segs")

   # Model deviance
  start = time.time()
  out[:,12:15] = bn.nanstd(out[:,0:mi].reshape(-1, n_predictions, n_classes), axis=-2).round(2) * 100
  ttprint(f"Tile {tile_id} - Model deviance: {(time.time() - start):.2f} segs")

  # Final map
  start = time.time()
  out[:,15] = np.argmax(out[:,9:12], axis=-1) + 1
  out[:,15][out[:,15] == 3] = 255
  ttprint(f"Tile {tile_id} - Argmax (hard_class): {(time.time() - start):.2f} segs")
  
  start = time.time()  
  out_exp = np.empty((array_mem_t.shape[0], out.shape[1]), dtype=np.float32)
  skmap_bindings.fillArray(out_exp, n_threads, 255.)
  
  skmap_bindings.expandArrayRows(out, n_threads, out_exp, selected_rows)
  ttprint(f"Tile {tile_id} - Reversing mask: {(time.time() - start):.2f} segs")

  start = time.time()
  out_idx = [ 9, 10, 12, 13, 15 ]
  out_t = np.empty((out_exp.shape[1],out_exp.shape[0]), dtype=np.float32)
  out_gdal = np.empty((len(out_idx) * len(years),x_size *y_size), dtype=np.float32)
  skmap_bindings.fillArray(out_gdal, n_threads, 255.)
  skmap_bindings.transposeArray(out_exp, n_threads, out_t)

  subrows = np.arange(0, len(years))
  rows = out_idx
  subrows_grid, rows_grid = np.meshgrid(subrows, rows)
  inverse_idx = np.empty((out_gdal.shape[0],2), dtype=np.uintc)
  inverse_idx[:,0] = rows_grid.flatten()
  inverse_idx[:,1] = subrows_grid.flatten()
  
  skmap_bindings.inverseReorderArray(out_t, n_threads, out_gdal, inverse_idx)
  ttprint(f"Tile {tile_id} - Transposing output: {(time.time() - start):.2f} segs")

  start = time.time()
  write_idx = range(0, out_gdal.shape[0])
  tmp_dir = str(make_tempdir(tile_id))
  base_raster = landsat_files[-1]
  
  out_files = [ f'gpw_eml.seeded.grass_30m_m_{year}0101_{year}1231_go_epsg.4326_v20240206' for year in years ] + \
             [ f'gpw_eml.semi.nat.grass_30m_m_{year}0101_{year}1231_go_epsg.4326_v20240206' for year in years ] + \
             [ f'gpw_eml.seeded.grass_30m_md_{year}0101_{year}1231_go_epsg.4326_v20240206' for year in years ] + \
             [ f'gpw_eml.semi.nat.grass_30m_md_{year}0101_{year}1231_go_epsg.4326_v20240206' for year in years ] + \
             [ f'gpw_eml.grass.type_30m_c_{year}0101_{year}1231_go_epsg.4326_v20240206' for year in years ]
  out_s3 = [ f'gaia/{s3_prefix}/{tile_id}' for o in out_files ]

  x_off_d, y_off_d = (2, 2)

  nodata_val = 255
  compression_command = f"gdal_translate -a_nodata {nodata_val} -co COMPRESS=deflate -co ZLEVEL=9 -co TILED=TRUE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024"

  skmap_bindings.writeByteData(out_gdal, n_threads, gdal_opts, base_raster, tmp_dir, out_files, write_idx,
        x_off_d, y_off_d, x_size, y_size, nodata_val, compression_command, out_s3)
  ttprint(f"Tile {tile_id} - Exporting output to S3: {(time.time() - start):.2f} segs")

  start = time.time()
  #sa.delete(array_fn)
  #sa.delete(out_fn)
  os.remove(array_fn.replace("file://",""))
  os.remove(out_fn.replace("file://",""))
  #ttprint(f"Tile {tile_id} - Cleaning SharedArray: {(time.time() - start):.2f} segs")
  ttprint(f"Tile {tile_id} - Result available in gaia {out_s3[-1]}")