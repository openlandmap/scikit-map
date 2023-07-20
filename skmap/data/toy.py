'''
Access to skmap toy demo datasets
'''

from pathlib import Path
from skmap.misc import find_files
from skmap.io import read_rasters, RasterData

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.joinpath('toy')
TOY_DATE_STEP = [109, 96, 80, 80]

def _static_raster():
  return find_files(DATA_DIR.joinpath('static'), '*.tif')

def _temporal_raster(type, subpath = None):
  
  raster_dir = DATA_DIR.joinpath(type)
  if subpath is not None:
    raster_dir = raster_dir.joinpath(subpath)
  
  base_dt = '20200913_20201201'
  
  raster_files = find_files(raster_dir, f'*{base_dt}*.tif')
  return str(raster_files[0]).replace(base_dt,'{dt}')

def rdata(verbose=True):
  return RasterData({
    'ndvi': _temporal_raster('ndvi', 'filled'),
    'swir1': _temporal_raster('swir1'),
    'static': _static_raster()
    }, verbose=verbose
  ).timespan('20141202', '20201201', 'days', 
    TOY_DATE_STEP, ignore_29feb=True
  ).read()

def ndvi_rdata(gappy=False, verbose=True):
  subpath = 'gappy' if gappy else 'filled'
  return RasterData( {
    'ndvi':_temporal_raster('ndvi', subpath)
    }, verbose=verbose
  ).timespan('20141202', '20201201', 'days', 
    TOY_DATE_STEP, ignore_29feb=True
  ).read()

def lc_samples():
  DATA_DIR.joinpath('lc').joinpath('samples.gpkg')