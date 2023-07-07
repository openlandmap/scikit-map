'''
Access to skmap toy demo datasets
'''

from pathlib import Path
from skmap.misc import find_files
from skmap.io import read_rasters, RasterData

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.joinpath('data')

def ndvi_files(gappy=False):
  subpath = 'gappy' if gappy else 'filled'
  return find_files(DATA_DIR.joinpath('ndvi').joinpath(subpath), '*.tif')

def ndvi_data(gappy=False, return_array = False, verbose=True):
  if return_array == False:
    subpath = 'gappy' if gappy else 'filled'
    base_dt = '20200913_20201201'
    ndvi_files = find_files(DATA_DIR.joinpath('ndvi').joinpath(subpath), f'*{base_dt}*.tif')
    ndvi_files = str(ndvi_files[0]).replace(base_dt,'{dt}')
    return RasterData(
              ndvi_files, verbose=verbose
            ).timespan('20141202', '20201201', 'days', 
              [109, 96, 80, 80], ignore_29feb=True
            ).read()
  else:
    return RasterData(ndvi_files(gappy))

def lc_samples():
  DATA_DIR.joinpath('lc').joinpath('samples.gpkg')