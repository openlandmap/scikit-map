'''
Access to skmap toy demo datasets
'''

from pathlib import Path
from skmap.misc import find_files
from skmap.io import read_rasters

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.joinpath('data')

def ndvi_files(gappy=False):
  subpath = 'gappy' if gappy else 'filled'
  return find_files(DATA_DIR.joinpath('ndvi').joinpath(subpath), '*.tif')

def read_ndvi(gappy=False):
  return read_rasters(
    ndvi_files(gappy=gappy)
  )

def read_samples():
  DATA_DIR.joinpath('lc').joinpath('samples.gpkg')