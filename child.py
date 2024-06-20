from mpi4py import MPI
import os
import shutil
os.environ['USE_PYGEOS'] = '0'
os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'
os.environ['NUMEXPR_MAX_THREADS'] = '48'
os.environ['NUMEXPR_NUM_THREADS'] = '48'
os.environ['OMP_THREAD_LIMIT'] = '48'
os.environ["OMP_NUM_THREADS"] = "48"
os.environ["OPENBLAS_NUM_THREADS"] = "48" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "48" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "48" # export VECLIB_MAXIMUM_THREADS=4
import gc
from datetime import datetime
from osgeo import gdal, gdal_array
from pathlib import Path
from typing import Callable, Iterator, List,        Union
import bottleneck as bn
import geopandas as gpd
import numpy as np
import pandas as pd
import skmap_bindings
import tempfile
import time
import sys
import random
import csv
from scipy.signal import savgol_coeffs
import numpy as np
from skmap.io import process
import matplotlib.pyplot as plt

def child_process(rank, tile_files, modis_mosaics, n_threads, n_pix):

	gdal_opts = {
	 'GDAL_HTTP_VERSION': '1.0',
	 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
	}
	warp_data = np.empty((n_pix,), dtype=np.float32)
	skmap_bindings.warpTile(warp_data, n_threads, gdal_opts, tile_files[rank], modis_mosaics[rank])
	
	return warp_data

def main():
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()

    tile_files = sys.argv[1].split(',')
    modis_mosaics = sys.argv[2].split(',')
    n_threads = int(sys.argv[3])
    n_pix = int(sys.argv[4])

    array = child_process(rank, tile_files, modis_mosaics, n_threads, n_pix)
    comm.Send(array, dest=0, tag=rank)

    comm.Disconnect()

if __name__ == "__main__":
    main()
