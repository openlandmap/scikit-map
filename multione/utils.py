# Various utils 

from numpy.typing import NDArray, ArrayLike


import subprocess
from datetime import datetime
import sys
from pathlib import Path
import random
import numpy as np
import sb_utils as sb


from settings import  n_imag_per_year, n_imag_per_year_agg, doy_start, doy_end, n_pix, x_size, y_size, gdal_opts, x_off, y_off
from settings import TMP_DIR, gaia_addrs, bands_prefix, landsat_file_ending, n_threads, no_data

# Function to set up S3 aliases using MinIO Client (mc)
# This function takes access key, secret key, and a list of Gaia addresses,
def s3_setup(access_key, secret_key, gaia_addrs) -> list[str]:
    s3_aliases = []
    s3_aliases = [f'g{i+1}' for i, _ in enumerate(gaia_addrs)]
    commands = [
        f'sudo mc alias set  g{i+1} {addr} {access_key} {secret_key} --api S3v4'
        for i, addr in enumerate(gaia_addrs)
    ]
    for cmd in commands:
        subprocess.run(cmd, shell=True, capture_output=False, text=True, check=True)
    return s3_aliases

def setup_gaia():
    
    from settings import gaia_s3_params

    s3_aliases = s3_setup(gaia_s3_params['s3_access_key'],
                gaia_s3_params['s3_secret_key'],
                gaia_s3_params['s3_addresses'])
    
    return s3_aliases

def ttprint(*args, **kwargs):    
    print(f'[{datetime.now():%H:%M:%S}] ', end='')
    print(*args, **kwargs, flush=True)

def make_tempdir(basedir='skmap', make_subdir = True) -> Path:
    import tempfile

    tempdir = Path(TMP_DIR).joinpath(basedir)
    if make_subdir: 
        name = Path(tempfile.NamedTemporaryFile().name).name
        tempdir = tempdir.joinpath(name)
    tempdir.mkdir(parents=True, exist_ok=True)
    return tempdir

def get_modis_ndvi_filename(year, doy_start, doy_end) -> str:
    """
    Get the filename for MODIS NDVI data based on year, start and end DOY, and band.
    """
    #return f'MOD13Q1.A{year}{doy_start}.{doy_end}.{band}.hdf'
    fn = f'/vsicurl/http://192.168.49.{random.randint(30,44)}:8333/global/veg/ndvi_mod13q1.v061_swa/ndvi_mod13q1.v061_m_250m_s_{year}{doy_start[m]}_{year}{doy_end[m]}_go_sinusoidal_v1.tif'
    return fn

def get_landsat_filenames(landsat_tile, years) -> NDArray[np.float32]:
    """
    Get the filename for Landsat data based on year, start and end month, and band.
    """
    landsat_files = []
    for b in bands_prefix:
        for year in years:
            for m in range(n_imag_per_year):
                landsat_files.append(f'{random.choice(gaia_addrs)}/prod-landsat-ard2/{landsat_tile}/raw/{b}.ard2_m_30m_s_{year}{doy_start[m]}_{year}{doy_end[m]}{landsat_file_ending}')

    n_years = len(years)
    n_spect_bands = len(bands_prefix) - 1  # Exclude
    n_s = n_years*n_imag_per_year
    n_s_agg = n_years*n_imag_per_year_agg
    landsat_data = np.empty((n_s*(n_spect_bands + 2), n_pix), dtype=np.float32)
    sb.readData(landsat_data, n_threads, landsat_files, range(len(landsat_files)), x_off, y_off, x_size, y_size, [1], gdal_opts, no_data, np.nan)
    
    return landsat_data

def get_modis_ndvi_data():
