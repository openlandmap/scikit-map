
from typing import Tuple, List
import os

n_threads = 96

os.environ['OMPI_MCA_rmaps_base_oversubscribe'] = '1'
os.environ['USE_PYGEOS'] = '0'
os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'
os.environ['NUMEXPR_MAX_THREADS'] = f'{n_threads}'
os.environ['NUMEXPR_NUM_THREADS'] = f'{n_threads}'
os.environ['OMP_THREAD_LIMIT'] = f'{n_threads}'
os.environ["OMP_NUM_THREADS"] = f'{n_threads}'
os.environ["OPENBLAS_NUM_THREADS"] = f'{n_threads}' # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = f'{n_threads}' # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = f'{n_threads}'

TMP_DIR = '/mnt/silva/tmp'

# Gaia S3 parameters
gaia_addrs = [f'http://192.168.49.{gaia_ip}:8333' for gaia_ip in range(30, 47)]
gaia_s3_params = {
        's3_addresses':gaia_addrs,
        's3_access_key':'iwum9G1fEQ920lYV4ol9',
        's3_secret_key':'GMBME3Wsm8S7mBXw3U4CNWurkzWMqGZ0n2rXHggS0',
        's3_prefix':'tmp-landsat-arco-v2',
    }

gaia_addrs = [f'http://192.168.49.{gaia_ip}:8333' for gaia_ip in range(30, 47)]

# S3 parameters for MinIO Client (mc)
s3_params = {
    's3_addresses':gaia_addrs,
    's3_access_key':'iwum9G1fEQ920lYV4ol9',
    's3_secret_key':'GMBME3Wsm8S7mBXw3U4CNWurkzWMqGZ0n2rXHggS0',
    's3_prefix':'tmp-landsat-arco-v2',
}

# Function to set up S3 aliases using MinIO Client (mc)
# This function takes access key, secret key, and a list of Gaia addresses,
def s3_setup(access_key, secret_key, gaia_addrs) -> List[str]:
    import subprocess

    s3_aliases = []
    s3_aliases = [f'g{i+1}' for i, _ in enumerate(gaia_addrs)]
    commands = [
        f'sudo mc alias set  g{i+1} {addr} {access_key} {secret_key} --api S3v4'
        for i, addr in enumerate(gaia_addrs)
    ]
    for cmd in commands:
        subprocess.run(cmd, shell=True, capture_output=False, text=True, check=True)
    return s3_aliases

s3_aliases = s3_setup(s3_params['s3_access_key'],
             s3_params['s3_secret_key'],
             s3_params['s3_addresses'])

# GDAL options for reading and writing
gdal_opts = {
 'GDAL_HTTP_VERSION': '1.0',
 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
}

no_data_out = 65000


gdal_co = ['TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE', 'BLOCKXSIZE=1024', 'BLOCKYSIZE=1024']

bands_prefix = ['red_glad',
                'nir_glad',
                'blue_glad',
                'green_glad',
                'swir1_glad',
                'swir2_glad',
                'thermal_glad',
                'qa_mask']

bands_prefix_out = ['red_glad',
                    'nir_glad',
                    'blue_glad',
                    'green_glad',
                    'swir1_glad',
                    'swir2_glad',
                    'thermal_glad']

file_ending_out = '_go_epsg.4326_v7'

# Landsat time-series parameters
doy_start = ['0101', '0117', '0202', '0218', '0305', '0321', '0406', '0422', '0508', '0524', '0609',
             '0625', '0711', '0727', '0812', '0828', '0913', '0929', '1015', '1031', '1116', '1202', '1218']

doy_end = ['0116', '0201', '0217', '0304', '0320', '0405', '0421', '0507', '0523', '0608', '0624',
           '0710', '0726', '0811', '0827', '0912', '0928', '1014', '1030', '1115', '1201', '1217', '1231']

month_start = ['0101', '0201', '0301', '0401', '0501', '0601', '0701', '0801', '0901', '1001', '1101', '1201']
month_end = ['0131', '0228', '0331', '0430', '0531', '0630', '0731', '0831', '0930', '1031', '1130', '1231']

# SWA time-series reconstruction parameters
att_env, att_seas, future_scaling = (20.0, 40.0, 0.1)

# MODIS NDVI filtering parameters
# diff_th, count_th = (3000, int(0.3*n_s))
# diff_th, count_th = (1000, 12*n_years)
#diff_th, count_th = (2500, 2*n_years)

resampling_strategy = "GRA_Bilinear"

x_off = y_off = 0  # GLAD and MODIS images are aligned to the top-left corner
x_size = y_size = 4004  # GLAD and MODIS images are 4004x4004 pixels
n_pix = x_size * y_size
n_imag_per_year = 23
n_imag_per_year_agg = 12
no_data = 0

landsat_file_ending = '_go_epsg.4326_v20240521.tif'

# Masking parameters
mask_band_scaling = 1/4e4
mask_result_scaling = 1e4
mask_result_offset = 0.

# MODIS NDVI filtering parameters
# diff_th, count_th = (3000, int(0.3*n_s))
# diff_th, count_th = (1000, 12*n_years)
#filter_diff_th, filter_count_th = (2500, 2*n_years)
def filter_params(n_years: int ) -> Tuple[int, int]:
    """
    Returns the filtering parameters based on the number of years.
    """
    filter_diff_th, filter_count_th = (2500, 2*n_years)
    return filter_diff_th, filter_count_th