from skmap.misc import ttprint
import os
import shutil
os.environ['OMPI_MCA_rmaps_base_oversubscribe'] = '1'
n_threads = 96
os.environ['USE_PYGEOS'] = '0'
os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'
os.environ['NUMEXPR_MAX_THREADS'] = f'{n_threads}'
os.environ['NUMEXPR_NUM_THREADS'] = f'{n_threads}'
os.environ['OMP_THREAD_LIMIT'] = f'{n_threads}'
os.environ["OMP_NUM_THREADS"] = f'{n_threads}'
os.environ["OPENBLAS_NUM_THREADS"] = f'{n_threads}' # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = f'{n_threads}' # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = f'{n_threads}' # export VECLIB_MAXIMUM_THREADS=4
import gc
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import skmap_bindings as sb
import tempfile
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import requests
import traceback

gdal_opts = {
 'GDAL_HTTP_VERSION': '1.0',
 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
}

def _landsat_files(band, tile, y0, y1):
    month_start = ['0101','0301','0501','0701','0901','1101']
    month_end = ['0228' ,'0430' ,'0630' ,'0831' ,'1031' ,'1231']
    raster_files = []
    for year in range(y0,y1+1):
        for bm in range(len(month_start)):
            raster_files += [f"http://192.168.49.{30+random.randint(0, 16)}:8333/prod-landsat-ard2/{tile}/agg/{band}.ard2_m_30m_s_{year}{month_start[bm]}_{year}{month_end[bm]}_go_epsg.4326_v20230908.tif"]
    return raster_files

def _out_urls(band, tile, y0, y1):
    month_start = ['0101','0301','0501','0701','0901','1101']
    month_end = ['0228' ,'0430' ,'0630' ,'0831' ,'1031' ,'1231']
    raster_files = []
    for year in range(y0,y1+1):
        for bm in range(len(month_start)):
            raster_files += [f"{band}.swa.ard2_m_30m_s_{year}{month_start[bm]}_{year}{month_end[bm]}_go_epsg.4326_v1"]
    return raster_files


def _out_year_urls(band, tile, y0, y1):
    raster_files = []
    for year in range(y0,y1+1):
        raster_files += [f"{band}.swa.ard2_p50_30m_s_{year}0101_{year}1231_go_epsg.4326_v1"]
    return raster_files


def _processed(tile, urls):
    return all(requests.head(f"http://192.168.49.30:8333/prod-landsat-ard2/{tile}/v1_masked/{url}.tif").status_code == 200 for url in urls)


    
def get_SWA_weights(att_env, att_seas, season_size, n_imag):
    conv_mat_row = np.zeros((n_imag))
    base_func = np.zeros((season_size,))
    period_y = season_size/2.0
    slope_y = att_seas/10/period_y
    for i in np.arange(season_size):
        if i <= period_y:
            base_func[i] = -slope_y*i
        else:
            base_func[i] = slope_y*(i-period_y)-att_seas/10
    # Compute the envelop to attenuate temporarly far images
    env_func = np.zeros((n_imag,))
    delta_e = n_imag
    slope_e = att_env/10/delta_e
    for i in np.arange(delta_e):
        env_func[i] = -slope_e*i
        conv_mat_row = 10.0**(np.resize(base_func,n_imag) + env_func)
    return conv_mat_row

n_threads = 96
x_size, y_size = 4004, 4004
x_off, y_off = 0, 0
y0 = 1997
y1 = 2024
n_years = y1 - y0 + 1
n_img_per_year = 6
n_s = n_years * n_img_per_year
n_pix = x_size * y_size
att_env, att_seas, future_scaling = (20.0, 40.0, 0.1)


bands_prefix = ['red_glad',
                'nir_glad',
                'blue_glad',
                'green_glad',
                'swir1_glad',
                'swir2_glad',
                'thermal_glad']

start_tile=int(sys.argv[1])
end_tile=int(sys.argv[2])

with open('/mnt/slurm/jobs/wri_gpp/zero_tiles.csv', 'r') as file:
    lines = file.readlines()
lines = [line.strip() for line in lines]
tiles_id = lines[start_tile:end_tile]

no_data_uint8 = 255

save_range_bimonthly = range(n_s-n_img_per_year*5,n_s)
save_range_yearly = range(n_years-5,n_years)
n_save_bimonthly = len(save_range_bimonthly)
n_save_yearly = len(save_range_yearly)



for tile in tiles_id:
    for band in bands_prefix:
        try:
            out_urls = _out_urls(band, tile, y0, y1)
            out_urls = [out_urls[o] for o in save_range_bimonthly]
            out_year_urls = _out_year_urls(band, tile, y0, y1)
            out_year_urls = [out_year_urls[o] for o in save_range_yearly]
            if True:
                ttprint(f"Tile {tile} - Starting. ")

                band_files = _landsat_files(band, tile, y0, y1)
                land_files = [f'http://192.168.49.30:8333/global/tiled.masks/mask_dsm.landmask/{tile}.tif']

                start = time.time()
                land_mask = np.empty((1, n_pix), dtype=np.float32)
                sb.readData(land_mask, n_threads, land_files, range(1), x_off, y_off, x_size, y_size, [1,], gdal_opts, 255., 0.)
                print(f"Tile {tile} - Read mask: {(time.time() - start):.2f} s")

                start = time.time()
                agg_band = np.empty((n_s, n_pix), dtype=np.float32)
                sb.readData(agg_band, n_threads, band_files, range(n_s), x_off, y_off, x_size, y_size, [1,], gdal_opts, 255., np.nan)
                agg_band_t = np.empty((n_pix, n_s), dtype=np.float32)
                sb.transposeArray(agg_band, n_threads, agg_band_t)
                print(f"Tile {tile} - Read daily data: {(time.time() - start):.2f} s")
                                                
                start = time.time()
                rec_band_t = np.empty((n_pix, n_s), dtype=np.float32)
                rec_band = np.empty((n_s, n_pix), dtype=np.float32)
                w_p = (get_SWA_weights(att_env, att_seas, n_img_per_year, n_s)[1:][::-1]).astype(np.float32)
                w_f = (get_SWA_weights(att_env, att_seas, n_img_per_year, n_s)[1:]).astype(np.float32)*future_scaling
                w_0 = 1.0
                sb.applyTsirf(agg_band_t, n_threads, rec_band_t, 0, w_0, w_p, w_f, True, 'v2', 'Matrix')
                sb.transposeArray(rec_band_t, n_threads, rec_band)
                print(f"Tile {tile} - Reconstructing time-series data: {(time.time() - start):.2f} s")

                start = time.time()
                rec_band_yearly_t = np.empty((n_pix, n_years), dtype=np.float32)
                rec_band_yearly = np.empty((n_years, n_pix), dtype=np.float32)
                for y in range(n_years):
                    sb.computePercentiles(rec_band_t, n_threads, range(y*n_img_per_year,(y+1)*n_img_per_year), rec_band_yearly_t, [y], [50.])
                sb.transposeArray(rec_band_yearly_t, n_threads, rec_band_yearly)
                print(f"Tile {tile} - Aggregation with p50: {(time.time() - start):.2f} s")

                start = time.time()
                band_out = np.empty((n_save_bimonthly+n_save_yearly, n_pix), dtype=np.float32)
                rec_band_save = np.empty((n_save_bimonthly, n_pix), dtype=np.float32)
                rec_band_yearly_save = np.empty((n_save_yearly, n_pix), dtype=np.float32)
                sb.extractArrayRows(rec_band, n_threads, rec_band_save, save_range_bimonthly)
                sb.expandArrayRows(rec_band_save, n_threads, band_out, range(n_save_bimonthly))
                sb.extractArrayRows(rec_band_yearly, n_threads, rec_band_yearly_save, save_range_yearly)
                sb.expandArrayRows(rec_band_yearly_save, n_threads, band_out, range(n_save_bimonthly, n_save_bimonthly+n_save_yearly))
                sb.maskDataRows(band_out, n_threads, range(band_out.shape[0]), land_mask, 0., np.nan)
                sb.maskNan(band_out, n_threads, range(band_out.shape[0]), no_data_uint8)
                print(f"Tile {tile} - Masking band data: {(time.time() - start):.2f} s")

                start = time.time()
                os.makedirs('out_data', exist_ok = True)
                out_dir = f'out_data/{tile}'
                os.makedirs(out_dir, exist_ok = True)            
                no_data_uint8 = 255
                compression_command_uint8 = f"gdal_translate -a_nodata {no_data_uint8} -co COMPRESS=deflate -co TILED=TRUE -co BLOCKXSIZE=2048 -co BLOCKYSIZE=2048"
                out_files = out_urls + out_year_urls
                base_files = band_files + band_files            
                out_s3 = [ f"g{random.randint(1, 17)}/prod-landsat-ard2/{tile}/v1_masked" for o in out_files ]
                out_range = range(n_save_bimonthly+n_save_yearly)
                sb.writeByteData(band_out, n_threads, gdal_opts, base_files[0:len(out_files)], out_dir, out_files, out_range,
                            x_off, y_off, x_size, y_size, no_data_uint8, compression_command_uint8, out_s3)
                os.rmdir(out_dir)
                print(f"Tile {tile} - Saving data: {(time.time() - start):.2f} s")
                print(f"Check {out_s3[0]}/{out_files[0]}")

                ttprint(f"Tile {tile} / band {band} - Done. ")

            else:
                ttprint(f"Tile {tile} / band {band} - Already exists. Skipping. ")    
        except:
            tb = traceback.format_exc()
            print(f"Tile {tile} / band {band} - Error")
            print(tb)
            continue


