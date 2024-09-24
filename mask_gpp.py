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

gdal_opts = {
 'GDAL_HTTP_VERSION': '1.0',
 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
}

grassland_tiles = pd.read_csv('grassland_tiles.csv')

def _gpp_files(tile, y0=2000, y1=2022):
    raster_files = []
    for year in range(2000,2022):
        raster_files += [
            f"http://192.168.49.{30+(year)%15}:8333/tmp-gpw/gpp_v202403/{tile}/gpw_gpp_lue.model_m_30m_s_{year}0101_{year}0228_go_epsg.4326_v20240603.tif",
            f"http://192.168.49.{30+(year)%15}:8333/tmp-gpw/gpp_v202403/{tile}/gpw_gpp_lue.model_m_30m_s_{year}0301_{year}0430_go_epsg.4326_v20240603.tif",
            f"http://192.168.49.{30+(year)%15}:8333/tmp-gpw/gpp_v202403/{tile}/gpw_gpp_lue.model_m_30m_s_{year}0501_{year}0630_go_epsg.4326_v20240603.tif",
            f"http://192.168.49.{30+(year)%15}:8333/tmp-gpw/gpp_v202403/{tile}/gpw_gpp_lue.model_m_30m_s_{year}0701_{year}0831_go_epsg.4326_v20240603.tif",
            f"http://192.168.49.{30+(year)%15}:8333/tmp-gpw/gpp_v202403/{tile}/gpw_gpp_lue.model_m_30m_s_{year}0901_{year}1031_go_epsg.4326_v20240603.tif",
            f"http://192.168.49.{30+(year)%15}:8333/tmp-gpw/gpp_v202403/{tile}/gpw_gpp_lue.model_m_30m_s_{year}1101_{year}1231_go_epsg.4326_v20240603.tif"
        ]
    return raster_files

def _grassland_files(tile, y0=2000, y1=2022):
    path = grassland_tiles.loc[grassland_tiles['tile'] == tile,'path'].iloc[0]
    return [
        f"http://192.168.49.{30+(year)%15}:8333/tmp-gpw/{path}/{tile}/gpw_grassland_rf.savgol.bthr_c_30m_{year}0101_{year}1231_go_epsg.4326_v1.tif"
        for year in range(2000,2022)
    ]

def _processed(tile):
    url = f'http://192.168.49.30:8333/tmp-gpw/gpp_v202403_masked/{tile}/gpw_ugpp_lue.model_m_30m_s_20000101_20001231_go_epsg.4326_v1.tif'
    r = requests.head(url)
    return (r.status_code == 200)


n_threads = 96
x_size, y_size = 4000, 4000
x_off, y_off = 2, 2
y0 = 2000
y1 = 2022
n_years = y1 - y0
n_img_per_year = 6


start_tile=int(sys.argv[1])
end_tile=int(sys.argv[2])
server_name=sys.argv[3]

base_dir = Path('/mnt/slurm/jobs/wri_gpp')
ids_fn = str(base_dir.joinpath('ard2_all_ids.csv'))
tiles_id = pd.read_csv(ids_fn)['TILE'][start_tile:end_tile]


for tile in tiles_id:
    try:
        if not _processed(tile):
            ttprint(f"Tile {tile} - Starting. ")
            
            gpp_files = _gpp_files(tile, y0=y0, y1=y1)
            grass_files = _grassland_files(tile, y0=y0, y1=y1)
            land_files = [f'http://192.168.49.30:8333/gpw/landmask/{tile}.tif']
            
            start = time.time()
            gpp_landmasked = np.empty((len(gpp_files), x_size * y_size), dtype=np.float32)
            gpp_grassmasked = np.empty((len(gpp_files), x_size * y_size), dtype=np.float32)
            sb.readData(gpp_landmasked, n_threads, gpp_files, range(len(gpp_files)), x_off, y_off, x_size, y_size, [1,], gdal_opts, 255., np.nan)
            sb.extractArrayRows(gpp_landmasked, n_threads, gpp_grassmasked, range(len(gpp_files)))
            grass_data = np.empty((len(grass_files), x_size * y_size), dtype=np.float32)
            sb.readData(grass_data, n_threads, grass_files, range(len(grass_files)), 0, 0, x_size, y_size, [1,], gdal_opts, 255., 0.)
            land_mask = np.empty((len(land_files), x_size * y_size), dtype=np.float32)
            sb.readData(land_mask, n_threads, land_files, range(len(land_files)), x_off, y_off, x_size, y_size, [1,], gdal_opts, 255., 0.)
            print(f"Tile {tile} - Read data: {(time.time() - start):.2f} s")
            
            start = time.time()
            sb.maskDataRows(gpp_landmasked, n_threads, range(gpp_landmasked.shape[0]), land_mask, 0., np.nan)
            for year in range(n_years):
                year_grass_mask = np.empty((1, x_size * y_size), dtype=np.float32)
                sb.extractArrayRows(grass_data, n_threads, year_grass_mask, [year])
                sb.maskDataRows(gpp_grassmasked, n_threads, range(n_img_per_year*year, n_img_per_year*(year+1)), year_grass_mask, 0., np.nan)
            print(f"Tile {tile} - Masking GPP for land and grassland: {(time.time() - start):.2f} s")
            
            
            start = time.time()
            gpp_landmasked_t = np.empty((x_size * y_size, len(gpp_files)), dtype=np.float32)
            gpp_grassmasked_t = np.empty((x_size * y_size, len(gpp_files)), dtype=np.float32)
            gpp_landmasked_yearly = np.empty((n_years, x_size * y_size), dtype=np.float32)
            gpp_grassmasked_yearly = np.empty((n_years, x_size * y_size), dtype=np.float32)
            gpp_landmasked_yearly_t = np.empty((x_size * y_size, n_years), dtype=np.float32)
            gpp_grassmasked_yearly_t = np.empty((x_size * y_size, n_years), dtype=np.float32)
            sb.transposeArray(gpp_landmasked, n_threads, gpp_landmasked_t)
            sb.transposeArray(gpp_grassmasked, n_threads, gpp_grassmasked_t)
            sb.averageAggregate(gpp_landmasked_t, n_threads, gpp_landmasked_yearly_t, n_img_per_year)
            sb.averageAggregate(gpp_grassmasked_t, n_threads, gpp_grassmasked_yearly_t, n_img_per_year)
            sb.transposeArray(gpp_landmasked_yearly_t, n_threads, gpp_landmasked_yearly)
            sb.transposeArray(gpp_grassmasked_yearly_t, n_threads, gpp_grassmasked_yearly)
            sb.offsetAndScale(gpp_landmasked_yearly, n_threads, 0., 36.5)
            sb.offsetAndScale(gpp_grassmasked_yearly, n_threads, 0., 36.5)
            print(f"Tile {tile} - GPP annual comulative aggregation for land and grassland: {(time.time() - start):.2f} s")
            
            
            start = time.time()
            bimonth_start = ['0101', '0301', '0501', '0701', '0901', '1101']
            bimonth_end = ['0228', '0430', '0630', '0831', '1031', '1231']
            os.makedirs('GPP_out_data', exist_ok = True)
            out_dir = f'GPP_out_data/{tile}'
            os.makedirs(out_dir, exist_ok = True)            
            no_data_uint8 = 255
            no_data_uint16 = int(65000)
            compression_command_uint8 = f"gdal_translate -a_nodata {no_data_uint8} -co COMPRESS=deflate -co ZLEVEL=9 -co TILED=TRUE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024"
            compression_command_uint16 = f"gdal_translate -a_nodata {no_data_uint16} -co COMPRESS=deflate -co ZLEVEL=9 -co TILED=TRUE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024"
            sb.maskNan(gpp_landmasked_yearly, n_threads, range(gpp_landmasked_yearly.shape[0]), no_data_uint16)
            sb.maskNan(gpp_grassmasked_yearly, n_threads, range(gpp_grassmasked_yearly.shape[0]), no_data_uint16)
            sb.maskNan(gpp_landmasked, n_threads, range(gpp_landmasked.shape[0]), no_data_uint8)
            sb.maskNan(gpp_grassmasked, n_threads, range(gpp_grassmasked.shape[0]), no_data_uint8)
            gpp_yearly = np.empty((n_years*2, x_size * y_size), dtype=np.float32)
            sb.expandArrayRows(gpp_landmasked_yearly, n_threads, gpp_yearly, range(0, n_years))
            sb.expandArrayRows(gpp_grassmasked_yearly, n_threads, gpp_yearly, range(n_years, n_years*2))
            gpp_bimonth = np.empty((len(gpp_files)*2, x_size * y_size), dtype=np.float32)
            sb.expandArrayRows(gpp_landmasked, n_threads, gpp_bimonth, range(0, len(gpp_files)))
            sb.expandArrayRows(gpp_grassmasked, n_threads, gpp_bimonth, range(len(gpp_files), len(gpp_files)*2))
            out_files_land = []
            out_files_grass = []
            out_files_land_year = []
            out_files_grass_year = []
            for year in range(y0, y1):
                out_files_land_year.append(f'gpw_ugpp_lue.model_m_30m_s_{year}0101_{year}1231_go_epsg.4326_v1')
                out_files_grass_year.append(f'gpw_gpp.grass_lue.model_m_30m_s_{year}0101_{year}1231_go_epsg.4326_v1')  
                for bm in range(n_img_per_year):
                    out_files_land.append(f'gpw_ugpp.daily_lue.model_m_30m_s_{year}{bimonth_start[bm]}_{year}{bimonth_end[bm]}_go_epsg.4326_v1')
                    out_files_grass.append(f'gpw_gpp.daily.grass_lue.model_m_30m_s_{year}{bimonth_start[bm]}_{year}{bimonth_end[bm]}_go_epsg.4326_v1')
            out_files_year = out_files_land_year + out_files_grass_year
            out_files_bimonth = out_files_land + out_files_grass
            base_files = gpp_files + gpp_files
            s3_prefix = 'tmp-gpw/gpp_v202403_masked'
            out_s3_year = [ f"g{1 + int.from_bytes(Path(o).stem.encode(), 'little') % 15}/{s3_prefix}/{tile}" for o in out_files_year ]
            out_s3_bimonth = [ f"g{1 + int.from_bytes(Path(o).stem.encode(), 'little') % 15}/{s3_prefix}/{tile}" for o in out_files_bimonth ]
            sb.writeUInt16Data(gpp_yearly, n_threads, gdal_opts, gpp_files[0:len(out_files_year)], out_dir, out_files_year, range(len(out_files_year)),
                        x_off, y_off, x_size, y_size, no_data_uint16, compression_command_uint16, out_s3_year)
            sb.writeByteData(gpp_bimonth, n_threads, gdal_opts, base_files[0:len(out_files_bimonth)], out_dir, out_files_bimonth, range(len(out_files_bimonth)),
                        x_off, y_off, x_size, y_size, no_data_uint8, compression_command_uint8, out_s3_bimonth)
            print(f"Tile {tile} - Saving data: {(time.time() - start):.2f} s")

            ttprint(f"Tile {tile} - Done. ")

        else:
            ttprint(f"Tile {tile} - Already exists. Skipping. ")    
    except:
        tb = traceback.format_exc()
        print(f"Tile {tile} - Error")
        print(tb)
        continue

