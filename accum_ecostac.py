import numpy as np
import pandas as pd
import sys
from datetime import datetime
import skmap_bindings
from pathlib import Path
import time
import os
from minio import Minio
from joblib import Parallel, delayed

os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'

gdal_opts = {
 'GDAL_HTTP_VERSION': '1.0',
 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
}
n_threads = 96
bands_list = [1,]
x_off = 0
y_off = 0
x_size, y_size = (4004, 4004)
nodata_val = -1
compression_command = f"gdal_translate -a_nodata {nodata_val} -co COMPRESS=deflate -co TILED=TRUE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 -co NUM_THREADS=8"
tmp_write_dir = '../tmp'


with open('input_file_list.txt') as file1:
    files_lists = [i.replace('\n', '') for i in file1.readlines()]

#with open('us48_tile.txt') as file1:
#    tiles_lists = [i for i in file1.readlines()][0].split(' ')

start_tile=int(sys.argv[1])
end_tile=int(sys.argv[2])
server_name=sys.argv[3]

tiles_lists=['089W_28N',
'095W_35N',
'114W_43N',
'095W_35N',
'084W_34N',
'103W_30N',
'108W_43N',
'120W_35N',
'102W_31N',
'084W_34N',
'095W_28N',
'086W_32N',
'088W_31N',
'105W_37N',
'095W_46N',
'085W_33N',
'084W_37N',
'083W_41N',
'092W_38N',
'083W_41N',
'090W_48N',
'094W_32N',
'119W_49N',
'114W_43N']

target_tiles = tiles_lists[start_tile:end_tile]
tmp_write_dir = f'/mnt/{server_name}/accum_ndvi_usa48/tmp'
os.makedirs(tmp_write_dir,exist_ok=True)


for tile in target_tiles:
    print(f'Processing {tile}')
    aug_list = [i.replace('{tile}',tile) for i in files_lists if '0701_' in i]
    ndvi_list = [i.replace('{tile}',tile) for i in files_lists if 'yearly' in i  and 'ndvi' in i]
    ndwi_list = [i.replace('{tile}',tile) for i in files_lists if 'yearly' in i  and 'ndwi' in i]
    print(f'peek in {ndwi_list[0]}')
    for files_list,layer_name in [(aug_list,'aug'),(ndvi_list,'ndvi'),(ndwi_list,'ndwi')]:
        n_rasters = len(files_list)

        file_order = np.arange(n_rasters)
        shape = (n_rasters, x_size * y_size)
        array = np.empty(shape, dtype=np.float32)
        skmap_bindings.readData(array, n_threads, files_list, file_order, x_off, y_off, x_size, y_size, bands_list, gdal_opts)
        for i in range(23):
            if i == 22:
                break
            array[22-i-1,array[22-i-1,:]==255]   =  array[22-i,array[22-i-1,:]==255]
        accum_array = np.cumsum(array,axis=0)
        
        base_raster = files_list
        if layer_name == 'aug':
            out_files = [f"accum.ndvi_glad.landsat.seasconv_m_30m_s_{file.split('/')[-1].split('_')[2]}_{file.split('/')[-1].split('_')[3][:-4]}_us_epsg.5070_v20240320" for file in files_list]
        elif layer_name == 'ndvi':
            out_files = [f"accum.ndvi_glad.landsat.seasconv.m.yearly_p50_30m_s_{file.split('/')[-1].split('_')[3]}_{file.split('/')[-1].split('_')[4][:-4]}_us_epsg.5070_v20240320" for file in files_list]
        elif layer_name == 'ndwi':
            out_files = [f"accum.ndwi_glad.landsat.seasconv.m.yearly_p50_30m_s_{file.split('/')[-1].split('_')[3]}_{file.split('/')[-1].split('_')[4][:-4]}_us_epsg.5070_v20240320" for file in files_list]

        write_idx = range(0, len(out_files))
        out_s3 = [ f'gaia/tmp-usa48-ard2/{tile}' for o in out_files ]
        skmap_bindings.writeInt16Data(accum_array, n_threads, gdal_opts, base_raster, tmp_write_dir, out_files, write_idx,
            x_off, y_off, x_size, y_size, nodata_val, compression_command,out_s3)