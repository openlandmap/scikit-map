import numpy as np
import pandas as pd
import sys
from datetime import datetime
sys.path.insert(0, '..')
import skmap_bindings
from pathlib import Path
import time
import os
os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'

gdal_opts = {
    'GDAL_HTTP_VERSION': '1.0',
    'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
}
n_threads = 3
bands_list = [1,]
x_off = 0
y_off = 0
x_size, y_size = (43200, 17923)
nodata_val = 255

files_list = ['https://s3.openlandmap.org/arco/land.use.land.cover_hilda.plus_c_1km_s_20170101_20171231_go_espg.4326_v1.0.tif',
              'https://s3.openlandmap.org/arco/land.use.land.cover_hilda.plus_c_1km_s_20180101_20181231_go_espg.4326_v1.0.tif',
              'https://s3.openlandmap.org/arco/land.use.land.cover_hilda.plus_c_1km_s_20190101_20191231_go_espg.4326_v1.0.tif']


start = time.time()
n_rasters = len(files_list)
file_order = np.arange(n_rasters)
shape = (n_rasters, x_size * y_size)
array = np.empty(shape, dtype=np.float32)
skmap_bindings.readData(array, n_threads, files_list, file_order, x_off, y_off, x_size, y_size, bands_list, gdal_opts)
print(f"Read files in {time.time() - start:.2f} s", flush=True)

import matplotlib.pyplot as plt
plt.imshow(array[1,:].reshape((17923, 43200)))
plt.show()