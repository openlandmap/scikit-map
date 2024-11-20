import os
from osgeo import gdal
import numpy as np
import subprocess
import skmap_bindings as sb
from skmap.misc import TimeTracker
from concurrent.futures import ProcessPoolExecutor
from skmap.catalog import DataCatalog, run_whales
from typing import List, Union, Callable, Optional
os.environ['USE_PYGEOS'] = '0'
os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'
n_threads = os.cpu_count() * 2
os.environ['OMPI_MCA_rmaps_base_oversubscribe'] = '1'
os.environ['USE_PYGEOS'] = '0'
os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'
os.environ['NUMEXPR_MAX_THREADS'] = f'{n_threads}'
os.environ['NUMEXPR_NUM_THREADS'] = f'{n_threads}'
os.environ['OMP_THREAD_LIMIT'] = f'{n_threads}'
os.environ["OMP_NUM_THREADS"] = f'{n_threads}'
os.environ["OPENBLAS_NUM_THREADS"] = f'{n_threads}' 
os.environ["MKL_NUM_THREADS"] = f'{n_threads}'
os.environ["VECLIB_MAXIMUM_THREADS"] = f'{n_threads}'
mask_aggregation_bash_script = '''#!/bin/bash
    if [ -z "$1" ]; then
        echo "Usage: $0 <input_tif_file>"
        exit 1
    fi
    tif_file="$1"
    basename_tif=$(basename "$tif_file" .tif)
    output_file="comp_${basename_tif}.tif"

    if [ -f "$output_file" ]; then
        rm "$output_file"
    fi
    xmin=$(gdalinfo $tif_file | grep "Upper Left" | awk -F'[(), ]+' '{print $3}')
    ymax=$(gdalinfo $tif_file | grep "Upper Left" | awk -F'[(), ]+' '{print $4}')
    xmax=$(gdalinfo $tif_file | grep "Lower Right" | awk -F'[(), ]+' '{print $3}')
    ymin=$(gdalinfo $tif_file | grep "Lower Right" | awk -F'[(), ]+' '{print $4}')
    tmp_out=$(gdalinfo "$tif_file" | grep "Pixel Size" | awk -F'[(),]' '{print $2, $3}')
    read -r pixel_size_x pixel_size_y <<< "$tmp_out"
    pixel_size_y=$(awk "BEGIN {print -1 * $pixel_size_y}")
    pixel_size_x_new=$(awk "BEGIN {print $pixel_size_x * 8}")
    pixel_size_y_new=$(awk "BEGIN {print $pixel_size_y * 8}")
    xmin_new=$(awk "BEGIN {print $xmin + 2 * $pixel_size_x}")
    ymin_new=$(awk "BEGIN {print $ymin + 2 * $pixel_size_y}")
    xmax_new=$(awk "BEGIN {print $xmax - 2 * $pixel_size_x}")
    ymax_new=$(awk "BEGIN {print $ymax - 2 * $pixel_size_y}")
    gdalwarp -te "$xmin_new" "$ymin_new" "$xmax_new" "$ymax_new" -tr "$pixel_size_x_new" "$pixel_size_y_new" -r max "$tif_file" "$output_file"
    '''


def warp_tile(rank, tile_file, mosaic_paths, n_threads, n_pix, resample, gdal_opts):
    os.environ['USE_PYGEOS'] = '0'
    os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'
    warp_data = np.empty((n_pix,), dtype=np.float32)
    try:
        sb.warpTile(warp_data, n_threads, gdal_opts, tile_file, mosaic_paths, resample)
    except:
        sb.fillArray(warp_data, n_threads, 0.0)
        print(f"Mosaic {mosaic_paths} has no data in {tile_file}, filling with 0.0")
    return warp_data

class TiledDataLoader():
    def __init__(self, catalog:DataCatalog, mask_template_path, gdal_opts, resampling_strategy, n_threads = os.cpu_count() * 2) -> None:
        self.catalog = catalog
        self.years = self.catalog.years
        self.num_years = len(self.years)
        self.mask_template_path = mask_template_path
        self.otf_const_idx = {}
        self.otf_indicator_params = []
        self.gdal_opts = gdal_opts
        self.array = None
        self.tile_id = None
        self.x_off = None
        self.y_off = None
        self.x_size = None
        self.y_size = None
        self.num_pixels = None
        self.mask = None
        self.resampling_strategy = resampling_strategy
        self._pixels_valid_idx = None
        self.num_pixels_valid = None
        self.n_threads = n_threads
        self.executor = ProcessPoolExecutor(max_workers=self.n_threads)
       
    def load_tile_data(self, tile_id, convert_nan_to_num, spatial_aggregation = False):
        self.tile_id = tile_id
        self.mask_path = self.mask_template_path.format(tile_id=tile_id)
        # @FIXME: this only work with our setting of Landsat data
        if spatial_aggregation:
            with open('mask_aggregation_bash_script.sh', 'w') as file:
                file.write(mask_aggregation_bash_script)
            subprocess.run(['chmod', '+x', 'mask_aggregation_bash_script.sh'])
            input_file = self.mask_path
            subprocess.run(['./mask_aggregation_bash_script.sh', input_file])
            self.mask_path = f"comp_{input_file.split('/')[-1].split('.')[0]}.tif"
            self.x_off, self.y_off, self.x_size, self.y_size = (0, 0, 500, 500)
        else:
            gdal_img = gdal.Open(self.mask_path)
            self.x_size = gdal_img.RasterXSize
            self.y_size = gdal_img.RasterYSize
            self.x_off, self.y_off = (0, 0)

        with TimeTracker(f"Tile {tile_id} - load tile", True):
            self.num_pixels = self.x_size * self.y_size
            # prepare mask
            with TimeTracker(f"Tile {self.tile_id} - prepare mask"):
                self.mask = np.zeros((1, self.num_pixels), dtype=np.float32)
                sb.readData(self.mask, 1, [self.mask_path], [0], self.x_off, self.y_off, self.x_size, self.y_size, [1], self.gdal_opts)
                self._pixels_valid_idx = np.arange(0, self.num_pixels)[self.mask[0, :].astype(int).astype(bool)]
                self.num_pixels_valid = int(np.sum(self.mask))
                if self.num_pixels_valid == 0:
                    return
                    
            # read rasters
            self.array = np.empty((self.catalog.num_features, self.num_pixels), dtype=np.float32)
            with TimeTracker(f"Tile {self.tile_id} - read images"):
                paths, paths_idx = self.catalog.get_paths()
                tile_paths, tile_idxs = zip(*[(path, idx) for path, idx in zip(paths, paths_idx) if '{tile_id}' in path])
                mosaic_paths, mosaic_idxs = zip(*[(path, idx) for path, idx in zip(paths, paths_idx) if '{tile_id}' not in path])
                tile_paths = [p.format(tile_id=self.tile_id) for p in tile_paths]
                
                # Reading resampled data
                # @FIXME we could copy locally the mask and then delete to reduce number of requests
                tile_template_paths = [self.mask_path for i in range(len(mosaic_paths))]
                futures = [self.executor.submit(warp_tile, i, tile_template_paths[i], mosaic_paths[i], self.n_threads, 
                    self.num_pixels, self.resampling_strategy, self.gdal_opts)
                    for i in range(len(mosaic_paths))]
                for i, future in enumerate(futures):
                    self.array[mosaic_idxs[i], :] = future.result()

                # Reading tiled data
                if spatial_aggregation:
                    N = len(tile_paths)
                    tmp_data = np.empty((N,4000*4000), dtype=np.float32)
                    sb.readData(tmp_data, self.n_threads, tile_paths, range(N), 2, 2, 4000, 4000, [1], self.gdal_opts, 255, 0)
                    arr_reshaped = tmp_data.reshape(N, 4000, 4000)
                    arr_aggregated = arr_reshaped.reshape(N, 500, 8, 500, 8).mean(axis=(2, 4))
                    arr_final = arr_aggregated.reshape(N, 500*500)
                    self.array[tile_idxs,:] = arr_final[:,:]
                else:
                    sb.readData(self.array, self.n_threads, tile_paths, tile_idxs, self.x_off, self.y_off, self.x_size, self.y_size, [1], self.gdal_opts)
                run_whales(self.catalog, self.array, self.n_threads)
                if convert_nan_to_num:
                    sb.maskNan(self.array, self.n_threads, range(self.array.shape[0]), 0.0)
    def get_pixels_valid_idx(self, num_years):
        return np.concatenate([self._pixels_valid_idx + self.num_pixels * i for i in range(num_years)]).tolist()
    def create_image_template(self, dtype, nodata, out_dir):
        return _create_image_template(self.mask_path, self.tiles, self.tile_id, self.x_size, self.y_size, dtype, nodata, out_dir)
    def fill_otf_constant(self, otf_name, otf_const):
        otf_idx = self.catalog.get_otf_idx()
        assert(otf_name in otf_idx)
        otf_name_idx = otf_idx[otf_name]
        self.array[otf_name_idx] = otf_const

