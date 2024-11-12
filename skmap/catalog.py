import os
os.environ['USE_PYGEOS'] = '0'
os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'
import skmap_bindings as sb
import rasterio
from osgeo import gdal
import numpy as np
import joblib
import json
import re
from datetime import datetime
import time
import random
import subprocess
import shutil
import sys
import re
import skmap.misc
from skmap.misc import TimeTracker
n_threads = os.cpu_count() * 2
os.environ['OMPI_MCA_rmaps_base_oversubscribe'] = '1'
os.environ['USE_PYGEOS'] = '0'
os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'
os.environ['NUMEXPR_MAX_THREADS'] = f'{n_threads}'
os.environ['NUMEXPR_NUM_THREADS'] = f'{n_threads}'
os.environ['OMP_THREAD_LIMIT'] = f'{n_threads}'
os.environ["OMP_NUM_THREADS"] = f'{n_threads}'
os.environ["OPENBLAS_NUM_THREADS"] = f'{n_threads}' # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = f'{n_threads}' # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = f'{n_threads}' # export VECLIB_MAXIMUM_THREADS=4
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

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

def get_whale_dependencies(whale, key, main_catalog):
    func_name, params = parse_template_whale(whale)
    dep_tags = []
    dep_names = []
    dep_paths = []
    if func_name == 'percentileAggregation':
        if (params['entry_template'].startswith("@")):
            tag = params['entry_template'][1:]
            for dt in params['dt']:
                dep_tags += [tag.format(dt=dt)]
    elif func_name == 'computeNormalizedDifference':
        if (params['idx_plus'].startswith("@")):
            dep_tags += [params['idx_plus'][1:]]
        if (params['idx_minus'].startswith("@")):
            dep_tags += [params['idx_minus'][1:]]
    elif func_name == 'computeSavi':
        if (params['idx_red'].startswith("@")):
            dep_tags += [params['idx_red'][1:]]
        if (params['idx_nir'].startswith("@")):
            dep_tags += [params['idx_nir'][1:]]
    for dep_tag in dep_tags:
        dep_path = main_catalog[key][dep_tag]['path']
        dep_paths += [dep_path]
        dep_names += [dep_tag]
        if dep_path.startswith("/whale"):
            rec_dep_names, rec_dep_paths = get_whale_dependencies(dep_path, key, main_catalog)
            dep_paths += rec_dep_paths
            dep_names += rec_dep_names
    return dep_names, dep_paths

def parse_template_whale(whale):
    func_name_match = re.search(r'/whale/([^?]+)', whale)
    func_name = func_name_match.group(1) if func_name_match else None
    query_params_string = whale.split('?')[1]
    params = {}

    for param in query_params_string.split('&'):
        key, value = param.split('=')
        # Split by commas to handle lists
        if ',' in value:
            params[key] = value.split(',')
        else:
            params[key] = value
    return func_name, params
#
def read_json(path):
    with open(path, 'r') as file:
        items = json.load(file)
    return items
#



def _s3_computed_files(out_s3):
    bash_command = f"mc ls {out_s3}"
    process = subprocess.Popen(bash_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    output = output.decode('utf-8')
    error = error.decode('utf-8')
    assert (error == ''), f"Error in checking if the tile in S3 `{out_s3}` was already computed. \nError: {error}"
    return len(output.splitlines())
#
def s3_list_files(s3_aliases, s3_prefix, tile_id, file_prefix):
    if len(s3_aliases) == 0: return []
    bash_cmd = f"mc ls {s3_aliases[0]}{s3_prefix}/{tile_id}/{file_prefix}"
    print(f'Checking `{bash_cmd}`...')
    process = subprocess.Popen(bash_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    stderr = stderr.decode('utf-8')
    assert stderr == '', f"Error listing S3 `{s3_aliases[0]}{s3_prefix}/{tile_id}/{file_prefix}`. \nError: {stderr}"
    stdout = stdout.decode('utf-8')
    return stdout.splitlines()
#
def _get_out_files_depths(out_files_prefix, out_files_suffix, tile_id, depths, num_depths, years, num_years, num_stats):
    assert(len(out_files_prefix) == len(out_files_suffix))
    assert(len(out_files_prefix) == num_stats)
    assert(len(depths) >= num_depths)
    assert(len(years) >= num_years)
    out_files = []
    for i in range(num_depths):
        for k in range(num_stats):
            for j in range(num_years):
                if num_years < len(years):
                    y1 = years[j]
                    y2 = years[j + len(years) - num_years]
                    if num_depths < len(depths):
                        d1 = depths[i]
                        d2 = depths[i + len(depths) - num_depths]
                        file = f'{out_files_prefix[k]}_b{d1}cm..{d2}cm_{y1}0101_{y2}1231_tile.{tile_id}_{out_files_suffix[k]}'
                    else:
                        d1 = depths[i]
                        file = f'{out_files_prefix[k]}_b{d1}cm_{y1}0101_{y2}1231_tile.{tile_id}_{out_files_suffix[k]}'
                else:
                    y1 = years[j]
                    if num_depths < len(depths):
                        d1 = depths[i]
                        d2 = depths[i + len(depths) - num_depths]
                        file = f'{out_files_prefix[k]}_b{d1}cm..{d2}cm_{y1}0101_{y1}1231_tile.{tile_id}_{out_files_suffix[k]}'
                    else:
                        d1 = depths[i]
                        file = f'{out_files_prefix[k]}_b{d1}cm_{y1}0101_{y1}1231_tile.{tile_id}_{out_files_suffix[k]}'
                out_files.append(file)
    return out_files
#
def _get_out_files(out_files_prefix, out_files_suffix, tile_id, years, num_stats):
    out_files = []
    for k in range(num_stats):
        for year in years:
            file = f'{out_files_prefix[k]}_s_{year}0101_{year}1231_tile.{tile_id}_{out_files_suffix[k]}'
            out_files.append(file)
    return out_files
#
def _create_image_template(base_img_path, tiles, tile_id, x_size, y_size, dtype, nodata, out_dir):
    out_dir = _make_dir(out_dir)
    template_tif = f"{out_dir}/tile_{tile_id}_{dtype}.tif"
    if os.path.exists(template_tif):
        os.remove(template_tif)
    if not os.path.isfile(template_tif):
        min_x, min_y, max_x, max_y = tiles[tiles['id'] == tile_id].iloc[0].geometry.bounds
        ds = rasterio.open(base_img_path)
        window = rasterio.windows.from_bounds(min_x, min_y, max_x, max_y, transform=ds.transform)
        transform = rasterio.windows.transform(window, ds.transform)
        with rasterio.open(
            fp=template_tif, 
            mode='w',
            driver='GTiff',
            height=y_size,
            width=x_size,
            count=1,
            dtype=dtype,
            crs=ds.crs,
            compress='deflate',
            transform=transform,
            nodata=nodata
        ) as dst:
            dst.write(np.zeros((1, x_size * y_size), dtype=dtype), 1)
    return template_tif
#
def s3_setup(have_to_register_s3, access_key, secret_key, n_gaia):
    s3_aliases = []
    if not have_to_register_s3:
        return s3_aliases
    s3_aliases = [f'g{i}' for i in range(1, n_gaia+1)]
    commands = [
        f'mc alias set  g{i} http://192.168.49.{i+29}:8333 {access_key} {secret_key} --api S3v4'
        for i in range(1, n_gaia+1)
    ]
    for cmd in commands:
        subprocess.run(cmd, shell=True, capture_output=False, text=True, check=True)
    return s3_aliases
#
def s3_have_to_compute_tile(models_pool, tile_id, s3_aliases, years):
    compute_tile = False
    if len(s3_aliases) == 0:
        return True
    for model in models_pool:
        if model['s3_prefix'] is None:
            compute_tile = True
            break
        # check if files were already produced
        n_out_files = model.n_out_layers * years
        # generate file output names
        for k in range(model.n_out_stats):
            print(f'Checking `mc ls {s3_aliases[0]}{model.s3_prefix}/{tile_id}/{model.out_files_prefix[k]}_`')
            if _s3_computed_files(f'{s3_aliases[0]}{model.s3_prefix}/{tile_id}/{model.out_files_prefix[k]}_') < n_out_files:
                compute_tile = True
                break
        if compute_tile:
            break
    return compute_tile
#
class DataCatalog():
    def __init__(self, catalog_name, data, years, features, num_features) -> None:
        self.catalog_name = catalog_name
        self.data = data
        self.years = years
        self.features = features
        self.num_features = num_features
    @staticmethod
    def _get_years(json_data):
        return list({k for k in json_data.keys() if k != 'static'})
    @staticmethod
    def _get_features(json_data):
        return list({k for v in json_data.values() for k in v.keys()})
    @classmethod
    def read_catalog(cls, catalog_name, path, additional_otf_names = None):
        json_data = read_json(path)
        years = cls._get_years(json_data)
        features = cls._get_features(json_data)
        # features - populate static and temporal entries
        data = {}
        entries = ['static'] + years
        num_features = 0
        for k in entries:
            for f in features:
                if f not in json_data[k]:
                    continue
                if k not in data:
                    data[k] = {}
                data[k][f] = {'path': json_data[k][f], 'idx': num_features}
                num_features += 1
        if additional_otf_names is not None:
            for k in additional_otf_names:
                for otf in additional_otf_names[k]:
                    data[k][otf] = {'path': additional_otf_names[k][otf], 'idx': num_features}
                    num_features += 1
        return DataCatalog(catalog_name, data, years, features, num_features)
    def get_paths(self):
        # prepare temporal and static paths and indexes
        paths = []
        idx = []
        for k in self.data:
            if k == 'otf':
                continue
            for f in self.data[k]:
                if not self.data[k][f]['path'].startswith("/whale"):
                    paths += [self.data[k][f]['path']]
                    idx += [self.data[k][f]['idx']]
        # modify paths of non VRT files
        paths = [path if path is None or path.endswith('vrt') else f'/vsicurl/{path}' for path in paths]
        return paths, idx
    def get_whales(self):
        # prepare temporal and static paths and indexes
        whales = []
        keys = []
        idx = []
        for k in self.data:
            if k == 'otf':
                continue
            for f in self.data[k]:
                if self.data[k][f]['path'].startswith("/whale"):
                    whales += [self.data[k][f]['path']]
                    idx += [self.data[k][f]['idx']]
                    keys += [k]
        return whales, idx, keys
    def get_otf_idx(self):
        otf_idx = {}
        if 'otf' in self.data:
            for y in self.years:
                for f in self.features:
                    if f in self.data['otf'][y]:
                        if f not in otf_idx:
                            otf_idx[f] = []
                        otf_idx[f] += [self.data['otf'][y][f]['idx']]
        return otf_idx
    def query(self, catalog_name, years, features):
        data = {}
        entries = ['static'] + years
        num_features = 0
        # features - populate static and temporal entries
        for k in entries:
            for f in features:
                if f in self.data[k]:
                    if k not in data:
                        data[k] = {}
                    data[k][f] = {'path': self.data[k][f]['path'], 'idx': num_features}
                    num_features += 1
        # features - populate OTF entries
        for y in years:
            for f in features:
                if f not in self.features:
                    if 'otf' not in data:
                        data['otf'] = {}
                    if y not in data['otf']:
                        data['otf'][y] = {}
                    data['otf'][y][f] = {'path': None, 'idx': num_features}
                    num_features += 1
        tmp_catalog = DataCatalog(catalog_name, data, years, features, num_features)
        # recurisvely get the dependencies here and create a new catalog that includes them
        data, num_features = tmp_catalog.expand_whales_dependencies(self.data, data, num_features)
        return DataCatalog(catalog_name, data, years, features, num_features)
    def expand_whales_dependencies(self, main_catalog, data, num_features):
        whales_paths, _, keys = self.get_whales()
        for i, whale in enumerate(whales_paths):
            dep_names, dep_paths = get_whale_dependencies(whale, keys[i], main_catalog)
            for dep_name, dep_path in zip(dep_names, dep_paths):
                if dep_name not in data[keys[i]]:
                    data[keys[i]][dep_name] = {'path': dep_path, 'idx': num_features}
                    num_features += 1
        return data, num_features
    def _get_covs_idx(self, covs_lst):
        covs_idx = np.zeros((len(covs_lst), len(self.years)), np.int32)
        for j in range(len(self.years)):
            k = self.years[j]
            for i in range(len(covs_lst)):
                c = covs_lst[i]
                if c in self.data['static']:
                    covs_idx[i, j] = self.data['static'][c]['idx']
                elif c in self.data[k]:
                    covs_idx[i, j] = self.data[k][c]['idx']
                else:
                    covs_idx[i, j] = self.data['otf'][k][c]['idx']
        return covs_idx
#
def print_catalog_statistics(catalog:DataCatalog):
    print(f'[{catalog.catalog_name}]')
    entries = list(catalog.data.keys())
    entries.sort()
    print(f'catalog entries: {entries}')
    print(f'- features: {catalog.num_features}')
    print(f'- rasters to read: {len(catalog.get_paths()[0])}')
    print(f'- on-the-fly features: {len(catalog.get_otf_idx())}')
    if len(catalog.get_otf_idx()) > 0:
        otf_list = list(catalog.get_otf_idx().keys())
        otf_list.sort()
        print(f'- otf list: {otf_list}')
    print('')
#

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

class DataLoaderTiled():
    def __init__(self, catalog:DataCatalog, mask_template_path, gdal_opts, num_gaia, resampling_strategy, threads = os.cpu_count() * 2) -> None:
        self.catalog = catalog
        self.years = self.catalog.years
        self.num_years = len(self.years)
        self.mask_template_path = mask_template_path
        self.otf_const_idx = {}
        self.otf_indicator_params = []
        self.gdal_opts = gdal_opts
        self.cache = None
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
        self.threads = threads
        self.num_gaia = num_gaia
        self.executor = ProcessPoolExecutor(max_workers=self.threads)
       
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

        with TimeTracker(f"Tile {tile_id}/catalog {self.catalog.catalog_name} - load tile", True):
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
            self.cache = np.empty((self.catalog.num_features, self.num_pixels), dtype=np.float32)
            with TimeTracker(f"Tile {self.tile_id} - read images"):
                paths, paths_idx = self.catalog.get_paths()
                tile_paths, tile_idxs = zip(*[(path, idx) for path, idx in zip(paths, paths_idx) if '{tile_id}' in path])
                mosaic_paths, mosaic_idxs = zip(*[(path, idx) for path, idx in zip(paths, paths_idx) if '{tile_id}' not in path])
                tile_paths = [p.format(tile_id=self.tile_id) for p in tile_paths]
                
                # Reading resampled data
                # @FIXME we could copy locally the mask and then delete to reduce number of requests
                tile_template_paths = [self.mask_path for i in range(len(mosaic_paths))]
                futures = [self.executor.submit(warp_tile, i, tile_template_paths[i], mosaic_paths[i], self.threads, 
                    self.num_pixels, self.resampling_strategy, self.gdal_opts)
                    for i in range(len(mosaic_paths))]
                for i, future in enumerate(futures):
                    self.cache[mosaic_idxs[i], :] = future.result()

                # Reading tiled data
                if spatial_aggregation:
                    N = len(tile_paths)
                    tmp_data = np.empty((N,4000*4000), dtype=np.float32)
                    sb.readData(tmp_data, self.threads, tile_paths, range(N), 2, 2, 4000, 4000, [1], self.gdal_opts, 255, 0)
                    arr_reshaped = tmp_data.reshape(N, 4000, 4000)
                    arr_aggregated = arr_reshaped.reshape(N, 500, 8, 500, 8).mean(axis=(2, 4))
                    arr_final = arr_aggregated.reshape(N, 500*500)
                    self.cache[tile_idxs,:] = arr_final[:,:]
                else:
                    sb.readData(self.cache, self.threads, tile_paths, tile_idxs, self.x_off, self.y_off, self.x_size, self.y_size, [1], self.gdal_opts)
                
                # Computing on the fly coovariates
                whale_paths, whale_idxs, whale_keys = self.catalog.get_whales()        
                max_exec_order = 0
                for i, whale in enumerate(whale_paths):
                    _, params = parse_template_whale(whale)
                    max_exec_order = max(max_exec_order, int(params['exec_order']))
                
                for exec_order in range(max_exec_order + 1):
                    for i, whale in enumerate(whale_paths):
                        func_name, params = parse_template_whale(whale)
                        if exec_order != int(params['exec_order']):
                            continue
                        else:
                            # @FIXME currently it is assued that only tags are used and not paths
                            if func_name == 'percentileAggregation':                                
                                tag = params['entry_template'][1:]
                                in_idxs = []
                                for dt in params['dt']:
                                    in_idxs += [self.catalog.data[whale_keys[i]][tag.format(dt=dt)]['idx']]
                                # @FIXME with this setting we do the sorting for percentiles N times for each used percentile
                                # @FIXME implement the percentiles without the need of transposition
                                array_sb = np.empty((len(in_idxs),self.cache.shape[1]), dtype=np.float32)
                                array_sb_t = np.empty((array_sb.shape[1],array_sb.shape[0]), dtype=np.float32)
                                array_pct_t = np.empty((array_sb.shape[1],1), dtype=np.float32)                               
                                sb.extractArrayRows(self.cache, self.threads, array_sb, in_idxs)
                                sb.transposeArray(array_sb, self.threads, array_sb_t)
                                sb.computePercentiles(array_sb_t, self.threads, range(len(in_idxs)),
                                        array_pct_t, [0], [float(params['percentile'])])
                                self.cache[whale_idxs[i], :] = array_pct_t[:,0]
                            elif func_name == 'computeNormalizedDifference':
                                sb.computeNormalizedDifference(self.cache, self.threads,
                                    [self.catalog.data[whale_keys[i]][params['idx_plus'][1:]]['idx']],
                                    [self.catalog.data[whale_keys[i]][params['idx_minus'][1:]]['idx']],
                                    [whale_idxs[i]], float(params['scale_plus']), float(params['scale_minus']),
                                    float(params['scale_result']), float(params['offset_result']),
                                    [float(params['clip'][0]), float(params['clip'][1])])
                            elif func_name == 'computeSavi':                                
                                sb.computeSavi(self.cache, self.threads,
                                    [self.catalog.data[whale_keys[i]][params['idx_red'][1:]]['idx']],
                                    [self.catalog.data[whale_keys[i]][params['idx_nir'][1:]]['idx']],
                                    [whale_idxs[i]], float(params['scale_plus']), float(params['scale_minus']),
                                    float(params['scale_result']), float(params['offset_result']),
                                    [float(params['clip'][0]), float(params['clip'][1])])
                            else:
                                sys.exit(f"The whale function {func_name} is not available")
                if convert_nan_to_num:
                    sb.maskNan(self.cache, self.threads, range(self.cache.shape[0]), 0.0)
    def get_pixels_valid_idx(self, num_years):
        return np.concatenate([self._pixels_valid_idx + self.num_pixels * i for i in range(num_years)]).tolist()
    def create_image_template(self, dtype, nodata, out_dir):
        return _create_image_template(self.mask_path, self.tiles, self.tile_id, self.x_size, self.y_size, dtype, nodata, out_dir)
    def fill_otf_constant(self, otf_name, otf_const):
        otf_idx = self.catalog.get_otf_idx()
        assert(otf_name in otf_idx)
        otf_name_idx = otf_idx[otf_name]
        self.cache[otf_name_idx] = otf_const
#


class DataLoader():
    def __init__(self, catalog:DataCatalog, tiles, mask_path, valid_mask_value) -> None:
        self.catalog = catalog
        self.years = self.catalog.years
        self.num_years = len(self.years)
        self.tiles = tiles
        assert('id' in tiles.columns)
        self.mask_path = mask_path
        gdal_img = gdal.Open(self.mask_path)
        self.total_x_size = gdal_img.RasterXSize
        self.total_y_size = gdal_img.RasterYSize
        self.gt = gdal_img.GetGeoTransform()
        self.gti = gdal.InvGeoTransform(self.gt)
        self.valid_mask_value = valid_mask_value
        self.otf_const_idx = {}
        self.otf_indicator_params = []
        self.cache = None
        self.tile_id = None
        self.x_off = None
        self.y_off = None
        self.x_size = None
        self.y_size = None
        self.num_pixels = None
        self.mask = None
        self._pixels_valid_idx = None
        self.num_pixels_valid = None
        self.threads = None
    def _get_block(self):
        tile = self.tiles[self.tiles['id'] == self.tile_id]
        assert(len(tile) > 0)
        min_x, min_y, max_x, max_y = tile.iloc[0].geometry.bounds
        x_off, y_off = gdal.ApplyGeoTransform(self.gti, min_x, max_y)
        x_off, y_off = int(x_off), int(y_off)
        x_size, y_size = int(abs(max_x - min_x) / self.gt[1]) , int(abs(max_y - min_y) / self.gt[1])
        return x_off, y_off, min(x_size, self.total_x_size - x_off), min(y_size, self.total_y_size - y_off)
    def free(self):
        self.cache = None
    def load_tile_data(self, tile_id, threads, gdal_opts, x_size = None, y_size = None):
        with TimeTracker(f"tile {tile_id}/catalog {self.catalog.catalog_name} - load tile", True):
            self.tile_id = tile_id
            self.threads = threads
            self.x_off, self.y_off, self.x_size, self.y_size = self._get_block()
            if x_size is not None:
                self.x_size = x_size
            if y_size is not None:
                self.y_size = y_size
            self.num_pixels = self.x_size * self.y_size
            # prepare mask
            with TimeTracker(f"tile {self.tile_id} - prepare mask ({threads} threads)"):
                self.mask = np.zeros((1, self.num_pixels), dtype=np.float32)
                sb.readData(self.mask, 1, [self.mask_path], [0], self.x_off, self.y_off, self.x_size, self.y_size, [1], gdal_opts)
                self.mask = (self.mask == self.valid_mask_value)
                self._pixels_valid_idx = np.arange(0, self.num_pixels)[self.mask[0, :]]
                self.num_pixels_valid = np.sum(self.mask)
            # read rasters
            self.cache = np.empty((self.catalog.num_features, self.num_pixels), dtype=np.float32)
            with TimeTracker(f"tile {self.tile_id} - read images ({threads} threads)"):
                paths, paths_idx = self.catalog.get_paths()
                sb.readData(self.cache, self.threads, paths, paths_idx, self.x_off, self.y_off, 
                                        self.x_size, self.y_size, [1], gdal_opts)
            # fill otf
            if len(self.otf_indicator_params) > 0:
                with TimeTracker(f"tile {self.tile_id} - fill otf features"):
                    for i in range(len(self.otf_indicator_params)):
                        self.fill_otf_indicator(**self.otf_indicator_params[i])
    def get_pixels_valid_idx(self, num_years):
        return np.concatenate([self._pixels_valid_idx + self.num_pixels * i for i in range(num_years)]).tolist()
    def create_image_template(self, dtype, nodata, out_dir):
        return _create_image_template(self.mask_path, self.tiles, self.tile_id, self.x_size, self.y_size, dtype, nodata, out_dir)
    def prep_otf_indicator(self, otf_path, otf_name, otf_code, gdal_opts):
        self.otf_indicator_params.append({'otf_path':otf_path,'otf_name':otf_name,'otf_code':otf_code,'gdal_opts':gdal_opts})
    def fill_otf_indicator(self, otf_path, otf_name, otf_code, gdal_opts):
        otf_idx = self.catalog.get_otf_idx()
        if not otf_name in otf_idx: return
        print(f'catalog {self.catalog.catalog_name} - fill otf {otf_name}')
        otf_name_idx = otf_idx[otf_name]
        otf_path = [otf_path for _ in otf_name_idx]
        sb.readData(self.cache, 1, otf_path, otf_name_idx, self.x_off, self.y_off, self.x_size, self.y_size, [1,], gdal_opts)
        self.cache[otf_name_idx] = (self.cache[otf_name_idx] == otf_code) * 1.0
    def fill_otf_constant(self, otf_name, otf_const):
        otf_idx = self.catalog.get_otf_idx()
        assert(otf_name in otf_idx)
        otf_name_idx = otf_idx[otf_name]
        self.cache[otf_name_idx] = otf_const
#

