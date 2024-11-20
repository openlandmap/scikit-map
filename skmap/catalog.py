from typing import List, Union, Callable, Optional
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
import pandas as pd
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
os.environ["OPENBLAS_NUM_THREADS"] = f'{n_threads}' 
os.environ["MKL_NUM_THREADS"] = f'{n_threads}'
os.environ["VECLIB_MAXIMUM_THREADS"] = f'{n_threads}'
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

def get_whale_dependencies(whale, key, main_catalog, whale_layer_name):
    func_name, params = parse_template_whale(whale)
    dep_tags = []
    dep_names = []
    dep_paths = []
    dep_exec_orders = []
    if func_name == 'percentileAggregation':
        tag = params['entry_template']
        for dt in params['dt']:
            dep_tags += [tag.format(dt=dt)]
    elif func_name == 'computeNormalizedDifference':
        dep_tags += [params['idx_plus']]
        dep_tags += [params['idx_minus']]
    elif func_name == 'computeSavi':
        dep_tags += [params['idx_red']]
        dep_tags += [params['idx_nir']]
    for dep_tag in dep_tags:
        dep_path = main_catalog[key][dep_tag]['path']
        if dep_path.startswith("/whale"):
            rec_dep_names, rec_dep_paths, rec_dep_exec_orders = get_whale_dependencies(dep_path, key, main_catalog, dep_tag)
            dep_paths += rec_dep_paths
            dep_names += rec_dep_names
            dep_exec_orders += rec_dep_exec_orders
        else:
            dep_paths += [dep_path]
            dep_names += [dep_tag]
            dep_exec_orders += [0]
    dep_paths += [whale]
    dep_names += [whale_layer_name]
    if dep_exec_orders:
        dep_exec_orders += [max(dep_exec_orders) + 1]
    else:
        dep_exec_orders += [0]
    return dep_names, dep_paths, dep_exec_orders

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


def _create_dict_catalog(
        catalog_def:Union[pd.DataFrame, str],
        years
    ):
    if not isinstance(catalog_def, pd.DataFrame):
        covar = pd.read_csv(catalog_def)
    else:
        covar = catalog_def

    # Replace placeholders in `layer_name` and `path`
    def replace_layer_name_placeholders(value):
        replacements = {
            "{year}": "YYYY",
            "{year_plus_one}": "YYPO",
            "{year_minus_one}": "YYMO",
            "{start_month}": "{start_month}",
            "{end_month}": "{end_month}",
            "{perc}": "{perc}",
            "{dt}": "{{dt}}",
        }
        if isinstance(value, str):
            for old, new in replacements.items():
                value = value.replace(old, new)
        return value

    covar['layer_name'] = covar['layer_name'].apply(replace_layer_name_placeholders)
    covar['path'] = covar['path'].apply(lambda x: replace_layer_name_placeholders(x) if '/whale/' in x else x)


    perc_mask = (
        covar['layer_name'].str.contains(r'\{perc\}') | 
        covar['path'].str.contains(r'\{perc\}')
    )
    perc_expanded_rows = []
    for _, row in covar[perc_mask].iterrows():
        perc_values = [p.strip() for p in (row['perc'].split(',') if pd.notna(row['perc']) else [None])]
        for perc in perc_values:
            new_row = row.copy()
            if perc:
                new_row['layer_name'] = new_row['layer_name'].replace('{perc}', perc)
                new_row['path'] = new_row['path'].replace('{perc}', perc)
            perc_expanded_rows.append(new_row)
    perc_expanded_df = pd.DataFrame(perc_expanded_rows)
    # Combine expanded rows with the rest of the dataframe
    covar = pd.concat([covar[~perc_mask], perc_expanded_df], ignore_index=True)

    month_mask = (
        covar['layer_name'].str.contains(r'\{start_month\}') | 
        covar['path'].str.contains(r'\{start_month\}') | 
        covar['layer_name'].str.contains(r'\{end_month\}') | 
        covar['path'].str.contains(r'\{end_month\}')
    )
    month_expanded_rows = []
    for _, row in covar[month_mask].iterrows():
        start_month_values = [sm.strip() for sm in (row['start_month'].split(',') if pd.notna(row['start_month']) else [None])]
        end_month_values = [em.strip() for em in (row['end_month'].split(',') if pd.notna(row['end_month']) else [None])]
        max_len = max(len(start_month_values), len(end_month_values))
        start_month_values = start_month_values or [None] * max_len
        end_month_values = end_month_values or [None] * max_len
        for start_month, end_month in zip(start_month_values, end_month_values):
            new_row = row.copy()
            if start_month:
                new_row['layer_name'] = new_row['layer_name'].replace('{start_month}', start_month)
                new_row['path'] = new_row['path'].replace('{start_month}', start_month)
            if end_month:
                new_row['layer_name'] = new_row['layer_name'].replace('{end_month}', end_month)
                new_row['path'] = new_row['path'].replace('{end_month}', end_month)
            month_expanded_rows.append(new_row)
    month_expanded_df = pd.DataFrame(month_expanded_rows)
    covar = pd.concat([covar[~month_mask], month_expanded_df], ignore_index=True)

    # Separate common and temporal data
    covar_comm = covar[covar['type'] == 'common'].reset_index(drop=True)
    covar_temp = covar[covar['type'] == 'temporal'].reset_index(drop=True)
    # Create the comm part of the catalog
    url_comm = covar_comm.set_index('layer_name')['path'].to_dict()
    comm = {'common': {layer_name: path for layer_name, path in url_comm.items()}}

    def calculate_year_placeholders(year, start_year, end_year, tmp_layer_name):
        valid_year = min(max(year, int(start_year)), int(end_year))
        if year != valid_year:
            print(f"Year {year} not available for layer {tmp_layer_name}, propagating year {valid_year}")
        return {
            "year": str(valid_year),
            "year_plus_one": str(valid_year + 1),
            "year_minus_one": str(valid_year - 1),
            "tile_id": '{tile_id}',
            "base_path": '{base_path}'
        }
    # Create the temporal part of the catalog
    url_temp = covar_temp.set_index('layer_name')['path'].to_dict()

    temporal = {}
    for year in years:
        year_dict = {}
        for i, (layer_name, path) in enumerate(url_temp.items()):
            year_placeholders = calculate_year_placeholders(
                year, 
                covar_temp.loc[i, 'start_year'], 
                covar_temp.loc[i, 'end_year'],
                layer_name
            )
            year_dict[layer_name] = path.format(**year_placeholders)
        temporal[str(year)] = year_dict
    catalog = {**comm, **temporal}
    return catalog




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
    def __init__(self, data, data_size):
        self.data = data
        self.data_size = data_size
    @classmethod
    def create_catalog(cls, 
                    catalog_def:Union[pd.DataFrame, str],
                    years:List[int]):
        catalog_dict = _create_dict_catalog(catalog_def, years)
        data = {}
        groups = ['common'] + [str(year) for year in years]
        features_names = cls.get_features_names(catalog_dict)
        data_size = 0
        for k in groups:
            for f in features_names:
                if f not in catalog_dict[k]:
                    continue
                if k not in data:
                    data[k] = {}
                data[k][f] = {'path': catalog_dict[k][f], 'idx': data_size}
                data_size += 1
        data, data_size = cls.expand_whales_dependencies(data, data, data_size)
        return cls(data, data_size)
    def save_json(self, json_out_path):
        if json_out_path is not None:
            with open(json_out_path, "w") as f:
                json.dump(self.data, f, indent=4)
    def get_groups(self):
        return sorted(list({k for k in self.data.keys()}))
    def copy(self):
        return DataCatalog(self.data.copy(), int(self.data_size))
    @staticmethod
    def get_features_names(catalog_dict):
        return {layer_name for _,inner_dict in catalog_dict.items() for layer_name,_ in inner_dict.items()}
    def get_features(self):
        sorted_keys = []
        for key, inner_dict in self.data.items():
            key_l2_with_idx = [(key_l2, inner_dict[key_l2]['idx']) for key_l2 in inner_dict]
            sorted_key_l2 = sorted(key_l2_with_idx, key=lambda x: x[1])
            sorted_keys.extend([key for key, _ in sorted_key_l2])
        return list(set(sorted_keys))
    def get_paths(self):
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
        paths = [path if path is None or path.endswith('vrt') or path.startswith('/vsicurl/') \
        else f'/vsicurl/{path}' for path in paths]
        return paths, idx
    @staticmethod
    def get_whales(data):
        whales, idx, keys, whales_layer_names = zip(*[
            (data[k][f]['path'], data[k][f]['idx'], k, f)
            for k in data if k != 'otf'
            for f in data[k]
            if data[k][f]['path'].startswith("/whale")
        ])
        return list(whales), list(idx), list(keys), list(whales_layer_names)
    def _get_whales(self):
        return self.get_whales(self.data)
    def query(self, groups, query_features_names):
        old_data = self.data.copy()
        self.data = {}
        self.data_size = 0
        for k in groups:
            for f in query_features_names:
                if f in old_data[k]:
                    if k not in self.data:
                        self.data[k] = {}
                    self.data[k][f] = {'path': old_data[k][f]['path'], 'idx': self.data_size}
                    self.data_size += 1                    
        self._expand_whales_dependencies(old_data)
        missing_features_names = [feature for feature in query_features_names if feature not in set(self.get_features())]
        for missing_feat_feature in missing_features_names:
            print(f'Feature {missing_feat_feature} is missing in the original catalog, adding is in the otf (on the fly) common group')
        if missing_features_names:
            self.add_otf_features(missing_features_names)
    def add_otf_features(self, otf_features):
        if 'otf' not in self.data:
            self.data['otf'] = {}
        for otf_feature in otf_features:
            self.data['otf'][otf_feature] = {'path': None, 'idx': self.data_size}
            self.data_size += 1
    @classmethod
    def expand_whales_dependencies(cls, reference_catalog_data, data, data_size):
        whales_paths, _, groups, whales_layer_names = cls.get_whales(data)
        for i, (whale_path, whale_layer_name) in enumerate(zip(whales_paths, whales_layer_names)):
            dep_names, dep_paths, dep_exec_orders = get_whale_dependencies(whale_path, groups[i], reference_catalog_data, whales_layer_names[i])
            for dep_name, dep_path, dep_exec_order in zip(dep_names, dep_paths, dep_exec_orders):
                if dep_name not in data[groups[i]]:
                    data[groups[i]][dep_name] = {'path': dep_path, 'idx': data_size, 'exec_order': dep_exec_order}
                    data_size += 1
                elif dep_name == whale_layer_name:
                    data[groups[i]][dep_name]['exec_order'] = dep_exec_order
        return data, data_size
    def _expand_whales_dependencies(self, reference_catalog_data):
        self.data, self.data_size = self.expand_whales_dependencies(reference_catalog_data, self.data, self.data_size)
    def get_otf_idx(self):
        otf_idx = {}
        if 'otf' in self.data:
            for f in self.data['otf']:
                if f not in otf_idx:
                    otf_idx[f] = []
                otf_idx[f] += [self.data['otf'][f]['idx']]
        return otf_idx
    def _get_covs_idx(self, covs_lst):
        covs_idx = np.zeros((len(covs_lst), len(self.years)), np.int32)
        for j in range(len(self.years)):
            k = self.years[j]
            for i in range(len(covs_lst)):
                c = covs_lst[i]
                if c in self.data['common']:
                    covs_idx[i,j] = self.data['common'][c]['idx']
                elif c in self.data[k]:
                    covs_idx[i,j] = self.data[k][c]['idx']
                else:
                    covs_idx[i,j] = self.data['otf'][k][c]['idx']
        return covs_idx
#
def print_catalog_statistics(catalog:DataCatalog):
    groups = list(catalog.data.keys())
    groups.sort()
    print(f'catalog groups: {groups}')
    print(f'- rasters to read: {len(catalog.get_paths()[0])}')
    print(f'- whales: {len(catalog._get_whales())}')
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
                whale_paths, whale_idxs, whale_keys, _ = self.catalog._get_whales()        
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

