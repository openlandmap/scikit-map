import os
os.environ['USE_PYGEOS'] = '0'
os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'
import skmap_bindings
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
import treelite_runtime
from trees_rf import cast_tree_rf
#
def read_json(path):
    with open(path, 'r') as file:
        items = json.load(file)
    return items
#
class TimeTracker(object):
    def __init__(self, task, enter_enabled=False, exit_enabled=True):
        self.task = task
        self.enter_enabled = enter_enabled
        self.exit_enabled = exit_enabled
        self.tracks = []
    @staticmethod
    def _print(*args, **kwargs):
        print(f'[{datetime.now():%H:%M:%S}] ', end='')
        print(*args, **kwargs, flush=True)
    def __enter__(self):
        self.tracks.append(time.time())
        if self.enter_enabled:
            print(f"{self.task}")
        return self.tracks[-1]
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and self.exit_enabled:
            self._print(f"{self.task}: {(time.time() - self.tracks.pop()):.2f} secs")
        else:
            self.tracks.pop()
#
def _read_model(model_name, model_path, model_covs_path):
    if model_path.endswith('.joblib'):
        model = joblib.load(model_path)
    elif model_path.endswith('.so'):
        model = treelite_runtime.Predictor(model_path)
    else:
        raise ValueError(f"Invalid model path extension '{model_path}'")
    if model_covs_path is not None:
        with open(model_covs_path, 'r') as file:
            model_covs = [line.strip() for line in file]
    elif hasattr(model, "feature_names_in_"):
        model_covs = list(model.feature_names_in_)
    elif hasattr(model, 'feature_names_'):
        model_covs = model.feature_names_
    else:
        raise ValueError(f"No feature names was found for model {model_name}")
    return (model, model_covs)
#
def _make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
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
def s3_setup(have_to_register_s3, access_key, secret_key):
    s3_aliases = []
    if not have_to_register_s3:
        return s3_aliases
    s3_aliases = [f'g{i}' for i in range(1, 14)]
    commands = [
        f'mc alias set  g{i} http://192.168.49.{i+29}:8333 {access_key} {secret_key} --api S3v4'
        for i in range(1, 14)
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
    def read_catalog(cls, catalog_name, path):
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
        return DataCatalog(catalog_name, data, years, features, num_features)
    def get_paths(self):
        # prepare temporal and static paths and indexes
        paths = []
        idx = []
        for k in self.data:
            if k == 'otf':
                continue
            for f in self.data[k]:
                paths += [self.data[k][f]['path']]
                idx += [self.data[k][f]['idx']]
        # modify paths of non VRT files
        paths = [path if path is None or path.endswith('vrt') else f'/vsicurl/{path}' for path in paths]
        return paths, idx
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
        return DataCatalog(catalog_name, data, years, features, num_features)
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
                skmap_bindings.readData(self.mask, 1, [self.mask_path], [0], self.x_off, self.y_off, self.x_size, self.y_size, [1], gdal_opts)
                self.mask = (self.mask == self.valid_mask_value)
                self._pixels_valid_idx = np.arange(0, self.num_pixels)[self.mask[0, :]]
                self.num_pixels_valid = np.sum(self.mask)
            # read rasters
            self.cache = np.empty((self.catalog.num_features, self.num_pixels), dtype=np.float32)
            with TimeTracker(f"tile {self.tile_id} - read images ({threads} threads)"):
                paths, paths_idx = self.catalog.get_paths()
                skmap_bindings.readData(self.cache, self.threads, paths, paths_idx, self.x_off, self.y_off, 
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
        skmap_bindings.readData(self.cache, 1, otf_path, otf_name_idx, self.x_off, self.y_off, self.x_size, self.y_size, [1,], gdal_opts)
        self.cache[otf_name_idx] = (self.cache[otf_name_idx] == otf_code) * 1.0
    def fill_otf_constant(self, otf_name, otf_const):
        otf_idx = self.catalog.get_otf_idx()
        assert(otf_name in otf_idx)
        otf_name_idx = otf_idx[otf_name]
        self.cache[otf_name_idx] = otf_const
#
class RFRegressorDepths():
    def __init__(self, 
                 model_name, 
                 model_path, 
                 model_covs_path, 
                 depth_var, 
                 depths, 
                 predict_fn) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.model_covs_path = model_covs_path
        model, features = _read_model(self.model_name, self.model_path, self.model_covs_path)
        assert(hasattr(model, 'estimators_'))
        self.model = cast_tree_rf(model)
        self.num_trees = len(self.model.estimators_)
        self.model_features = features
        self.depth_var = depth_var
        self.depths = depths
        self.predict_fn = predict_fn
        self.in_covs_t = None
        self.in_covs = None
        self.in_covs_valid = None
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.in_covs_t = None
        self.in_covs = None
        self.in_covs_valid = None
    def predict(self, data:DataLoader):
        # prepare input and output arrays
        self.in_covs_t = np.empty((len(self.model_features), len(data.years) * data.num_pixels), dtype=np.float32)
        self.in_covs = np.empty((len(data.years) * data.num_pixels, len(self.model_features)), dtype=np.float32)
        self.in_covs_valid = np.empty((len(data.years) * data.num_pixels_valid, len(self.model_features)), dtype=np.float32)
        # create output result
        result = PredictedDepths(
            model_name=self.model_name, 
            depths=self.depths,
            num_depths=len(self.depths),
            years=data.years,
            num_years=len(data.years),
            num_trees=self.num_trees,
            data=data
        )
        # traverse depths
        matrix_idx = data.catalog._get_covs_idx(self.model_features)
        for i in range(len(self.depths)):
            depth = self.depths[i]
            data.fill_otf_constant(self.depth_var, depth)
            # transpose data
            # TODO create a class for skmap matrix with threads in constructor
            skmap_bindings.reorderArray(data.cache, data.threads, self.in_covs_t, matrix_idx)
            skmap_bindings.transposeArray(self.in_covs_t, data.threads, self.in_covs)
            skmap_bindings.selArrayRows(self.in_covs, data.threads, self.in_covs_valid, data.get_pixels_valid_idx(data.num_years))
            # predict
            # TODO ?implement a base class without depth and use repeatedly here?
            # self.predicted_trees shape: (num_depths, num_trees, num_years, num_pixels)
            result._out_trees_valid[i,:,:,:] = self.predict_fn(self.model, self.in_covs_valid).reshape(result.num_trees, result.num_years, result._data.num_pixels_valid)
        return result # shape: (n_depths, n_trees, n_samples)
    def predictDepth(self, data:DataLoader, i):
        # prepare input and output arrays
        self.in_covs_t = np.empty((len(self.model_features), len(data.years) * data.num_pixels), dtype=np.float32)
        self.in_covs = np.empty((len(data.years) * data.num_pixels, len(self.model_features)), dtype=np.float32)
        self.in_covs_valid = np.empty((len(data.years) * data.num_pixels_valid, len(self.model_features)), dtype=np.float32)
        # create output result
        result = PredictedDepths(
            model_name=self.model_name, 
            depths=self.depths,
            num_depths=len(self.depths),
            years=data.years,
            num_years=len(data.years),
            num_trees=self.num_trees,
            data=data
        )
        # traverse depths
        matrix_idx = data.catalog._get_covs_idx(self.model_features)
        depth = self.depths[i]
        data.fill_otf_constant(self.depth_var, depth)
        # transpose data
        # TODO create a class for skmap matrix with threads in constructor
        skmap_bindings.reorderArray(data.cache, data.threads, self.in_covs_t, matrix_idx)
        skmap_bindings.transposeArray(self.in_covs_t, data.threads, self.in_covs)
        skmap_bindings.selArrayRows(self.in_covs, data.threads, self.in_covs_valid, data.get_pixels_valid_idx(data.num_years))
        # predict
        # TODO ?implement a base class without depth and use repeatedly here?
        # self.predicted_trees shape: (num_depths, num_trees, num_years, num_pixels)
        result = self.predict_fn(self.model, self.in_covs_valid)
        return result # shape: (n_trees, num_years * num_pixels)
#
class RFClassifierProbs():
    def __init__(self, 
                 model_name, 
                 model_path, 
                 model_covs_path, 
                 num_class,
                 predict_fn) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.model_covs_path = model_covs_path
        model, features = _read_model(self.model_name, self.model_path, self.model_covs_path)
        self.model = model
        self.model_features = features
        self.num_class = num_class
        self.predict_fn = predict_fn
        self.in_covs_t = None
        self.in_covs = None
        self.in_covs_valid = None
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.in_covs_t = None
        self.in_covs = None
        self.in_covs_valid = None
    def predict(self, data:DataLoader, round=False):
        with TimeTracker(f"tile {data.tile_id}/model {self.model_name} - predict ({len(self.model_features)} input features)", True):
            # prepare input and output arrays
            self.in_covs_t = np.empty((len(self.model_features), len(data.years) * data.num_pixels), dtype=np.float32)
            self.in_covs = np.empty((len(data.years) * data.num_pixels, len(self.model_features)), dtype=np.float32)
            self.in_covs_valid = np.empty((len(data.years) * data.num_pixels_valid, len(self.model_features)), dtype=np.float32)
            # create output result
            result = PredictedProbs(
                data=data, 
                model_name=self.model_name, 
                num_class=self.num_class 
            )
            # transpose data
            matrix_idx = data.catalog._get_covs_idx(self.model_features)
            with TimeTracker(f"tile {data.tile_id}/model {self.model_name} - transpose data ({data.threads} threads)"):
                skmap_bindings.reorderArray(data.cache, data.threads, self.in_covs_t, matrix_idx)
                skmap_bindings.transposeArray(self.in_covs_t, data.threads, self.in_covs)
                skmap_bindings.selArrayRows(self.in_covs, data.threads, self.in_covs_valid, data.get_pixels_valid_idx(data.num_years))
            # predict
            with TimeTracker(f"tile {data.tile_id}/model {self.model_name} - model prediction ({data.threads} threads)"):
                result._out_probs_valid[:,:] = self.predict_fn(self.model, self.in_covs_valid) * 100
                if round:
                    np.round(result._out_probs_valid, out=result._out_probs_valid)
        return result # shape: (n_samples, n_classes)
#
# TODO copy all metadata needed from DataLoader as parameter of constructor
class PredictedDepths():
    def __init__(self, 
                 model_name, 
                 depths,
                 num_depths,
                 years,
                 num_years, 
                 num_trees,
                 data:DataLoader) -> None:
        self.model_name = model_name
        self.depths = depths
        self.num_depths = num_depths
        self.years = years
        self.num_years = num_years
        self.num_trees = num_trees
        self._data = data
        # TODO optimize shape of self.out_trees
        self._out_trees_valid = np.empty((self.num_depths, self.num_trees, self.num_years, self._data.num_pixels_valid), dtype=np.float32)
        self._out_stats_valid = None
        self._out_stats = None
        self._out_stats_t = None
        self._out_stats_gdal = None
        self.num_stats = None
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self._out_trees_valid = None
        self._out_stats_valid = None
        self._out_stats = None
        self._out_stats_t = None
        self._out_stats_gdal = None
    @property
    def predicted_trees(self):
        # self.predicted_trees shape: (num_depths, num_trees, num_years, num_pixels_valid)
        return self._out_trees_valid[:self.num_depths, :, :self.num_years, :]
    @property
    def predicted_stats(self):
        # self.predicted_stats shape: (num_years, num_pixels_valid, num_depths, num_stats) 
        if self._out_stats_valid is not None:
            return self._out_stats_valid.reshape((self.num_years, self._data.num_pixels_valid, self.num_depths, self.num_stats))
    def create_empty_copy(self, model_name):
        pred = PredictedDepths(
            model_name=model_name, 
            depths=self.depths,
            num_depths=self.num_depths,
            years=self.years,
            num_years=self.num_years,
            num_trees=self.num_trees,
            data=self._data
        )
        return pred
    def average_trees_depth_ranges(self):
        assert(self.num_depths > 1)
        self.num_depths -= 1
        for i in range(self.num_depths):
            # self.predicted_trees shape: (num_depths, num_trees, num_years, num_pixels_valid)
            self._out_trees_valid[i,:,:self.num_years,:] += self._out_trees_valid[i + 1,:,:self.num_years,:]
            self._out_trees_valid[i,:,:self.num_years,:] /= 2
    def average_trees_year_ranges(self):
        assert(self.num_years > 1)
        self.num_years -= 1
        for j in range(self.num_years):
            # self.predicted_trees shape: (num_depths, num_trees, num_years, num_pixels_valid)
            self._out_trees_valid[:self.num_depths,:,j,:] += self._out_trees_valid[:self.num_depths,:,j + 1,:]
            self._out_trees_valid[:self.num_depths,:,j,:] /= 2
    def compute_stats(self, mean=True, quantiles=[0.025, 0.975], expm1=False, scale=1):
        quantile_idx = 1 if mean else 0
        self.num_stats = quantile_idx + len(quantiles)
        assert(self.num_stats > 0)
        self._out_stats_valid = np.empty((self.num_years * self._data.num_pixels_valid, self.num_depths * self.num_stats), dtype=np.float32)
        # compute stats
        for i in range(self.num_depths):
            if mean:
                # self.predicted_stats shape: (num_years, num_pixels_valid, num_depths, num_stats) 
                # self.predicted_trees shape: (num_depths, num_trees, num_years, num_pixels_valid)
                self.predicted_stats[:,:,i,0] = np.mean(self._out_trees_valid[i,:,:self.num_years,:], axis=0)
            if len(quantiles) > 0:
                q = np.quantile(self._out_trees_valid[i,:,:self.num_years,:], quantiles, axis=0)
                self.predicted_stats[:,:,i,quantile_idx:] = q.transpose((1, 2, 0))

        # compute inverse log1p
        if expm1:
            np.expm1(self._out_stats_valid, out=self._out_stats_valid)
        # compute scale
        if scale != 1:
            self._out_stats_valid[:] = self._out_stats_valid * scale
    def save_stats_layers(self, 
                          base_dir,
                          nodata,
                          dtype,
                          out_files_prefix,
                          out_files_suffix, 
                          s3_prefix, 
                          s3_aliases, 
                          gdal_opts,
                          threads):
        assert(self._out_stats_valid is not None)
        assert(dtype == 'int16' or dtype == 'uint8')
        assert(len(out_files_prefix) == len(out_files_suffix))
        assert(len(out_files_prefix) == self.num_stats)
        assert(s3_prefix is None or len(s3_aliases) > 0)
        # create and transpose output
        with TimeTracker(f"tile {self._data.tile_id}/model {self.model_name} - transpose data for final output ({threads} threads)"):
            # expand to original number of pixels
            self._out_stats = np.empty((self.num_years * self._data.num_pixels, self.num_depths * self.num_stats), dtype=np.float32)
            skmap_bindings.fillArray(self._out_stats, threads, nodata)
            skmap_bindings.expandArrayRows(self._out_stats_valid, threads, self._out_stats, self._data.get_pixels_valid_idx(self.num_years))
            # transpose expanded array
            self._out_stats_t = np.empty((self.num_depths * self.num_stats, self.num_years * self._data.num_pixels), dtype=np.float32)
            skmap_bindings.transposeArray(self._out_stats, threads, self._out_stats_t)
            # rearrange years and stats
            # TODO ? could this be replaced by just self._out_stats_t.reshape((self.num_depths * self.num_stats * self.num_years, self.model._data.n_pixels))?
            self._out_stats_gdal = np.empty((self.num_depths * self.num_stats * self.num_years, self._data.num_pixels), dtype=np.float32)
            skmap_bindings.fillArray(self._out_stats_gdal, threads, nodata)
            inverse_idx = np.empty((self.num_depths * self.num_stats * self.num_years, 2), dtype=np.uintc)
            subrows_grid, rows_grid = np.meshgrid(np.arange(self.num_years), list(range(self.num_depths * self.num_stats)))
            inverse_idx[:,0] = rows_grid.flatten()
            inverse_idx[:,1] = subrows_grid.flatten()
            skmap_bindings.inverseReorderArray(self._out_stats_t, threads, self._out_stats_gdal, inverse_idx)
        # write outputs
        with TimeTracker(f"tile {self._data.tile_id}/model {self.model_name} - write output ({threads} threads)"):
            out_dir = _make_dir(f"{base_dir}/{self._data.tile_id}/{self.model_name}")
            # TODO implement filenames function as an class function
            out_files = _get_out_files_depths(
                out_files_prefix=out_files_prefix, 
                out_files_suffix=out_files_suffix, 
                tile_id=self._data.tile_id, 
                depths=self.depths, 
                num_depths=self.num_depths, 
                years=self.years, 
                num_years=self.num_years,
                num_stats=self.num_stats
            )
            # TODO change the need for base image in skmap_bindings.writeByteData and skmap_bindings.writeInt16Data
            temp_dir = f"{base_dir}/.skmap"
            temp_tif = self._data.create_image_template(dtype, nodata, temp_dir)
            write_idx = range(self.num_depths * self.num_stats * self.num_years)
            compress_cmd = f"gdal_translate -a_nodata {nodata} -co COMPRESS=deflate -co ZLEVEL=9 -co TILED=TRUE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024"
            s3_out = None
            if s3_prefix is not None:
                s3_out = [f'{s3_aliases[random.randint(0, len(s3_aliases) - 1)]}{s3_prefix}/{self._data.tile_id}' for _ in range(len(out_files))]
            if dtype == 'int16':
                skmap_bindings.writeInt16Data(self._out_stats_gdal, threads, gdal_opts, [temp_tif for _ in range(len(out_files))], out_dir, out_files, 
                                              write_idx, 0, 0, self._data.x_size, self._data.y_size, int(nodata), compress_cmd, s3_out)
            elif dtype == 'uint8':
                skmap_bindings.writeByteData(self._out_stats_gdal, threads, gdal_opts, [temp_tif for _ in range(len(out_files))], out_dir, out_files, 
                                             write_idx, 0, 0, self._data.x_size, self._data.y_size, int(nodata), compress_cmd, s3_out)
        # show final message and remove local files after sent to s3 backend
        if s3_prefix is not None:
            for k in range(self.num_stats):
                print(f'List results with `mc ls {s3_aliases[0]}{s3_prefix}/{self._data.tile_id}/{out_files_prefix[k]}_`')
            shutil.rmtree(out_dir)
            os.remove(temp_tif)
            return s3_out
        return [f"{out_dir}/{file}" for file in out_files]
#
class PredictedProbs():
    def __init__(self, 
                 data:DataLoader, 
                 model_name, 
                 num_class) -> None:
        self._data = data
        assert(self._data is not None)
        self.model_name = model_name
        self.num_class = num_class
        self.years = self._data.years
        self.num_years = len(self.years)
        self._out_probs_valid = np.empty((self.num_years * self._data.num_pixels_valid, self.num_class), dtype=np.float32)
        self._out_probs = None
        self._out_probs_t = None
        self._out_probs_gdal = None
        self._out_cls_valid = None
        self._out_cls = None
        self._out_cls_t = None
        self._out_cls_gdal = None
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self._out_probs_valid = None
        self._out_probs = None
        self._out_probs_t = None
        self._out_probs_gdal = None
        self._out_cls_valid = None
        self._out_cls = None
        self._out_cls_t = None
        self._out_cls_gdal = None
    @property
    def predicted_probs(self):
        return self._out_probs_valid.reshape((self.num_years, self._data.num_pixels_valid, self.num_class))
    @property
    def predicted_class(self):
        return self._out_cls_valid.reshape((self.num_years, self._data.num_pixels_valid))
    def _compute_classes(self):
        # TODO split dimensions
        self._out_cls_valid = np.empty((self.num_years * self._data.num_pixels_valid, 1), dtype=np.float32)
        # compute stats
        with TimeTracker(f"tile {self._data.tile_id}/model {self.model_name} - compute classes"):
            self._out_cls_valid[:, 0] = np.argmax(self._out_probs_valid[:,:], axis=-1)
    def save_class_layer(self,
                         base_dir, 
                         nodata, 
                         dtype, 
                         out_files_prefix, 
                         out_files_suffix, 
                         s3_prefix, 
                         s3_aliases, 
                         gdal_opts,
                         threads):
        self._compute_classes()
        assert(self._out_cls_valid is not None)
        assert(dtype == 'int16' or dtype == 'uint8')
        assert(len(out_files_prefix) == len(out_files_suffix))
        assert(len(out_files_prefix) == 1)
        assert(s3_prefix is None or len(s3_aliases) > 0)
        # create and transpose output
        with TimeTracker(f"tile {self._data.tile_id}/model {self.model_name} - transpose data for final output ({threads} threads)"):
            # expand to original number of pixels
            self._out_cls = np.empty((self.num_years * self._data.num_pixels, self._out_cls_valid.shape[1]), dtype=np.float32)
            skmap_bindings.fillArray(self._out_cls, threads, nodata)
            skmap_bindings.expandArrayRows(self._out_cls_valid, threads, self._out_cls, self._data.get_pixels_valid_idx(self.num_years))
            # transpose expanded array
            self._out_cls_t = np.empty((self._out_cls_valid.shape[1], self.num_years * self._data.num_pixels), dtype=np.float32)
            skmap_bindings.transposeArray(self._out_cls, threads, self._out_cls_t)
            # rearrange years and stats
            self._out_cls_gdal = np.empty((self._out_cls_valid.shape[1] * self.num_years, self._data.num_pixels), dtype=np.float32)
            skmap_bindings.fillArray(self._out_cls_gdal, threads, nodata)
            inverse_idx = np.empty((self._out_cls_gdal.shape[0], 2), dtype=np.uintc)
            subrows_grid, rows_grid = np.meshgrid(np.arange(self.num_years), list(range(self._out_cls_valid.shape[1])))
            inverse_idx[:,0] = rows_grid.flatten()
            inverse_idx[:,1] = subrows_grid.flatten()
            skmap_bindings.inverseReorderArray(self._out_cls_t, threads, self._out_cls_gdal, inverse_idx)
        # write outputs
        with TimeTracker(f"tile {self._data.tile_id}/model {self.model_name} - write class images ({threads} threads)"):
            out_dir = _make_dir(f"{base_dir}/{self._data.tile_id}/{self.model_name}")
            out_files = _get_out_files(
                out_files_prefix=out_files_prefix, 
                out_files_suffix=out_files_suffix, 
                tile_id=self._data.tile_id, 
                years=self.years, 
                num_stats=1
            )
            temp_dir = f"{base_dir}/.skmap"
            temp_tif = self._data.create_image_template(dtype, nodata, temp_dir)
            write_idx = range(self._out_cls_gdal.shape[0])
            compress_cmd = f"gdal_translate -a_nodata {nodata} -co COMPRESS=deflate -co ZLEVEL=9 -co TILED=TRUE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024"
            s3_out = None
            if s3_prefix is not None:
                s3_out = [f'{s3_aliases[random.randint(0, len(s3_aliases) - 1)]}{s3_prefix}/{self._data.tile_id}' for _ in range(len(out_files))]
            if dtype == 'int16':
                skmap_bindings.writeInt16Data(self._out_cls_gdal, threads, gdal_opts, [temp_tif for _ in range(len(out_files))], out_dir, out_files, 
                                              write_idx, 0, 0, self._data.x_size, self._data.y_size, int(nodata), compress_cmd, s3_out)
            elif dtype == 'uint8':
                skmap_bindings.writeByteData(self._out_cls_gdal, threads, gdal_opts, [temp_tif for _ in range(len(out_files))], out_dir, out_files, 
                                             write_idx, 0, 0, self._data.x_size, self._data.y_size, int(nodata), compress_cmd, s3_out)
        if s3_prefix is not None:
            print(f'List results with `mc ls {s3_aliases[0]}{s3_prefix}/{self._data.tile_id}/{out_files_prefix[0]}_`')
            shutil.rmtree(out_dir)
            os.remove(temp_tif)
            return s3_out
        return [f"{out_dir}/{file}" for file in out_files]
    def save_probs_layers(self, 
                          base_dir, 
                          nodata, 
                          dtype, 
                          out_files_prefix, 
                          out_files_suffix, 
                          s3_prefix, 
                          s3_aliases, 
                          gdal_opts,
                          threads):
        assert(self._out_probs_valid is not None)
        assert(dtype == 'int16' or dtype == 'uint8')
        assert(len(out_files_prefix) == len(out_files_suffix))
        assert(len(out_files_prefix) == self.num_class)
        assert(s3_prefix is None or len(s3_aliases) > 0)
        # create and transpose output
        with TimeTracker(f"tile {self._data.tile_id}/model {self.model_name} - transpose data for final output ({threads} threads)"):
            # expand to original number of pixels
            self._out_probs = np.empty((self.num_years * self._data.num_pixels, self.num_class), dtype=np.float32)
            skmap_bindings.fillArray(self._out_probs, threads, nodata)
            skmap_bindings.expandArrayRows(self._out_probs_valid, threads, self._out_probs, self._data.get_pixels_valid_idx(self.num_years))
            # transpose expanded array
            self._out_probs_t = np.empty((self.num_class, self.num_years * self._data.num_pixels), dtype=np.float32)
            skmap_bindings.transposeArray(self._out_probs, threads, self._out_probs_t)
            # rearrange years and stats
            self._out_probs_gdal = np.empty((self.num_class * self.num_years, self._data.num_pixels), dtype=np.float32)
            skmap_bindings.fillArray(self._out_probs_gdal, threads, nodata)
            inverse_idx = np.empty((self._out_probs_gdal.shape[0], 2), dtype=np.uintc)
            subrows_grid, rows_grid = np.meshgrid(np.arange(self.num_years), list(range(self.num_class)))
            inverse_idx[:,0] = rows_grid.flatten()
            inverse_idx[:,1] = subrows_grid.flatten()
            skmap_bindings.inverseReorderArray(self._out_probs_t, threads, self._out_probs_gdal, inverse_idx)
        # write outputs
        with TimeTracker(f"tile {self._data.tile_id}/model {self.model_name} - write probs images ({threads} threads)"):
            out_dir = _make_dir(f"{base_dir}/{self._data.tile_id}/{self.model_name}")
            out_files = _get_out_files(
                out_files_prefix=out_files_prefix, 
                out_files_suffix=out_files_suffix, 
                tile_id=self._data.tile_id, 
                years=self.years, 
                num_stats=self.num_class
            )
            temp_dir = f"{base_dir}/.skmap"
            temp_tif = self._data.create_image_template(dtype, nodata, temp_dir)
            write_idx = range(self._out_probs_gdal.shape[0])
            compress_cmd = f"gdal_translate -a_nodata {nodata} -co COMPRESS=deflate -co ZLEVEL=9 -co TILED=TRUE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024"
            s3_out = None
            if s3_prefix is not None:
                s3_out = [f'{s3_aliases[random.randint(0, len(s3_aliases) - 1)]}{s3_prefix}/{self._data.tile_id}' for _ in range(len(out_files))]
            if dtype == 'int16':
                skmap_bindings.writeInt16Data(self._out_probs_gdal, threads, gdal_opts, [temp_tif for _ in range(len(out_files))], out_dir, out_files, 
                                              write_idx, 0, 0, self._data.x_size, self._data.y_size, int(nodata), compress_cmd, s3_out)
            elif dtype == 'uint8':
                skmap_bindings.writeByteData(self._out_probs_gdal, threads, gdal_opts, [temp_tif for _ in range(len(out_files))], out_dir, out_files, 
                                             write_idx, 0, 0, self._data.x_size, self._data.y_size, int(nodata), compress_cmd, s3_out)
        if s3_prefix is not None:
            print(f'List results with `mc ls {s3_aliases[0]}{s3_prefix}/{self._data.tile_id}/{out_files_prefix[0]}_`')
            shutil.rmtree(out_dir)
            os.remove(temp_tif)
            return s3_out
        return [f"{out_dir}/{file}" for file in out_files]
