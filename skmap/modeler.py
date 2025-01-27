import treelite_runtime 
from typing import List, Callable, Optional
import os
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from joblib import Parallel, delayed
import joblib
import threading
import numpy as np
import tl2cgen
from skmap.loader import TiledDataLoader
import skmap_bindings as sb
from skmap.misc import _make_dir, TimeTracker, _rm_dir
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
os.environ["OMP_DYNAMIC"] = f'TRUE'
os.environ['TREELITE_BIND_THREADS'] = '0'


from time import time

def _single_prediction(predict, X, out, i, lock):
    prediction = predict(X, check_input=False)
    with lock:
        out[i, :] = prediction

def cast_tree_rf(model):
    model.__class__ = TreesRandomForestRegressor
    return model

def _get_out_files_depths(out_files_prefix, out_files_suffix, tile_id, depths, n_depths, years, n_years, n_stats):
    assert(len(out_files_prefix) == len(out_files_suffix))
    assert(len(out_files_prefix) == n_stats)
    assert(len(depths) >= n_depths)
    assert(len(years) >= n_years)
    out_files = []
    for i in range(n_depths):
        for k in range(n_stats):
            for j in range(n_years):
                if n_years < len(years):
                    y1 = years[j]
                    y2 = years[j + len(years) - n_years]
                    if n_depths < len(depths):
                        d1 = depths[i]
                        d2 = depths[i + len(depths) - n_depths]
                        file = f'{out_files_prefix[k]}_b{d1}cm..{d2}cm_{y1}0101_{y2}1231_tile.{tile_id}_{out_files_suffix[k]}'
                    else:
                        d1 = depths[i]
                        file = f'{out_files_prefix[k]}_b{d1}cm_{y1}0101_{y2}1231_tile.{tile_id}_{out_files_suffix[k]}'
                else:
                    y1 = years[j]
                    if n_depths < len(depths):
                        d1 = depths[i]
                        d2 = depths[i + len(depths) - n_depths]
                        file = f'{out_files_prefix[k]}_b{d1}cm..{d2}cm_{y1}0101_{y1}1231_tile.{tile_id}_{out_files_suffix[k]}'
                    else:
                        d1 = depths[i]
                        file = f'{out_files_prefix[k]}_b{d1}cm_{y1}0101_{y1}1231_tile.{tile_id}_{out_files_suffix[k]}'
                out_files.append(file)
    return out_files

class Regressor():
    def __init__(self, 
                 model_name, 
                 model_path, 
                 model_covs_path, 
                 predict_fn) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.model_covs_path = model_covs_path
        model, features = _read_model(self.model_name, self.model_path, self.model_covs_path)
        assert(hasattr(model, 'estimators_'))
        self.model = model
        self.n_trees = len(self.model.estimators_)
        self.model_features = features
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
    def predict(self, data:TiledDataLoader):
        # prepare input and output arrays
        self.in_covs_t = np.empty((len(self.model_features), len(data.catalog.get_groups()) * data.n_pixels), dtype=np.float32)
        self.in_covs = np.empty((len(data.catalog.get_groups()) * data.n_pixels, len(self.model_features)), dtype=np.float32)
        self.in_covs_valid = np.empty((len(data.catalog.get_groups()) * data.n_pixels_valid, len(self.model_features)), dtype=np.float32)
        # create output result
        result = Predicted(
            data=data,
            model_name=self.model_name
        )
        # traverse depths
        matrix_idx = data.catalog._get_covs_idx(self.model_features)
        i=0
        # transpose data
        # TODO create a class for skmap matrix with threads in constructor
        sb.reorderArray(data.array, data.n_threads, self.in_covs_t, matrix_idx)
        sb.transposeArray(self.in_covs_t, data.n_threads, self.in_covs)
        sb.selArrayRows(self.in_covs, data.n_threads, self.in_covs_valid, data.get_pixels_valid_idx(data.n_years))
        # predict
        # self.predicted_trees shape: (n_years, n_pixels)
        result._out_valid[:,:] = self.predict_fn(self.model, self.in_covs_valid).reshape(result.n_groups, result.data.n_pixels_valid)
        return result # shape: (n_samples)

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
        self.n_trees = len(self.model.estimators_)
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
    def predict(self, data:TiledDataLoader):
        # prepare input and output arrays
        self.in_covs_t = np.empty((len(self.model_features), len(data.catalog.get_groups()) * data.n_pixels), dtype=np.float32)
        self.in_covs = np.empty((len(data.catalog.get_groups()) * data.n_pixels, len(self.model_features)), dtype=np.float32)
        self.in_covs_valid = np.empty((len(data.catalog.get_groups()) * data.n_pixels_valid, len(self.model_features)), dtype=np.float32)
        # create output result
        result = PredictedDepths(
            data=data,
            model_name=self.model_name, 
            depths=self.depths,
            n_depths=len(self.depths),
            n_trees=self.n_trees
        )
        # traverse depths
        matrix_idx = data.catalog._get_covs_idx(self.model_features)
        for i in range(len(self.depths)):
            depth = self.depths[i]
            data.fill_otf_constant(self.depth_var, depth)
            # transpose data
            # TODO create a class for skmap matrix with threads in constructor
            sb.reorderArray(data.array, data.n_threads, self.in_covs_t, matrix_idx)
            sb.transposeArray(self.in_covs_t, data.n_threads, self.in_covs)
            sb.selArrayRows(self.in_covs, data.n_threads, self.in_covs_valid, data.get_pixels_valid_idx(data.n_years))
            # predict
            # TODO ?implement a base class without depth and use repeatedly here?
            # self.predicted_trees shape: (n_depths, n_trees, n_years, n_pixels)
            result._out_trees_valid[i,:,:,:] = self.predict_fn(self.model, self.in_covs_valid).reshape(result.n_trees, result.n_groups, result.data.n_pixels_valid)
        return result # shape: (n_depths, n_trees, n_samples)
    def predictDepth(self, data:TiledDataLoader, i):
        # prepare input and output arrays
        self.in_covs_t = np.empty((len(self.model_features), len(data.catalog.get_groups()) * data.n_pixels), dtype=np.float32)
        self.in_covs = np.empty((len(data.catalog.get_groups()) * data.n_pixels, len(self.model_features)), dtype=np.float32)
        self.in_covs_valid = np.empty((len(data.catalog.get_groups()) * data.n_pixels_valid, len(self.model_features)), dtype=np.float32)
        # create output result
        result = PredictedDepths(
            data=data,
            model_name=self.model_name, 
            depths=self.depths,
            n_depths=len(self.depths),
            n_trees=self.n_trees
        )
        # traverse depths
        matrix_idx = data.catalog._get_covs_idx(self.model_features)
        depth = self.depths[i]
        data.fill_otf_constant(self.depth_var, depth)
        # transpose data
        # TODO create a class for skmap matrix with threads in constructor
        sb.reorderArray(data.array, data.n_threads, self.in_covs_t, matrix_idx)
        sb.transposeArray(self.in_covs_t, data.n_threads, self.in_covs)
        sb.selArrayRows(self.in_covs, data.n_threads, self.in_covs_valid, data.get_pixels_valid_idx(len(data.catalog.get_groups())))
        # predict
        # TODO ?implement a base class without depth and use repeatedly here?
        # self.predicted_trees shape: (n_depths, n_trees, n_years, n_pixels)
        result = self.predict_fn(self.model, self.in_covs_valid)
        return result # shape: (n_trees, n_years * n_pixels)
#
class Classifier:
    def __init__(self, 
                 model_name: str, 
                 model_path: str, 
                 model_covs_path: str, 
                 n_class: int,
                 predict_fn:Callable=lambda predictor, data: predictor.predict_proba(data)) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.model_covs_path = model_covs_path
        self.n_class = n_class
        self.predict_fn = predict_fn
        self.in_covs_t = None
        self.in_covs = None
        self.in_covs_valid = None
        self.model = None
        self.model_covs = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.in_covs_t = None
        self.in_covs = None
        self.in_covs_valid = None
        self.model = None
        self.model_covs = None
    
    def _load_model(self):
        if self.model_path.endswith(('.joblib', '.lz4')):
            model = joblib.load(self.model_path)
        elif self.model_path.endswith('.so'):
            model = treelite_runtime.Predictor(self.model_path, nthread=n_threads)
            os.environ['TREELITE_BIND_THREADS'] = '0'
        else:
            raise ValueError(f"Invalid model path extension '{self.model_path}'")
        if self.model_covs_path is not None:
            with open(self.model_covs_path, 'r') as file:
                model_covs = [line.strip() for line in file]
        elif hasattr(model, "feature_names_in_"):
            model_covs = list(model.feature_names_in_)
        elif hasattr(model, 'feature_names_'):
            model_covs = model.feature_names_
        elif hasattr(model, 'feature_name'):
            model_covs = model.feature_name()
        else:
            raise ValueError(f"No feature names was found for model {self.model_name}")
        self.model = model
        self.model_covs = model_covs
    
    def predict():
        raise NotImplementedError()
#
class RFClassifier(Classifier):
    def __init__(self, 
                 model_name: str, 
                 model_path, 
                 model_covs_path, 
                 n_class,
                 predict_fn:Callable=lambda predictor, data: predictor.predict_proba(data)) -> None:
        super().__init__(model_name, model_path, model_covs_path, n_class, predict_fn)
        self._load_model()
    
    def predict(self, data:TiledDataLoader):
        with TimeTracker(f"tile {data.tile_id}/model {self.model_name} - predict ({len(self.model_covs)} input features)", True):
            # prepare input and output arrays
            n_groups = len(data.catalog.get_groups())
            n_samples = n_groups * data.n_pixels
            n_samples_valid = n_groups * data.n_pixels_valid
            self.in_covs_t = np.empty((len(self.model_covs), n_samples), dtype=np.float32)
            self.in_covs = np.empty((n_samples, len(self.model_covs)), dtype=np.float32)
            self.in_covs_valid = np.empty((n_samples_valid, len(self.model_covs)), dtype=np.float32)
            # transpose data
            matrix_idx = data.catalog._get_covs_idx(self.model_covs)
            with TimeTracker(f"tile {data.tile_id}/model {self.model_name} - transpose data ({data.n_threads} threads)"):
                sb.reorderArray(data.array, data.n_threads, self.in_covs_t, matrix_idx)
                sb.transposeArray(self.in_covs_t, data.n_threads, self.in_covs)
                sb.selArrayRows(self.in_covs, data.n_threads, self.in_covs_valid, data.get_pixels_valid_idx(n_groups))
            # create output result
            result = PredictedProbs(data=data, model_name=self.model_name, n_class=self.n_class)
            # predict
            with TimeTracker(f"tile {data.tile_id}/model {self.model_name} - model prediction ({data.n_threads} threads)"):
                tmp_res = self.predict_fn(self.model, self.in_covs_valid)
                if tmp_res.dtype == np.float64:
                    sb.castFloat64ToFloat32(tmp_res, data.n_threads, result._out_probs_valid)
                elif tmp_res.dtype == np.float32:
                    result._out_probs_valid = tmp_res
                else:
                    print("Result prediction are not in float32 nor float64, converting with python (can be slow)")
                    result._out_probs_valid = tmp_res.astype(np.float32)
        return result # shape: (n_samples, n_classes)
#
class Predicted():
    def __init__(self,
                 data:TiledDataLoader, 
                 model_name, 
                 depths) -> None:
        self.data = data
        assert(self.data is not None)
        self.model_name = model_name
        self.depths = depths
        self.groups = self.data.catalog.get_groups()
        self.n_groups = len(self.groups)
        # TODO optimize shape of self.out_trees
        self._out_valid = np.empty((self.n_groups, self.data.n_pixels_valid), dtype=np.float32)
        self._out_stats_valid = None
        self._out_stats = None
        self._out_stats_t = None
        self._out_stats_gdal = None
        self.n_stats = None
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self._out_valid = None
        self._out_stats_valid = None
        self._out_stats = None
        self._out_stats_t = None
        self._out_stats_gdal = None
    @property
    def predicted_trees(self):
        # self.predicted_trees shape: (n_depths, n_trees, n_years, n_pixels_valid)
        return self._out_valid[:self.n_depths, :, :self.n_groups, :]
    @property
    def predicted_stats(self):
        # self.predicted_stats shape: (n_years, n_pixels_valid, n_depths, n_stats) 
        if self._out_stats_valid is not None:
            return self._out_stats_valid.reshape((self.n_groups, self.data.n_pixels_valid, self.n_depths, self.n_stats))
    def create_empty_copy(self, model_name):
        pred = PredictedDepths(
            model_name=model_name, 
            depths=self.depths,
            n_depths=self.n_depths,
            years=self.groups,
            n_years=self.n_groups,
            n_trees=self.n_trees,
            data=self.data
        )
        return pred
    def average_trees_depth_ranges(self):
        assert(self.n_depths > 1)
        self.n_depths -= 1
        for i in range(self.n_depths):
            # self.predicted_trees shape: (n_depths, n_trees, n_years, n_pixels_valid)
            self._out_valid[i,:,:self.n_groups,:] += self._out_valid[i + 1,:,:self.n_groups,:]
            self._out_valid[i,:,:self.n_groups,:] /= 2
    def average_trees_year_ranges(self):
        assert(self.n_groups > 1)
        self.n_groups -= 1
        for j in range(self.n_groups):
            # self.predicted_trees shape: (n_depths, n_trees, n_years, n_pixels_valid)
            self._out_valid[:self.n_depths,:,j,:] += self._out_valid[:self.n_depths,:,j + 1,:]
            self._out_valid[:self.n_depths,:,j,:] /= 2
    def compute_stats(self, mean=True, quantiles=[0.025, 0.975], expm1=False, scale=1):
        quantile_idx = 1 if mean else 0
        self.n_stats = quantile_idx + len(quantiles)
        assert(self.n_stats > 0)
        self._out_stats_valid = np.empty((self.n_groups * self.data.n_pixels_valid, self.n_depths * self.n_stats), dtype=np.float32)
        # compute stats
        for i in range(self.n_depths):
            if mean:
                # self.predicted_stats shape: (n_years, n_pixels_valid, n_depths, n_stats) 
                # self.predicted_trees shape: (n_depths, n_trees, n_years, n_pixels_valid)
                self.predicted_stats[:,:,i,0] = np.mean(self._out_valid[i,:,:self.n_groups,:], axis=0)
            if len(quantiles) > 0:
                q = np.quantile(self._out_valid[i,:,:self.n_groups,:], quantiles, axis=0)
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
        assert(len(out_files_prefix) == self.n_stats)
        assert(s3_prefix is None or len(s3_aliases) > 0)
        # create and transpose output
        with TimeTracker(f"tile {self.data.tile_id}/model {self.model_name} - transpose data for final output ({threads} threads)"):
            # expand to original number of pixels
            self._out_stats = np.empty((self.n_groups * self.data.n_pixels, self.n_depths * self.n_stats), dtype=np.float32)
            sb.fillArray(self._out_stats, threads, nodata)
            sb.expandArrayRows(self._out_stats_valid, threads, self._out_stats, self.data.get_pixels_valid_idx(self.n_groups))
            # transpose expanded array
            self._out_stats_t = np.empty((self.n_depths * self.n_stats, self.n_groups * self.data.n_pixels), dtype=np.float32)
            sb.transposeArray(self._out_stats, threads, self._out_stats_t)
            # rearrange years and stats
            # TODO ? could this be replaced by just self._out_stats_t.reshape((self.n_depths * self.n_stats * self.n_years, self.model._data.n_pixels))?
            self._out_stats_gdal = np.empty((self.n_depths * self.n_stats * self.n_groups, self.data.n_pixels), dtype=np.float32)
            sb.fillArray(self._out_stats_gdal, threads, nodata)
            inverse_idx = np.empty((self.n_depths * self.n_stats * self.n_groups, 2), dtype=np.uintc)
            subrows_grid, rows_grid = np.meshgrid(np.arange(self.n_groups), list(range(self.n_depths * self.n_stats)))
            inverse_idx[:,0] = rows_grid.flatten()
            inverse_idx[:,1] = subrows_grid.flatten()
            sb.inverseReorderArray(self._out_stats_t, threads, self._out_stats_gdal, inverse_idx)
        # write outputs
        with TimeTracker(f"tile {self.data.tile_id}/model {self.model_name} - write output ({threads} threads)"):
            out_dir = _make_dir(f"{base_dir}/{self.data.tile_id}/{self.model_name}")
            # TODO implement filenames function as an class function
            out_files = _get_out_files_depths(
                out_files_prefix=out_files_prefix, 
                out_files_suffix=out_files_suffix, 
                tile_id=self.data.tile_id, 
                depths=self.depths, 
                n_depths=self.n_depths, 
                years=self.groups, 
                n_years=self.n_groups,
                n_stats=self.n_stats
            )
            # TODO change the need for base image in sb.writeByteData and sb.writeInt16Data
            temp_dir = f"{base_dir}/.skmap"
            temp_tif = self.data.create_image_template(dtype, nodata, temp_dir)
            write_idx = range(self.n_depths * self.n_stats * self.n_groups)
            compress_cmd = f"gdal_translate -a_nodata {nodata} -co COMPRESS=deflate -co PREDICTOR=2 -co TILED=TRUE -co BLOCKXSIZE=2048 -co BLOCKYSIZE=2048"
            s3_out = None
            if s3_prefix is not None:
                s3_out = [f'{s3_aliases[random.randint(0, len(s3_aliases) - 1)]}{s3_prefix}/{self.data.tile_id}' for _ in range(len(out_files))]
            if dtype == 'int16':
                sb.writeInt16Data(self._out_stats_gdal, threads, gdal_opts, [temp_tif for _ in range(len(out_files))], out_dir, out_files, 
                                              write_idx, 0, 0, self.data.x_size, self.data.y_size, int(nodata), compress_cmd, s3_out)
            elif dtype == 'uint8':
                sb.writeByteData(self._out_stats_gdal, threads, gdal_opts, [temp_tif for _ in range(len(out_files))], out_dir, out_files, 
                                             write_idx, 0, 0, self.data.x_size, self.data.y_size, int(nodata), compress_cmd, s3_out)
        # show final message and remove local files after sent to s3 backend
        if s3_prefix is not None:
            for k in range(self.n_stats):
                print(f'List results with `mc ls {s3_aliases[0]}{s3_prefix}/{self.data.tile_id}/{out_files_prefix[k]}_`')
            _rm_dir(out_dir)
            os.remove(temp_tif)
            return s3_out
        return [f"{out_dir}/{file}" for file in out_files]
#
# TODO copy all metadata needed from TiledDataLoader as parameter of constructor
class PredictedDepths():
    def __init__(self,
                 data:TiledDataLoader, 
                 model_name, 
                 depths,
                 n_depths,
                 n_trees) -> None:
        self.data = data
        assert(self.data is not None)
        self.model_name = model_name
        self.depths = depths
        self.n_depths = n_depths
        self.groups = self.data.catalog.get_groups()
        self.n_groups = len(self.groups)
        self.n_trees = n_trees
        # TODO optimize shape of self.out_trees
        self._out_trees_valid = np.empty((self.n_depths, self.n_trees, self.n_groups, self.data.n_pixels_valid), dtype=np.float32)
        self._out_stats_valid = None
        self._out_stats = None
        self._out_stats_t = None
        self._out_stats_gdal = None
        self.n_stats = None
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
        # self.predicted_trees shape: (n_depths, n_trees, n_years, n_pixels_valid)
        return self._out_trees_valid[:self.n_depths, :, :self.n_groups, :]
    @property
    def predicted_stats(self):
        # self.predicted_stats shape: (n_years, n_pixels_valid, n_depths, n_stats) 
        if self._out_stats_valid is not None:
            return self._out_stats_valid.reshape((self.n_groups, self.data.n_pixels_valid, self.n_depths, self.n_stats))
    def create_empty_copy(self, model_name):
        pred = PredictedDepths(
            model_name=model_name, 
            depths=self.depths,
            n_depths=self.n_depths,
            years=self.groups,
            n_years=self.n_groups,
            n_trees=self.n_trees,
            data=self.data
        )
        return pred
    def average_trees_depth_ranges(self):
        assert(self.n_depths > 1)
        self.n_depths -= 1
        for i in range(self.n_depths):
            # self.predicted_trees shape: (n_depths, n_trees, n_years, n_pixels_valid)
            self._out_trees_valid[i,:,:self.n_groups,:] += self._out_trees_valid[i + 1,:,:self.n_groups,:]
            self._out_trees_valid[i,:,:self.n_groups,:] /= 2
    def average_trees_year_ranges(self):
        assert(self.n_groups > 1)
        self.n_groups -= 1
        for j in range(self.n_groups):
            # self.predicted_trees shape: (n_depths, n_trees, n_years, n_pixels_valid)
            self._out_trees_valid[:self.n_depths,:,j,:] += self._out_trees_valid[:self.n_depths,:,j + 1,:]
            self._out_trees_valid[:self.n_depths,:,j,:] /= 2
    def compute_stats(self, mean=True, quantiles=[0.025, 0.975], expm1=False, scale=1):
        quantile_idx = 1 if mean else 0
        self.n_stats = quantile_idx + len(quantiles)
        assert(self.n_stats > 0)
        self._out_stats_valid = np.empty((self.n_groups * self.data.n_pixels_valid, self.n_depths * self.n_stats), dtype=np.float32)
        # compute stats
        for i in range(self.n_depths):
            if mean:
                # self.predicted_stats shape: (n_years, n_pixels_valid, n_depths, n_stats) 
                # self.predicted_trees shape: (n_depths, n_trees, n_years, n_pixels_valid)
                self.predicted_stats[:,:,i,0] = np.mean(self._out_trees_valid[i,:,:self.n_groups,:], axis=0)
            if len(quantiles) > 0:
                q = np.quantile(self._out_trees_valid[i,:,:self.n_groups,:], quantiles, axis=0)
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
        assert(len(out_files_prefix) == self.n_stats)
        assert(s3_prefix is None or len(s3_aliases) > 0)
        # create and transpose output
        with TimeTracker(f"tile {self.data.tile_id}/model {self.model_name} - transpose data for final output ({threads} threads)"):
            # expand to original number of pixels
            self._out_stats = np.empty((self.n_groups * self.data.n_pixels, self.n_depths * self.n_stats), dtype=np.float32)
            sb.fillArray(self._out_stats, threads, nodata)
            sb.expandArrayRows(self._out_stats_valid, threads, self._out_stats, self.data.get_pixels_valid_idx(self.n_groups))
            # transpose expanded array
            self._out_stats_t = np.empty((self.n_depths * self.n_stats, self.n_groups * self.data.n_pixels), dtype=np.float32)
            sb.transposeArray(self._out_stats, threads, self._out_stats_t)
            # rearrange years and stats
            # TODO ? could this be replaced by just self._out_stats_t.reshape((self.n_depths * self.n_stats * self.n_years, self.model._data.n_pixels))?
            self._out_stats_gdal = np.empty((self.n_depths * self.n_stats * self.n_groups, self.data.n_pixels), dtype=np.float32)
            sb.fillArray(self._out_stats_gdal, threads, nodata)
            inverse_idx = np.empty((self.n_depths * self.n_stats * self.n_groups, 2), dtype=np.uintc)
            subrows_grid, rows_grid = np.meshgrid(np.arange(self.n_groups), list(range(self.n_depths * self.n_stats)))
            inverse_idx[:,0] = rows_grid.flatten()
            inverse_idx[:,1] = subrows_grid.flatten()
            sb.inverseReorderArray(self._out_stats_t, threads, self._out_stats_gdal, inverse_idx)
        # write outputs
        with TimeTracker(f"tile {self.data.tile_id}/model {self.model_name} - write output ({threads} threads)"):
            out_dir = _make_dir(f"{base_dir}/{self.data.tile_id}/{self.model_name}")
            # TODO implement filenames function as an class function
            out_files = _get_out_files_depths(
                out_files_prefix=out_files_prefix, 
                out_files_suffix=out_files_suffix, 
                tile_id=self.data.tile_id, 
                depths=self.depths, 
                n_depths=self.n_depths, 
                years=self.groups, 
                n_years=self.n_groups,
                n_stats=self.n_stats
            )
            # TODO change the need for base image in sb.writeByteData and sb.writeInt16Data
            temp_dir = f"{base_dir}/.skmap"
            temp_tif = self.data.create_image_template(dtype, nodata, temp_dir)
            write_idx = range(self.n_depths * self.n_stats * self.n_groups)
            compress_cmd = f"gdal_translate -a_nodata {nodata} -co COMPRESS=deflate -co PREDICTOR=2 -co TILED=TRUE -co BLOCKXSIZE=2048 -co BLOCKYSIZE=2048"
            s3_out = None
            if s3_prefix is not None:
                s3_out = [f'{s3_aliases[random.randint(0, len(s3_aliases) - 1)]}{s3_prefix}/{self.data.tile_id}' for _ in range(len(out_files))]
            if dtype == 'int16':
                sb.writeInt16Data(self._out_stats_gdal, threads, gdal_opts, [temp_tif for _ in range(len(out_files))], out_dir, out_files, 
                                              write_idx, 0, 0, self.data.x_size, self.data.y_size, int(nodata), compress_cmd, s3_out)
            elif dtype == 'uint8':
                sb.writeByteData(self._out_stats_gdal, threads, gdal_opts, [temp_tif for _ in range(len(out_files))], out_dir, out_files, 
                                             write_idx, 0, 0, self.data.x_size, self.data.y_size, int(nodata), compress_cmd, s3_out)
        # show final message and remove local files after sent to s3 backend
        if s3_prefix is not None:
            for k in range(self.n_stats):
                print(f'List results with `mc ls {s3_aliases[0]}{s3_prefix}/{self.data.tile_id}/{out_files_prefix[k]}_`')
            _rm_dir(out_dir)
            os.remove(temp_tif)
            return s3_out
        return [f"{out_dir}/{file}" for file in out_files]
#
class PredictedProbs():
    def __init__(self, 
                 data:TiledDataLoader, 
                 model_name, 
                 n_class) -> None:
        self.data = data
        assert(self.data is not None)
        self.model_name = model_name
        self.n_class = n_class
        self.groups = self.data.catalog.get_groups()
        self.n_groups = len(self.groups)
        self._out_probs_valid = np.empty((self.n_groups * self.data.n_pixels_valid, self.n_class), dtype=np.float32)
        self._out_cls_valid = np.empty((self.n_groups * self.data.n_pixels_valid, 1), dtype=np.float32)
        self._out_probs = None
        self._out_probs_t = None
        self._out_probs_gdal = None
        self._out_cls = None
        self._out_cls_t = None
        self._out_cls_gdal = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._out_probs = None
        self._out_probs_t = None
        self._out_probs_gdal = None
        self._out_cls = None
        self._out_cls_t = None
        self._out_cls_gdal = None
    
    @property
    def predicted_probs(self):
        return self._out_probs_valid.reshape((self.n_groups, self.data.n_pixels_valid, self.n_class))
    
    @property
    def predicted_class(self):
        return self._out_cls_valid.reshape((self.n_groups, self.data.n_pixels_valid))
    
    def compute_class(self):
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
                         n_threads):
        self.compute_class()
        assert(self._out_cls_valid is not None)
        assert(dtype == 'int16' or dtype == 'uint8')
        assert(len(out_files_prefix) == len(out_files_suffix))
        assert(len(out_files_prefix) == 1)
        assert(s3_prefix is None or len(s3_aliases) > 0)
        # create and transpose output
        with TimeTracker(f"tile {self.data.tile_id}/model {self.model_name} - transpose data for final output ({n_threads} threads)"):
            # expand to original number of pixels
            self._out_cls = np.empty((self.n_groups * self.data.n_pixels, self._out_cls_valid.shape[1]), dtype=np.float32)
            sb.fillArray(self._out_cls, n_threads, nodata)
            sb.expandArrayRows(self._out_cls_valid, n_threads, self._out_cls, self.data.get_pixels_valid_idx(self.n_groups))
            # transpose expanded array
            self._out_cls_t = np.empty((self._out_cls.shape[1], self._out_cls.shape[0]), dtype=np.float32)
            sb.transposeArray(self._out_cls, n_threads, self._out_cls_t)
            # rearrange groups
            self._out_cls_gdal = np.empty((self._out_cls_t.shape[0] * self.n_groups, self.data.n_pixels), dtype=np.float32)
            sb.fillArray(self._out_cls_gdal, n_threads, nodata)
            inverse_idx = np.empty((self._out_cls_gdal.shape[0], 2), dtype=np.uintc)
            subrows_grid, rows_grid = np.meshgrid(np.arange(self.n_groups), list(range(self._out_cls_valid.shape[1])))
            inverse_idx[:,0] = rows_grid.flatten()
            inverse_idx[:,1] = subrows_grid.flatten()
            sb.inverseReorderArray(self._out_cls_t, n_threads, self._out_cls_gdal, inverse_idx)
        # write outputs
        with TimeTracker(f"tile {self.data.tile_id}/model {self.model_name} - write class images ({n_threads} threads)"):
            out_dir = _make_dir(f"{base_dir}/{self.data.tile_id}/{self.model_name}")
            out_files = _get_out_files(out_files_prefix, out_files_suffix, self.groups)
            temp_tif = [self.data.mask_path for _ in range(len(out_files))]
            write_idx = range(self._out_cls_gdal.shape[0])
            compress_cmd = f"gdal_translate -a_nodata {nodata} -co COMPRESS=deflate -co PREDICTOR=2 -co TILED=TRUE -co BLOCKXSIZE=2048 -co BLOCKYSIZE=2048"
            s3_out = None
            if s3_prefix is not None:
                s3_out = [f'{s3_aliases[random.randint(0, len(s3_aliases) - 1)]}{s3_prefix}/{self.data.tile_id}' for _ in range(len(out_files))]
            if dtype == 'int16':
                sb.writeInt16Data(self._out_cls_gdal, n_threads, gdal_opts, temp_tif, out_dir, out_files,
                                  write_idx, 0, 0, self.data.x_size, self.data.y_size, int(nodata), compress_cmd, s3_out)
            elif dtype == 'uint8':
                sb.writeByteData(self._out_cls_gdal, n_threads, gdal_opts, temp_tif, out_dir, out_files,
                                 write_idx, 0, 0, self.data.x_size, self.data.y_size, int(nodata), compress_cmd, s3_out)
        if s3_prefix is not None:
            print(f'List results with `mc ls {s3_aliases[0]}{s3_prefix}/{self.data.tile_id}/{out_files_prefix[0]}_`')
            _rm_dir(out_dir)
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
                          n_threads):
        assert(self._out_probs_valid is not None)
        assert(dtype == 'int16' or dtype == 'uint8')
        assert(len(out_files_prefix) == len(out_files_suffix))
        assert(len(out_files_prefix) == self.n_class)
        assert(s3_prefix is None or len(s3_aliases) > 0)
        # create and transpose output
        with TimeTracker(f"tile {self.data.tile_id}/model {self.model_name} - transpose data for final output ({n_threads} threads)"):
            # expand to original number of pixels
            self._out_probs = np.empty((self.n_groups * self.data.n_pixels, self.n_class), dtype=np.float32)
            sb.fillArray(self._out_probs, n_threads, nodata)
            sb.expandArrayRows(self._out_probs_valid, n_threads, self._out_probs, self.data.get_pixels_valid_idx(self.n_groups))
            # transpose expanded array
            self._out_probs_t = np.empty((self.n_class, self.n_groups * self.data.n_pixels), dtype=np.float32)
            sb.transposeArray(self._out_probs, n_threads, self._out_probs_t)
            # rearrange years and stats
            self._out_probs_gdal = np.empty((self.n_class * self.n_groups, self.data.n_pixels), dtype=np.float32)
            sb.fillArray(self._out_probs_gdal, n_threads, nodata)
            inverse_idx = np.empty((self._out_probs_gdal.shape[0], 2), dtype=np.uintc)
            subrows_grid, rows_grid = np.meshgrid(np.arange(self.n_groups), list(range(self.n_class)))
            inverse_idx[:,0] = rows_grid.flatten()
            inverse_idx[:,1] = subrows_grid.flatten()
            sb.inverseReorderArray(self._out_probs_t, n_threads, self._out_probs_gdal, inverse_idx)
        # write outputs
        with TimeTracker(f"tile {self.data.tile_id}/model {self.model_name} - write probs images ({n_threads} threads)"):
            out_dir = _make_dir(f"{base_dir}/{self.data.tile_id}/{self.model_name}")
            out_files = _get_out_files(out_files_prefix, out_files_suffix, self.groups, self.n_class)
            temp_tif = [self.data.mask_path for _ in range(len(out_files))]
            write_idx = range(self._out_probs_gdal.shape[0])
            compress_cmd = f"gdal_translate -a_nodata {nodata} -co COMPRESS=deflate -co PREDICTOR=2 -co TILED=TRUE -co BLOCKXSIZE=2048 -co BLOCKYSIZE=2048"
            s3_out = None
            if s3_prefix is not None:
                s3_out = [f'{s3_aliases[random.randint(0, len(s3_aliases) - 1)]}{s3_prefix}/{self.data.tile_id}' for _ in range(len(out_files))]
            if dtype == 'int16':
                sb.writeInt16Data(self._out_probs_gdal, n_threads, gdal_opts, temp_tif, out_dir, out_files, 
                                  write_idx, 0, 0, self.data.x_size, self.data.y_size, int(nodata), compress_cmd, s3_out)
            elif dtype == 'uint8':
                sb.writeByteData(self._out_probs_gdal, n_threads, gdal_opts, temp_tif, out_dir, out_files, 
                                 write_idx, 0, 0, self.data.x_size, self.data.y_size, int(nodata), compress_cmd, s3_out)
        if s3_prefix is not None:
            print(f'List results with `mc ls {s3_aliases[0]}{s3_prefix}/{self.data.tile_id}/{out_files_prefix[0]}_`')
            _rm_dir(out_dir)
            return s3_out
        return [f"{out_dir}/{file}" for file in out_files]
#
class TreesRandomForestRegressor(RandomForestRegressor):
    def predict(self, X):
        """
        Predict regression target for X.

        The predicted regression target of an input sample is computed according
        to a list of functions that receives the predicted regression targets of each 
        single tree in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        s : an ndarray of shape (n_estimators, n_samples)
            The predicted values for each single tree.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # store the output of every estimator
        assert(self.n_outputs_ == 1)
        pred_t = np.empty((len(self.estimators_), X.shape[0]), dtype=np.float32)
        # Assign chunk of trees to jobs
        n_jobs = min(self.n_estimators, os.cpu_count() * 2)
        # Parallel loop prediction
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_single_prediction)(self.estimators_[i].predict, X, pred_t, i, lock)
            for i in range(len(self.estimators_))
        )
        return pred_t
#
class Reducer():
    def __init__(self, 
                 reducer_name:str, 
                 reducer_features:List[str],
                 reducer_fn:Callable) -> None:
        self.reducer_name = reducer_name
        self.reducer_features = reducer_features
        self.reducer_fn = reducer_fn
        self.in_covs_t = None
        self.in_covs = None
        self.in_covs_valid = None
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.in_covs_t = None
        self.in_covs = None
        self.in_covs_valid = None
    def reduce(self, data:TiledDataLoader):
        with TimeTracker(f"tile {data.tile_id} - predict ({len(self.reducer_features)} input features)", True):
            # prepare input and output arrays
            n_groups = len(data.catalog.get_groups())
            n_samples = n_groups * data.n_pixels
            n_samples_valid = n_groups * data.n_pixels_valid
            self.in_covs_t = np.empty((len(self.reducer_features), n_samples), dtype=np.float32)
            self.in_covs = np.empty((n_samples, len(self.reducer_features)), dtype=np.float32)
            self.in_covs_valid = np.empty((n_samples_valid, len(self.reducer_features)), dtype=np.float32)
            # transpose data
            matrix_idx = data.catalog._get_covs_idx(self.reducer_features)
            with TimeTracker(f"tile {data.tile_id} - transpose data ({data.n_threads} threads)"):
                sb.reorderArray(data.array, data.n_threads, self.in_covs_t, matrix_idx)
                sb.transposeArray(self.in_covs_t, data.n_threads, self.in_covs)
                sb.selArrayRows(self.in_covs, data.n_threads, self.in_covs_valid, data.get_pixels_valid_idx(n_groups))
            # create output result
            result = ReducedValues(data=data, reducer_name=self.reducer_name)
            # compute
            with TimeTracker(f"tile {data.tile_id} - model prediction ({data.n_threads} threads)"):
                result._out_reduc_valid[:] = self.reducer_fn(self.in_covs_valid)
        return result # shape: (n_samples, 1)
#
class ReducedValues():
    def __init__(self, data:TiledDataLoader, reducer_name:str) -> None:
        self.data = data
        assert(self.data is not None)
        self.reducer_name = reducer_name
        self.groups = self.data.catalog.get_groups()
        self.n_groups = len(self.groups)
        self._out_reduc_valid = np.empty((self.n_groups * self.data.n_pixels_valid), dtype=np.float32)
        self._out_reduc = None
        self._out_reduc_t = None
        self._out_reduc_gdal = None
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self._out_reduc_valid = None
        self._out_reduc = None
        self._out_reduc_t = None
        self._out_reduc_gdal = None
    @property
    def reduced_values(self):
        return self._out_reduc_valid.reshape((self.n_groups, self.data.n_pixels_valid))
    def save_reduced_layer(self, 
                           base_dir, 
                           nodata, 
                           dtype, 
                           out_files_prefix, 
                           out_files_suffix, 
                           s3_prefix, 
                           s3_aliases, 
                           gdal_opts,
                           threads):
        assert(self._out_reduc_valid is not None)
        assert(dtype == 'int16' or dtype == 'uint8')
        assert(len(out_files_prefix) == len(out_files_suffix))
        assert(len(out_files_prefix) == 1)
        assert(s3_prefix is None or len(s3_aliases) > 0)
        # create and transpose output
        with TimeTracker(f"tile {self.data.tile_id} - transpose data for final output ({threads} threads)"):
            # expand to original number of pixels
            self._out_reduc = np.empty((self.n_groups * self.data.n_pixels, 1), dtype=np.float32)
            sb.fillArray(self._out_reduc, threads, nodata)
            sb.expandArrayRows(self._out_reduc_valid, threads, self._out_reduc, self.data.get_pixels_valid_idx(self.n_groups))
            # transpose expanded array
            self._out_reduc_t = np.empty((1, self.n_groups * self.data.n_pixels), dtype=np.float32)
            sb.transposeArray(self._out_reduc, threads, self._out_reduc_t)
            # rearrange groups
            self._out_reduc_gdal = np.empty((1 * self.n_groups, self.data.n_pixels), dtype=np.float32)
            sb.fillArray(self._out_reduc_gdal, threads, nodata)
            inverse_idx = np.empty((self._out_reduc_gdal.shape[0], 2), dtype=np.uintc)
            subrows_grid, rows_grid = np.meshgrid(np.arange(self.n_groups), list(range(1)))
            inverse_idx[:,0] = rows_grid.flatten()
            inverse_idx[:,1] = subrows_grid.flatten()
            sb.inverseReorderArray(self._out_reduc_t, threads, self._out_reduc_gdal, inverse_idx)
        # write outputs
        with TimeTracker(f"tile {self.data.tile_id} - write probs images ({threads} threads)"):
            out_dir = _make_dir(f"{base_dir}/{self.data.tile_id}/{self.reducer_name}")
            out_files = _get_out_files(out_files_prefix, out_files_suffix, self.groups)
            temp_tif = [self.data.mask_path for _ in range(len(out_files))]
            write_idx = range(self._out_reduc_gdal.shape[0])
            compress_cmd = f"gdal_translate -a_nodata {nodata} -co COMPRESS=deflate -co PREDICTOR=2 -co TILED=TRUE -co BLOCKXSIZE=2048 -co BLOCKYSIZE=2048"
            s3_out = None
            if s3_prefix is not None:
                s3_out = [f'{s3_aliases[random.randint(0, len(s3_aliases) - 1)]}{s3_prefix}/{self.data.tile_id}' for _ in range(len(out_files))]
            if dtype == 'int16':
                sb.writeInt16Data(self._out_reduc_gdal, threads, gdal_opts, temp_tif, out_dir, out_files, 
                                  write_idx, 0, 0, self.data.x_size, self.data.y_size, int(nodata), compress_cmd, s3_out)
            elif dtype == 'uint8':
                sb.writeByteData(self._out_reduc_gdal, threads, gdal_opts, temp_tif, out_dir, out_files, 
                                 write_idx, 0, 0, self.data.x_size, self.data.y_size, int(nodata), compress_cmd, s3_out)
        if s3_prefix is not None:
            print(f'List results with `mc ls {s3_aliases[0]}{s3_prefix}/{self.data.tile_id}/{out_files_prefix[0]}_`')
            _rm_dir(out_dir)
            return s3_out
        return [f"{out_dir}/{file}" for file in out_files]
#
def _read_model(model_name:str, model_path:str, model_covs_path:str):
    if model_path.endswith('.joblib'):
        model = joblib.load(model_path)
    elif model_path.endswith('.so'):
        model = tl2cgen.Predictor(model_path)
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
def _get_out_files(out_files_prefix:List[str], out_files_suffix:List[str], groups:List[str], n_class:Optional[int]=None):
    if n_class is None:
        n_class = 1
    out_files = []
    if 'common' in groups:
        for _ in groups:
            for i in range(n_class):
                file = f'{out_files_prefix[i]}_{out_files_suffix[i]}'
                out_files.append(file)
    else:
        for group in groups:
            for i in range(n_class):
                file = f'{out_files_prefix[i]}_{group}0101_{group}1231_{out_files_suffix[i]}'
                out_files.append(file)
    return out_files
