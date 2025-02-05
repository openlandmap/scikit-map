from typing import List, Callable, Optional
import os
import random
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
import joblib
import threading
import tl2cgen
import numpy as np
import treelite
from skmap.tiled_data import TiledData, TiledDataLoader
import skmap_bindings as sb
from skmap.misc import _make_dir, TimeTracker, _rm_dir
os.environ['USE_PYGEOS'] = '0'
os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'
n_cpus = os.cpu_count()
n_threads = n_cpus * 2
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

def _tree_based_load_model(model_path):
    if model_path.endswith(('.joblib', '.lz4')):
        model = joblib.load(model_path)
        predict_fn = lambda predictor, data: predictor.predict(data)
    elif model_path.endswith('.so'):
        model = tl2cgen.Predictor(model_path, nthread=n_cpus)
        def predict_tl2cgen(predictor, data):
            dmat = tl2cgen.DMatrix(data, dtype='float32')
            res = predictor.predict(dmat)
            if res.shape[-1] == 1:
                res = np.squeeze(res, axis=-1)
            if res.shape[-1] == 1:
                res = np.squeeze(res, axis=-1)
            return res
        predict_fn = predict_tl2cgen
        os.environ['TREELITE_BIND_THREADS'] = '0'
    else:
        raise ValueError(f"Invalid model path extension '{model_path}'")
    return model, predict_fn

class Modeler():
    def __init__(self, 
             model_path: str, 
             model_covs_path: str = None, 
             predict_fn:Callable=lambda predictor, data: predictor.predict(data)) -> None:
        self.model_path = model_path
        self.model_covs_path = model_covs_path
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
        else:
            raise ValueError(f"Invalid model path extension '{self.model_path}'")
        self.model = model
        
    def _load_covs(self):
        if self.model_covs_path is not None:
            with open(self.model_covs_path, 'r') as file:
                model_covs = [line.strip() for line in file]
        elif hasattr(self.model, "feature_names_in_"):
            model_covs = list(self.model.feature_names_in_)
        elif hasattr(self.model, 'feature_names_'):
            model_covs = self.model.feature_names_
        elif hasattr(self.model, 'feature_name'):
            model_covs = self.model.feature_name()
        else:
            base, _ = os.path.splitext(self.model_path)
            covs_path = base + '.covs'
            if os.path.exists(covs_path):
                with open(covs_path, 'r') as file:
                    print(covs_path)
                    model_covs = [line.strip() for line in file]
            else:
                raise ValueError(f"No feature names was found")
        self.model_covs = model_covs
        
    def _prepare_covariates(self, data:TiledDataLoader):
        # prepare input and output arrays
        n_groups = len(data.catalog.get_groups())
        n_samples = n_groups * data.n_pixels
        n_samples_valid = n_groups * data.n_pixels_valid
        self.in_covs_t = np.empty((len(self.model_covs), n_samples), dtype=np.float32)
        self.in_covs = np.empty((n_samples, len(self.model_covs)), dtype=np.float32)
        self.in_covs_valid = np.empty((n_samples_valid, len(self.model_covs)), dtype=np.float32)
        # transpose data
        matrix_idx = data.catalog._get_covs_idx(self.model_covs)
        sb.reorderArray(data.array, data.n_threads, self.in_covs_t, matrix_idx)
        sb.transposeArray(self.in_covs_t, data.n_threads, self.in_covs)
        sb.selArrayRows(self.in_covs, data.n_threads, self.in_covs_valid, data.get_pixels_valid_idx(n_groups))
    
    def predict():
        raise NotImplementedError()

#################################################################################################################################
########################################  Regressors     ########################################################################
#################################################################################################################################

class Regressor(Modeler):
    def __init__(self, 
             model_path: str, 
             model_covs_path: str = None,
             n_responses: int = 1,
             predict_fn:Callable = None) -> None:
        super().__init__(model_path, model_covs_path, predict_fn)
        self.n_responses = n_responses
#
class RFRegressor(Regressor):
    def __init__(self, 
             model_path: str, 
             model_covs_path: str = None,
             n_responses: int = 1,
             predict_fn:Callable = None) -> None:
        super().__init__(model_path, model_covs_path, n_responses, predict_fn)
        self.model, self.predict_fn = _tree_based_load_model(model_path)
        if predict_fn:
            self.predict_fn = predict_fn
        self._load_covs()

    def predict(self, data:TiledData):
        with TimeTracker(f"    Predict ({len(self.model_covs)} input features)", True):
            # prepare input and output arrays
            with TimeTracker(f"        Transpose data ({data.n_threads} threads)"):
                self._prepare_covariates(data)
            # predict
            with TimeTracker(f"        Model prediction ({data.n_threads} threads)"):
                result = TiledData(self.n_responses, self.in_covs_valid.shape[0])
                assert self.n_responses == 1, "Do not yet manage the case for multiple respounces"
                result.array[0,:] = self.predict_fn(self.model, self.in_covs_valid).astype(np.float32)
        return result # shape: (n_responses, n_samples)
    
#################################################################################################################################
########################################  Classifiers    ########################################################################
#################################################################################################################################

class Classifier(Modeler):
    def __init__(self, 
             model_path: str, 
             model_covs_path: str = None, 
             n_class: int = 1,
             predict_fn:Callable = lambda predictor, data: predictor.predict_proba(data)) -> None:
        super().__init__(model_path, model_covs_path, predict_fn)
        self.n_class = n_class
#
class RFClassifier(Classifier):
    def __init__(self, 
             model_path: str, 
             model_covs_path: str = None, 
             n_class: int = 1,
             predict_fn:Callable = None) -> None:
        super().__init__(model_path, model_covs_path, n_class, predict_fn)
        self.model, _ = _tree_based_load_model(model_covs_path)
        if predict_fn:
            self.predict_fn = predict_fn
        self._load_covs()
        
    def predict(self, data:TiledData):
        with TimeTracker(f"    Predict ({len(self.model_covs)} input features)", True):
            # prepare input and output arrays
            self._prepare_covariates(data)
            # create output result
            result = PredictedProbs(data=data, n_class=self.n_class)
            # predict
            with TimeTracker(f"        Model prediction ({data.n_threads} threads)"):
                tmp_res = self.predict_fn(self.model, self.in_covs_valid)
                if tmp_res.dtype == np.float64:
                    sb.castFloat64ToFloat32(tmp_res, data.n_threads, result._out_probs_valid)
                elif tmp_res.dtype == np.float32:
                    result._out_probs_valid = tmp_res
                else:
                    print("Result prediction are not in float32 nor float64, converting with python (can be slow)")
                    result._out_probs_valid = tmp_res.astype(np.float32)
        return result # shape: (n_samples, n_classes)

    
#################################################################################################################################
########################################  Others         ########################################################################
#################################################################################################################################

class Predicted():
    def __init__(self,
                 data:TiledDataLoader, 
                  depths) -> None:
        self.data = data
        assert(self.data is not None)
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
        with TimeTracker(f"    Tile {self.data.tile_id}/model {self.model_name} - transpose data for final output ({threads} threads)"):
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
        with TimeTracker(f"    Tile {self.data.tile_id}/model {self.model_name} - write output ({threads} threads)"):
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
        with TimeTracker(f"    Tile {self.data.tile_id}/model {self.model_name} - transpose data for final output ({n_threads} threads)"):
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
        with TimeTracker(f"    Tile {self.data.tile_id}/model {self.model_name} - write class images ({n_threads} threads)"):
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
        with TimeTracker(f"    Tile {self.data.tile_id}/model {self.model_name} - transpose data for final output ({n_threads} threads)"):
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
        with TimeTracker(f"    Tile {self.data.tile_id}/model {self.model_name} - write probs images ({n_threads} threads)"):
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
        with TimeTracker(f"    Tile {data.tile_id} - predict ({len(self.reducer_features)} input features)", True):
            # prepare input and output arrays
            n_groups = len(data.catalog.get_groups())
            n_samples = n_groups * data.n_pixels
            n_samples_valid = n_groups * data.n_pixels_valid
            self.in_covs_t = np.empty((len(self.reducer_features), n_samples), dtype=np.float32)
            self.in_covs = np.empty((n_samples, len(self.reducer_features)), dtype=np.float32)
            self.in_covs_valid = np.empty((n_samples_valid, len(self.reducer_features)), dtype=np.float32)
            # transpose data
            matrix_idx = data.catalog._get_covs_idx(self.reducer_features)
            with TimeTracker(f"    Tile {data.tile_id} - transpose data ({data.n_threads} threads)"):
                sb.reorderArray(data.array, data.n_threads, self.in_covs_t, matrix_idx)
                sb.transposeArray(self.in_covs_t, data.n_threads, self.in_covs)
                sb.selArrayRows(self.in_covs, data.n_threads, self.in_covs_valid, data.get_pixels_valid_idx(n_groups))
            # create output result
            result = ReducedValues(data=data, reducer_name=self.reducer_name)
            # compute
            with TimeTracker(f"    Tile {data.tile_id} - model prediction ({data.n_threads} threads)"):
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
        with TimeTracker(f"    Tile {self.data.tile_id} - transpose data for final output ({threads} threads)"):
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
        with TimeTracker(f"    Tile {self.data.tile_id} - write probs images ({threads} threads)"):
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
