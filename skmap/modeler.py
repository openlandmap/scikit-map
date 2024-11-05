from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
import skmap_bindings as skb
from joblib import Parallel, delayed
import joblib
import threading
import numpy as np
import skmap.misc
import treelite_runtime
from skmap.catalog import DataLoader
import skmap_bindings as sb

def _single_prediction(predict, X, out, i, lock):
    prediction = predict(X, check_input=False)
    with lock:
        out[i, :] = prediction

def cast_tree_rf(model):
    model.__class__ = TreesRandomForestRegressor
    return model


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
            sb.reorderArray(data.cache, data.threads, self.in_covs_t, matrix_idx)
            sb.transposeArray(self.in_covs_t, data.threads, self.in_covs)
            sb.selArrayRows(self.in_covs, data.threads, self.in_covs_valid, data.get_pixels_valid_idx(data.num_years))
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
        sb.reorderArray(data.cache, data.threads, self.in_covs_t, matrix_idx)
        sb.transposeArray(self.in_covs_t, data.threads, self.in_covs)
        sb.selArrayRows(self.in_covs, data.threads, self.in_covs_valid, data.get_pixels_valid_idx(data.num_years))
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
                sb.reorderArray(data.cache, data.threads, self.in_covs_t, matrix_idx)
                sb.transposeArray(self.in_covs_t, data.threads, self.in_covs)
                sb.selArrayRows(self.in_covs, data.threads, self.in_covs_valid, data.get_pixels_valid_idx(data.num_years))
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
            sb.fillArray(self._out_stats, threads, nodata)
            sb.expandArrayRows(self._out_stats_valid, threads, self._out_stats, self._data.get_pixels_valid_idx(self.num_years))
            # transpose expanded array
            self._out_stats_t = np.empty((self.num_depths * self.num_stats, self.num_years * self._data.num_pixels), dtype=np.float32)
            sb.transposeArray(self._out_stats, threads, self._out_stats_t)
            # rearrange years and stats
            # TODO ? could this be replaced by just self._out_stats_t.reshape((self.num_depths * self.num_stats * self.num_years, self.model._data.n_pixels))?
            self._out_stats_gdal = np.empty((self.num_depths * self.num_stats * self.num_years, self._data.num_pixels), dtype=np.float32)
            sb.fillArray(self._out_stats_gdal, threads, nodata)
            inverse_idx = np.empty((self.num_depths * self.num_stats * self.num_years, 2), dtype=np.uintc)
            subrows_grid, rows_grid = np.meshgrid(np.arange(self.num_years), list(range(self.num_depths * self.num_stats)))
            inverse_idx[:,0] = rows_grid.flatten()
            inverse_idx[:,1] = subrows_grid.flatten()
            sb.inverseReorderArray(self._out_stats_t, threads, self._out_stats_gdal, inverse_idx)
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
            # TODO change the need for base image in sb.writeByteData and sb.writeInt16Data
            temp_dir = f"{base_dir}/.skmap"
            temp_tif = self._data.create_image_template(dtype, nodata, temp_dir)
            write_idx = range(self.num_depths * self.num_stats * self.num_years)
            compress_cmd = f"gdal_translate -a_nodata {nodata} -co COMPRESS=deflate -co ZLEVEL=9 -co TILED=TRUE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024"
            s3_out = None
            if s3_prefix is not None:
                s3_out = [f'{s3_aliases[random.randint(0, len(s3_aliases) - 1)]}{s3_prefix}/{self._data.tile_id}' for _ in range(len(out_files))]
            if dtype == 'int16':
                sb.writeInt16Data(self._out_stats_gdal, threads, gdal_opts, [temp_tif for _ in range(len(out_files))], out_dir, out_files, 
                                              write_idx, 0, 0, self._data.x_size, self._data.y_size, int(nodata), compress_cmd, s3_out)
            elif dtype == 'uint8':
                sb.writeByteData(self._out_stats_gdal, threads, gdal_opts, [temp_tif for _ in range(len(out_files))], out_dir, out_files, 
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
            sb.fillArray(self._out_cls, threads, nodata)
            sb.expandArrayRows(self._out_cls_valid, threads, self._out_cls, self._data.get_pixels_valid_idx(self.num_years))
            # transpose expanded array
            self._out_cls_t = np.empty((self._out_cls_valid.shape[1], self.num_years * self._data.num_pixels), dtype=np.float32)
            sb.transposeArray(self._out_cls, threads, self._out_cls_t)
            # rearrange years and stats
            self._out_cls_gdal = np.empty((self._out_cls_valid.shape[1] * self.num_years, self._data.num_pixels), dtype=np.float32)
            sb.fillArray(self._out_cls_gdal, threads, nodata)
            inverse_idx = np.empty((self._out_cls_gdal.shape[0], 2), dtype=np.uintc)
            subrows_grid, rows_grid = np.meshgrid(np.arange(self.num_years), list(range(self._out_cls_valid.shape[1])))
            inverse_idx[:,0] = rows_grid.flatten()
            inverse_idx[:,1] = subrows_grid.flatten()
            sb.inverseReorderArray(self._out_cls_t, threads, self._out_cls_gdal, inverse_idx)
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
                sb.writeInt16Data(self._out_cls_gdal, threads, gdal_opts, [temp_tif for _ in range(len(out_files))], out_dir, out_files, 
                                              write_idx, 0, 0, self._data.x_size, self._data.y_size, int(nodata), compress_cmd, s3_out)
            elif dtype == 'uint8':
                sb.writeByteData(self._out_cls_gdal, threads, gdal_opts, [temp_tif for _ in range(len(out_files))], out_dir, out_files, 
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
            sb.fillArray(self._out_probs, threads, nodata)
            sb.expandArrayRows(self._out_probs_valid, threads, self._out_probs, self._data.get_pixels_valid_idx(self.num_years))
            # transpose expanded array
            self._out_probs_t = np.empty((self.num_class, self.num_years * self._data.num_pixels), dtype=np.float32)
            sb.transposeArray(self._out_probs, threads, self._out_probs_t)
            # rearrange years and stats
            self._out_probs_gdal = np.empty((self.num_class * self.num_years, self._data.num_pixels), dtype=np.float32)
            sb.fillArray(self._out_probs_gdal, threads, nodata)
            inverse_idx = np.empty((self._out_probs_gdal.shape[0], 2), dtype=np.uintc)
            subrows_grid, rows_grid = np.meshgrid(np.arange(self.num_years), list(range(self.num_class)))
            inverse_idx[:,0] = rows_grid.flatten()
            inverse_idx[:,1] = subrows_grid.flatten()
            sb.inverseReorderArray(self._out_probs_t, threads, self._out_probs_gdal, inverse_idx)
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
                sb.writeInt16Data(self._out_probs_gdal, threads, gdal_opts, [temp_tif for _ in range(len(out_files))], out_dir, out_files, 
                                              write_idx, 0, 0, self._data.x_size, self._data.y_size, int(nodata), compress_cmd, s3_out)
            elif dtype == 'uint8':
                sb.writeByteData(self._out_probs_gdal, threads, gdal_opts, [temp_tif for _ in range(len(out_files))], out_dir, out_files, 
                                             write_idx, 0, 0, self._data.x_size, self._data.y_size, int(nodata), compress_cmd, s3_out)
        if s3_prefix is not None:
            print(f'List results with `mc ls {s3_aliases[0]}{s3_prefix}/{self._data.tile_id}/{out_files_prefix[0]}_`')
            shutil.rmtree(out_dir)
            os.remove(temp_tif)
            return s3_out
        return [f"{out_dir}/{file}" for file in out_files]
    
    
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
        n_jobs = min(self.n_estimators, self.n_jobs)
        # Parallel loop prediction
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_single_prediction)(self.estimators_[i].predict, X, pred_t, i, lock)
            for i in range(len(self.estimators_))
        )
        return pred_t

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