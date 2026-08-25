import os
import random
import threading
from typing import Callable, List, NoReturn, Optional

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestRegressor

import skmap.set_env  # noqa: F401
import skmap_bindings as sb
from skmap.misc import TimeTracker, _make_dir, _rm_dir, s3_upload_file, sb_arr
from skmap.tiled_data import TiledData, TiledDataLoader


def _get_out_files_depths(
    out_files_prefix,
    out_files_suffix,
    tile_id,
    depths,
    n_depths,
    years,
    n_years,
    n_stats,
):
    assert len(out_files_prefix) == len(out_files_suffix)
    assert len(out_files_prefix) == n_stats
    assert len(depths) >= n_depths
    assert len(years) >= n_years
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
                        file = f"{out_files_prefix[k]}_b{d1}cm..{d2}cm_{y1}0101_{y2}1231_tile.{tile_id}_{out_files_suffix[k]}"
                    else:
                        d1 = depths[i]
                        file = f"{out_files_prefix[k]}_b{d1}cm_{y1}0101_{y2}1231_tile.{tile_id}_{out_files_suffix[k]}"
                else:
                    y1 = years[j]
                    if n_depths < len(depths):
                        d1 = depths[i]
                        d2 = depths[i + len(depths) - n_depths]
                        file = f"{out_files_prefix[k]}_b{d1}cm..{d2}cm_{y1}0101_{y1}1231_tile.{tile_id}_{out_files_suffix[k]}"
                    else:
                        d1 = depths[i]
                        file = f"{out_files_prefix[k]}_b{d1}cm_{y1}0101_{y1}1231_tile.{tile_id}_{out_files_suffix[k]}"
                out_files.append(file)
    return out_files


def _s3_upload_outputs(
    s3_clients, s3_prefix, tile_id, out_dir, out_files, temp_tif=None
):
    """Upload written GeoTIFFs to S3 via the MinIO client and clean up local files."""
    client = random.choice(s3_clients)
    bucket, prefix_path = s3_prefix.partition("/")
    s3_out = []
    for out_file in out_files:
        local = f"{out_dir}/{out_file}.tif"
        object_name = (
            f"{prefix_path}/{tile_id}/{out_file}.tif"
            if prefix_path
            else f"{tile_id}/{out_file}.tif"
        )
        s3_upload_file(client, bucket, object_name, local)
        s3_out.append(object_name)
    _rm_dir(out_dir)
    if temp_tif is not None:
        os.remove(temp_tif)
    return s3_out


def _write_output_layers(
    data_gdal,
    out_dir,
    out_files,
    base_files,
    threads,
    gdal_opts,
    nodata,
    dtype,
    x_size,
    y_size,
    s3_prefix,
    s3_clients,
    tile_id,
    temp_file=None,
):
    """Write a (layers x pixels) array to GeoTIFFs and optionally upload to S3.

    Shared by ``Predicted``, ``PredictedProbs`` and ``ReducedValues``; the five
    ``save_*_layer`` methods only differ in how they build ``out_files`` and
    ``base_files``.
    """
    creation_options = [
        "COMPRESS=deflate",
        "PREDICTOR=2",
        "TILED=TRUE",
        "BLOCKXSIZE=2048",
        "BLOCKYSIZE=2048",
    ]
    write_idx = range(data_gdal.shape[0])
    if dtype == "int16":
        sb.writeInt16Data(
            data_gdal,
            threads,
            gdal_opts,
            base_files,
            out_dir,
            out_files,
            write_idx,
            0,
            0,
            x_size,
            y_size,
            int(nodata),
            creation_options,
        )
    elif dtype == "uint8":
        sb.writeByteData(
            data_gdal,
            threads,
            gdal_opts,
            base_files,
            out_dir,
            out_files,
            write_idx,
            0,
            0,
            x_size,
            y_size,
            int(nodata),
            creation_options,
        )
    if s3_prefix is not None:
        return _s3_upload_outputs(
            s3_clients, s3_prefix, tile_id, out_dir, out_files, temp_file
        )
    return [f"{out_dir}/{file}" for file in out_files]


def _tree_based_load_model(model_path):
    if model_path.endswith((".joblib", ".lz4", ".pkl")):
        model = joblib.load(model_path)

        def predict_fn(predictor, data):
            return predictor.predict(data)
    elif model_path.endswith(".so"):
        import tl2cgen

        try:
            tl2cgen.util.check_if_fast()
        except:
            print(
                "The current installation of tl2cgen is not the one with parallel DMatrix and can be slow"
            )
        model = tl2cgen.Predictor(model_path, nthread=os.cpu_count())

        def predict_tl2cgen(predictor, data):
            dmat = tl2cgen.DMatrix(data, dtype="float32")
            res = predictor.predict(dmat)
            for a in range(len(res.shape)):
                if res.shape[a] == 1:
                    res = np.squeeze(res, axis=a)
                    break
            for a in range(len(res.shape)):
                if res.shape[a] == 1:
                    res = np.squeeze(res, axis=a)
                    break
            return res

        predict_fn = predict_tl2cgen
    else:
        raise ValueError(f"Invalid model path extension '{model_path}'")
    return model, predict_fn


class Modeler:
    """Base class wrapping a fitted model, its covariate names, and tile/feature preparation."""

    def __init__(
        self,
        model_path: str,
        model_covs_path: str = None,
        predict_fn: Callable = lambda predictor, data: predictor.predict(data),
    ) -> None:
        assert os.path.exists(model_path), f"Model path {model_path} do not exist"
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

    def _load_model(self) -> None:
        if self.model_path.endswith((".joblib", ".lz4", ".pkl")):
            model = joblib.load(self.model_path)
        else:
            raise ValueError(f"Invalid model path extension '{self.model_path}'")
        self.model = model

    def _load_covs(self) -> None:
        if self.model_covs_path is not None:
            with open(self.model_covs_path, "r") as file:
                model_covs = [line.strip() for line in file]
        elif hasattr(self.model, "feature_names_in_"):
            model_covs = list(self.model.feature_names_in_)
        elif hasattr(self.model, "feature_names_"):
            model_covs = self.model.feature_names_
        elif hasattr(self.model, "feature_name"):
            model_covs = self.model.feature_name()
        else:
            base, _ = os.path.splitext(self.model_path)
            covs_path = base + ".covs"
            if os.path.exists(covs_path):
                with open(covs_path, "r") as file:
                    model_covs = [line.strip() for line in file]
            else:
                raise ValueError("No feature names was found")
        self.model_covs = model_covs
        # sklearn convention: expose the feature names seen during fit.
        self.feature_names_in_ = np.asarray(model_covs, dtype=object)

    def _prepare_covariates(self, data: TiledDataLoader) -> None:
        # prepare input and output arrays
        n_groups = len(data.catalog.get_groups())
        n_samples = n_groups * data.n_pixels
        n_samples_valid = n_groups * data.n_pixels_valid
        self.in_covs_t = sb_arr(len(self.model_covs), n_samples)
        self.in_covs = sb_arr(n_samples, len(self.model_covs))
        self.in_covs_valid = sb_arr(n_samples_valid, len(self.model_covs))
        # transpose data
        matrix_idx = data.catalog._get_covs_idx(self.model_covs)
        sb.reorderArray(data.array, data.n_threads, self.in_covs_t, matrix_idx)
        sb.transposeArray(self.in_covs_t, data.n_threads, self.in_covs)
        sb.selArrayRows(
            self.in_covs,
            data.n_threads,
            self.in_covs_valid,
            data.get_pixels_valid_idx(n_groups),
        )

    def predict(self, X) -> np.ndarray:
        """sklearn-style prediction on a 2D array of covariates.

        Tile-oriented prediction (which reads rasters and transposes data) is
        available as ``predict_tile`` on the concrete subclasses.
        """
        return self.predict_fn(self.model, X)

    def predict_raster(self, rdata, valid_only: bool = True) -> np.ndarray:
        """Predict on a pre-loaded :class:`~skmap.io.RasterData` (spatial).

        Maps the model's covariate names to RasterData bands (via
        :meth:`RasterData._get_covs_idx`), predicts once per group (year),
        and returns a numpy array of shape
        ``(n_groups, H*W[, n_responses/n_class])`` with ``NaN`` for invalid
        pixels.

        :param rdata: a RasterData whose bands are named after the model's
          covariate features (``common`` group falls back for shared layers).
        :param valid_only: if ``True`` (default) only non-NaN pixels are
          predicted and the result is a dense ``(n_groups, H*W, ...)`` array
          with ``NaN`` elsewhere.
        """
        groups = rdata.get_groups()
        covs_idx = rdata._get_covs_idx(self.model_covs)  # (n_covs, n_groups)
        arr = rdata.array.get()
        n_pixels = arr.shape[1]

        predictions = []
        for j in range(len(groups)):
            X = arr[covs_idx[:, j], :].T  # (n_pixels, n_covs)
            if valid_only:
                valid = ~np.isnan(X).any(axis=1)
                pred_valid = np.asarray(
                    self.predict_fn(self.model, X[valid]), dtype=np.float32
                )
                shape = (n_pixels,) + pred_valid.shape[1:]
                pred = np.full(shape, np.nan, dtype=np.float32)
                pred[valid] = pred_valid
            else:
                pred = np.asarray(self.predict_fn(self.model, X), dtype=np.float32)
            predictions.append(pred)

        return np.stack(predictions)
    def predict_tile(self, data: TiledData) -> NoReturn:
        """Run the model over a :class:`TiledData` tile and return a :class:`TiledData` of predictions."""

        raise NotImplementedError()


#################################################################################################################################
########################################  Regressors     ########################################################################
#################################################################################################################################


class Regressor(Modeler, RegressorMixin, BaseEstimator):
    """Base regressor wrapping a fitted model (inherits scikit-learn :class:`RegressorMixin`)."""

    def __init__(
        self,
        model_path: str,
        model_covs_path: str = None,
        n_responses: int = 1,
        predict_fn: Callable = None,
    ) -> None:
        super().__init__(model_path, model_covs_path, predict_fn)
        self.n_responses = n_responses


#
class RFRegressor(Regressor):
    """Tree-based regressor (joblib or tl2cgen) predicting a single response per tile."""

    def __init__(
        self,
        model_path: str,
        model_covs_path: str = None,
        n_responses: int = 1,
        predict_fn: Callable = None,
    ) -> None:
        super().__init__(model_path, model_covs_path, n_responses, predict_fn)
        self.model, self.predict_fn = _tree_based_load_model(model_path)
        if predict_fn:
            self.predict_fn = predict_fn
        self._load_covs()

    def predict_tile(self, data: TiledData):
        # prepare input and output arrays
        """Predict one response for every valid pixel of ``data`` and return a :class:`TiledData`."""

        with TimeTracker("          Transpose data", False):
            self._prepare_covariates(data)
        # predict
        with TimeTracker("          Model prediction", False):
            result = TiledData(
                self.n_responses, self.in_covs_valid.shape[0], data.tile_id
            )
            assert self.n_responses == 1, (
                "Do not yet manage the case for multiple respounces"
            )
            result.array[0, :] = self.predict_fn(self.model, self.in_covs_valid).astype(
                np.float32
            )
        return result  # shape: (n_responses, n_samples)


#
class RFRegressorTrees(Regressor):
    """Regressor returning per-tree predictions of a scikit-learn ``RandomForestRegressor``."""

    def __init__(
        self,
        model_path: str,
        model_covs_path: str = None,
        n_responses: int = 1,
        predict_fn: Callable = None,
    ) -> None:
        super().__init__(model_path, model_covs_path, n_responses, predict_fn)
        self._load_model()
        assert isinstance(self.model, RandomForestRegressor), (
            "The model must be of type sklearn.ensemble.RandomForestRegressor"
        )
        self.n_trees = self.model.n_estimators
        self._load_covs()

    def predict_tile(self, data: TiledData):
        # prepare input and output arrays
        """Predict one row per tree for every valid pixel of ``data`` and return a :class:`TiledData`."""

        with TimeTracker("          Transpose data", False):
            self._prepare_covariates(data)
        # predict
        with TimeTracker("          Model prediction", False):

            def _single_prediction(predict, X, out, i, lock) -> None:
                prediction = predict(X, check_input=False)
                with lock:
                    out[i, :] = prediction

            self.in_covs_valid = self.model._validate_X_predict(self.in_covs_valid)
            assert self.model.n_outputs_ == 1
            result = TiledData(self.n_trees, self.in_covs_valid.shape[0], data.tile_id)
            # Assign chunk of trees to jobs
            n_jobs = min(self.n_trees, data.n_threads)
            # Parallel loop prediction
            lock = threading.Lock()
            Parallel(n_jobs=n_jobs, verbose=self.model.verbose, require="sharedmem")(
                delayed(_single_prediction)(
                    self.model.estimators_[i].predict,
                    self.in_covs_valid,
                    result.array,
                    i,
                    lock,
                )
                for i in range(self.n_trees)
            )
        return result  # shape: (n_responses, n_samples)


#################################################################################################################################
########################################  Classifiers    ########################################################################
#################################################################################################################################


class Classifier(Modeler, ClassifierMixin, BaseEstimator):
    """Base classifier wrapping a fitted model (inherits scikit-learn :class:`ClassifierMixin`)."""

    def __init__(
        self,
        model_path: str,
        model_covs_path: str = None,
        n_class: int = 1,
        predict_fn: Callable = lambda predictor, data: predictor.predict_proba(data),
    ) -> None:
        super().__init__(model_path, model_covs_path, predict_fn)
        self.n_class = n_class


#
class RFClassifier(Classifier):
    """Tree-based classifier (joblib or tl2cgen) returning class labels or probabilities."""

    def __init__(
        self,
        model_path: str,
        model_covs_path: str = None,
        n_class: int = 1,
        predict_fn: Callable = None,
    ) -> None:
        super().__init__(model_path, model_covs_path, n_class, predict_fn)
        self.model, self.predict_fn = _tree_based_load_model(model_path)
        if predict_fn:
            self.predict_fn = predict_fn
        self._load_covs()

    def predict_tile(self, data: TiledData):
        # prepare input and output arrays
        """Predict class labels (or probabilities) for every valid pixel of ``data`` and return a :class:`TiledData`."""

        with TimeTracker("          Transpose data", False):
            self._prepare_covariates(data)
        # predict
        with TimeTracker("          Model prediction", False):
            result = TiledData(self.n_class, self.in_covs_valid.shape[0], data.tile_id)
            if self.n_class == 1:
                result.array[0, :] = self.predict_fn(
                    self.model, self.in_covs_valid
                ).astype(np.float32)
            else:
                tmp_res_t = self.predict_fn(self.model, self.in_covs_valid)
        if self.n_class != 1:
            with TimeTracker("          Convert and back transpose data", False):
                if tmp_res_t.dtype == np.float64:
                    tmp_res_cast_t = sb_arr(tmp_res_t.shape[0], tmp_res_t.shape[1])
                    sb.castFloat64ToFloat32(tmp_res_t, data.n_threads, tmp_res_cast_t)
                    sb.transposeArray(tmp_res_cast_t, data.n_threads, result.array)
                elif tmp_res_t.dtype == np.float32:
                    sb.transposeArray(tmp_res_t, data.n_threads, result.array)
                else:
                    print(
                        "Result prediction are not in float32 nor float64, converting with python (can be slow)"
                    )
                    tmp_res_t = tmp_res_t.astype(np.float32)
                    sb.transposeArray(tmp_res_t, data.n_threads, result.array)

        return result  # shape: (n_class, n_samples)


#################################################################################################################################
########################################  Others         ########################################################################
#################################################################################################################################


class Predicted:
    """Container for tree-level predictions, used to derive per-depth/year statistics."""

    def __init__(self, data: TiledDataLoader, depths) -> None:
        self.data = data
        assert self.data is not None
        self.depths = depths
        self.groups = self.data.catalog.get_groups()
        self.n_groups = len(self.groups)
        # TODO optimize shape of self.out_trees
        self._out_valid = np.empty(
            (self.n_groups, self.data.n_pixels_valid), dtype=np.float32
        )
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
        """Per-depth tree predictions reshaped as ``(n_depths, n_trees, n_years, n_pixels_valid)``."""

        return self._out_valid[: self.n_depths, :, : self.n_groups, :]

    @property
    def predicted_stats(self):
        # self.predicted_stats shape: (n_years, n_pixels_valid, n_depths, n_stats)
        """Per-year statistics reshaped as ``(n_years, n_pixels_valid, n_depths, n_stats)``."""

        if self._out_stats_valid is not None:
            return self._out_stats_valid.reshape(
                (self.n_groups, self.data.n_pixels_valid, self.n_depths, self.n_stats)
            )

    def average_trees_depth_ranges(self) -> None:
        """Average tree predictions across consecutive depth ranges (in place)."""

        assert self.n_depths > 1
        self.n_depths -= 1
        for i in range(self.n_depths):
            # self.predicted_trees shape: (n_depths, n_trees, n_years, n_pixels_valid)
            self._out_valid[i, :, : self.n_groups, :] += self._out_valid[
                i + 1, :, : self.n_groups, :
            ]
            self._out_valid[i, :, : self.n_groups, :] /= 2

    def average_trees_year_ranges(self) -> None:
        """Average tree predictions across consecutive year ranges (in place)."""

        assert self.n_groups > 1
        self.n_groups -= 1
        for j in range(self.n_groups):
            # self.predicted_trees shape: (n_depths, n_trees, n_years, n_pixels_valid)
            self._out_valid[: self.n_depths, :, j, :] += self._out_valid[
                : self.n_depths, :, j + 1, :
            ]
            self._out_valid[: self.n_depths, :, j, :] /= 2

    def compute_stats(
        self, mean=True, quantiles=[0.025, 0.975], expm1=False, scale=1
    ) -> None:
        """Compute mean and quantile statistics over the tree predictions."""

        quantile_idx = 1 if mean else 0
        self.n_stats = quantile_idx + len(quantiles)
        assert self.n_stats > 0
        self._out_stats_valid = np.empty(
            (self.n_groups * self.data.n_pixels_valid, self.n_depths * self.n_stats),
            dtype=np.float32,
        )
        # compute stats
        for i in range(self.n_depths):
            if mean:
                # self.predicted_stats shape: (n_years, n_pixels_valid, n_depths, n_stats)
                # self.predicted_trees shape: (n_depths, n_trees, n_years, n_pixels_valid)
                self.predicted_stats[:, :, i, 0] = np.mean(
                    self._out_valid[i, :, : self.n_groups, :], axis=0
                )
            if len(quantiles) > 0:
                q = np.quantile(
                    self._out_valid[i, :, : self.n_groups, :], quantiles, axis=0
                )
                self.predicted_stats[:, :, i, quantile_idx:] = q.transpose((1, 2, 0))

        # compute inverse log1p
        if expm1:
            np.expm1(self._out_stats_valid, out=self._out_stats_valid)
        # compute scale
        if scale != 1:
            self._out_stats_valid[:] = self._out_stats_valid * scale

    def save_stats_layers(
        self,
        base_dir,
        nodata,
        dtype,
        out_files_prefix,
        out_files_suffix,
        s3_prefix,
        s3_clients,
        gdal_opts,
        threads,
    ):
        """Write the computed statistics to GeoTIFF layers (and optionally upload to S3)."""

        assert self._out_stats_valid is not None
        assert dtype == "int16" or dtype == "uint8"
        assert len(out_files_prefix) == len(out_files_suffix)
        assert len(out_files_prefix) == self.n_stats
        assert s3_prefix is None or len(s3_clients) > 0
        # create and transpose output
        with TimeTracker(
            f"    Tile {self.data.tile_id} - transpose data for final output ({threads} threads)"
        ):
            # expand to original number of pixels
            self._out_stats = np.empty(
                (self.n_groups * self.data.n_pixels, self.n_depths * self.n_stats),
                dtype=np.float32,
            )
            sb.fillArray(self._out_stats, threads, nodata)
            sb.expandArrayRows(
                self._out_stats_valid,
                threads,
                self._out_stats,
                self.data.get_pixels_valid_idx(self.n_groups),
            )
            # transpose expanded array
            self._out_stats_t = np.empty(
                (self.n_depths * self.n_stats, self.n_groups * self.data.n_pixels),
                dtype=np.float32,
            )
            sb.transposeArray(self._out_stats, threads, self._out_stats_t)
            # rearrange years and stats
            # TODO ? could this be replaced by just self._out_stats_t.reshape((self.n_depths * self.n_stats * self.n_years, self.model._data.n_pixels))?
            self._out_stats_gdal = np.empty(
                (self.n_depths * self.n_stats * self.n_groups, self.data.n_pixels),
                dtype=np.float32,
            )
            sb.fillArray(self._out_stats_gdal, threads, nodata)
            inverse_idx = np.empty(
                (self.n_depths * self.n_stats * self.n_groups, 2), dtype=np.uintc
            )
            subrows_grid, rows_grid = np.meshgrid(
                np.arange(self.n_groups), list(range(self.n_depths * self.n_stats))
            )
            inverse_idx[:, 0] = rows_grid.flatten()
            inverse_idx[:, 1] = subrows_grid.flatten()
            sb.inverseReorderArray(
                self._out_stats_t, threads, self._out_stats_gdal, inverse_idx
            )
        # write outputs
        with TimeTracker(
            f"    Tile {self.data.tile_id} - write output ({threads} threads)"
        ):
            out_dir = _make_dir(f"{base_dir}/{self.data.tile_id}")
            # TODO implement filenames function as an class function
            out_files = _get_out_files_depths(
                out_files_prefix=out_files_prefix,
                out_files_suffix=out_files_suffix,
                tile_id=self.data.tile_id,
                depths=self.depths,
                n_depths=self.n_depths,
                years=self.groups,
                n_years=self.n_groups,
                n_stats=self.n_stats,
            )
            # TODO change the need for base image in sb.writeByteData and sb.writeInt16Data
            temp_dir = f"{base_dir}/.skmap"
            temp_tif = self.data.create_image_template(dtype, nodata, temp_dir)
            return _write_output_layers(
                self._out_stats_gdal,
                out_dir,
                out_files,
                [temp_tif for _ in range(len(out_files))],
                threads,
                gdal_opts,
                nodata,
                dtype,
                self.data.x_size,
                self.data.y_size,
                s3_prefix,
                s3_clients,
                self.data.tile_id,
                temp_file=temp_tif,
            )


#
class PredictedProbs:
    """Container for class probability predictions and derived class/KL-divergence outputs."""

    def __init__(self, data: TiledDataLoader, legend) -> None:
        self.data = data
        assert self.data is not None
        self.legend = legend
        assert isinstance(self.legend, dict)
        try:
            self.legend_codes = np.array(list(self.legend.keys()), dtype=np.float32)
        except ValueError as e:
            raise ValueError("Legend keys must be convertible to float32") from e
        self.n_class = len(self.legend)
        self.groups = self.data.catalog.get_groups()
        self.n_groups = len(self.groups)
        self._out_probs_valid = np.empty(
            (self.n_groups * self.data.n_pixels_valid, self.n_class), dtype=np.float32
        )
        self._out_cls_valid = np.empty(
            (self.n_groups * self.data.n_pixels_valid, 1), dtype=np.float32
        )
        self._out_kld_valid = np.empty(
            (self.n_groups * self.data.n_pixels_valid, 1), dtype=np.float32
        )
        self._out_probs = None
        self._out_probs_t = None
        self._out_probs_gdal = None
        self._out_cls = None
        self._out_cls_t = None
        self._out_cls_gdal = None
        self._out_kld = None
        self._out_kld_t = None
        self._out_kld_gdal = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._out_probs = None
        self._out_probs_t = None
        self._out_probs_gdal = None
        self._out_cls = None
        self._out_cls_t = None
        self._out_cls_gdal = None
        self._out_kld = None
        self._out_kld_t = None
        self._out_kld_gdal = None

    @property
    def predicted_probs(self):
        """Class probabilities reshaped as ``(n_groups, n_pixels_valid, n_class)``."""

        return self._out_probs_valid.reshape(
            (self.n_groups, self.data.n_pixels_valid, self.n_class)
        )

    @property
    def predicted_class(self):
        """Dominant class per pixel reshaped as ``(n_groups, n_pixels_valid)``."""

        return self._out_cls_valid.reshape((self.n_groups, self.data.n_pixels_valid))

    def compute_class(self) -> None:
        """Compute the dominant class per pixel from the predicted probabilities."""

        self._out_cls_valid[:, 0] = self.legend_codes[
            np.argmax(self._out_probs_valid[:, :], axis=-1).astype(int)
        ]

    def compute_kl_divergence(self) -> None:
        """Compute the per-pixel Kullback-Leibler divergence of the predicted probabilities."""

        adjusted_data = np.where(
            self._out_probs_valid[:, :] > 0,
            self._out_probs_valid[:, :],
            1 / self.n_class,
        )
        self._out_kld_valid[:, 0] = np.sum(
            adjusted_data * np.log2(adjusted_data * self.n_class), axis=-1
        )

    def save_class_layer(
        self,
        base_dir,
        nodata,
        dtype,
        out_files_prefix,
        out_files_suffix,
        s3_prefix,
        s3_clients,
        gdal_opts,
        n_threads,
    ):
        """Write the dominant-class layer to a GeoTIFF (and optionally upload to S3)."""

        self.compute_class()
        assert self._out_cls_valid is not None
        assert dtype == "int16" or dtype == "uint8"
        assert len(out_files_prefix) == len(out_files_suffix)
        assert len(out_files_prefix) == 1
        assert s3_prefix is None or len(s3_clients) > 0
        # create and transpose output
        with TimeTracker(
            f"    Tile {self.data.tile_id} - transpose data for final output"
        ):
            # expand to original number of pixels
            self._out_cls = np.empty(
                (self.n_groups * self.data.n_pixels, self._out_cls_valid.shape[1]),
                dtype=np.float32,
            )
            sb.fillArray(self._out_cls, n_threads, nodata)
            sb.expandArrayRows(
                self._out_cls_valid,
                n_threads,
                self._out_cls,
                self.data.get_pixels_valid_idx(self.n_groups),
            )
            # transpose expanded array
            self._out_cls_t = np.empty(
                (self._out_cls.shape[1], self._out_cls.shape[0]), dtype=np.float32
            )
            sb.transposeArray(self._out_cls, n_threads, self._out_cls_t)
            # rearrange groups
            self._out_cls_gdal = np.empty(
                (self._out_cls_t.shape[0] * self.n_groups, self.data.n_pixels),
                dtype=np.float32,
            )
            sb.fillArray(self._out_cls_gdal, n_threads, nodata)
            inverse_idx = np.empty((self._out_cls_gdal.shape[0], 2), dtype=np.uintc)
            subrows_grid, rows_grid = np.meshgrid(
                np.arange(self.n_groups), list(range(self._out_cls_valid.shape[1]))
            )
            inverse_idx[:, 0] = rows_grid.flatten()
            inverse_idx[:, 1] = subrows_grid.flatten()
            sb.inverseReorderArray(
                self._out_cls_t, n_threads, self._out_cls_gdal, inverse_idx
            )
        # write outputs
        with TimeTracker(f"    Tile {self.data.tile_id} - write class images"):
            out_dir = _make_dir(f"{base_dir}/{self.data.tile_id}")
            out_files = _get_out_files(out_files_prefix, out_files_suffix, self.groups)
            temp_tif = [self.data.mask_path for _ in range(len(out_files))]
            return _write_output_layers(
                self._out_cls_gdal,
                out_dir,
                out_files,
                temp_tif,
                n_threads,
                gdal_opts,
                nodata,
                dtype,
                self.data.x_size,
                self.data.y_size,
                s3_prefix,
                s3_clients,
                self.data.tile_id,
                temp_file=None,
            )

    def save_kl_divergence_layer(
        self,
        base_dir,
        nodata,
        dtype,
        scale,
        out_files_prefix,
        out_files_suffix,
        s3_prefix,
        s3_clients,
        gdal_opts,
        n_threads,
    ):
        """Write the KL-divergence layer to a GeoTIFF (and optionally upload to S3)."""

        self.compute_kl_divergence()
        assert self._out_kld_valid is not None
        assert dtype == "int16" or dtype == "uint8"
        assert len(out_files_prefix) == len(out_files_suffix)
        assert len(out_files_prefix) == 1
        assert s3_prefix is None or len(s3_clients) > 0
        # create and transpose output
        with TimeTracker(f"tile {self.data.tile_id} - transpose data for final output"):
            sb.offsetAndScale(self._out_kld_valid, n_threads, 0.5 / scale, scale)
            # expand to original number of pixels
            self._out_kld = np.empty(
                (self.n_groups * self.data.n_pixels, self._out_kld_valid.shape[1]),
                dtype=np.float32,
            )
            sb.fillArray(self._out_kld, n_threads, nodata)
            sb.expandArrayRows(
                self._out_kld_valid,
                n_threads,
                self._out_kld,
                self.data.get_pixels_valid_idx(self.n_groups),
            )
            # transpose expanded array
            self._out_kld_t = np.empty(
                (self._out_kld.shape[1], self._out_kld.shape[0]), dtype=np.float32
            )
            sb.transposeArray(self._out_kld, n_threads, self._out_kld_t)
            # rearrange groups
            self._out_kld_gdal = np.empty(
                (self._out_kld_t.shape[0] * self.n_groups, self.data.n_pixels),
                dtype=np.float32,
            )
            sb.fillArray(self._out_kld_gdal, n_threads, nodata)
            inverse_idx = np.empty((self._out_kld_gdal.shape[0], 2), dtype=np.uintc)
            subrows_grid, rows_grid = np.meshgrid(
                np.arange(self.n_groups), list(range(self._out_kld_valid.shape[1]))
            )
            inverse_idx[:, 0] = rows_grid.flatten()
            inverse_idx[:, 1] = subrows_grid.flatten()
            sb.inverseReorderArray(
                self._out_kld_t, n_threads, self._out_kld_gdal, inverse_idx
            )
        # write outputs
        with TimeTracker(
            f"tile {self.data.tile_id} - write Kullback-Leibler divergence image"
        ):
            out_dir = _make_dir(f"{base_dir}/{self.data.tile_id}")
            out_files = _get_out_files(out_files_prefix, out_files_suffix, self.groups)
            temp_tif = [self.data.mask_path for _ in range(len(out_files))]
            return _write_output_layers(
                self._out_kld_gdal,
                out_dir,
                out_files,
                temp_tif,
                n_threads,
                gdal_opts,
                nodata,
                dtype,
                self.data.x_size,
                self.data.y_size,
                s3_prefix,
                s3_clients,
                self.data.tile_id,
                temp_file=None,
            )

    def save_probs_layers(
        self,
        base_dir,
        nodata,
        dtype,
        scale,
        out_files_prefix,
        out_files_suffix,
        s3_prefix,
        s3_clients,
        gdal_opts,
        n_threads,
    ):
        """Write the per-class probability layers to GeoTIFFs (and optionally upload to S3)."""

        assert self._out_probs_valid is not None
        assert dtype == "int16" or dtype == "uint8"
        assert len(out_files_prefix) == len(out_files_suffix)
        assert len(out_files_prefix) == self.n_class
        assert s3_prefix is None or len(s3_clients) > 0
        # create and transpose output
        with TimeTracker(
            f"    Tile {self.data.tile_id} - transpose data for final output"
        ):
            sb.offsetAndScale(self._out_probs_valid, n_threads, 0.5 / scale, scale)
            # expand to original number of pixels
            self._out_probs = np.empty(
                (self.n_groups * self.data.n_pixels, self.n_class), dtype=np.float32
            )
            sb.fillArray(self._out_probs, n_threads, nodata)
            sb.expandArrayRows(
                self._out_probs_valid,
                n_threads,
                self._out_probs,
                self.data.get_pixels_valid_idx(self.n_groups),
            )
            # transpose expanded array
            self._out_probs_t = np.empty(
                (self.n_class, self.n_groups * self.data.n_pixels), dtype=np.float32
            )
            sb.transposeArray(self._out_probs, n_threads, self._out_probs_t)
            # rearrange years and stats
            self._out_probs_gdal = np.empty(
                (self.n_class * self.n_groups, self.data.n_pixels), dtype=np.float32
            )
            sb.fillArray(self._out_probs_gdal, n_threads, nodata)
            inverse_idx = np.empty((self._out_probs_gdal.shape[0], 2), dtype=np.uintc)
            subrows_grid, rows_grid = np.meshgrid(
                np.arange(self.n_groups), list(range(self.n_class))
            )
            inverse_idx[:, 0] = rows_grid.flatten()
            inverse_idx[:, 1] = subrows_grid.flatten()
            sb.inverseReorderArray(
                self._out_probs_t, n_threads, self._out_probs_gdal, inverse_idx
            )
        # write outputs
        with TimeTracker(f"    Tile {self.data.tile_id} - write probs images"):
            out_dir = _make_dir(f"{base_dir}/{self.data.tile_id}")
            out_files = _get_out_files(
                out_files_prefix, out_files_suffix, self.groups, self.n_class
            )
            temp_tif = [self.data.mask_path for _ in range(len(out_files))]
            return _write_output_layers(
                self._out_probs_gdal,
                out_dir,
                out_files,
                temp_tif,
                n_threads,
                gdal_opts,
                nodata,
                dtype,
                self.data.x_size,
                self.data.y_size,
                s3_prefix,
                s3_clients,
                self.data.tile_id,
                temp_file=None,
            )


#
class Reducer:
    """Apply a user-defined reduction function over the prepared covariates of a tile."""

    def __init__(self, reducer_features: List[str], reducer_fn: Callable) -> None:
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

    def reduce(self, data: TiledDataLoader):
        """Prepare the tile covariates and apply ``reducer_fn`` to the valid-pixel matrix."""

        with TimeTracker(
            f"    Tile {data.tile_id} - predict ({len(self.reducer_features)} input features)",
            True,
        ):
            # prepare input and output arrays
            n_groups = len(data.catalog.get_groups())
            n_samples = n_groups * data.n_pixels
            n_samples_valid = n_groups * data.n_pixels_valid
            self.in_covs_t = np.empty(
                (len(self.reducer_features), n_samples), dtype=np.float32
            )
            self.in_covs = np.empty(
                (n_samples, len(self.reducer_features)), dtype=np.float32
            )
            self.in_covs_valid = np.empty(
                (n_samples_valid, len(self.reducer_features)), dtype=np.float32
            )
            # transpose data
            matrix_idx = data.catalog._get_covs_idx(self.reducer_features)
            with TimeTracker(f"    Tile {data.tile_id} - transpose data"):
                sb.reorderArray(data.array, data.n_threads, self.in_covs_t, matrix_idx)
                sb.transposeArray(self.in_covs_t, data.n_threads, self.in_covs)
                sb.selArrayRows(
                    self.in_covs,
                    data.n_threads,
                    self.in_covs_valid,
                    data.get_pixels_valid_idx(n_groups),
                )
            # create output result
            result = ReducedValues(data=data)
            # compute
            with TimeTracker(f"    Tile {data.tile_id} - model prediction"):
                result._out_reduc_valid[:] = self.reducer_fn(self.in_covs_valid)
        return result  # shape: (n_samples, 1)


#
class ReducedValues:
    """Container for the scalar output of a :class:`Reducer` run, ready to be saved."""

    def __init__(self, data: TiledDataLoader) -> None:
        self.data = data
        assert self.data is not None
        self.groups = self.data.catalog.get_groups()
        self.n_groups = len(self.groups)
        self._out_reduc_valid = np.empty(
            (self.n_groups * self.data.n_pixels_valid), dtype=np.float32
        )
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
        """Reduced values reshaped as ``(n_groups, n_pixels_valid)``."""

        return self._out_reduc_valid.reshape((self.n_groups, self.data.n_pixels_valid))

    def save_reduced_layer(
        self,
        base_dir,
        nodata,
        dtype,
        out_files_prefix,
        out_files_suffix,
        s3_prefix,
        s3_clients,
        gdal_opts,
        threads,
    ):
        """Write the reduced values to a GeoTIFF layer (and optionally upload to S3)."""

        assert self._out_reduc_valid is not None
        assert dtype == "int16" or dtype == "uint8"
        assert len(out_files_prefix) == len(out_files_suffix)
        assert len(out_files_prefix) == 1
        assert s3_prefix is None or len(s3_clients) > 0
        # create and transpose output
        with TimeTracker(
            f"    Tile {self.data.tile_id} - transpose data for final output ({threads} threads)"
        ):
            # expand to original number of pixels
            self._out_reduc = np.empty(
                (self.n_groups * self.data.n_pixels, 1), dtype=np.float32
            )
            sb.fillArray(self._out_reduc, threads, nodata)
            sb.expandArrayRows(
                self._out_reduc_valid,
                threads,
                self._out_reduc,
                self.data.get_pixels_valid_idx(self.n_groups),
            )
            # transpose expanded array
            self._out_reduc_t = np.empty(
                (1, self.n_groups * self.data.n_pixels), dtype=np.float32
            )
            sb.transposeArray(self._out_reduc, threads, self._out_reduc_t)
            # rearrange groups
            self._out_reduc_gdal = np.empty(
                (1 * self.n_groups, self.data.n_pixels), dtype=np.float32
            )
            sb.fillArray(self._out_reduc_gdal, threads, nodata)
            inverse_idx = np.empty((self._out_reduc_gdal.shape[0], 2), dtype=np.uintc)
            subrows_grid, rows_grid = np.meshgrid(
                np.arange(self.n_groups), list(range(1))
            )
            inverse_idx[:, 0] = rows_grid.flatten()
            inverse_idx[:, 1] = subrows_grid.flatten()
            sb.inverseReorderArray(
                self._out_reduc_t, threads, self._out_reduc_gdal, inverse_idx
            )
        # write outputs
        with TimeTracker(
            f"    Tile {self.data.tile_id} - write probs images ({threads} threads)"
        ):
            out_dir = _make_dir(f"{base_dir}/{self.data.tile_id}")
            out_files = _get_out_files(out_files_prefix, out_files_suffix, self.groups)
            temp_tif = [self.data.mask_path for _ in range(len(out_files))]
            return _write_output_layers(
                self._out_reduc_gdal,
                out_dir,
                out_files,
                temp_tif,
                threads,
                gdal_opts,
                nodata,
                dtype,
                self.data.x_size,
                self.data.y_size,
                s3_prefix,
                s3_clients,
                self.data.tile_id,
                temp_file=None,
            )


#
def _get_out_files(
    out_files_prefix: List[str],
    out_files_suffix: List[str],
    groups: List[str],
    n_class: Optional[int] = None,
):
    if n_class is None:
        n_class = 1
    out_files = []
    if "common" in groups:
        for _ in groups:
            for i in range(n_class):
                file = f"{out_files_prefix[i]}_{out_files_suffix[i]}"
                out_files.append(file)
    else:
        for group in groups:
            for i in range(n_class):
                file = f"{out_files_prefix[i]}_{group}0101_{group}1231_{out_files_suffix[i]}"
                out_files.append(file)
    return out_files
