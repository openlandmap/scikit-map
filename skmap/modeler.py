import os
from typing import Callable

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestRegressor

import skmap.set_env  # noqa: F401


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
    """Base class wrapping a fitted model, its covariate names, and feature preparation."""

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

    def predict(self, X) -> np.ndarray:
        """sklearn-style prediction on a 2D array of covariates.

        Spatial prediction (which reads rasters and maps covariate names to
        bands) is available as :meth:`predict_raster`.
        """
        return self.predict_fn(self.model, X)

    def predict_raster(self, rdata, valid_only=True) -> np.ndarray:
        """Predict on a :class:`~skmap.io.RasterData` for **all years at once**.

        All available years are concatenated into a single feature matrix
        (``(n_years·n_pixels, n_covs)``) with static covariates repeated for
        every year, and the model is invoked **once**.  The result is reshaped
        to ``(n_out·n_years, H*W)`` — out-band major, year minor (band ``k`` →
        output ``k // n_years``, year ``k % n_years``) — so it can be written
        directly as one raster per (output, year).

        Years come from the temporal layers' ``start_date``; a static-only
        catalogue yields ``n_years == 1`` (a single prediction).

        :param rdata: a RasterData whose bands are named after the model's
          covariate features.  Temporal covariates are matched by name **and**
          year (year-agnostic names such as ``ndvi_winter`` repeat per year);
          ``common`` (static) covariates are reused for every year.
        :param valid_only: select which pixels to predict.

          * ``True`` (default) — per-(year, pixel) NaN validity: a pixel is
            predicted for a year only if none of that year's covariates is NaN.
          * ``False`` — predict every pixel for every year.
          * a 1-D boolean ``np.ndarray`` of shape ``(H*W,)`` — a static land
            mask, applied to **all** years (the same pixels are predicted in
            every year; the rest are ``NaN``).

        :return: ``(n_out·n_years, H*W)`` float32 array, ``NaN`` where not
          predicted.  ``n_out`` is 1 for label/single-response models, or the
          number of classes/responses for probability / multi-output models.
        """
        covs_idx, years = rdata._get_covs_idx_by_year(self.model_covs)
        arr = rdata.array.get()
        n_pixels = arr.shape[1]
        n_years = len(years)
        n_covs = len(self.model_covs)

        # Build the concatenated feature matrix (n_years·n_pixels, n_covs),
        # rows ordered year-major: year0_pixel0.., year1_pixel0..  Static
        # covariates repeat because their column in covs_idx is the same band
        # for every year.
        X = np.concatenate(
            [arr[covs_idx[:, j], :].T for j in range(n_years)], axis=0
        ).astype(np.float32, copy=False)

        if isinstance(valid_only, np.ndarray):
            # static land mask: same pixels across all years.
            valid = np.tile(np.asarray(valid_only, dtype=bool), n_years)
        elif valid_only:
            valid = ~np.isnan(X).any(axis=1)
        else:
            valid = np.ones(X.shape[0], dtype=bool)

        pred_full = np.full((n_years * n_pixels, 1), np.nan, dtype=np.float32)
        if valid.any():
            pred_valid = np.asarray(
                self.predict_fn(self.model, X[valid]), dtype=np.float32
            )
            if pred_valid.ndim == 1:
                pred_valid = pred_valid.reshape(-1, 1)
            n_out = pred_valid.shape[1]
            pred_full = np.full(
                (n_years * n_pixels, n_out), np.nan, dtype=np.float32
            )
            pred_full[valid] = pred_valid
        else:
            n_out = pred_full.shape[1]

        # (n_years, n_pixels, n_out) -> (n_out, n_years, n_pixels) ->
        # (n_out·n_years, n_pixels), out-band major, year minor.
        pred = (
            pred_full.reshape(n_years, n_pixels, n_out)
            .transpose(2, 0, 1)
            .reshape(n_out * n_years, n_pixels)
        )
        return pred

    def predict_raster_to_file(
        self,
        rdata,
        out_files: list,
        valid_only=True,
        base_raster: str = None,
        **save_kwargs,
    ) -> list:
        """Run :meth:`predict_raster` and write one raster per (output, year).

        ``out_files`` must list ``n_out·n_years`` paths ordered out-band
        major, year minor — i.e. matching the layout returned by
        :meth:`predict_raster`::

            years = rdata.get_years() or [None]
            out_files = [
                f"{out_dir}/{out_name}_{year}.tif"
                for out_name in out_names
                for year in years
            ]

        :param rdata: passed to :meth:`predict_raster`.
        :param out_files: list of output GeoTIFF paths.
        :param valid_only: passed to :meth:`predict_raster`.
        :param base_raster: reference raster for the geo-transform/CRS;
          defaults to ``rdata._base_raster()``.
        :param save_kwargs: forwarded to :func:`skmap.io.save_rasters`.
        :return: ``out_files``.
        """
        from skmap.io.base import save_rasters

        pred = self.predict_raster(rdata, valid_only=valid_only)
        if len(out_files) != pred.shape[0]:
            raise ValueError(
                f"expected {pred.shape[0]} output files (n_out·n_years), "
                f"got {len(out_files)}"
            )
        if base_raster is None:
            base_raster = rdata._base_raster()
        save_rasters(base_raster, out_files, pred, **save_kwargs)
        return out_files


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
    """Tree-based regressor (joblib or tl2cgen) predicting a single response."""

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


#
class RFRegressorTrees(Regressor):
    """Regressor wrapping a scikit-learn ``RandomForestRegressor``."""

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
