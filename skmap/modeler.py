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

    def predict_raster(self, rdata, valid_only: bool = True) -> np.ndarray:
        """Predict on a :class:`~skmap.io.RasterData` (spatial).

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
