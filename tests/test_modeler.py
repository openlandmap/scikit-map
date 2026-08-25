import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from skmap.modeler import RFClassifier, RFRegressor, _write_output_layers

# The sklearn-API fixtures below use random data on purpose: they test the
# estimator *wrapper* conformance (is_regressor, feature_names_in_, get_params),
# not ML quality. The end-to-end test at the bottom uses real toy data.


@pytest.fixture
def _toy_covariates():
    """Extract elev+slope at toy land-cover sample points via SpaceOverlay."""
    from pathlib import Path

    from skmap.catalog import DataCatalog
    from skmap.data import toy
    from skmap.overlay import SpaceOverlay

    toy_dir = Path(__file__).parent.parent / "skmap" / "data" / "toy"
    elev = "elev.lowestmode_gedi.eml_mf_30m_s_20000101_20181231_nl_epsg.3035_v0.3"
    slope = "slope.percent_gedi.eml_m_30m_s_20000101_20181231_nl_epsg.3035_v0.3"
    catalog = DataCatalog.create_catalog(
        catalog_def=pd.DataFrame({
            "layer_name": ["elev", "slope"],
            "path": ["{base_path}/" + elev + ".tif", "{base_path}/" + slope + ".tif"],
            "type": ["common", "common"],
        }),
        years=[2020],
        base_path=str(toy_dir / "static"),
    )
    pts = toy.lc_samples()
    so = SpaceOverlay(points=pts, catalog=catalog)
    res = so.run(max_ram_mb=512, out_file_name=None)
    return res


@pytest.fixture
def regressor_model(tmp_path):
    X = pd.DataFrame(np.random.RandomState(0).rand(30, 3), columns=["a", "b", "c"])
    y = np.random.RandomState(1).rand(30)
    model = RandomForestRegressor(n_estimators=3, random_state=0).fit(X, y)
    path = str(tmp_path / "reg.joblib")
    joblib.dump(model, path)
    return path, X


@pytest.fixture
def classifier_model(tmp_path):
    X = pd.DataFrame(np.random.RandomState(0).rand(30, 3), columns=["a", "b", "c"])
    y = (np.random.RandomState(1).rand(30) > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
    path = str(tmp_path / "cls.joblib")
    joblib.dump(model, path)
    return path, X


def test_rf_regressor_sklearn_api(regressor_model):
    path, X = regressor_model
    est = RFRegressor(path)
    assert is_regressor(est)
    assert list(est.feature_names_in_) == ["a", "b", "c"]
    assert set(est.get_params().keys()) == {
        "model_path",
        "model_covs_path",
        "n_responses",
        "predict_fn",
    }
    pred = est.predict(X.values[:5])
    assert pred.shape == (5,)


def test_rf_classifier_sklearn_api(classifier_model):
    path, X = classifier_model
    est = RFClassifier(path)
    assert is_classifier(est)
    assert list(est.feature_names_in_) == ["a", "b", "c"]
    pred = est.predict(X.values[:5])
    assert pred.shape == (5,)  # class labels


def test_write_output_layers(tmp_path):
    import skmap_bindings as sb

    base = os.path.join(
        os.path.dirname(__file__),
        "..",
        "skmap",
        "data",
        "toy",
        "swir1",
        "swir1_landsat.ard1_p50_30m_s_20181202_20190320_nl_epsg.3035_v20230720.tif",
    )
    base = os.path.abspath(base)
    out_dir = str(tmp_path / "tile1")
    os.makedirs(out_dir)
    data = np.full((2, 256 * 256), 100.0, dtype=np.float32)
    res = _write_output_layers(
        data,
        out_dir,
        ["a", "b"],
        [base, base],
        1,
        {},
        -9999,
        "int16",
        256,
        256,
        None,
        None,
        "tile1",
    )
    assert res == [f"{out_dir}/a", f"{out_dir}/b"]
    assert os.path.exists(f"{out_dir}/a.tif")
    assert os.path.exists(f"{out_dir}/b.tif")


def test_rf_regressor_end_to_end_toy(_toy_covariates, tmp_path):
    """End-to-end: fit RF on real toy covariates, dump, reload via RFRegressor, predict."""
    df = _toy_covariates
    X = df[["elev", "slope"]]
    y = df["target"].astype(float)

    model = RandomForestRegressor(n_estimators=5, random_state=0).fit(X, y)
    path = str(tmp_path / "reg_toy.joblib")
    joblib.dump(model, path)

    est = RFRegressor(path)
    assert is_regressor(est)
    assert list(est.feature_names_in_) == ["elev", "slope"]
    pred = est.predict(X.values[:10])
    assert pred.shape == (10,)
    assert np.isfinite(pred).all()


def test_rf_classifier_end_to_end_toy(_toy_covariates, tmp_path):
    """End-to-end: fit RF classifier on real toy covariates, dump, reload, predict."""
    df = _toy_covariates
    X = df[["elev", "slope"]]
    y = df["target"].astype(int)

    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    path = str(tmp_path / "cls_toy.joblib")
    joblib.dump(model, path)

    est = RFClassifier(path)
    assert is_classifier(est)
    pred = est.predict(X.values[:10])
    assert pred.shape == (10,)
    # predicted labels must be a subset of the training labels
    assert set(np.unique(pred)).issubset(set(np.unique(y)))
