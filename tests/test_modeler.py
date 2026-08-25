import joblib
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from skmap.modeler import RFClassifier, RFRegressor

# The sklearn-API fixtures below use random data on purpose: they test the
# estimator *wrapper* conformance (is_regressor, feature_names_in_, get_params),
# not ML quality. The end-to-end test at the bottom uses real toy data.


@pytest.fixture
def _toy_covariates():
    """Extract elev+slope at toy land-cover sample points via lazy SpaceOverlay."""
    from skmap.data import toy
    from skmap.io import RasterData
    from skmap.overlay import SpaceOverlay

    toy_dir = toy.DATA_DIR
    elev = "elev.lowestmode_gedi.eml_mf_30m_s_20000101_20181231_nl_epsg.3035_v0.3.tif"
    slope = "slope.percent_gedi.eml_m_30m_s_20000101_20181231_nl_epsg.3035_v0.3.tif"
    rdata = RasterData(
        {
            "common": [
                str(toy_dir / "static" / elev),
                str(toy_dir / "static" / slope),
            ]
        }
    )
    rdata.info["name"] = ["elev", "slope"]
    pts = toy.lc_samples()
    so = SpaceOverlay(points=pts, rasterdata=rdata, verbose=False)
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


def _toy_raster_rdata():
    """A RasterData with named covariate bands (elev, slope) for predict_raster."""
    from pathlib import Path

    from skmap.data import toy
    from skmap.io import RasterData

    toy_dir = toy.DATA_DIR
    elev = "elev.lowestmode_gedi.eml_mf_30m_s_20000101_20181231_nl_epsg.3035_v0.3.tif"
    slope = "slope.percent_gedi.eml_m_30m_s_20000101_20181231_nl_epsg.3035_v0.3.tif"
    rdata = RasterData(
        {"common": [str(toy_dir / "static" / elev), str(toy_dir / "static" / slope)]}
    ).read()
    rdata.info["name"] = ["elev", "slope"]
    return rdata


def test_predict_raster_regressor(tmp_path):
    """predict_raster maps covariate names to bands and predicts every pixel."""
    rdata = _toy_raster_rdata()
    arr = rdata.array.get()  # (2, 65536)
    X = pd.DataFrame(arr.T, columns=["elev", "slope"])
    rng = np.random.default_rng(0)
    y = X["elev"] * 2 + X["slope"] * 0.5

    idx = rng.choice(len(y), 3000, replace=False)
    model = RandomForestRegressor(n_estimators=3, random_state=0).fit(
        X.iloc[idx], y.iloc[idx]
    )
    path = str(tmp_path / "reg_raster.joblib")
    joblib.dump(model, path)

    est = RFRegressor(path)
    assert list(est.feature_names_in_) == ["elev", "slope"]
    pred = est.predict_raster(rdata)
    assert pred.shape == (1, 256 * 256)
    assert np.isfinite(pred).all()


def test_predict_raster_classifier(tmp_path):
    """predict_raster on a classifier returns class labels per pixel."""
    rdata = _toy_raster_rdata()
    arr = rdata.array.get()
    X = pd.DataFrame(arr.T, columns=["elev", "slope"])
    y = (X["elev"] > 100).astype(int)

    model = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
    path = str(tmp_path / "cls_raster.joblib")
    joblib.dump(model, path)

    est = RFClassifier(path)
    pred = est.predict_raster(rdata)
    assert pred.shape == (1, 256 * 256)
    assert set(np.unique(pred[~np.isnan(pred)])).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Multi-year predict_raster (all years concatenated, one model call)
# ---------------------------------------------------------------------------


def _toy_multiyear_rdata():
    """YAML toy: filled NDVI + SWIR1 + statics for years 2015-2019.

    Year-agnostic season names (ndvi_winter ...) repeat once per year,
    distinguished only by start_date — the case predict_raster must handle.
    """
    from skmap.data import toy
    from skmap.io import RasterData

    return (
        RasterData.from_yaml(str(toy.LAYERS_YAML), base_path=str(toy.DATA_DIR))
        .filter("variant != 'gappy'")
        .filter_date(
            "2015-01-01",
            "2019-12-31",
            include_non_temporal=True,
            by_start_date=True,
        )
        .read()
    )


_TY_FEATURES = [
    "ndvi_winter", "ndvi_spring", "ndvi_summer", "ndvi_fall",
    "swir1_winter", "swir1_spring", "swir1_summer", "swir1_fall",
    "elev", "slope",
]


class _CallCounter:
    """Mock model: counts predict calls and remembers the last X."""

    def __init__(self, n_out=1):
        self.calls = 0
        self.X = None
        self.n_out = n_out

    def predict(self, X):
        self.calls += 1
        self.X = X
        rng = np.random.default_rng(0)
        if self.n_out == 1:
            return np.zeros(len(X), dtype=np.float32)
        return rng.random((len(X), self.n_out), dtype=np.float32)


def _mock_modeler(n_out=1):
    from skmap.modeler import Modeler

    est = Modeler.__new__(Modeler)
    est.model = _CallCounter(n_out=n_out)
    est.predict_fn = lambda m, X: m.predict(X)
    est.model_covs = _TY_FEATURES
    return est


def test_get_years_and_covs_idx_by_year():
    rdata = _toy_multiyear_rdata()
    assert rdata.get_years() == [2015, 2016, 2017, 2018, 2019]

    covs_idx, years = rdata._get_covs_idx_by_year(_TY_FEATURES)
    assert years == [2015, 2016, 2017, 2018, 2019]
    assert covs_idx.shape == (len(_TY_FEATURES), 5)

    # static covariates repeat the same band for every year
    elev_i = _TY_FEATURES.index("elev")
    assert len(set(covs_idx[elev_i].tolist())) == 1
    # temporal covariates get a distinct band per year
    ndvi_w = _TY_FEATURES.index("ndvi_winter")
    assert len(set(covs_idx[ndvi_w].tolist())) == 5


def test_predict_raster_multiyear_one_call():
    """All years predicted in a single model call; statics repeated."""
    rdata = _toy_multiyear_rdata()
    n_pix = rdata.array.shape[1]
    est = _mock_modeler(n_out=1)

    pred = est.predict_raster(rdata, valid_only=False)

    assert est.model.calls == 1
    assert pred.shape == (5, n_pix)  # n_out(=1) * n_years

    # the model saw all years concatenated: (n_years * n_pixels, n_covs)
    X = est.model.X.reshape(5, n_pix, len(_TY_FEATURES))
    # statics (elev, slope) identical across years
    for static in ("elev", "slope"):
        i = _TY_FEATURES.index(static)
        assert all(np.array_equal(X[0, :, i], X[y, :, i]) for y in range(5))


def test_predict_raster_multiyear_classifier_layout():
    """n_out>1 yields (n_out*n_years, n_pixels), out-band major, year minor."""
    rdata = _toy_multiyear_rdata()
    n_pix = rdata.array.shape[1]
    n_class = 3
    est = _mock_modeler(n_out=n_class)

    pred = est.predict_raster(rdata, valid_only=False)
    assert pred.shape == (n_class * 5, n_pix)
    # band k -> output k // n_years, year k % n_years
    assert est.model.calls == 1


def test_predict_raster_static_land_mask():
    """A 1-D boolean land mask selects the same pixels for every year."""
    rdata = _toy_multiyear_rdata()
    n_pix = rdata.array.shape[1]
    est = _mock_modeler(n_out=1)

    mask = ~np.isnan(rdata.array.get()).any(axis=0)
    assert mask.any()
    pred = est.predict_raster(rdata, valid_only=mask)
    assert pred.shape == (5, n_pix)
    # the mask is tiled across years: the same pixels are valid in every year
    valid_per_year = (~np.isnan(pred)).sum(axis=1)
    assert np.all(valid_per_year == mask.sum())


def test_predict_raster_temporal_without_dates_raises():
    """Temporal layers without start_date are rejected (dates required)."""
    from skmap.data import toy

    # dict-built toy carries start_date; drop it to simulate the undated case.
    rdata = toy.ndvi_rdata()
    rdata.info = rdata.info.drop(columns=["start_date", "end_date"], errors="ignore")
    assert rdata.info["temporal"].any()  # temporal layers present
    assert "start_date" not in rdata.info.columns

    est = _mock_modeler(n_out=1)
    est.model_covs = [rdata.info["name"].iloc[0]]  # any covariate; date check fires first
    with pytest.raises(ValueError, match="start_date"):
        est.predict_raster(rdata, valid_only=False)


def test_predict_raster_to_file(tmp_path):
    """predict_raster_to_file writes one GeoTIFF per (output, year)."""
    import rasterio

    from skmap.data import toy
    from skmap.io import RasterData
    from skmap.modeler import RFRegressor

    rdata = _toy_multiyear_rdata()
    n_pix = rdata.array.shape[1]

    # a real regressor trained on the overlaid samples (year-agnostic features)
    from skmap.overlay import SpaceTimeOverlay

    samples = toy.lc_samples()
    overlay = SpaceTimeOverlay(
        points=samples,
        col_date="date",
        rasterdata=rdata,
        date_ranges=[(f"{y}-01-01", f"{y}-12-31") for y in range(2015, 2020)],
        raster_tiles=None,
        verbose=False,
    )
    train = overlay.run(max_ram_mb=512, out_file_name=None)
    X = train[_TY_FEATURES]
    y = (train["target"]).astype(float)
    model = RandomForestRegressor(n_estimators=3, random_state=0).fit(X, y)
    path = str(tmp_path / "myr.joblib")
    joblib.dump(model, path)
    est = RFRegressor(path)

    years = rdata.get_years()
    out_files = [str(tmp_path / f"pred_{y}.tif") for y in years]  # n_out=1
    ret = est.predict_raster_to_file(rdata, out_files, valid_only=False)
    assert ret == out_files
    for f in out_files:
        assert os.path.exists(f)
        with rasterio.open(f) as ds:
            assert ds.read(1).shape == (256, 256)
    assert est.predict_raster(rdata, valid_only=False).shape == (5, n_pix)
