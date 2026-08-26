"""Tests for RasterData.predict / predict_raster / predict_raster_to_file."""

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from skmap.data import toy
from skmap.io import RasterData


def _toy_raster_rdata():
    """A RasterData with named covariate bands (elev, slope)."""
    toy_dir = toy.DATA_DIR
    elev = "elev.lowestmode_gedi.eml_mf_30m_s_20000101_20181231_nl_epsg.3035_v0.3.tif"
    slope = "slope.percent_gedi.eml_m_30m_s_20000101_20181231_nl_epsg.3035_v0.3.tif"
    rdata = RasterData(
        {"common": [str(toy_dir / "static" / elev), str(toy_dir / "static" / slope)]}
    ).read()
    rdata.info["name"] = ["elev", "slope"]
    return rdata


def _toy_multiyear_rdata():
    """YAML toy: filled NDVI + SWIR1 + statics for years 2015-2019."""
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

    def predict_proba(self, X):
        self.calls += 1
        self.X = X
        rng = np.random.default_rng(0)
        return rng.random((len(X), self.n_out), dtype=np.float32)


# ---------------------------------------------------------------------------
# predict (2-D)
# ---------------------------------------------------------------------------


def test_predict_regressor():
    rdata = _toy_raster_rdata()
    arr = rdata.array.get()
    X = pd.DataFrame(arr.T, columns=["elev", "slope"])
    y = X["elev"] * 2 + X["slope"] * 0.5
    model = RandomForestRegressor(n_estimators=3, random_state=0).fit(X, y)
    pred = rdata.predict(model, X.values[:5])
    assert pred.shape == (5,)


def test_predict_classifier_proba():
    rdata = _toy_raster_rdata()
    arr = rdata.array.get()
    X = pd.DataFrame(arr.T, columns=["elev", "slope"])
    y = (X["elev"] > 100).astype(int)
    model = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
    proba = rdata.predict(model, X.values[:5], predict_proba=True)
    assert proba.shape == (5, 2)


# ---------------------------------------------------------------------------
# predict_raster (single static catalogue)
# ---------------------------------------------------------------------------


def test_predict_raster_regressor():
    rdata = _toy_raster_rdata()
    arr = rdata.array.get()
    X = pd.DataFrame(arr.T, columns=["elev", "slope"])
    y = X["elev"] * 2 + X["slope"] * 0.5
    model = RandomForestRegressor(n_estimators=3, random_state=0).fit(X, y)
    pred = rdata.predict_raster(model, feature_names=["elev", "slope"])
    assert pred.shape == (1, 256 * 256)
    assert np.isfinite(pred).all()


def test_predict_raster_classifier():
    rdata = _toy_raster_rdata()
    arr = rdata.array.get()
    X = pd.DataFrame(arr.T, columns=["elev", "slope"])
    y = (X["elev"] > 100).astype(int)
    model = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
    pred = rdata.predict_raster(model, feature_names=["elev", "slope"])
    assert pred.shape == (1, 256 * 256)
    assert set(np.unique(pred[~np.isnan(pred)])).issubset({0.0, 1.0})


def test_predict_raster_proba():
    rdata = _toy_raster_rdata()
    arr = rdata.array.get()
    X = pd.DataFrame(arr.T, columns=["elev", "slope"])
    y = (X["elev"] > 100).astype(int)
    model = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
    proba = rdata.predict_raster(
        model, feature_names=["elev", "slope"], predict_proba=True
    )
    assert proba.shape == (2, 256 * 256)  # n_class * n_years(=1)


def test_predict_raster_missing_feature_names_raises():
    rdata = _toy_raster_rdata()
    model = _CallCounter(n_out=1)  # no feature_names_in_
    with pytest.raises(ValueError, match="feature_names"):
        rdata.predict_raster(model)


# ---------------------------------------------------------------------------
# multi-year predict_raster (all years concatenated, one model call)
# ---------------------------------------------------------------------------


def test_get_years_and_covs_idx_by_year():
    rdata = _toy_multiyear_rdata()
    assert rdata.get_years() == [2015, 2016, 2017, 2018, 2019]

    covs_idx, years = rdata._get_covs_idx_by_year(_TY_FEATURES)
    assert years == [2015, 2016, 2017, 2018, 2019]
    assert covs_idx.shape == (len(_TY_FEATURES), 5)

    elev_i = _TY_FEATURES.index("elev")
    assert len(set(covs_idx[elev_i].tolist())) == 1  # static repeats
    ndvi_w = _TY_FEATURES.index("ndvi_winter")
    assert len(set(covs_idx[ndvi_w].tolist())) == 5  # temporal per year


def test_predict_raster_multiyear_one_call():
    """All years predicted in a single model call; statics repeated."""
    rdata = _toy_multiyear_rdata()
    n_pix = rdata.array.shape[1]
    model = _CallCounter(n_out=1)

    pred = rdata.predict_raster(model, feature_names=_TY_FEATURES, valid_only=False)

    assert model.calls == 1
    assert pred.shape == (5, n_pix)  # n_out(=1) * n_years

    X = model.X.reshape(5, n_pix, len(_TY_FEATURES))
    for static in ("elev", "slope"):
        i = _TY_FEATURES.index(static)
        assert all(np.array_equal(X[0, :, i], X[y, :, i]) for y in range(5))


def test_predict_raster_multiyear_classifier_layout():
    """predict_proba yields (n_class*n_years, n_pixels), out-band major."""
    rdata = _toy_multiyear_rdata()
    n_pix = rdata.array.shape[1]
    n_class = 3
    model = _CallCounter(n_out=n_class)

    pred = rdata.predict_raster(
        model, feature_names=_TY_FEATURES, predict_proba=True, valid_only=False
    )
    assert pred.shape == (n_class * 5, n_pix)
    assert model.calls == 1


def test_predict_raster_static_land_mask():
    """A 1-D boolean land mask selects the same pixels for every year."""
    rdata = _toy_multiyear_rdata()
    n_pix = rdata.array.shape[1]
    model = _CallCounter(n_out=1)

    mask = ~np.isnan(rdata.array.get()).any(axis=0)
    assert mask.any()
    pred = rdata.predict_raster(model, feature_names=_TY_FEATURES, valid_only=mask)
    assert pred.shape == (5, n_pix)
    valid_per_year = (~np.isnan(pred)).sum(axis=1)
    assert np.all(valid_per_year == mask.sum())


def test_predict_raster_temporal_without_dates_raises():
    """Temporal layers without start_date are rejected (dates required)."""
    rdata = toy.ndvi_rdata()
    rdata.info = rdata.info.drop(columns=["start_date", "end_date"], errors="ignore")
    assert rdata.info["temporal"].any()
    assert "start_date" not in rdata.info.columns

    model = _CallCounter(n_out=1)
    with pytest.raises(ValueError, match="start_date"):
        rdata.predict_raster(model, feature_names=[rdata.info["name"].iloc[0]])


def test_predict_raster_to_file(tmp_path):
    """predict_raster_to_file writes one GeoTIFF per (output, year)."""
    import rasterio

    from skmap.overlay import SpaceTimeOverlay

    rdata = _toy_multiyear_rdata()
    n_pix = rdata.array.shape[1]

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
    y = train["target"].astype(float)
    model = RandomForestRegressor(n_estimators=3, random_state=0).fit(X, y)

    years = rdata.get_years()
    out_files = [str(tmp_path / f"pred_{y}.tif") for y in years]  # n_out=1
    ret = rdata.predict_raster_to_file(
        model, out_files, feature_names=_TY_FEATURES, valid_only=False
    )
    assert ret == out_files
    for f in out_files:
        assert os.path.exists(f)
        with rasterio.open(f) as ds:
            assert ds.read(1).shape == (256, 256)
    assert rdata.predict_raster(model, feature_names=_TY_FEATURES, valid_only=False).shape == (5, n_pix)
