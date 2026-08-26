"""Tests for the Prediction runner (skmap.io.process.Prediction)."""

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from skmap.data import toy
from skmap.io import RasterData
from skmap.io.process import Prediction


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


def _prediction_bands(rdata):
    """Return the appended prediction bands (n_out·n_years, H*W)."""
    idx = rdata.info[rdata.info["group"] == "prediction"].index
    return rdata.array.get()[idx]


# ---------------------------------------------------------------------------
# single static catalogue
# ---------------------------------------------------------------------------


def test_prediction_regressor():
    rdata = _toy_raster_rdata()
    arr = rdata.array.get()
    X = pd.DataFrame(arr.T, columns=["elev", "slope"])
    y = X["elev"] * 2 + X["slope"] * 0.5
    model = RandomForestRegressor(n_estimators=3, random_state=0).fit(X, y)

    rdata.run(Prediction(model=model, feature_names=["elev", "slope"]))

    pred_info = rdata.info[rdata.info["group"] == "prediction"]
    assert len(pred_info) == 1
    assert pred_info["name"].iloc[0] == "prediction"
    assert not pred_info["temporal"].iloc[0]

    pred = _prediction_bands(rdata)
    assert pred.shape == (1, 256 * 256)
    assert np.isfinite(pred).all()


def test_prediction_classifier():
    rdata = _toy_raster_rdata()
    arr = rdata.array.get()
    X = pd.DataFrame(arr.T, columns=["elev", "slope"])
    y = (X["elev"] > 100).astype(int)
    model = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)

    rdata.run(Prediction(model=model, feature_names=["elev", "slope"]))

    pred = _prediction_bands(rdata)
    assert pred.shape == (1, 256 * 256)
    assert set(np.unique(pred[~np.isnan(pred)])).issubset({0.0, 1.0})


def test_prediction_proba():
    rdata = _toy_raster_rdata()
    arr = rdata.array.get()
    X = pd.DataFrame(arr.T, columns=["elev", "slope"])
    y = (X["elev"] > 100).astype(int)
    model = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)

    rdata.run(
        Prediction(model=model, feature_names=["elev", "slope"], predict_proba=True)
    )

    pred_info = rdata.info[rdata.info["group"] == "prediction"]
    assert list(pred_info["name"]) == [
        "prediction_prob_0", "prediction_prob_1", "prediction"
    ]
    pred = _prediction_bands(rdata)
    assert pred.shape == (3, 256 * 256)  # n_class + dominant, n_years=1


def test_prediction_missing_feature_names_raises():
    rdata = _toy_raster_rdata()
    model = _CallCounter(n_out=1)  # no feature_names_in_
    with pytest.raises(ValueError, match="feature_names"):
        rdata.run(Prediction(model=model))


# ---------------------------------------------------------------------------
# multi-year (all years concatenated, one model call)
# ---------------------------------------------------------------------------


def test_prediction_multiyear_one_call():
    """All years predicted in a single model call; statics repeated."""
    rdata = _toy_multiyear_rdata()
    n_pix = rdata.array.shape[1]
    model = _CallCounter(n_out=1)

    rdata.run(Prediction(model=model, feature_names=_TY_FEATURES, valid_only=False))

    assert model.calls == 1
    pred = _prediction_bands(rdata)
    assert pred.shape == (5, n_pix)  # n_out(=1) * n_years

    pred_info = rdata.info[rdata.info["group"] == "prediction"]
    assert list(pred_info["name"]) == ["prediction"] * 5
    assert list(pred_info["start_date"].dt.year) == [2015, 2016, 2017, 2018, 2019]

    X = model.X.reshape(5, n_pix, len(_TY_FEATURES))
    for static in ("elev", "slope"):
        i = _TY_FEATURES.index(static)
        assert all(np.array_equal(X[0, :, i], X[y, :, i]) for y in range(5))


def test_prediction_multiyear_classifier_layout():
    """predict_proba yields proba (n_class*n_years) + dominant (n_years)."""
    rdata = _toy_multiyear_rdata()
    n_pix = rdata.array.shape[1]
    n_class = 3
    model = _CallCounter(n_out=n_class)

    rdata.run(
        Prediction(
            model=model,
            feature_names=_TY_FEATURES,
            predict_proba=True,
            valid_only=False,
        )
    )
    pred = _prediction_bands(rdata)
    assert pred.shape == (n_class * 5 + 5, n_pix)  # proba + dominant
    assert model.calls == 1

    pred_info = rdata.info[rdata.info["group"] == "prediction"]
    assert list(pred_info["name"]) == [
        f"prediction_prob_{i}" for i in range(n_class) for _ in range(5)
    ] + ["prediction"] * 5


def test_prediction_static_land_mask():
    """A 1-D boolean land mask selects the same pixels for every year."""
    rdata = _toy_multiyear_rdata()
    n_pix = rdata.array.shape[1]
    model = _CallCounter(n_out=1)

    mask = ~np.isnan(rdata.array.get()).any(axis=0)
    assert mask.any()
    rdata.run(
        Prediction(model=model, feature_names=_TY_FEATURES, valid_only=mask)
    )
    pred = _prediction_bands(rdata)
    assert pred.shape == (5, n_pix)
    valid_per_year = (~np.isnan(pred)).sum(axis=1)
    assert np.all(valid_per_year == mask.sum())


def test_prediction_temporal_without_dates_raises():
    """Temporal layers without start_date are rejected (dates required)."""
    rdata = toy.ndvi_rdata()
    rdata.info = rdata.info.drop(columns=["start_date", "end_date"], errors="ignore")
    assert rdata.info["temporal"].any()
    assert "start_date" not in rdata.info.columns

    model = _CallCounter(n_out=1)
    with pytest.raises(ValueError, match="start_date"):
        rdata.run(
            Prediction(model=model, feature_names=[rdata.info["name"].iloc[0]])
        )


# ---------------------------------------------------------------------------
# target_names
# ---------------------------------------------------------------------------


def test_prediction_target_names():
    rdata = _toy_multiyear_rdata()
    model = _CallCounter(n_out=3)

    rdata.run(
        Prediction(
            model=model,
            feature_names=_TY_FEATURES,
            predict_proba=True,
            target_names=["water", "pasture", "forest"],
            valid_only=False,
        )
    )
    pred_info = rdata.info[rdata.info["group"] == "prediction"]
    assert list(pred_info["name"]) == [
        f"prediction_{t}" for t in ("water", "pasture", "forest") for _ in range(5)
    ] + ["prediction"] * 5


def test_prediction_target_names_wrong_length_raises():
    rdata = _toy_multiyear_rdata()
    model = _CallCounter(n_out=3)
    with pytest.raises(ValueError, match="target_names"):
        rdata.run(
            Prediction(
                model=model,
                feature_names=_TY_FEATURES,
                predict_proba=True,
                target_names=["water", "pasture"],
                valid_only=False,
            )
        )


# ---------------------------------------------------------------------------
# year column (Q1)
# ---------------------------------------------------------------------------


def test_prediction_year_column():
    rdata = _toy_multiyear_rdata()
    model = _CallCounter(n_out=1)
    rdata.run(Prediction(model=model, feature_names=_TY_FEATURES, valid_only=False))
    pred_info = rdata.info[rdata.info["group"] == "prediction"]
    assert list(pred_info["year"]) == [2015, 2016, 2017, 2018, 2019]


def test_prediction_year_column_static():
    rdata = _toy_raster_rdata()
    model = _CallCounter(n_out=1)
    rdata.run(Prediction(model=model, feature_names=["elev", "slope"]))
    pred_info = rdata.info[rdata.info["group"] == "prediction"]
    assert len(pred_info) == 1
    assert pred_info["year"].iloc[0] is None or pd.isna(pred_info["year"].iloc[0])


# ---------------------------------------------------------------------------
# drop_input (Q2)
# ---------------------------------------------------------------------------


def test_prediction_drop_input():
    rdata = _toy_multiyear_rdata()
    model = _CallCounter(n_out=1)
    rdata.run(
        Prediction(model=model, feature_names=_TY_FEATURES, valid_only=False),
        drop_input=True,
    )
    assert set(rdata.info["group"]) == {"prediction"}
    assert rdata.array.shape[0] == 5  # n_out(=1) * n_years


def test_prediction_drop_input_proba():
    rdata = _toy_multiyear_rdata()
    model = _CallCounter(n_out=3)
    rdata.run(
        Prediction(
            model=model, feature_names=_TY_FEATURES,
            predict_proba=True, valid_only=False,
        ),
        drop_input=True,
    )
    assert set(rdata.info["group"]) == {"prediction"}
    # proba (n_class*5) + dominant (5)
    assert rdata.array.shape[0] == 3 * 5 + 5


# ---------------------------------------------------------------------------
# dominant class argmax (Q3)
# ---------------------------------------------------------------------------


def test_prediction_dominant_argmax():
    """Dominant class = argmax over proba classes; NaN where invalid."""
    rdata = _toy_multiyear_rdata()
    n_pix = rdata.array.shape[1]
    n_class = 3
    model = _CallCounter(n_out=n_class)
    mask = ~np.isnan(rdata.array.get()).any(axis=0)

    rdata.run(
        Prediction(
            model=model, feature_names=_TY_FEATURES,
            predict_proba=True, valid_only=mask,
        )
    )
    pred = _prediction_bands(rdata)
    proba = pred[: n_class * 5].reshape(n_class, 5, n_pix)
    dominant = pred[n_class * 5 :].reshape(5, n_pix)

    valid_yp = ~np.isnan(proba).all(axis=0)  # (5, n_pix)
    assert np.array_equal(
        dominant[valid_yp].astype(int), proba[:, valid_yp].argmax(axis=0)
    )
    if (~valid_yp).any():
        assert np.isnan(dominant[~valid_yp]).all()


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


def test_prediction_to_file(tmp_path):
    """Prediction bands can be written to one GeoTIFF per (output, year)."""
    import rasterio

    from skmap.io import save_rasters
    from skmap.overlay import SpaceTimeOverlay

    rdata = _toy_multiyear_rdata()

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

    rdata.run(Prediction(model=model, feature_names=_TY_FEATURES, valid_only=False))
    pred = _prediction_bands(rdata)  # (n_out=1 * n_years, H*W)

    years = rdata.get_years()
    out_files = [str(tmp_path / f"pred_{y}.tif") for y in years]
    save_rasters(rdata.base_raster, out_files, pred)
    for f in out_files:
        assert os.path.exists(f)
        with rasterio.open(f) as ds:
            assert ds.read(1).shape == (256, 256)


# ---------------------------------------------------------------------------
# plot titles (img_title_text templates)
# ---------------------------------------------------------------------------


def test_get_titles_template():
    """{field} templates format against each prediction info row."""
    rdata = _toy_multiyear_rdata()
    model = _CallCounter(n_out=1)
    rdata.run(Prediction(model=model, feature_names=_TY_FEATURES, valid_only=False))

    pred_info = rdata.info[rdata.info["group"] == "prediction"]
    titles = rdata._get_titles("Land Cover {year} {name}", ["prediction"])
    assert len(titles) == len(pred_info)
    assert titles[0] == "Land Cover 2015 prediction"
    assert titles[-1] == "Land Cover 2019 prediction"


def test_get_titles_date_convenience():
    """{date} expands to 'start - end'."""
    rdata = _toy_multiyear_rdata()
    model = _CallCounter(n_out=1)
    rdata.run(Prediction(model=model, feature_names=_TY_FEATURES, valid_only=False))

    titles = rdata._get_titles("{date}", ["prediction"])
    assert titles[0].startswith("2015-01-01")
    assert "2015-12-31" in titles[0]


def test_get_titles_literal():
    """A plain literal (no braces) is repeated for every frame."""
    rdata = _toy_multiyear_rdata()
    model = _CallCounter(n_out=1)
    rdata.run(Prediction(model=model, feature_names=_TY_FEATURES, valid_only=False))

    pred_info = rdata.info[rdata.info["group"] == "prediction"]
    titles = rdata._get_titles("Land Cover", ["prediction"])
    assert titles == ["Land Cover"] * len(pred_info)
