"""Fluent-interface tests for RasterData (toy data)."""

import warnings

import numpy as np
import pytest

from skmap.data import toy
from skmap.io import RasterData
from skmap.io.process import SeasConvFill


@pytest.fixture
def rdata():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return toy.ndvi_rdata(gappy=True, verbose=False)


# ---------------------------------------------------------------------------
# Chaining / return-self semantics
# ---------------------------------------------------------------------------


def test_chaining_read_filter_run(rdata):
    """Mutating methods return self so they chain."""
    result = rdata.run(SeasConvFill(season_size=4, verbose=False))
    assert result is rdata


def test_drop_returns_self(rdata):
    rdata.run(SeasConvFill(season_size=4, verbose=False))
    result = rdata.drop("ndvi")
    assert result is rdata


def test_rename_returns_self(rdata):
    result = rdata.rename({"ndvi": "ndvi_renamed"})
    assert result is rdata
    assert "ndvi_renamed" in rdata.info["group"].values


def test_timespan_read_chain():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = (
            RasterData({"ndvi": toy._temporal_raster("ndvi", "filled")}, verbose=False)
            .timespan("20141202", "20201201", "days", toy.TOY_DATE_STEP, ignore_29feb=True)
            .read()
        )
    assert r.array.shape[2] == 24


# ---------------------------------------------------------------------------
# Non-mutating select (the fluent filter)
# ---------------------------------------------------------------------------


def test_select_returns_new_object(rdata):
    sub = rdata.filter("index < 3")
    assert sub is not rdata


def test_select_does_not_share_date_args(rdata):
    """Regression: filtered copies must not share the date_args dict."""
    sub = rdata.filter("index < 3")
    sub.date_args["ndvi"]["date_format"] = "%Y"
    # original untouched
    assert rdata.date_args["ndvi"]["date_format"] != "%Y"


def test_select_does_not_share_raster_files(rdata):
    sub = rdata.filter("index < 3")
    sub.raster_files["extra"] = ["x.tif"]
    assert "extra" not in rdata.raster_files


def test_select_preserves_backend(rdata):
    from skmap.compute import NumbaBackend

    rdata.backend = NumbaBackend()
    sub = rdata.filter("index < 3")
    assert isinstance(sub.backend, NumbaBackend)


def test_select_original_array_unchanged(rdata):
    orig = rdata.array.copy()
    sub = rdata.filter("index < 3")
    np.testing.assert_array_equal(rdata.array, orig)


def test_filter_date_returns_new_object(rdata):
    sub = rdata.filter_date("2015-01-01", "2019-01-01")
    assert sub is not rdata
    assert sub.array.shape[2] < rdata.array.shape[2]


def test_filter_contains_returns_new_object(rdata):
    sub = rdata.filter_contains("ndvi")
    assert sub is not rdata


# ---------------------------------------------------------------------------
# Terminal methods return artefacts
# ---------------------------------------------------------------------------


def test_plot_returns_figure(rdata):
    from matplotlib.figure import Figure

    fig = rdata.plot()
    assert isinstance(fig, Figure)


def test_point_query_returns_data(rdata):
    import geopandas as gpd

    samples = toy.lc_samples()
    data = rdata.point_query(
        x=samples.geometry.x.to_list()[:3],
        y=samples.geometry.y.to_list()[:3],
        return_data=True,
    )
    assert data.shape[0] == 3