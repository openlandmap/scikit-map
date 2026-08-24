"""Tests for the deduplicated _base_raster and read side-effects."""

import warnings

import pytest

from skmap.data import toy
from skmap.io import RasterData


@pytest.fixture
def rdata():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return toy.ndvi_rdata(gappy=False, verbose=False)


def test_base_raster_returns_path(rdata):
    """_base_raster returns a reachable raster path string, not a bool."""
    path = rdata._base_raster()
    assert isinstance(path, str)
    assert path.endswith(".tif")


def test_has_base_raster_returns_bool(rdata):
    assert rdata._has_base_raster() is True


def test_has_base_raster_false_for_missing():
    r = RasterData({"g": ["nonexistent.tif"]})
    assert r._has_base_raster() is False


def test_base_raster_raises_when_none():
    r = RasterData({"g": ["nonexistent.tif"]})
    with pytest.raises(Exception, match="No base raster"):
        r._base_raster()


def test_read_sets_side_effects():
    """read() must set window, bounds, base_raster and array."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = RasterData(
            {"ndvi": toy._temporal_raster("ndvi", "filled")}, verbose=False
        ).timespan("20141202", "20201201", "days", toy.TOY_DATE_STEP, ignore_29feb=True)
    r.read()
    assert hasattr(r, "window")
    assert hasattr(r, "bounds")
    assert isinstance(r.base_raster, str)
    assert r.array.ndim == 2
    assert r.array.shape[0] == 24


def test_only_one_base_raster_definition():
    """Ensure _base_raster is defined exactly once (no shadowing)."""
    import skmap.io.base as base

    # count definitions in the class
    defs = [
        n
        for n, v in vars(base.RasterData).items()
        if n == "_base_raster" and callable(v)
    ]
    assert len(defs) == 1