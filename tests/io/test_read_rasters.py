"""Tests for ``read_rasters`` (python backend) using real toy NDVI rasters.

Ground truth comes from reading the same file with ``rasterio`` directly, so
exact-value assertions are preserved — just sourced from real rasters (real
CRS, real nodata, real temporal gaps).
"""

import numpy as np
import pytest
import rasterio
from rasterio.windows import Window
from numpy.testing import assert_equal

from skmap.data import toy
from skmap.io.base import read_rasters


@pytest.fixture(scope="module")
def ndvi_filled():
    return [str(p) for p in toy.ndvi_files()]

@pytest.fixture(scope="module")
def ndvi_gappy():
    return [str(p) for p in toy.ndvi_files(gappy=True)]


def _rasterio_ground_truth(path, dtype="float32", window=None):
    """Read a toy raster with rasterio and return the (possibly windowed) 2-D array."""
    with rasterio.open(path) as src:
        if window is not None:
            return src.read(1, window=window).astype(dtype)
        return src.read(1).astype(dtype)


def test_read_rasters_single_file(ndvi_filled):
    """Read a single toy NDVI band and compare to rasterio ground truth."""
    path = ndvi_filled[0]
    res = read_rasters(raster_files=[path], backend="python", verbose=False)
    assert res.shape == (1, 256 * 256)
    expected = _rasterio_ground_truth(path)
    assert_equal(res.get()[0].reshape(256, 256), expected)


def test_read_rasters_multiple_files(ndvi_filled):
    """Read multiple toy NDVI bands and compare each to rasterio ground truth."""
    paths = ndvi_filled[:3]
    res = read_rasters(raster_files=paths, backend="python", verbose=False)
    assert res.shape == (3, 256 * 256)
    for idx, path in enumerate(paths):
        expected = _rasterio_ground_truth(path)
        assert_equal(res.get()[idx].reshape(256, 256), expected)


def test_read_rasters_with_window(ndvi_filled):
    """Read a window of a toy NDVI band and compare to rasterio window read."""
    path = ndvi_filled[0]
    window = Window(0, 0, 64, 64)
    res = read_rasters(raster_files=[path], window=window, backend="python", verbose=False)
    assert res.shape == (1, 64 * 64)
    expected = _rasterio_ground_truth(path, window=window)
    assert_equal(res.get()[0].reshape(64, 64), expected)


def test_read_rasters_with_dtype(ndvi_filled):
    """Read a toy NDVI band with float64 dtype."""
    path = ndvi_filled[0]
    res = read_rasters(raster_files=[path], dtype="float64", backend="python", verbose=False)
    assert res.dtype == np.float64
    expected = _rasterio_ground_truth(path, dtype="float64")
    assert_equal(res.get(), expected.reshape(1, -1))


def test_read_rasters_gappy_has_nan(ndvi_gappy):
    """Gappy toy NDVI contains real NaN gaps after nodata conversion."""
    res = read_rasters(raster_files=ndvi_gappy, backend="python", verbose=False)
    assert res.shape == (24, 256 * 256)
    assert np.isnan(res.get()).any(), "gappy data should contain NaN gaps"


def test_read_rasters_filled_no_nan(ndvi_filled):
    """Filled toy NDVI has no NaN after nodata conversion."""
    res = read_rasters(raster_files=ndvi_filled, backend="python", verbose=False)
    assert res.shape == (24, 256 * 256)
    assert not np.isnan(res.get()).any(), "filled data should have no NaN"


def test_read_rasters_empty_list():
    """Reading an empty list of raster files raises IndexError."""
    with pytest.raises(IndexError):
        read_rasters(raster_files=[], backend="python", verbose=False)