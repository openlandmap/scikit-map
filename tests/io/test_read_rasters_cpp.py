"""Tests for ``read_rasters`` (cpp backend) using real toy NDVI rasters.

The cpp path reads uint8 disk data as float32; ground truth comes from
``rasterio``. Non-float32 requests fall back to the python backend.
"""

import numpy as np
import pytest
import rasterio
import warnings
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
    with rasterio.open(path) as src:
        if window is not None:
            return src.read(1, window=window).astype(dtype)
        return src.read(1).astype(dtype)


def test_read_rasters_cpp_single_file(ndvi_filled):
    """Read a single toy NDVI band via the C++ backend."""
    path = ndvi_filled[0]
    res = read_rasters(raster_files=[path], backend="cpp", verbose=False)
    assert res.shape == (1, 256 * 256)
    assert res.dtype == np.float32
    expected = _rasterio_ground_truth(path)
    assert_equal(res.get()[0].reshape(256, 256), expected)


def test_read_rasters_cpp_multiple_files(ndvi_filled):
    """Read multiple toy NDVI bands via the C++ backend."""
    paths = ndvi_filled[:3]
    res = read_rasters(raster_files=paths, backend="cpp", verbose=False)
    assert res.shape == (3, 256 * 256)
    assert res.dtype == np.float32
    for idx, path in enumerate(paths):
        expected = _rasterio_ground_truth(path)
        assert_equal(res.get()[idx].reshape(256, 256), expected)


def test_read_rasters_cpp_with_window(ndvi_filled):
    """Read a window of a toy NDVI band via the C++ backend."""
    path = ndvi_filled[0]
    window = Window(0, 0, 64, 64)
    res = read_rasters(raster_files=[path], window=window, backend="cpp", verbose=False)
    assert res.shape == (1, 64 * 64)
    expected = _rasterio_ground_truth(path, window=window)
    assert_equal(res.get()[0].reshape(64, 64), expected)


def test_read_rasters_cpp_with_dtype_falls_back(ndvi_filled):
    """Non-float32 dtype falls back to the python backend (no error)."""
    path = ndvi_filled[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = read_rasters(raster_files=[path], dtype=np.float64, backend="cpp", verbose=False)
    assert res.dtype == np.float64
    expected = _rasterio_ground_truth(path, dtype="float64")
    assert_equal(res.get(), expected.reshape(1, -1))


def test_read_rasters_cpp_gappy_has_nan(ndvi_gappy):
    """Gappy toy NDVI contains real NaN gaps via the C++ backend."""
    res = read_rasters(raster_files=ndvi_gappy, backend="cpp", verbose=False)
    assert res.shape == (24, 256 * 256)
    assert np.isnan(res.get()).any(), "gappy data should contain NaN gaps"


def test_read_rasters_cpp_empty_list():
    """Reading an empty list of raster files raises IndexError."""
    with pytest.raises(IndexError):
        read_rasters(raster_files=[], backend="cpp", verbose=False)