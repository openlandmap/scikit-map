from numpy.testing import assert_equal
import numpy as np
import pytest
from rasterio.windows import Window
from skmap.io.base import read_rasters
from conftest import temp_raster_file, temp_multi_raster_files

def test_read_rasters_single_file(temp_raster_file):
    """Test reading a single raster file."""
    data, path = temp_raster_file
    res = read_rasters(raster_files=[path], backend="python", verbose=True)
    assert res.shape == (1, 100 * 100)  # (N, H*W)
    assert_equal(res[0].reshape(100, 100), data)

def test_read_rasters_multiple_files(temp_multi_raster_files):
    """Test reading multiple raster files."""
    datas, paths = temp_multi_raster_files
    res = read_rasters(raster_files=paths, backend="python", verbose=True, scale=1)
    assert res.shape == (3, 100 * 100)  # (N, H*W)
    for idx, data in enumerate(datas):
        assert_equal(res[idx].reshape(100, 100), datas[idx])

def test_read_rasters_with_window(temp_raster_file):
    """Test reading a raster with a window."""
    data, path = temp_raster_file
    window = Window(0, 0, 50, 50)  # Read top-left 50x50 pixels
    res = read_rasters(raster_files=[path], window=window, backend="python", verbose=True)
    assert res.shape == (1, 50 * 50)  # (N, H*W)
    assert_equal(res[0].reshape(50, 50), data[:50, :50])

def test_read_rasters_with_dtype(temp_raster_file):
    """Test reading a raster with a specific dtype."""
    data, path = temp_raster_file
    res = read_rasters(raster_files=[path], dtype="float64", backend="python", verbose=True)
    assert res.dtype == np.float64
    assert_equal(res, data.astype(np.float64).reshape(1, -1))

def test_read_rasters_empty_list():
    """Test reading an empty list of raster files."""
    with pytest.raises(IndexError):
        read_rasters(raster_files=[], verbose=True)
