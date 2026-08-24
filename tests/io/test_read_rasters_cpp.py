from numpy.testing import assert_equal
import numpy as np
import pytest
from rasterio.windows import Window
from skmap.io.base import read_rasters

def test_read_rasters_cpp_single_file(temp_raster_file):
    """Test reading a single raster file via the C++ backend."""
    data, path = temp_raster_file
    res = read_rasters(raster_files=[path], backend="cpp", verbose=True)
    assert res.shape == (1, 100 * 100)  # (N, H*W)
    assert res.dtype == np.float32
    assert_equal(res.get()[0].reshape(100, 100), data)

def test_read_rasters_cpp_multiple_files(temp_multi_raster_files):
    """Test reading multiple raster files via the C++ backend."""
    datas, paths = temp_multi_raster_files
    res = read_rasters(raster_files=paths, backend="cpp", verbose=True)
    assert res.shape == (3, 100 * 100)  # (N, H*W)
    assert res.dtype == np.float32
    for idx, data in enumerate(datas):
        assert_equal(res.get()[idx].reshape(100, 100), data)

def test_read_rasters_cpp_with_window(temp_raster_file):
    """Test reading a raster with a window via the C++ backend."""
    data, path = temp_raster_file
    window = Window(0, 0, 50, 50)
    res = read_rasters(raster_files=[path], window=window, backend="cpp", verbose=True)
    assert res.shape == (1, 50 * 50)
    assert_equal(res.get()[0].reshape(50, 50), data[:50, :50])

def test_read_rasters_cpp_with_dtype_falls_back(temp_raster_file):
    """Non-float32 dtype falls back to the python backend (no error)."""
    data, path = temp_raster_file
    res = read_rasters(raster_files=[path], dtype=np.float64, backend="cpp", verbose=True)
    assert res.dtype == np.float64
    assert_equal(res.get(), data.astype(np.float64).reshape(1, -1))

def test_read_rasters_cpp_empty_list():
    """Test reading an empty list of raster files."""
    with pytest.raises(IndexError):
        read_rasters(raster_files=[], backend="cpp", verbose=True)
