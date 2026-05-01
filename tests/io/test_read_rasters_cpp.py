from numpy.testing import assert_equal
import numpy as np
import pytest
from rasterio.windows import Window
from skmap.io.base import read_rasters_cpp

def test_read_rasters_cpp_single_file(temp_raster_file):
    """Test reading a single raster file."""
    data, path = temp_raster_file
    res = read_rasters_cpp(raster_files=[path], verbose=True)
    assert res.shape == (1, 100 * 100,)  # Flattened shape: (n_pixels,)
    assert res.dtype == np.float32
    assert_equal(res, data.reshape(res.shape))

def test_read_rasters_cpp_multiple_files(temp_multi_raster_files):
    """Test reading multiple raster files."""
    datas, paths = temp_multi_raster_files
    res = read_rasters_cpp(raster_files=paths, verbose=True)
    assert res.shape == (3, 100 * 100,)  # Flattened shape: (n_files * n_pixels,)
    assert res.dtype == np.float32
    for idx,data in enumerate(datas):
        assert_equal(res[idx,:], data.reshape((100*100)))


def test_read_rasters_cpp_with_window(temp_raster_file):
    """Test reading a raster with a window."""
    data, path = temp_raster_file
    window = Window(0, 0, 50, 50)  # Read top-left 50x50 pixels
    res = read_rasters_cpp(raster_files=[path], window=window, verbose=True)
    assert res.shape == (1, 50 * 50)  # Flattened shape: (window_height * window_width,)
    assert_equal(res, data[:50,:50].reshape(res.shape))

@pytest.mark.xfail(reason="TODO: readData only supports float32, see writeData for dispatch pattern")
def test_read_rasters_cpp_with_dtype(temp_raster_file):
    """Test reading a raster with a specific dtype."""
    data, path = temp_raster_file
    res = read_rasters_cpp(raster_files=[path], dtype=np.float64, verbose=True)
    assert res.dtype == np.float64
    assert_equal(res, data.astype(np.float64).reshape(res.shape))

def test_read_rasters_cpp_with_out_data(temp_raster_file):
    """Test reading into a pre-allocated array."""
    data, path = temp_raster_file
    out_data = np.empty((1,100 * 100), dtype=np.float32)
    res = read_rasters_cpp(raster_files=[path], out_data=out_data, verbose=True)
    assert res is out_data  # Should use the provided array
    assert res.shape == (1, 100 * 100,)
    assert_equal(res, data.reshape(res.shape))

def test_read_rasters_cpp_empty_list():
    """Test reading an empty list of raster files."""
    with pytest.raises(IndexError):  # Or whatever error your function raises
        read_rasters_cpp(raster_files=[], verbose=True)
