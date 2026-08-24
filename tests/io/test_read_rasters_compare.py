from numpy.testing import assert_equal
import numpy as np
from rasterio.windows import Window
from skmap.io.base import read_rasters

def test_read_rasters_vs_cpp_single_file(temp_raster_file):
    """Compare python and cpp backends for a single file."""
    data, path = temp_raster_file
    data_py = read_rasters(raster_files=[path], backend="python", verbose=True)
    data_cpp = read_rasters(raster_files=[path], backend="cpp", verbose=True)
    assert data_py.shape == data_cpp.shape == (1, 100 * 100)
    assert_equal(data_py, data_cpp)

def test_read_rasters_vs_cpp_multiple_files(temp_multi_raster_files):
    """Compare python and cpp backends for multiple files."""
    datas, paths = temp_multi_raster_files
    data_py = read_rasters(raster_files=paths, backend="python", verbose=False)
    data_cpp = read_rasters(raster_files=paths, backend="cpp", verbose=False)
    assert data_py.shape == data_cpp.shape == (3, 100 * 100)
    assert_equal(data_py, data_cpp)

def test_read_rasters_vs_cpp_with_window(temp_raster_file):
    """Compare python and cpp backends with a window."""
    data, path = temp_raster_file
    window = Window(0, 0, 50, 50)
    data_py = read_rasters(raster_files=[path], window=window, backend="python", verbose=True)
    data_cpp = read_rasters(raster_files=[path], window=window, backend="cpp", verbose=True)
    assert data_py.shape == data_cpp.shape == (1, 50 * 50)
    assert_equal(data_py, data_cpp)

def test_auto_backend_selects_cpp_for_float32(temp_raster_file):
    """backend=None auto-selects cpp for a plain float32 request."""
    data, path = temp_raster_file
    res = read_rasters(raster_files=[path], verbose=False)
    assert res.dtype == np.float32
    assert res.shape == (1, 100 * 100)
