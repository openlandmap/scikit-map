from numpy.testing import assert_equal
import numpy as np
import pytest
from rasterio.windows import Window
from skmap.io.base import read_rasters, read_rasters_cpp

def test_read_rasters_vs_cpp_single_file(temp_raster_file):
    """Compare outputs of read_rasters and read_rasters_cpp for a single file."""
    data, path = temp_raster_file
    # Read with read_rasters
    data_py = read_rasters(raster_files=[path], verbose=True)

    # Read with read_rasters_cpp
    data_cpp = read_rasters_cpp(raster_files=[path], verbose=True)

    # Compare results
    assert_equal(data_py.reshape(data_cpp.shape), data_cpp)

def test_read_rasters_vs_cpp_multiple_files(temp_multi_raster_files):
    """Compare outputs of read_rasters (Python) and read_rasters_cpp (C++).

    The two implementations use different array conventions:
      - Python: (height, width, n_files) = (100, 100, 3), spatial-first, bands last
      - C++:    (n_files, height * width) = (3, 10000), files-first, pixels flattened

    To compare, we convert Python output to C++ convention:
      (100, 100, 3) -> transpose to (3, 100, 100) -> reshape to (3, 10000)
    """
    datas, paths = temp_multi_raster_files

    data_py = read_rasters(raster_files=paths, verbose=False)
    data_cpp = read_rasters_cpp(raster_files=paths, verbose=False)

    assert data_py.shape == (100, 100, 3), f"Python shape mismatch: {data_py.shape}"
    assert data_cpp.shape == (3, 10000), f"C++ shape mismatch: {data_cpp.shape}"

    # Convert Python convention -> C++ convention for comparison:
    # (H, W, N) -> (N, H*W)
    # transpose is a permutation array, reshape flattens spatially
    data_py_converted = data_py.transpose(2, 0, 1).reshape(3, -1)

    assert_equal(data_py_converted, data_cpp)

# def test_read_rasters_vs_cpp_multiple_files(temp_multi_raster_files):
#     """Compare outputs of read_rasters and read_rasters_cpp for multiple files."""
#     datas, paths = temp_multi_raster_files
#     # Read with read_rasters
#     data_py = read_rasters(raster_files=paths, verbose=True)

#     # Read with read_rasters_cpp
#     data_cpp = read_rasters_cpp(raster_files=paths, verbose=True)

#     # Compare results
#     # assert_equal(data_py.reshape(data_cpp.shape), data_cpp)

def test_read_rasters_vs_cpp_with_window(temp_raster_file):
    """Compare outputs of read_rasters and read_rasters_cpp with a window."""
    data, path = temp_raster_file
    window = Window(0, 0, 50, 50)

    # Read with read_rasters
    data_py = read_rasters(raster_files=[path], window=window, verbose=True)
    data_py = data_py.reshape(-1)  # Flatten to match read_rasters_cpp output

    # Read with read_rasters_cpp
    data_cpp = read_rasters_cpp(raster_files=[path], window=window, verbose=True)

    # Compare results
    assert_equal(data_py.reshape(data_cpp.shape), data_cpp)
