"""Compare python and cpp backends of ``read_rasters`` on real toy NDVI data.

Real data (real CRS, real nodata, real temporal gaps) makes the equivalence
test catch dtype/nodata-handling bugs that synthetic EPSG:4326 rasters miss.
"""

import numpy as np
import pytest
from rasterio.windows import Window
from numpy.testing import assert_equal

from skmap.data import toy
from skmap.io.base import read_rasters


@pytest.fixture(scope="module")
def ndvi_filled():
    return [str(p) for p in toy.ndvi_files()]


def test_read_rasters_vs_cpp_single_file(ndvi_filled):
    """Compare python and cpp backends for a single toy NDVI file."""
    path = ndvi_filled[0]
    data_py = read_rasters(raster_files=[path], backend="python", verbose=False)
    data_cpp = read_rasters(raster_files=[path], backend="cpp", verbose=False)
    assert data_py.shape == data_cpp.shape == (1, 256 * 256)
    assert_equal(data_py.get(), data_cpp.get())


def test_read_rasters_vs_cpp_multiple_files(ndvi_filled):
    """Compare python and cpp backends for multiple toy NDVI files."""
    paths = ndvi_filled[:3]
    data_py = read_rasters(raster_files=paths, backend="python", verbose=False)
    data_cpp = read_rasters(raster_files=paths, backend="cpp", verbose=False)
    assert data_py.shape == data_cpp.shape == (3, 256 * 256)
    assert_equal(data_py.get(), data_cpp.get())


def test_read_rasters_vs_cpp_with_window(ndvi_filled):
    """Compare python and cpp backends with a window on toy NDVI."""
    path = ndvi_filled[0]
    window = Window(0, 0, 64, 64)
    data_py = read_rasters(raster_files=[path], window=window, backend="python", verbose=False)
    data_cpp = read_rasters(raster_files=[path], window=window, backend="cpp", verbose=False)
    assert data_py.shape == data_cpp.shape == (1, 64 * 64)
    assert_equal(data_py.get(), data_cpp.get())


def test_read_rasters_vs_cpp_gappy(ndvi_filled):
    """Compare python and cpp backends on gappy toy NDVI (real NaN gaps)."""
    paths = [str(p) for p in toy.ndvi_files(gappy=True)]
    data_py = read_rasters(raster_files=paths, backend="python", verbose=False)
    data_cpp = read_rasters(raster_files=paths, backend="cpp", verbose=False)
    assert data_py.shape == data_cpp.shape == (24, 256 * 256)
    assert_equal(data_py.get(), data_cpp.get())


def test_auto_backend_selects_cpp_for_float32(ndvi_filled):
    """backend=None auto-selects cpp for a plain float32 request on toy data."""
    path = ndvi_filled[0]
    res = read_rasters(raster_files=[path], verbose=False)
    assert res.dtype == np.float32
    assert res.shape == (1, 256 * 256)


def test_cpp_backend_read_is_local_no_ray(ndvi_filled):
    """backend='cpp' reads via the C++ bindings and stays in-process (no Ray)."""
    import ray

    if ray.is_initialized():
        ray.shutdown()

    data = read_rasters(raster_files=[ndvi_filled[0]], backend="cpp", verbose=False)
    assert isinstance(data.ref, np.ndarray)  # local array, not an ObjectRef
    assert not ray.is_initialized()

    data_ray = read_rasters(
        raster_files=[ndvi_filled[0]], backend="python", verbose=False
    )
    assert_equal(data.get(), data_ray.get())


def test_rasterdata_cpp_backend_read_skips_ray():
    """RasterData(backend='cpp').read() must not initialize Ray."""
    import ray

    if ray.is_initialized():
        ray.shutdown()

    r = toy.rdata(backend="cpp")
    assert isinstance(r.array.ref, np.ndarray)
    assert not ray.is_initialized()
    assert r.array.shape[0] == 50  # 24 ndvi + 24 swir1 + 2 static


def test_rasterdata_default_backend_keeps_ray_object_store():
    """The default (numpy) backend keeps the Ray object-store memory model."""
    import ray

    if ray.is_initialized():
        ray.shutdown()

    r = toy.rdata(backend="numpy")
    assert isinstance(r.array.ref, ray.ObjectRef)
    assert r.array.shape[0] == 50