import skmap.data.toy as toy
from skmap.io.base import RasterData


def test__static_raster() -> None:
    assert len(toy._static_raster()) == 2


def test_ndvi_rdata() -> None:
    rdata: RasterData = toy.ndvi_rdata()
    assert list(vars(rdata).keys()) == [
        "backend",
        "raster_files",
        "verbose",
        "raster_mask",
        "raster_mask_val",
        "info",
        "date_args",
        "_active_group",
        "array",
        "base_raster",
        "window",
        "overview",
        "extent",
        "extent_epsg",
        "max_rasters",
        "_spatial_shape",
    ]


def test_ndvi_rdata_backend() -> None:
    """The data accessors forward the backend to RasterData."""
    from skmap.compute import CppBackend, NumbaBackend, NumpyBackend

    assert isinstance(toy.ndvi_rdata(backend="numpy").backend, NumpyBackend)
    assert isinstance(toy.ndvi_rdata(backend="numba").backend, NumbaBackend)
    assert isinstance(toy.ndvi_rdata(backend="cpp").backend, CppBackend)


def test_rdata_backend() -> None:
    """rdata() also forwards the backend."""
    from skmap.compute import NumbaBackend

    assert isinstance(toy.rdata(backend="numba").backend, NumbaBackend)
