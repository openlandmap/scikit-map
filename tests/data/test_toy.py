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
        "bounds",
        "max_rasters",
        "_spatial_shape",
    ]
