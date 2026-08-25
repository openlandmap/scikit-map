"""
Access to skmap toy demo datasets
"""

from pathlib import Path

from geopandas import read_file

from skmap.io import RasterData
from skmap.misc import find_files

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.joinpath("toy")
TOY_DATE_STEP = [109, 96, 80, 80]
LAYERS_YAML = DATA_DIR.joinpath("layers.yaml")


def _static_raster():
    return find_files(DATA_DIR.joinpath("static"), "*.tif")


def _temporal_raster(type, subpath=None):
    raster_dir = DATA_DIR.joinpath(type)
    if subpath is not None:
        raster_dir = raster_dir.joinpath(subpath)

    base_dt = "20200913_20201201"

    raster_files = find_files(raster_dir, f"*{base_dt}*.tif")
    return str(raster_files[0]).replace(base_dt, "{dt}")


def rdata(verbose=True, backend="numpy"):
    """Return a small example static raster array (two layers) for testing and demos.

    :param backend: compute backend (``"numpy"``/``"numba"``/``"cpp"``) used by
        subsequent :meth:`RasterData.run` calls; see :mod:`skmap.compute`.
    """

    return (
        RasterData(
            {
                "ndvi": _temporal_raster("ndvi", "filled"),
                "swir1": _temporal_raster("swir1"),
                "static": _static_raster(),
            },
            verbose=verbose,
            backend=backend,
        )
        .timespan("20141202", "20201201", "days", TOY_DATE_STEP, ignore_29feb=True)
        .read()
    )


def ndvi_rdata(gappy=False, verbose=True, backend="numpy"):
    """Return a small example quarterly NDVI time-series raster, optionally with gaps.

    :param backend: compute backend (``"numpy"``/``"numba"``/``"cpp"``) used by
        subsequent :meth:`RasterData.run` calls; see :mod:`skmap.compute`.
    """

    subpath = "gappy" if gappy else "filled"
    return (
        RasterData({"ndvi": _temporal_raster("ndvi", subpath)}, verbose=verbose, backend=backend)
        .timespan("20141202", "20201201", "days", TOY_DATE_STEP, ignore_29feb=True)
        .read()
    )


def ndvi_files(gappy=False):
    """Return a sorted list of concrete NDVI toy-raster file paths."""

    subpath = "gappy" if gappy else "filled"
    return find_files(DATA_DIR.joinpath("ndvi", subpath), "*.tif")


def swir1_files():
    """Return a sorted list of concrete SWIR1 toy-raster file paths."""

    return find_files(DATA_DIR.joinpath("swir1"), "*.tif")


def lc_samples():
    """Return a small example land-cover sample ``GeoDataFrame`` for testing and demos."""

    return read_file(DATA_DIR.joinpath("samples").joinpath("samples.gpkg"))
