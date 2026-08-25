"""Tests for COG overview reads, RAM-fit auto-selection, and extent reads."""

import os

import numpy as np
import pytest
import rasterio

from skmap.data import toy
from skmap.io import RasterData
from skmap.io.base import _resolve_read_params, read_rasters, save_rasters


@pytest.fixture
def cog(tmp_path):
    """A 64x64 float32 COG with overviews [2, 4]."""
    p = str(tmp_path / "cog.tif")
    data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=1,
        dtype="float32",
        transform=rasterio.transform.from_origin(0, 64, 1, 1),
        crs="EPSG:4326",
    ) as ds:
        ds.write(data, 1)
        ds.build_overviews([2, 4], rasterio.enums.Resampling.average)
    return p


@pytest.fixture
def cog_no_overviews(tmp_path):
    """A 64x64 float32 raster with no overviews."""
    p = str(tmp_path / "noov.tif")
    data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=1,
        dtype="float32",
        transform=rasterio.transform.from_origin(0, 64, 1, 1),
        crs="EPSG:4326",
    ) as ds:
        ds.write(data, 1)
    return p


def _static_path():
    return str(toy.DATA_DIR / "static" / os.listdir(toy.DATA_DIR / "static")[0])


# ---------------------------------------------------------------------------
# Overview reads
# ---------------------------------------------------------------------------


def test_overview_read_cpp_python_match(cog):
    cpp = read_rasters([cog], overview=2, backend="cpp", verbose=False)
    py = read_rasters([cog], overview=2, backend="python", verbose=False)
    with rasterio.open(cog) as ds:
        gt = ds.read(1, out_shape=(32, 32))
    assert np.allclose(cpp.get().reshape(32, 32), gt)
    assert np.allclose(py.get().reshape(32, 32), gt)


def test_invalid_overview_raises(cog):
    with pytest.raises(ValueError, match="Overview 3 is invalid"):
        read_rasters([cog], overview=3, backend="python", verbose=False)


def test_overview_write_transform(cog, tmp_path):
    arr = read_rasters([cog], overview=2, backend="cpp", verbose=False)
    out = str(tmp_path / "out.tif")
    save_rasters(cog, [out], arr.get(), overview=2, verbose=False)
    with rasterio.open(cog) as src, rasterio.open(out) as dst:
        assert dst.transform.a == src.transform.a * 2
        assert (dst.width, dst.height) == (32, 32)
        assert np.allclose(dst.read(1), arr.get().reshape(32, 32))


def test_rasterdata_read_overview(cog):
    r = RasterData({"common": [cog]}).read(overview=2)
    assert r.overview == 2
    assert r._spatial_shape == (32, 32)
    assert r.array.shape == (1, 32 * 32)


# ---------------------------------------------------------------------------
# RAM-fit auto-selection
# ---------------------------------------------------------------------------


def test_ram_fit_auto_selects_overview(cog, monkeypatch):
    import psutil

    # full read = 64*64*4 = 16384 bytes; force available to 10000 so the
    # finest overview that fits (x2 -> 32*32*4 = 4096) is chosen.
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: type("V", (), {"available": 10000})()
    )
    _, ov, oh, ow = _resolve_read_params(
        [cog], [1], None, None, "float32", 1, None, 0.7, False
    )
    assert ov == 2
    assert (oh, ow) == (32, 32)


def test_ram_fit_no_overviews_raises(cog_no_overviews, monkeypatch):
    import psutil

    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: type("V", (), {"available": 10000})()
    )
    with pytest.raises(MemoryError, match="no COG overviews"):
        _resolve_read_params(
            [cog_no_overviews], [1], None, None, "float32", 1, None, 0.7, False
        )


def test_ram_fit_full_res_when_fits(cog, monkeypatch):
    import psutil

    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: type("V", (), {"available": 10**9})()
    )
    _, ov, oh, ow = _resolve_read_params(
        [cog], [1], None, None, "float32", 1, None, 0.7, False
    )
    assert ov is None
    assert (oh, ow) == (64, 64)


# ---------------------------------------------------------------------------
# extent reads
# ---------------------------------------------------------------------------


def test_extent_read_matches_rasterio():
    path = _static_path()
    with rasterio.open(path) as src:
        b = src.bounds
        extent = (b.left, b.bottom, b.left + (b.right - b.left) / 2, b.top)
        win = rasterio.windows.from_bounds(*extent, src.transform).round_lengths()
        expected = src.read(1, window=win)

    r = RasterData({"common": [path]}).read(extent=extent, extent_epsg=src.crs)
    got = r.array.get()[0].reshape(r._spatial_shape)
    assert got.shape == expected.shape
    assert np.allclose(got, expected, equal_nan=True)


def test_extent_read_epsg4326():
    path = _static_path()
    with rasterio.open(path) as src:
        b4326 = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        extent = (b4326[0], b4326[1], b4326[0] + (b4326[2] - b4326[0]) / 2, b4326[3])
        b_rast = rasterio.warp.transform_bounds("EPSG:4326", src.crs, *extent)
        win = rasterio.windows.from_bounds(*b_rast, src.transform).round_lengths()
        # clip to the raster grid (same as _resolve_read_params)
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        win = rasterio.windows.Window(int(win.col_off), int(win.row_off), int(win.width), int(win.height))
        expected = src.read(1, window=win)

    r = RasterData({"common": [path]}).read(extent=extent, extent_epsg="EPSG:4326")
    got = r.array.get()[0].reshape(r._spatial_shape)
    assert got.shape == expected.shape
    assert np.allclose(got, expected, equal_nan=True)
