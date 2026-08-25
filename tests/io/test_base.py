"""Tests for ``_new_raster`` and ``save_rasters``/``read_rasters`` roundtrip.

Uses real toy rasters (real CRS, real nodata) as the base raster and data
source. The one non-square roundtrip test keeps a synthetic array because all
toy rasters are 256x256 (square) — a non-square array is needed to catch
silent height/width axis swaps.
"""

import rasterio
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from skmap.data import toy
from skmap.io import save_rasters, read_rasters
from skmap.io.base import _new_raster


def test_with_statement_closes_file():
    """_new_raster opens a GeoTIFF for writing and closes it on context exit."""
    base_file = str(toy._static_raster()[0])
    output_file = "/tmp/skmap_test_new_raster.tif"

    band = np.full((256, 256), 42.0, dtype=np.float32)
    with _new_raster(base_file, output_file, band) as new_raster:
        assert new_raster.shape == (256, 256)
        new_raster.write(band, 1)

    # After the with block the file must be closed (re-openable in r+ mode)
    with rasterio.open(output_file, "r+") as src:
        assert src.read(1).shape == (256, 256)

    import os
    os.remove(output_file)


def test_new_raster_non_square_roundtrip():
    """_new_raster preserves shape for non-square arrays (catches H/W swaps).

    # synthetic: toy rasters are all 256x256 (square); a non-square synthetic
    # array is required to catch silent height/width axis swaps.
    """
    H, W = 30, 70
    original = np.random.rand(H, W).astype(np.float32)
    base_raster_path = "/tmp/skmap_base_nonsquare.tif"
    output_path = "/tmp/skmap_output_nonsquare.tif"

    with rasterio.open(
        base_raster_path, "w", driver="GTiff",
        height=H, width=W, count=1, dtype="float32",
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(original, 1)

    with _new_raster(base_raster_path, output_path, original) as ds:
        ds.write(original, 1)

    with rasterio.open(output_path) as src:
        loaded = src.read(1)

    assert loaded.shape == original.shape, \
        f"Shape mismatch: wrote {original.shape}, got {loaded.shape}"
    assert_array_equal(original, loaded)

    import os
    os.remove(base_raster_path)
    os.remove(output_path)


def test_save_read_rasters_roundtrip(tmp_path):
    """Round-trip: read real toy NDVI bands -> save_rasters -> read_rasters.

    Uses a toy static raster as the base raster (real CRS, real transform).
    save_rasters expects (N, H*W); read_rasters returns (N, H*W).
    """
    ndvi_paths = [str(p) for p in toy.ndvi_files()[:3]]
    flat = read_rasters(raster_files=ndvi_paths, backend="python", verbose=False)
    N, H, W = 3, 256, 256
    assert flat.shape == (N, H * W)

    base_raster_path = str(toy._static_raster()[0])
    output_paths = [str(tmp_path / f"out_{i}.tif") for i in range(N)]

    save_rasters(base_raster_path, output_paths, flat, n_jobs=1)

    loaded = read_rasters(raster_files=output_paths, backend="python", n_jobs=1, verbose=False)
    assert loaded.shape == (N, H * W), \
        f"Shape mismatch: saved {flat.shape}, loaded {loaded.shape}"
    assert_array_equal(loaded.get(), flat.get())