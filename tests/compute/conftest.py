"""Shared fixtures for compute-backend tests.

``toy_arr`` is a (1024, 24) float32 slice of real gappy toy NDVI data
(Window(0,0,32,32) → 24 bands × 1024 pixels, transposed so time is on the
last axis). It has real NaN gaps and matches the (pixels, N) layout that
runners feed to the backends. Edge-case tests that need constructed inputs
(all-NaN rows, specific matrices) stay synthetic.
"""

import numpy as np
import pytest
import rasterio
from rasterio.windows import Window

from skmap.data import toy


@pytest.fixture(scope="module")
def toy_arr():
    """(1024, 24) float32 real gappy NDVI, time on the last axis."""
    path = str(toy.ndvi_files(gappy=True)[0])
    with rasterio.open(path) as src:
        data = src.read(1, window=Window(0, 0, 32, 32)).astype(np.float32)
    # data is (32, 32) for one band; build the full 24-band time series
    files = toy.ndvi_files(gappy=True)
    bands = []
    for f in files:
        with rasterio.open(str(f)) as src:
            bands.append(src.read(1, window=Window(0, 0, 32, 32)).astype(np.float32))
    # nodata 255 -> NaN (matching RasterData.read behaviour)
    stack = np.stack(bands, axis=0)  # (24, 32, 32)
    stack[stack == 255] = np.nan
    return stack.reshape(24, -1).T  # (1024, 24) — pixels first, time last


@pytest.fixture(scope="module")
def toy_arr_filled():
    """(1024, 24) float32 filled NDVI (no NaN) — for convolution equivalence."""
    files = toy.ndvi_files(gappy=False)
    bands = []
    for f in files:
        with rasterio.open(str(f)) as src:
            bands.append(src.read(1, window=Window(0, 0, 32, 32)).astype(np.float32))
    stack = np.stack(bands, axis=0)  # (24, 32, 32)
    stack[stack == 255] = np.nan
    return stack.reshape(24, -1).T  # (1024, 24)