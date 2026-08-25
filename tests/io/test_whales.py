"""Tests for the whale (on-the-fly feature) runners."""

import warnings

import numpy as np
import pytest

from skmap.data import toy
from skmap.io import RasterData, process


@pytest.fixture
def static_rdata():
    """A RasterData with short covariate names elev/slope in a common group."""

    elev = "elev.lowestmode_gedi.eml_mf_30m_s_20000101_20181231_nl_epsg.3035_v0.3.tif"
    slope = "slope.percent_gedi.eml_m_30m_s_20000101_20181231_nl_epsg.3035_v0.3.tif"
    toy_dir = toy.DATA_DIR
    rdata = RasterData(
        {"common": [str(toy_dir / "static" / elev), str(toy_dir / "static" / slope)]}
    ).read()
    rdata.info["name"] = ["elev", "slope"]
    return rdata


def test_normalized_difference(static_rdata):
    r = static_rdata
    r = r.run(process.NormalizedDifference("elev", "slope"), outname="nd")
    assert r.array.shape == (3, 256 * 256)
    assert r.info["name"].tolist()[-1] == "nd"
    vals = r.array.get()[-1]
    assert np.isfinite(vals).all()
    assert vals.min() >= -1.0 and vals.max() <= 1.0


def test_normalized_difference_matches_cpp():
    import skmap_bindings as sb

    rng = np.random.default_rng(0)
    arr = (rng.random((3, 1000)) * 100).astype(np.float32)
    sp, sm, sr, off = 1.0, 1.0, 1000.0, 0.0
    clip = [0.0, 1000.0]

    p = arr[0] * sp
    m = arr[1] * sm
    with np.errstate(divide="ignore", invalid="ignore"):
        val = (p - m) / (p + m) * sr + off
    val = np.round(val)
    val = np.where(val == -np.inf, -sr + off, val)
    val = np.where(val == np.inf, sr + off, val)
    val = np.clip(val, clip[0], clip[1])

    data = arr.copy()
    sb.computeNormalizedDifference(data, 1, [0], [1], [2], sp, sm, sr, off, clip)
    np.testing.assert_array_equal(val, data[2])


def test_extract_indicator(static_rdata):
    r = static_rdata
    r = r.run(process.ExtractIndicator("elev", 70.0), outname="elev_gt70")
    vals = r.array.get()[-1]
    assert set(np.unique(vals)) <= {0.0, 1.0}


def test_percentile_aggregation():
    files = [str(p) for p in toy.ndvi_files(gappy=True)[:6]]
    r = RasterData({"ndvi": files}).read()
    names = [f"ndvi_{i}" for i in range(6)]
    r.info["name"] = names
    r = r.run(process.PercentileAggregation(names, 50.0), outname="ndvi_p50")
    vals = r.array.get()[-1]
    sub = r.array.get()[:6]
    valid = ~np.isnan(vals)
    assert np.all(vals[valid] >= np.nanmin(sub, axis=0)[valid] - 1e-6)
    assert np.all(vals[valid] <= np.nanmax(sub, axis=0)[valid] + 1e-6)


def test_get_latitude(static_rdata):
    r = static_rdata
    r = r.run(process.GetLatitude())
    vals = r.array.get()[-1]
    assert vals.shape == (256 * 256,)
    # toy data is EPSG:3035, y between ~3210145 and ~3217795
    assert 3210000 <= np.nanmin(vals) <= np.nanmax(vals) <= 3220000


def test_geometric_temperature(static_rdata):
    r = static_rdata.run(process.GetLatitude())
    r = r.run(
        process.GeometricTemperature("latitude", "elev", day_of_year_mmdd="0101"),
        outname="gt",
    )
    vals = r.array.get()[-1]
    assert vals.shape == (256 * 256,)
    assert np.isfinite(vals).all()
