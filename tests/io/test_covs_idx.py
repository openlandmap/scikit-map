"""Tests for RasterData covariate-index mapping and valid-pixel selection."""

import numpy as np
import pytest

from skmap.data import toy
from skmap.io import RasterData


@pytest.fixture
def static_rdata():
    """A RasterData with a ``common`` group and short covariate names."""

    elev = "elev.lowestmode_gedi.eml_mf_30m_s_20000101_20181231_nl_epsg.3035_v0.3.tif"
    slope = "slope.percent_gedi.eml_m_30m_s_20000101_20181231_nl_epsg.3035_v0.3.tif"
    toy_dir = toy.DATA_DIR
    rdata = RasterData(
        {"common": [str(toy_dir / "static" / elev), str(toy_dir / "static" / slope)]}
    ).read()
    rdata.info["name"] = ["elev", "slope"]
    return rdata


@pytest.fixture
def multi_group_rdata():
    """A RasterData with two year groups + a common group."""

    elev = "elev.lowestmode_gedi.eml_mf_30m_s_20000101_20181231_nl_epsg.3035_v0.3.tif"
    slope = "slope.percent_gedi.eml_m_30m_s_20000101_20181231_nl_epsg.3035_v0.3.tif"
    toy_dir = toy.DATA_DIR
    rdata = RasterData(
        {
            "2019": [str(toy_dir / "static" / elev)],
            "2020": [str(toy_dir / "static" / slope)],
            "common": [str(toy_dir / "static" / elev), str(toy_dir / "static" / slope)],
        }
    ).read()
    rdata.info["name"] = ["elev", "slope", "elev", "slope"]
    return rdata


def test_get_groups_common_only(static_rdata):
    assert static_rdata.get_groups() == ["common"]


def test_get_groups_multi(multi_group_rdata):
    assert multi_group_rdata.get_groups() == ["2019", "2020"]


def test_band_index(static_rdata):
    assert static_rdata._band_index("elev") == 0
    assert static_rdata._band_index("slope") == 1
    with pytest.raises(KeyError):
        static_rdata._band_index("nope")


def test_covs_idx_common_only(static_rdata):
    idx = static_rdata._get_covs_idx(["elev", "slope"])
    assert idx.shape == (2, 1)
    assert idx.tolist() == [[0], [1]]


def test_covs_idx_multi_group(multi_group_rdata):
    idx = multi_group_rdata._get_covs_idx(["elev", "slope"])
    # 2019: elev=0, slope falls back to common=3
    # 2020: elev falls back to common=2, slope=1
    assert idx.tolist() == [[0, 2], [3, 1]]


def test_covs_idx_missing_raises(static_rdata):
    with pytest.raises(KeyError):
        static_rdata._get_covs_idx(["missing"])


def test_valid_pixels_and_roundtrip(static_rdata):
    vp = static_rdata.valid_pixels
    assert vp.shape == (256 * 256,)
    assert vp.sum() == 256 * 256  # no NaN in static toy rasters

    sel = static_rdata.select_valid()
    assert sel.shape == (256 * 256, 2)
    assert not np.isnan(sel).any()

    exp = static_rdata.expand_valid(np.arange(sel.shape[0], dtype=np.float32), nodata=-9999)
    assert exp.shape == (256 * 256,)
    assert (exp != -9999).sum() == sel.shape[0]
