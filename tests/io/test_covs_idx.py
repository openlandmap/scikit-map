"""Tests for RasterData covariate-index mapping and valid-pixel selection."""

import numpy as np
import pandas as pd
import pytest

from skmap import parallel
from skmap.data import toy
from skmap.io import RasterData


@pytest.fixture
def static_rdata():
    """A RasterData with two common features (elev, slope)."""

    elev = "elev.lowestmode_gedi.eml_mf_30m_s_20000101_20181231_nl_epsg.3035_v0.3.tif"
    slope = "slope.percent_gedi.eml_m_30m_s_20000101_20181231_nl_epsg.3035_v0.3.tif"
    toy_dir = toy.DATA_DIR
    return RasterData(
        {
            "elev": [str(toy_dir / "static" / elev)],
            "slope": [str(toy_dir / "static" / slope)],
        }
    ).read()


@pytest.fixture
def multi_feature_rdata():
    """Two temporal features (ndvi, swir1) + two common features (elev, slope)."""

    rows = [
        {"feature": "ndvi", "name": "ndvi_2019", "input_path": "/t/ndvi.tif",
         "input_band": 1, "start_date": pd.Timestamp("2019-01-01"),
         "end_date": pd.Timestamp("2019-12-31"), "is_common": False},
        {"feature": "swir1", "name": "swir1_2019", "input_path": "/t/swir1.tif",
         "input_band": 1, "start_date": pd.Timestamp("2019-01-01"),
         "end_date": pd.Timestamp("2019-12-31"), "is_common": False},
        {"feature": "elev", "name": "elev", "input_path": "/t/elev.tif",
         "input_band": 1, "start_date": None, "end_date": None, "is_common": True},
        {"feature": "slope", "name": "slope", "input_path": "/t/slope.tif",
         "input_band": 1, "start_date": None, "end_date": None, "is_common": True},
    ]
    rdata = RasterData.from_info(pd.DataFrame(rows))
    rdata._set_array(
        parallel.put_shared(np.arange(40, dtype=np.float32).reshape(4, 10), local=True)
    )
    rdata.base_raster = "/t/base.tif"
    return rdata


def test_get_features_common_only(static_rdata):
    assert static_rdata.get_features() == ["elev", "slope"]


def test_get_features_multi(multi_feature_rdata):
    assert multi_feature_rdata.get_features() == ["elev", "ndvi", "slope", "swir1"]


def test_band_index(static_rdata):
    assert static_rdata._band_index("elev") == 0
    assert static_rdata._band_index("slope") == 1
    with pytest.raises(KeyError):
        static_rdata._band_index("nope")


def test_covs_idx_common_only(static_rdata):
    # no temporal features -> empty (0, 0) matrix
    idx = static_rdata._get_covs_idx(["elev", "slope"])
    assert idx.shape == (2, 0)


def test_covs_idx_multi_feature(multi_feature_rdata):
    idx = multi_feature_rdata._get_covs_idx(["elev", "slope"])
    # ndvi: elev/slope fall back to common (rows 2, 3)
    # swir1: elev/slope fall back to common (rows 2, 3)
    assert idx.tolist() == [[2, 2], [3, 3]]


def test_covs_idx_missing_raises(multi_feature_rdata):
    with pytest.raises(KeyError):
        multi_feature_rdata._get_covs_idx(["missing"])


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
