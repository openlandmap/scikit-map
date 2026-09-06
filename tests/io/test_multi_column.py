"""Tests for feature-based runner resolution (match / by / select).

These exercise resolving input bands and partitioning work by the mandatory
``feature`` column (plus optional columns), which is how YAML/STAC catalogues
organise layers.
"""

import numpy as np
import pandas as pd
import pytest

from skmap import parallel
from skmap.io import RasterData
from skmap.io.process import Calc, NormalizedDifference


def _temporal_rdata():
    """Two temporal features (red, nir) over two intervals, plus a year column."""

    rows = []
    for year in [2020, 2021]:
        for feat in ["red", "nir"]:
            rows.append(
                {
                    "feature": feat,
                    "name": f"{feat}_{year}",
                    "year": year,
                    "input_path": f"/tmp/{feat}_{year}.tif",
                    "input_band": 1,
                    "start_date": pd.Timestamp(f"{year}-01-01"),
                    "end_date": pd.Timestamp(f"{year}-12-31"),
                    "is_common": False,
                }
            )
    rdata = RasterData.from_info(pd.DataFrame(rows))
    arr = np.arange(40, dtype=np.float32).reshape(4, 10)
    rdata._set_array(parallel.put_shared(arr, local=True))
    rdata.base_raster = "/tmp/base.tif"
    return rdata


def test_band_index_feature_ambiguous():
    rdata = _temporal_rdata()
    # red appears in two intervals -> ambiguous without a disambiguating match
    with pytest.raises(KeyError, match="matched 2 rows"):
        rdata._band_index("red")
    idx = rdata._band_index("red", match={"year": 2020})
    assert idx == 0
    idx = rdata._band_index("nir", match={"year": 2021})
    assert idx == 3


def test_band_index_unknown_column():
    rdata = _temporal_rdata()
    with pytest.raises(KeyError, match="not in RasterData.info"):
        rdata._band_index("red", match_col="nope")


def test_select_fluent_filter():
    rdata = _temporal_rdata()
    sub = rdata.select(feature="red")
    assert sub.info["feature"].tolist() == ["red", "red"]
    assert sub.array.shape == (2, 10)


def test_normalized_difference_interval_partition():
    """Temporal inputs default to one output band per time interval."""
    rdata = _temporal_rdata()
    r = rdata.run(NormalizedDifference("red", "nir"), outname="ndvi")
    assert r.array.shape == (6, 10)
    assert r.info["feature"].tolist()[-2:] == ["ndvi", "ndvi"]
    assert r.info["name"].tolist()[-2:] == [
        "ndvi_20200101_20201231",
        "ndvi_20210101_20211231",
    ]
    assert r.info["is_common"].tolist()[-2:] == [False, False]


def test_normalized_difference_by_year():
    rdata = _temporal_rdata()
    r = rdata.run(NormalizedDifference("red", "nir"), outname="ndvi", by="year")
    assert r.array.shape == (6, 10)
    assert r.info["name"].tolist()[-2:] == ["ndvi_2020", "ndvi_2021"]


def test_calc_feature_namespace():
    rows = []
    for feat in ["red", "nir"]:
        rows.append(
            {
                "feature": feat,
                "name": f"{feat}_2020",
                "input_path": f"/tmp/{feat}_2020.tif",
                "input_band": 1,
                "start_date": pd.Timestamp("2020-01-01"),
                "end_date": pd.Timestamp("2020-12-31"),
                "is_common": False,
            }
        )
    rdata = RasterData.from_info(pd.DataFrame(rows))
    rdata._set_array(
        parallel.put_shared(np.arange(20, dtype=np.float32).reshape(2, 10), local=True)
    )
    rdata.base_raster = "/tmp/base.tif"

    r = rdata.run(Calc({"nd": "red - nir"}))
    assert r.array.shape[0] == 2 + 1  # 2 inputs + 1 output
    assert "nd" in r.info["feature"].tolist()
