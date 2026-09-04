"""Tests for multi-column runner support (match_col / match / by / select).

These exercise the ability to resolve input bands and partition work by
arbitrary ``info`` columns (e.g. ``band``) instead of the predefined
``group`` column, which is how YAML/STAC catalogues organise layers.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from skmap import parallel
from skmap.io import RasterData
from skmap.io.process import Calc, NormalizedDifference


def _band_rdata():
    """A synthetic RasterData organised by year, with a ``band`` variable column.

    Two bands (red, nir) per year, names are ``{band}_{year}``.
    """
    rows = []
    for year in [2020, 2021]:
        for band in ["red", "nir"]:
            rows.append(
                {
                    "group": str(year),
                    "name": f"{band}_{year}",
                    "band": band,
                    "year": year,
                    "input_path": f"/tmp/{band}_{year}.tif",
                    "input_band": 1,
                    "start_date": None,
                    "end_date": None,
                    "temporal": False,
                }
            )
    rdata = RasterData.from_info(pd.DataFrame(rows))
    # 4 bands x 10 pixels; red=rows 0,2 nir=rows 1,3
    arr = np.arange(40, dtype=np.float32).reshape(4, 10)
    rdata._set_array(parallel.put_shared(arr, local=True))
    rdata.base_raster = "/tmp/base.tif"
    return rdata


def test_band_index_match_col():
    rdata = _band_rdata()
    # red appears in two years -> ambiguous without a disambiguating match
    with pytest.raises(KeyError, match="matched 2 rows"):
        rdata._band_index("red", match_col="band")
    # disambiguate by year
    idx = rdata._band_index("red", match_col="band", match={"year": 2020})
    assert idx == 0
    idx = rdata._band_index("nir", match_col="band", match={"year": 2021})
    assert idx == 3


def test_band_index_unknown_column():
    rdata = _band_rdata()
    with pytest.raises(KeyError, match="not in RasterData.info"):
        rdata._band_index("red", match_col="nope")


def test_select_fluent_filter():
    rdata = _band_rdata()
    sub = rdata.select(band="red")
    assert sub.info["band"].tolist() == ["red", "red"]
    assert sub.array.shape == (2, 10)


def test_normalized_difference_single_with_match_col():
    rdata = _band_rdata()
    r = rdata.run(
        NormalizedDifference("red", "nir"),
        match_col="band",
        match={"year": 2020},
    )
    assert r.array.shape == (5, 10)
    assert r.info["name"].tolist()[-1] == "normalizeddifference"
    assert r.info["group"].tolist()[-1] == "2020"


def test_normalized_difference_grouped_by_year():
    rdata = _band_rdata()
    r = rdata.run(NormalizedDifference("red", "nir"), match_col="band", by="year")
    assert r.array.shape == (6, 10)
    assert r.info["name"].tolist()[-2:] == [
        "normalizeddifference_2020",
        "normalizeddifference_2021",
    ]
    assert r.info["group"].tolist()[-2:] == ["2020", "2021"]


def test_calc_match_col():
    # single year: red and nir are unambiguous within the one date group
    rows = []
    for band in ["red", "nir"]:
        rows.append(
            {
                "group": "2020",
                "name": f"{band}_2020",
                "band": band,
                "year": 2020,
                "input_path": f"/tmp/{band}_2020.tif",
                "input_band": 1,
                "start_date": pd.Timestamp("2020-01-01"),
                "end_date": pd.Timestamp("2020-12-31"),
                "temporal": True,
            }
        )
    rdata = RasterData.from_info(pd.DataFrame(rows))
    rdata._set_array(
        parallel.put_shared(np.arange(20, dtype=np.float32).reshape(2, 10), local=True)
    )
    rdata.base_raster = "/tmp/base.tif"

    r = rdata.run(Calc({"nd": "red - nir"}, match_col="band"))
    assert r.array.shape[0] == 2 + 1  # 2 inputs + 1 output
    assert "nd" in r.info["group"].tolist()
