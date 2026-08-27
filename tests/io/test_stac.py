"""Tests for the STAC source (skmap.io.sources.StacSource) and RasterData.from_stac.

The unit tests stub the HTTP layer (``_fetch_item_dicts`` / ``_root_url``) so
they never touch the network.  A single opt-in ``@pytest.mark.network`` test
hits the live ecodatacube endpoint and is skipped by default.
"""

from datetime import datetime

import pandas as pd
import pytest

from skmap.io import RasterData
from skmap.io import sources as src_mod
from skmap.io.sources import StacSource

NDVI_CID = "ndvi_glad.landsat.ard2.seasconv_eu_ecodatacube"
CH4_CID = "ch4.vmr_s5p.l2.m.seasconv_eu_ecodatacube"
CATALOG_URL = "https://stac.opengeohub.org/v1/cat/ecodatacube"
ROOT_URL = "https://stac.opengeohub.org/v1"


def _item(cid, start, end, assets):
    return {
        "id": f"{cid}_{start.replace('-', '')}_{end.replace('-', '')}",
        "properties": {
            "start_datetime": f"{start}T00:00:00Z",
            "end_datetime": f"{end}T00:00:00Z",
            "proj:epsg": 3035,
            "gsd": 30.0,
        },
        "assets": assets,
    }


def _data_asset(key, href=None):
    return {key: {"href": href or f"https://s3.ecodatacube.eu/arco/{key}.tif",
                  "roles": ["data"]}}


NDVI_ITEMS = [
    _item(NDVI_CID, "2022-11-01", "2022-12-31",
          {**_data_asset("ndvi_m_30m_s"),
           "thumbnail": {"href": "https://x/thumb.png", "roles": ["thumbnail"]},
           "ndvi_qml_1": {"href": "https://x/style.qml", "roles": ["style"]}}),
    _item(NDVI_CID, "2022-09-01", "2022-10-31",
          _data_asset("ndvi_m_30m_s")),
    _item(NDVI_CID, "2021-11-01", "2021-12-31",
          _data_asset("ndvi_m_30m_s")),
]

CH4_ITEMS = [
    _item(CH4_CID, "2022-12-01", "2022-12-31",
          {**_data_asset("ch4_p10_2km_a"), **_data_asset("ch4_p50_2km_a"),
           **_data_asset("ch4_p90_2km_a")}),
]


class _StubStacSource(StacSource):
    """StacSource with the HTTP layer replaced by canned items."""

    def __init__(self, items_by_collection, **kwargs):
        self._items = items_by_collection
        super().__init__(**kwargs)

    def _root_url(self):
        return ROOT_URL

    def _fetch_item_dicts(self, collection_id):
        return self._items.get(collection_id, [])


def _stub(items_by_collection, **kwargs):
    kwargs.setdefault("url", CATALOG_URL)
    kwargs.setdefault("collections", list(items_by_collection))
    return _StubStacSource(items_by_collection, **kwargs)


class TestIterSpecs:
    def test_skips_non_data_assets(self):
        s = _stub({NDVI_CID: NDVI_ITEMS})
        specs = list(s.iter_specs())
        # thumbnail + style assets are skipped; only the data asset is kept
        assert len(specs) == 3
        assert all(sp.name == "ndvi_m_30m_s" for sp in specs)
        assert all(sp.path.endswith(".tif") for sp in specs)

    def test_multiple_data_assets_one_row_each(self):
        s = _stub({CH4_CID: CH4_ITEMS})
        specs = list(s.iter_specs())
        assert [sp.name for sp in specs] == [
            "ch4_p10_2km_a", "ch4_p50_2km_a", "ch4_p90_2km_a"
        ]
        assert [sp.vars["asset"] for sp in specs] == [
            "ch4_p10_2km_a", "ch4_p50_2km_a", "ch4_p90_2km_a"
        ]

    def test_group_is_collection(self):
        s = _stub({NDVI_CID: NDVI_ITEMS})
        specs = list(s.iter_specs())
        assert all(sp.group == NDVI_CID for sp in specs)

    def test_dates_from_start_end_datetime(self):
        s = _stub({NDVI_CID: NDVI_ITEMS})
        specs = list(s.iter_specs())
        assert specs[0].start_date == datetime(2022, 11, 1)
        assert specs[0].end_date == datetime(2022, 12, 31)
        assert specs[0].temporal is True

    def test_date_filter_client_side(self):
        s = _stub({NDVI_CID: NDVI_ITEMS}, datetime="2022-09-01/2022-11-30")
        specs = list(s.iter_specs())
        # 2021 item is dropped; the two 2022 items are kept
        assert len(specs) == 2
        assert {sp.start_date.year for sp in specs} == {2022}

    def test_bands_filter(self):
        s = _stub({CH4_CID: CH4_ITEMS}, bands=["ch4_p50_2km_a"])
        specs = list(s.iter_specs())
        assert [sp.name for sp in specs] == ["ch4_p50_2km_a"]

    def test_vars_columns(self):
        s = _stub({NDVI_CID: NDVI_ITEMS})
        sp = list(s.iter_specs())[0]
        assert sp.vars == {
            "collection": NDVI_CID,
            "asset": "ndvi_m_30m_s",
            "year": 2022,
            "gsd": 30.0,
            "epsg": 3035,
        }


class TestFromStac:
    def test_lazy_info(self, monkeypatch):
        monkeypatch.setattr(StacSource, "_root_url", lambda self: ROOT_URL)
        monkeypatch.setattr(
            StacSource, "_fetch_item_dicts", lambda self, cid: NDVI_ITEMS
        )
        r = RasterData.from_stac(
            url=CATALOG_URL, collections=NDVI_CID, datetime="2022-09-01/2022-11-30"
        )
        assert r.array is None  # lazy
        assert len(r.info) == 2

        for col in ["collection", "asset", "year", "gsd", "epsg"]:
            assert col in r.info.columns

        assert set(r.info["group"].unique()) == {NDVI_CID}
        assert set(r.info["asset"].unique()) == {"ndvi_m_30m_s"}
        assert set(r.info["year"].unique()) == {2022}

        # date_args populated for the (dated) collection group
        assert r.date_args[NDVI_CID]["date_style"] == "interval"
        assert r.date_args[NDVI_CID]["date_format"] == "%Y%m%d"

    def test_filter_by_collection_column(self, monkeypatch):
        monkeypatch.setattr(StacSource, "_root_url", lambda self: ROOT_URL)
        monkeypatch.setattr(
            StacSource,
            "_fetch_item_dicts",
            lambda self, cid: NDVI_ITEMS if cid == NDVI_CID else CH4_ITEMS,
        )
        r = RasterData.from_stac(url=CATALOG_URL, collections=[NDVI_CID, CH4_CID])
        assert set(r.get_groups()) == {NDVI_CID, CH4_CID}
        ndvi = r.filter(f"collection == '{NDVI_CID}'")
        assert set(ndvi.info["collection"].unique()) == {NDVI_CID}

    def test_filter_date_works(self, monkeypatch):
        monkeypatch.setattr(StacSource, "_root_url", lambda self: ROOT_URL)
        monkeypatch.setattr(
            StacSource, "_fetch_item_dicts", lambda self, cid: NDVI_ITEMS
        )
        r = RasterData.from_stac(url=CATALOG_URL, collections=NDVI_CID)
        sub = r.filter_date("2022-01-01", "2022-12-31")
        assert len(sub.info) == 2
        assert set(sub.info["year"].unique()) == {2022}


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class TestPagination:
    def test_follows_next_link(self, monkeypatch):
        page1 = {
            "features": [NDVI_ITEMS[0]],
            "links": [{"rel": "next",
                       "href": f"{ROOT_URL}/collections/{NDVI_CID}/items?limit=1&token=abc"}],
        }
        page2 = {"features": [NDVI_ITEMS[1]], "links": []}
        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append((url, params))
            return _FakeResp(page2 if "token" in url else page1)

        monkeypatch.setattr(src_mod.requests, "get", fake_get)
        s = StacSource(url=CATALOG_URL, collections=[NDVI_CID], limit=1)
        s._root_url = lambda: ROOT_URL

        items = s._fetch_item_dicts(NDVI_CID)
        assert len(items) == 2
        # first page: params carry limit (and no datetime); second: next href
        assert calls[0][0].endswith(f"/collections/{NDVI_CID}/items")
        assert calls[0][1] == {"limit": 1}
        assert "token" in calls[1][0]
        assert calls[1][1] is None

    def test_no_datetime_param_sent(self, monkeypatch):
        """The datetime filter is client-side: never sent to the server."""
        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append(params)
            return _FakeResp({"features": [], "links": []})

        monkeypatch.setattr(src_mod.requests, "get", fake_get)
        s = StacSource(url=CATALOG_URL, collections=[NDVI_CID],
                       datetime="2022-01-01/2022-12-31", bbox=[10, 45, 12, 47])
        s._root_url = lambda: ROOT_URL
        s._fetch_item_dicts(NDVI_CID)
        assert calls[0] == {"limit": 500, "bbox": "10,45,12,47"}

    def test_max_items_caps_fetch(self, monkeypatch):
        monkeypatch.setattr(src_mod.requests, "get",
                            lambda *a, **k: _FakeResp(
                                {"features": NDVI_ITEMS, "links": []}))
        s = StacSource(url=CATALOG_URL, collections=[NDVI_CID], max_items=2)
        s._root_url = lambda: ROOT_URL
        items = s._fetch_item_dicts(NDVI_CID)
        assert len(items) == 2


class TestParseRange:
    def test_range(self):
        a, b = StacSource._parse_range("2022-01-01/2022-12-31")
        assert a == datetime(2022, 1, 1)
        assert b == datetime(2022, 12, 31)

    def test_single_date(self):
        a, b = StacSource._parse_range("2022-01-01")
        assert a == b == datetime(2022, 1, 1)

    def test_open_end(self):
        a, b = StacSource._parse_range("2022-01-01/..")
        assert a == datetime(2022, 1, 1)
        assert b is None

    def test_none(self):
        assert StacSource._parse_range(None) == (None, None)

    def test_tuple(self):
        a, b = StacSource._parse_range(["2022-01-01", "2022-12-31"])
        assert a == datetime(2022, 1, 1)
        assert b == datetime(2022, 12, 31)


@pytest.mark.network
def test_network_ecodatacube():
    """Live smoke test (skipped by default; run with ``pytest -m network``)."""
    r = RasterData.from_stac(
        url=CATALOG_URL,
        collections=NDVI_CID,
        max_items=2,
    )
    assert r.array is None
    assert len(r.info) >= 1
    assert r.info["input_path"].str.endswith(".tif").all()
    assert set(r.info["group"].unique()) == {NDVI_CID}
    assert set(r.info["epsg"].unique()) == {3035}
