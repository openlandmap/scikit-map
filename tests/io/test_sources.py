"""Tests for the layer-source abstraction and the YAML driver.

The YAML driver expands ``{variable}`` placeholders in path templates into
concrete paths + dates and populates a lazy :class:`RasterData` (paths only,
no ``.read()``).  Fake paths are used throughout: RasterData is lazy, so no
files need to exist.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from skmap.io import RasterData
from skmap.io.sources import LayerSpec, TemplateExpander, YamlSource


YAML_TEXT = """
- layer: '{band}_test_{year}{start_month}_{year}{end_month}'
  path: '{base_path}/arco/{band}_test_{year}{start_month}_{year}{end_month}.tif'
  temporal_resolution: 'bimonthly'
  type: 'temporal'
  start_year: 2015
  end_year: 2016
  band: 'blue, green'
  start_month: '0101, 0301'
  end_month: '0228, 0430'

- layer: 'clm_{perc}'
  path: '{base_path}/clm/clm_{perc}.tif'
  temporal_resolution: 'longterm_or_static'
  type: 'common'
  perc: 'p50'

- layer: 'elev'
  path: '{base_path}/elev.tif'
  temporal_resolution: 'longterm_or_static'
  type: 'common'
"""


@pytest.fixture
def yaml_file(tmp_path):
    p = tmp_path / "layers.yaml"
    p.write_text(YAML_TEXT)
    return str(p)


def _specs_by_path(specs):
    return {s.path: s for s in specs}


class TestTemplateExpander:
    def test_bimonthly_expansion(self):
        entry = {
            "layer": "{band}_{year}{start_month}_{year}{end_month}",
            "path": "{base_path}/{band}_{year}{start_month}_{year}{end_month}.tif",
            "temporal_resolution": "bimonthly",
            "type": "temporal",
            "start_year": 2015,
            "end_year": 2016,
            "band": "blue, green",
            "start_month": "0101, 0301",
            "end_month": "0228, 0430",
        }
        specs = list(TemplateExpander().expand(entry, base_path="/data"))
        # 2 bands x 2 years x 2 month-pairs = 8
        assert len(specs) == 8

        by_path = _specs_by_path(specs)
        s = by_path["/data/blue_20150101_20150228.tif"]
        assert s.group == "2015"
        assert s.start_date == datetime(2015, 1, 1)
        assert s.end_date == datetime(2015, 2, 28)
        assert s.temporal is True
        assert s.vars == {
            "band": "blue",
            "year": 2015,
            "start_month": "0101",
            "end_month": "0228",
        }

        s = by_path["/data/green_20160301_20160430.tif"]
        assert s.group == "2016"
        assert s.start_date == datetime(2016, 3, 1)
        assert s.end_date == datetime(2016, 4, 30)

    def test_yearly_expansion(self):
        entry = {
            "path": "{base_path}/{band}_{year}.tif",
            "temporal_resolution": "yearly",
            "type": "temporal",
            "start_year": 2015,
            "end_year": 2016,
            "band": "ndvi",
        }
        specs = list(TemplateExpander().expand(entry, base_path="/data"))
        assert len(specs) == 2
        by_path = _specs_by_path(specs)
        assert by_path["/data/ndvi_2015.tif"].start_date == datetime(2015, 1, 1)
        assert by_path["/data/ndvi_2015.tif"].end_date == datetime(2015, 12, 31)
        assert by_path["/data/ndvi_2016.tif"].group == "2016"

    def test_monthly_expansion(self):
        entry = {
            "path": "{base_path}/{band}_{year}{start_month}.tif",
            "temporal_resolution": "monthly",
            "type": "temporal",
            "start_year": 2015,
            "end_year": 2015,
            "band": "ndvi",
            "start_month": "01, 02",
        }
        specs = list(TemplateExpander().expand(entry, base_path="/data"))
        assert len(specs) == 2
        by_path = _specs_by_path(specs)
        assert by_path["/data/ndvi_201501.tif"].start_date == datetime(2015, 1, 1)
        assert by_path["/data/ndvi_201501.tif"].end_date == datetime(2015, 1, 31)
        assert by_path["/data/ndvi_201502.tif"].end_date == datetime(2015, 2, 28)

    def test_static_expansion(self):
        entry = {
            "path": "{base_path}/elev.tif",
            "temporal_resolution": "longterm_or_static",
            "type": "common",
        }
        specs = list(TemplateExpander().expand(entry, base_path="/data"))
        assert len(specs) == 1
        s = specs[0]
        assert s.path == "/data/elev.tif"
        assert s.group == "common"
        assert s.start_date is None and s.end_date is None
        assert s.temporal is False

    def test_cross_year_end_month(self):
        entry = {
            "path": "{base_path}/{year}{start_month}_{year}{end_month}.tif",
            "temporal_resolution": "bimonthly",
            "type": "temporal",
            "start_year": 2015,
            "end_year": 2015,
            "start_month": "1101",
            "end_month": "0131",
        }
        specs = list(TemplateExpander().expand(entry, base_path="/data"))
        s = specs[0]
        assert s.start_date == datetime(2015, 11, 1)
        assert s.end_date == datetime(2016, 1, 31)

    def test_ignore_29feb_clamps(self):
        entry = {
            "path": "{base_path}/{year}{start_month}_{year}{end_month}.tif",
            "temporal_resolution": "bimonthly",
            "type": "temporal",
            "start_year": 2016,
            "end_year": 2016,
            "start_month": "0101",
            "end_month": "0229",
        }
        specs = list(TemplateExpander(ignore_29feb=True).expand(entry, base_path="/d"))
        assert specs[0].end_date == datetime(2016, 2, 28)

        specs = list(TemplateExpander(ignore_29feb=False).expand(entry, base_path="/d"))
        assert specs[0].end_date == datetime(2016, 2, 29)


class TestYamlSource:
    def test_iter_specs(self, yaml_file):
        src = YamlSource(yaml_file, base_path="/data")
        specs = list(src.iter_specs())
        # 8 temporal + 2 static = 10
        assert len(specs) == 10

        groups = sorted({s.group for s in specs})
        assert groups == ["2015", "2016", "common"]

        by_path = _specs_by_path(specs)
        assert "/data/arco/blue_test_20150101_20150228.tif" in by_path
        assert "/data/clm/clm_p50.tif" in by_path
        assert "/data/elev.tif" in by_path

    def test_base_path_required(self, tmp_path):
        p = tmp_path / "layers.yaml"
        p.write_text("- path: '{base_path}/elev.tif'\n  type: 'common'\n")
        with pytest.raises(ValueError, match="base_path"):
            YamlSource(str(p))

    def test_base_path_env(self, tmp_path, monkeypatch):
        p = tmp_path / "layers.yaml"
        p.write_text("- path: '{base_path}/elev.tif'\n  type: 'common'\n")
        monkeypatch.setenv("SKMAP_BASE_PATH", "/from-env")
        src = YamlSource(str(p))
        assert list(src.iter_specs())[0].path == "/from-env/elev.tif"


class TestFromYaml:
    def test_from_yaml_info(self, yaml_file):
        r = RasterData.from_yaml(yaml_file, base_path="/data")
        assert r.array is None  # lazy
        assert len(r.info) == 10

        # group = year for temporal, "common" for static
        assert set(r.info["group"].unique()) == {"2015", "2016", "common"}

        # extra variable columns present, NaN where absent
        for col in ["band", "year", "start_month", "end_month", "perc"]:
            assert col in r.info.columns

        blue_2015 = r.info[
            (r.info["band"] == "blue") & (r.info["year"] == 2015)
        ]
        assert len(blue_2015) == 2  # two month-pairs
        assert blue_2015["start_date"].iloc[0] == pd.Timestamp("2015-01-01")
        assert blue_2015["end_date"].iloc[0] == pd.Timestamp("2015-02-28")

        # static rows: no dates, no band/year
        elev = r.info[r.info["name"] == "elev"]
        assert len(elev) == 1
        assert elev["group"].iloc[0] == "common"
        assert pd.isna(elev["start_date"].iloc[0])
        assert pd.isna(elev["band"].iloc[0])

        # perc column only populated for the clm layer
        clm = r.info[r.info["name"] == "clm_p50"]
        assert clm["perc"].iloc[0] == "p50"

    def test_from_yaml_date_args(self, yaml_file):
        r = RasterData.from_yaml(
            yaml_file, base_path="/data", date_format="%Y-%m-%d", ignore_29feb=False
        )
        for g in ["2015", "2016"]:
            assert r.date_args[g]["date_format"] == "%Y-%m-%d"
            assert r.date_args[g]["ignore_29feb"] is False

    def test_to_rasterdata_roundtrip(self, yaml_file):
        src = YamlSource(yaml_file, base_path="/data")
        r = src.to_rasterdata(backend="numpy")
        assert isinstance(r, RasterData)
        assert r.array is None
        assert len(r.info) == 10
        assert r.get_groups() == ["2015", "2016"]

    def test_filter_by_extra_column(self, yaml_file):
        """Extra variable columns are queryable via filter()."""
        r = RasterData.from_yaml(yaml_file, base_path="/data")
        blue = r.filter("band == 'blue'")
        assert len(blue.info) == 4  # 2 years x 2 month-pairs
        assert set(blue.info["band"].unique()) == {"blue"}


class TestIntervalExpansion:
    ENTRY = {
        "name": "{band}_{season}",
        "path": "{base_path}/ndvi/{band}_s_{dt}.tif",
        "temporal_resolution": "interval",
        "type": "temporal",
        "start_date": "20141202",
        "end_date": "20201201",
        "date_unit": "days",
        "date_step": "109, 96, 80, 80",
        "ignore_29feb": True,
        "band": "ndvi",
        "variant": "filled",
        "season": "winter, spring, summer, fall",
    }

    def test_interval_expands_24_specs(self):
        specs = list(TemplateExpander().expand(self.ENTRY, base_path="/d"))
        assert len(specs) == 24
        s = specs[0]
        assert s.path == "/d/ndvi/ndvi_s_20141202_20150320.tif"
        assert s.start_date == datetime(2014, 12, 2)
        assert s.end_date == datetime(2015, 3, 20)
        assert s.group == "2014"
        assert s.temporal is True

    def test_interval_name_template_is_year_agnostic(self):
        specs = list(TemplateExpander().expand(self.ENTRY, base_path="/d"))
        # the same season name appears in every year (not date-specific)
        assert specs[0].name == "ndvi_winter"
        winter_names = {s.name for s in specs if s.vars["season"] == "winter"}
        assert winter_names == {"ndvi_winter"}

    def test_interval_cycling_season(self):
        specs = list(TemplateExpander().expand(self.ENTRY, base_path="/d"))
        seasons = [s.vars["season"] for s in specs[:8]]
        # season cycles with the 4 step values: winter, spring, summer, fall, ...
        assert seasons == [
            "winter", "spring", "summer", "fall",
            "winter", "spring", "summer", "fall",
        ]

    def test_interval_variant_column_not_in_path(self):
        """A scalar field not referenced in the path still becomes a column."""
        specs = list(TemplateExpander().expand(self.ENTRY, base_path="/d"))
        assert all(s.vars["variant"] == "filled" for s in specs)
        assert all(s.vars["band"] == "ndvi" for s in specs)

    def test_interval_cross_product_band(self):
        """A list variable with a length != date_step cross-products."""
        entry = dict(self.ENTRY)
        entry["band"] = "ndvi, swir1"  # length 2 != 4 -> cross-product
        entry["name"] = "{band}_{season}"
        entry["path"] = "{base_path}/{band}/{band}_s_{dt}.tif"
        specs = list(TemplateExpander().expand(entry, base_path="/d"))
        assert len(specs) == 48  # 2 bands x 24 intervals
        assert {s.name.split("_")[0] for s in specs} == {"ndvi", "swir1"}


class TestToyLayersYaml:
    def test_loads_all_toy_data(self):
        from skmap.data import toy

        r = RasterData.from_yaml(str(toy.LAYERS_YAML), base_path=str(toy.DATA_DIR))
        assert r.array is None  # lazy
        assert len(r.info) == 74  # 24+24 ndvi + 24 swir1 + 2 static
        assert set(r.get_groups()) == {str(y) for y in range(2014, 2021)}

    def test_season_names_are_year_agnostic(self):
        from skmap.data import toy

        r = RasterData.from_yaml(str(toy.LAYERS_YAML), base_path=str(toy.DATA_DIR))
        ndvi = r.info[r.info["band"] == "ndvi"]
        assert set(ndvi["name"].unique()) == {
            "ndvi_winter", "ndvi_spring", "ndvi_summer", "ndvi_fall"
        }

    def test_filled_filter_excludes_gappy(self):
        from skmap.data import toy

        r = RasterData.from_yaml(str(toy.LAYERS_YAML), base_path=str(toy.DATA_DIR))
        filled = r.filter("variant != 'gappy'")
        assert len(filled.info) == 50  # 24 ndvi filled + 24 swir1 + 2 static
        assert set(filled.info["band"].dropna().unique()) == {"ndvi", "swir1"}

    def test_unique_names_within_year(self):
        """Names are unique within each year group (SpaceOverlay requirement)."""
        from skmap.data import toy

        r = RasterData.from_yaml(str(toy.LAYERS_YAML), base_path=str(toy.DATA_DIR))
        filled = r.filter("variant != 'gappy'")
        for year in ["2015", "2016", "2019"]:
            sub = filled.filter(f"group == '{year}'")
            assert sub.info["name"].nunique() == len(sub.info)
