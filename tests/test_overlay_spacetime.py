from pathlib import Path

import geopandas as gpd
import numpy as np

from skmap.io import RasterData
from skmap.overlay import SpaceOverlay, SpaceTimeOverlay


class TestSpaceTimeOverlay:
    REPO_ROOT = Path(__file__).parent.parent
    TOY_DIR = REPO_ROOT / "skmap/data/toy"
    SAMPLES = TOY_DIR / "samples" / "samples.gpkg"
    SWIR1_DIR = TOY_DIR / "swir1"

    ELEV = (
        TOY_DIR / "static"
        / "elev.lowestmode_gedi.eml_mf_30m_s_20000101_20181231_nl_epsg.3035_v0.3.tif"
    ).as_posix()

    @classmethod
    def _swir1_template(cls) -> str:
        # the first swir1 file carries the 20141202_20150320 period; turn the
        # period into the {dt} placeholder so timespan() can expand it
        first = sorted(cls.SWIR1_DIR.glob("*.tif"))[0]
        return str(first).replace("20141202_20150320", "{dt}")

    @classmethod
    def _temporal_rdata(cls) -> RasterData:
        """A lazy RasterData with 24 dated swir1 bands + 1 static elev layer."""
        rdata = RasterData(
            {"swir1": cls._swir1_template(), "static": cls.ELEV}
        ).timespan("20141202", "20201201", "days", [109, 96, 80, 80])
        assert rdata.array is None  # must stay lazy
        return rdata

    def test_groups_points_by_year(self) -> None:
        pts = gpd.read_file(self.SAMPLES)
        rdata = self._temporal_rdata()
        sto = SpaceTimeOverlay(
            points=pts, col_date="date", rasterdata=rdata, raster_tiles=None
        )
        for year in range(2015, 2021):
            label = f"{year}-01-01..{year}-12-31"
            expected = pts[pts["date"].dt.year == year]
            assert len(sto.range_points[label]) == len(expected)
        # the caller's rdata must not be mutated
        assert rdata.array is None

    def test_run_returns_dataframe(self) -> None:
        pts = gpd.read_file(self.SAMPLES)
        rdata = self._temporal_rdata()
        sto = SpaceTimeOverlay(
            points=pts, col_date="date", rasterdata=rdata, raster_tiles=None
        )
        res = sto.run(max_ram_mb=512, out_file_name=None)
        assert res is not None
        assert res.shape[0] == len(pts)
        # the static elev layer must appear in the concatenated output
        assert any(c.startswith("elev") for c in res.columns)

    def test_static_layer_in_every_range(self) -> None:
        """The static (non-dated) layer is included in every date slice."""
        pts = gpd.read_file(self.SAMPLES)
        rdata = self._temporal_rdata()
        sto = SpaceTimeOverlay(
            points=pts, col_date="date", rasterdata=rdata, raster_tiles=None
        )
        for label, so in sto.overlay_objs.items():
            names = so.layer_names
            assert any(n.startswith("elev") for n in names), (
                f"static elev missing from slice {label}"
            )

    def test_explicit_date_ranges(self) -> None:
        """Explicit date_ranges select only the temporal layers in range."""
        pts = gpd.read_file(self.SAMPLES)
        rdata = self._temporal_rdata()
        sto = SpaceTimeOverlay(
            points=pts,
            col_date="date",
            rasterdata=rdata,
            date_ranges=[("2018-01-01", "2018-06-30")],
            raster_tiles=None,
        )
        assert sto.date_ranges == [("2018-01-01", "2018-06-30")]
        res = sto.run(max_ram_mb=512, out_file_name=None)
        assert res.shape[0] == len(pts[pts["date"].dt.year == 2018])
        # static elev present, swir1 bands only from the first half of 2018
        assert any(c.startswith("elev") for c in res.columns)
        swir_cols = [c for c in res.columns if c.startswith("swir1")]
        assert len(swir_cols) > 0
        # every swir1 column should start within the first half of 2018
        assert all("2018" in c for c in swir_cols)

    def test_year_alignment(self) -> None:
        pts = gpd.read_file(self.SAMPLES)
        rdata = self._temporal_rdata()
        sto = SpaceTimeOverlay(
            points=pts, col_date="date", rasterdata=rdata, raster_tiles=None
        )
        res = sto.run(max_ram_mb=512, out_file_name=None)

        pts_2019 = pts[pts["date"].dt.year == 2019].reset_index(drop=True)
        rdata_2019 = self._temporal_rdata()
        so_2019 = SpaceOverlay(
            points=pts_2019,
            rasterdata=rdata_2019.filter_date(
                "2019-01-01", "2019-12-31", include_non_temporal=True
            ),
        )
        res_2019 = so_2019.run(max_ram_mb=512, out_file_name=None)

        res_2019_st = res[res["date"].dt.year == 2019].reset_index(drop=True)
        swir_cols = [c for c in res_2019.columns if c.startswith("swir1")]
        for col in swir_cols:
            assert np.allclose(
                res_2019_st[col].to_numpy(),
                res_2019[col].to_numpy(),
                equal_nan=True,
            )

    def test_skips_range_with_no_layers(self, capsys):
        """A year with no matching layers is skipped, not crashed.

        Regression for KeyError: 'group' when a date range (here 2020) has
        points but the catalogue has no layers covering it.
        """
        pts = gpd.read_file(self.SAMPLES)
        # temporal-only swir1 (no static) ending before 2020 -> the 2020 range
        # has points but zero layers.
        rdata = RasterData(
            {"swir1": self._swir1_template()}
        ).timespan("20141202", "20190901", "days", [109, 96, 80, 80])
        sto = SpaceTimeOverlay(
            points=pts, col_date="date", rasterdata=rdata, raster_tiles=None
        )
        # 2020 has points but no layers -> skipped (not built, not crashed)
        assert "2020-01-01..2020-12-31" not in sto.overlay_objs
        res = sto.run(max_ram_mb=512, out_file_name=None)
        assert 2020 not in set(res["date"].dt.year)
        out = capsys.readouterr().out
        assert "No raster layers in [2020-01-01, 2020-12-31]" in out