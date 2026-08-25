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

    YEARS = [2015, 2016, 2017, 2018, 2019, 2020]

    @classmethod
    def _swir1_path(cls, year: int) -> str:
        fname = (
            f"swir1_landsat.ard1_p50_30m_s_{year}0321_{year}0624"
            "_nl_epsg.3035_v20230720.tif"
        )
        return (cls.SWIR1_DIR / fname).as_posix()

    @classmethod
    def _temporal_rdata(cls, years=None) -> RasterData:
        years = years if years is not None else cls.YEARS
        raster_files = {}
        names = []
        for year in years:
            raster_files[str(year)] = [cls._swir1_path(year)]
            names.append("swir1")
        rdata = RasterData(raster_files)
        rdata.info["name"] = names
        return rdata

    def test_groups_points_by_year(self) -> None:
        pts = gpd.read_file(self.SAMPLES)
        rdata = self._temporal_rdata()
        sto = SpaceTimeOverlay(
            points=pts, col_date="date", rasterdata=rdata, raster_tiles=None
        )
        for year in self.YEARS:
            expected = pts[pts["date"].dt.year == year]
            assert len(sto.year_points[str(year)]) == len(expected)
        # the caller's rdata must not be mutated
        assert rdata.get_groups() == [str(y) for y in self.YEARS]

    def test_run_returns_dataframe(self) -> None:
        pts = gpd.read_file(self.SAMPLES)
        rdata = self._temporal_rdata()
        sto = SpaceTimeOverlay(
            points=pts, col_date="date", rasterdata=rdata, raster_tiles=None
        )
        res = sto.run(max_ram_mb=512, out_file_name=None)
        assert res is not None
        assert res.shape[0] == len(pts)
        assert "swir1" in res.columns
        assert np.isfinite(res["swir1"]).all()

    def test_year_alignment(self) -> None:
        pts = gpd.read_file(self.SAMPLES)
        rdata = self._temporal_rdata()
        sto = SpaceTimeOverlay(
            points=pts, col_date="date", rasterdata=rdata, raster_tiles=None
        )
        res = sto.run(max_ram_mb=512, out_file_name=None)

        pts_2019 = pts[pts["date"].dt.year == 2019].reset_index(drop=True)
        rdata_2019 = self._temporal_rdata(years=[2019])
        so_2019 = SpaceOverlay(points=pts_2019, rasterdata=rdata_2019)
        res_2019 = so_2019.run(max_ram_mb=512, out_file_name=None)

        res_2019_st = res[res["date"].dt.year == 2019].reset_index(drop=True)
        assert np.allclose(
            res_2019_st["swir1"].to_numpy(),
            res_2019["swir1"].to_numpy(),
            equal_nan=True,
        )
