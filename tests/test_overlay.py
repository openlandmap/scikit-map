import contextlib
import io
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from skmap.io import RasterData
from skmap.overlay import (
    SpaceOverlay,
    _ParallelOverlay,
)


class TestOverlayBase:
    REPO_ROOT = Path(__file__).parent.parent
    TOY_DIR = REPO_ROOT / "skmap/data/toy"
    ELEV_NAME = "elev.lowestmode_gedi.eml_mf_30m_s_20000101_20181231_nl_epsg.3035_v0.3"
    ELEV_FILE = TOY_DIR / "static" / (ELEV_NAME + ".tif")
    SLOPE_NAME = "slope.percent_gedi.eml_m_30m_s_20000101_20181231_nl_epsg.3035_v0.3"
    SLOPE_FILE = TOY_DIR / "static" / (SLOPE_NAME + ".tif")

    PTS_X = [4021600, 4024200]
    PTS_Y = [3216130, 3215420]
    PTS_CRS = "3035"

    @pytest.fixture
    def po(self) -> _ParallelOverlay:
        return _ParallelOverlay(
            self.PTS_X,
            self.PTS_Y,
            raster_files=[self.ELEV_FILE.as_posix()],
            points_crs=self.PTS_CRS,
        )


class TestParallelOverlay(TestOverlayBase):
    def test__init__defaults(self, po: _ParallelOverlay) -> None:
        # initializing the overlay already does quite a lot of processing
        # elev_file serves as the reference

        # points in the same crs that lie within the tile
        # check with `gdalinfo file` what the coordinates and crs are
        assert po.verbose
        assert po.default_tile_id == ""
        assert po.tile_id_col == "tile_id"
        assert po.raster_tiles is None

    def test__init__files(self, po: _ParallelOverlay) -> None:
        assert po.raster_files == [self.ELEV_FILE.as_posix()]
        assert po.layers.to_dict() == {
            "name": {0: self.ELEV_NAME},
            "path": {0: self.ELEV_FILE.as_posix()},
            "group": {0: "8c36693f22356214e61afb7002635270"},
        }
        assert po.query_pixels["8c36693f22356214e61afb7002635270"].to_dict() == {
            "block_id": {0: 50, 1: 71},
            "x": {0: self.PTS_X[0], 1: self.PTS_X[1]},
            "y": {0: self.PTS_Y[0], 1: self.PTS_Y[1]},
            "block_col_off": {0: 32, 1: 112},
            "block_row_off": {0: 48, 1: 64},
            "block_width": {0: 16, 1: 16},
            "block_height": {0: 16, 1: 16},
            "sample_col": {0: 1, 1: 8},
            "sample_row": {0: 8, 1: 15},
        }
        assert len(po.query_pixels) == 1


class TestSpaceOverlay(TestOverlayBase):
    def setup_method(self) -> None:
        # lazy RasterData: paths only, no .read()
        self.rdata = RasterData({"common": [self.ELEV_FILE.as_posix()]})
        self.rdata.info["name"] = ["elev"]
        assert self.rdata.array is None  # must stay lazy
        self.so = SpaceOverlay(
            points=gpd.GeoDataFrame(
                geometry=[Point(x, y) for x, y in zip(self.PTS_X, self.PTS_Y)],
                crs=self.PTS_CRS,
            ),
            rasterdata=self.rdata,
        )

    def test__init__defaults(self, po: _ParallelOverlay) -> None:
        so = self.so

        assert so.verbose
        assert so.runners == []
        assert so.layer_paths == [self.ELEV_FILE.as_posix()]
        assert so.layer_idxs == [0]
        assert so.layer_names == ["elev"]
        assert so.pts.to_dict() == {
            "geometry": {0: Point(4021600, 3216130), 1: Point(4024200, 3215420)}
        }
        assert so.n_threads == os.cpu_count()

    def test__init__paralleloverlay(self, po: _ParallelOverlay) -> None:
        spo = self.so.parallelOverlay
        assert spo.verbose
        assert spo.default_tile_id == po.default_tile_id
        assert spo.tile_id_col == po.tile_id_col
        assert spo.raster_tiles is None
        assert spo.raster_files == po.raster_files
        assert spo.layers.equals(po.layers)
        for k, v in spo.query_pixels.items():
            assert po.query_pixels[k].equals(v)

    def test_run(self) -> None:
        res = self.so.run(max_ram_mb=512, out_file_name=None)
        assert res.shape == (2, 3)
        assert list(res.columns) == ["lon", "lat", "elev"]
        assert res["elev"].tolist() == [70.0, 284.0]

    def test_run_lazy_rasterdata(self) -> None:
        """SpaceOverlay works on a lazy (unread) RasterData: no .read() call."""
        rdata = RasterData(
            {
                "common": [
                    self.ELEV_FILE.as_posix(),
                    self.SLOPE_FILE.as_posix(),
                ]
            }
        )
        rdata.info["name"] = ["elev", "slope"]
        assert rdata.array is None

        so = SpaceOverlay(
            points=gpd.GeoDataFrame(
                geometry=[Point(x, y) for x, y in zip(self.PTS_X, self.PTS_Y)],
                crs=self.PTS_CRS,
            ),
            rasterdata=rdata,
            verbose=False,
        )
        res = so.run(max_ram_mb=512, out_file_name=None)
        assert res.shape == (2, 4)
        assert list(res.columns) == ["lon", "lat", "elev", "slope"]
        assert res["elev"].tolist() == [70.0, 284.0]
        assert res["slope"].tolist() == [31.0, 59.0]

    def test_run_multiple_layers_same_group(self) -> None:
        # Regression: a group with more than one layer used to return NaN for
        # every layer except the last one (extractOverlay hash-map collapse).
        rdata = RasterData(
            {
                "common": [
                    self.ELEV_FILE.as_posix(),
                    self.SLOPE_FILE.as_posix(),
                ]
            }
        )
        rdata.info["name"] = ["elev", "slope"]
        so = SpaceOverlay(
            points=gpd.GeoDataFrame(
                geometry=[Point(x, y) for x, y in zip(self.PTS_X, self.PTS_Y)],
                crs=self.PTS_CRS,
            ),
            rasterdata=rdata,
        )
        res = so.run(max_ram_mb=512, out_file_name=None)
        assert list(res.columns) == ["lon", "lat", "elev", "slope"]
        assert res["elev"].tolist() == [70.0, 284.0]
        assert res["slope"].tolist() == [31.0, 59.0]

    def test_run_with_whale_runners(self) -> None:
        """Whale runners compute derived columns on the sampled points."""
        from skmap.io import process

        rdata = RasterData(
            {
                "common": [
                    self.ELEV_FILE.as_posix(),
                    self.SLOPE_FILE.as_posix(),
                ]
            }
        )
        rdata.info["name"] = ["elev", "slope"]
        nd = process.NormalizedDifference("elev", "slope")
        nd.outname = "nd"
        lat = process.GetLatitude()
        lat.outname = "lat_deg"
        so = SpaceOverlay(
            points=gpd.GeoDataFrame(
                geometry=[Point(x, y) for x, y in zip(self.PTS_X, self.PTS_Y)],
                crs=self.PTS_CRS,
            ),
            rasterdata=rdata,
            runners=[nd, lat],
            verbose=False,
        )
        res = so.run(max_ram_mb=512, out_file_name=None)
        assert list(res.columns) == ["lon", "lat", "elev", "slope", "nd", "lat_deg"]
        # nd = round((elev - slope) / (elev + slope))
        assert res["nd"].tolist() == [0.0, 1.0]
        # GetLatitude uses the point y coordinate (same as the old lat_info)
        assert res["lat_deg"].tolist() == [self.PTS_Y[0], self.PTS_Y[1]]

    def test_out_of_extent_dropped(self) -> None:
        so = SpaceOverlay(
            points=gpd.GeoDataFrame(
                geometry=[
                    Point(self.PTS_X[0], self.PTS_Y[0]),
                    Point(0, 0),
                ],
                crs=self.PTS_CRS,
            ),
            rasterdata=self.rdata,
        )
        assert so.pts.shape[0] == 1

    def test_local_no_http_errors(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            SpaceOverlay(
                points=gpd.GeoDataFrame(
                    geometry=[
                        Point(x, y) for x, y in zip(self.PTS_X, self.PTS_Y)
                    ],
                    crs=self.PTS_CRS,
                ),
                rasterdata=self.rdata,
            )
        assert "Error checking URL" not in buf.getvalue()
