import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from skmap.catalog import DataCatalog
from skmap.overlay import (
    SpaceOverlay,
    _ParallelOverlay,
)


class TestOverlayBase:
    REPO_ROOT = Path(__file__).parent.parent
    TOY_DIR = REPO_ROOT / "skmap/data/toy"
    ELEV_NAME = "elev.lowestmode_gedi.eml_mf_30m_s_20000101_20181231_nl_epsg.3035_v0.3"
    ELEV_FILE = TOY_DIR / "static" / (ELEV_NAME + ".tif")

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
            "nodata": {0: -9999.0},
            "block_height": {0: 16},
            "block_width": {0: 16},
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
        self.catalog = DataCatalog.create_catalog(
            catalog_def=pd.DataFrame(
                {
                    "layer_name": ["elev"],
                    "path": ["{base_path}/" + self.ELEV_NAME + ".tif"],
                    "type": ["common"],
                }
            ),
            years=[2020],
            base_path=str(self.TOY_DIR / "static"),
        )
        self.so = SpaceOverlay(
            points=gpd.GeoDataFrame(
                geometry=[Point(x, y) for x, y in zip(self.PTS_X, self.PTS_Y)],
                crs=self.PTS_CRS,
            ),
            catalog=self.catalog,
        )

    def test__init__defaults(self, po: _ParallelOverlay) -> None:
        so = self.so

        # assert vars(so).keys() == []
        assert so.verbose
        assert so.catalog == self.catalog
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
        for k,v in spo.query_pixels.items():
            assert po.query_pixels[k].equals(v)
