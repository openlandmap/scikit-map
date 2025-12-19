import os
from pathlib import Path, PosixPath

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from skmap.catalog import DataCatalog
from skmap.overlay import (
    SpaceOverlay,
    _ParallelOverlay,
)

REPO_ROOT = Path(__file__).parent.parent
TOY_DIR = REPO_ROOT / "skmap/data/toy"
elev_name = "elev.lowestmode_gedi.eml_mf_30m_s_20000101_20181231_nl_epsg.3035_v0.3"
elev_file = TOY_DIR / "static" / (elev_name + ".tif")
p1_x = 4021600
p1_y = 3216130
p2_x = 4024200
p2_y = 3215420
pts_crs = "3035"
points_x: list[int] = [p1_x, p2_x]
points_y: list[int] = [p1_y, p2_y]


class TestParallelOverlay:
    def test__init__static(self) -> None:
        # initializing the overlay already does quite a lot of processing
        # elev_file serves as the reference

        # points in the same crs that lie within the tile
        # check with `gdalinfo file` what the coordinates and crs are
        po = _ParallelOverlay(
            points_x,
            points_y,
            raster_files=[elev_file],
            points_crs=pts_crs,
        )
        assert po.verbose
        assert po.default_tile_id == ""
        assert po.tile_id_col == "tile_id"
        assert po.raster_tiles is None
        assert po.raster_files == [elev_file]
        assert po.layers.to_dict() == {
            "name": {0: elev_name},
            "path": {0: PosixPath(elev_file)},
            "nodata": {0: -9999.0},
            "block_height": {0: 16},
            "block_width": {0: 16},
            "group": {0: "8c36693f22356214e61afb7002635270"},
        }
        assert po.query_pixels["8c36693f22356214e61afb7002635270"].to_dict() == {
            "block_id": {0: 50, 1: 71},
            "x": {0: p1_x, 1: p2_x},
            "y": {0: p1_y, 1: p2_y},
            "block_col_off": {0: 32, 1: 112},
            "block_row_off": {0: 48, 1: 64},
            "block_width": {0: 16, 1: 16},
            "block_height": {0: 16, 1: 16},
            "sample_col": {0: 1, 1: 8},
            "sample_row": {0: 8, 1: 15},
        }
        assert len(po.query_pixels) == 1


class TestSpaceOverlay:
    def test__init__elev(self) -> None:
        catalog = DataCatalog.create_catalog(
            catalog_def=pd.DataFrame(
                {
                    "layer_name": ["elev"],
                    "path": ["{base_path}/" + elev_name + ".tif"],
                    "type": ["common"],
                }
            ),
            years=[2020],
            base_path=str(TOY_DIR / "static"),
        )
        so = SpaceOverlay(
            points=gpd.GeoDataFrame(
                geometry=[Point(p1_x, p1_y), Point(p2_x, p2_y)], crs=pts_crs
            ),
            catalog=catalog,
        )

        # assert vars(so).keys() == []
        assert so.verbose
        assert so.catalog == catalog
        assert so.layer_paths == [elev_file.as_posix()]
        assert so.layer_idxs == [0]
        assert so.layer_names == ["elev"]
        assert so.pts.to_dict() == {"geometry": {0: Point(4021600, 3216130), 1: Point(4024200, 3215420)}}
        assert so.n_threads == os.cpu_count()

        po = so.parallelOverlay
        assert po.verbose
        assert po.default_tile_id == ""
        assert po.tile_id_col == "tile_id"
        assert po.raster_tiles is None
        assert po.raster_files == [elev_file.as_posix()]
        assert po.layers.to_dict() == {
            "name": {0: elev_name},
            "path": {0: elev_file.as_posix()},
            "nodata": {0: -9999.0},
            "block_height": {0: 16},
            "block_width": {0: 16},
            "group": {0: "8c36693f22356214e61afb7002635270"},
        }
        assert po.query_pixels["8c36693f22356214e61afb7002635270"].to_dict() == {
            "block_id": {0: 50, 1: 71},
            "x": {0: p1_x, 1: p2_x},
            "y": {0: p1_y, 1: p2_y},
            "block_col_off": {0: 32, 1: 112},
            "block_row_off": {0: 48, 1: 64},
            "block_width": {0: 16, 1: 16},
            "block_height": {0: 16, 1: 16},
            "sample_col": {0: 1, 1: 8},
            "sample_row": {0: 8, 1: 15},
        }
        assert len(po.query_pixels) == 1
