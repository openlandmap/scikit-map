from pathlib import Path

import pandas as pd

from skmap.overlay import (
    _ParallelOverlay as _po,
)

REPO_ROOT = Path(__file__).parent.parent


class TestParallelOverlay:
    def test__init__(self) -> None:
        file = (
            Path(REPO_ROOT)
            / "skmap/data/toy/static/elev.lowestmode_gedi.eml_mf_30m_s_20000101_20181231_nl_epsg.3035_v0.3.tif"
        )
        points_x = [4021600, 4024200]
        points_y = [3216130, 3215420]
        po = _po(
            points_x,
            points_y,
            raster_files=[file],
            points_crs="3035",
        )
        assert po.verbose
        assert po.default_tile_id == ""
        assert po.raster_tiles is None
        assert (
            (
                po.query_pixels["8c36693f22356214e61afb7002635270"]
                == pd.DataFrame(
                    {
                        "block_id": [50, 71],
                        "x": [4021600.0, 4024200.0],
                        "y": [3216130.0, 3215420.0],
                        "block_col_off": [32, 112],
                        "block_row_off": [48, 64],
                        "block_width": [16, 16],
                        "block_height": [16, 16],
                        "sample_col": [1, 8],
                        "sample_row": [8, 15],
                    }
                )
            )
            .all()
            .all()
        )
        assert len(po.query_pixels) == 1
        # assert False
