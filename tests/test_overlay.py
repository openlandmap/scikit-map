from pathlib import Path

import pandas as pd
import pytest

from skmap.overlay import (
    SpaceOverlay as so,
)
from skmap.overlay import (
    SpaceTimeOverlay as sto,
)
from skmap.overlay import (
    _ParallelOverlay as _po,
)


class TestParallelOverlay:
    def test__init__(self) -> None:
        po = _po(
            points_x=[],
            points_y=[],
            raster_files=[
                Path(
                    "../skmap/data/toy/static/elev.lowestmode_gedi.eml_mf_30m_s_20000101_20181231_nl_epsg.3035_v0.3.tif"
                )
            ],
            points_crs="4326",
        )
        assert po.verbose
        assert po.default_tile_id == ""
        assert po.raster_tiles == ""
