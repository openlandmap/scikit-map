from pathlib import Path

import pandas as pd
import pytest

from skmap.catalog import DataCatalog as c


class TestDataCatalog:
    def test_path(self):
        print(Path(".").absolute())
        assert Path(
            "skmap/data/toy/ndvi/gappy/ndvi_landsat.ard1_p50_30m_s_20141202_20150320_nl_epsg.3035_v20230720.tif"
        ).exists()

    def test__get_features_names(self):
        assert c._get_feature_names(
            {"a": {"hello": "hii"}, "b": {"world": "Gaia"}}
        ) == ["hello", "world"]

    def test_get_whales(self):
        assert c.get_whales(
            {
                "common": {
                    "example": {
                        "idx": 0,
                        "path": "/whale/expr",
                    }
                },
                "temporal": {
                    "layer": {
                        "idx": 1,
                        "path": "/not_whale/expr",
                    }
                },
            }
        ) == (["/whale/expr"], ["common"], ["example"])

    def test_create_catalog_minimal(self):
        catalog = c.create_catalog(
            catalog_def=pd.DataFrame(
                {
                    "layer_name": ["layer", "example"],
                    "path": ["wrong/file.tif", "sample/path.tif"],
                    "type": ["common", "common"],
                }
            ),
            years=[2020],
            base_path=str(Path(".").absolute()),
        )
        print(f"{catalog.data=!r}", repr(catalog), dir(catalog.data), range(2014, 2020))
        assert isinstance(catalog, c)
        assert catalog.data_size == 2
        assert catalog.data == {
            "common": {
                "layer": {"idx": 1, "path": "wrong/file.tif"},
                "example": {"idx": 0, "path": "sample/path.tif"},
            }
        }
        assert catalog.get_groups() == ["common"]
        assert catalog.get_otf_idx() == {}
        assert catalog.get_unrolled_catalog() == (
            ["example", "layer"],
            ["sample/path.tif", "wrong/file.tif"],
            [0, 1],
        )
        assert catalog.get_feature_names() == ["example", "layer"]

    def test_create_catalog_year(self):
        catalog = c.create_catalog(
            pd.DataFrame(
                {
                    "layer_name": ["layer_{year}"],
                    "path": ["path/to/param_{year}.tif"],
                    "type": ["temporal"],
                    "start_year": ["2014"],
                    "end_year": ["2015"],
                }
            ),
            years=[2014, 2015, 2016],
            base_path=str(Path(".").absolute()),
        )
        assert isinstance(catalog, c)
        assert catalog.data_size == 3
        assert catalog.data == {
            "2014": {
                "layer_YYYY": {
                    "path": "path/to/param_2014.tif",
                    "idx": 0,
                }
            },
            "2015": {
                "layer_YYYY": {
                    "path": "path/to/param_2015.tif",
                    "idx": 1,
                }
            },
            "2016": {
                "layer_YYYY": {
                    "path": "path/to/param_2015.tif",
                    "idx": 2,
                }
            },
        }
        assert catalog.get_groups() == ["2014", "2015", "2016"]
        assert catalog.get_otf_idx() == {}
        assert catalog.get_unrolled_catalog() == (
            [
                "layer_YYYY",
                "layer_YYYY",
                "layer_YYYY",
            ],
            [
                "path/to/param_2014.tif",
                "path/to/param_2015.tif",
                "path/to/param_2015.tif",
            ],
            [0, 1, 2],
        )
        assert catalog.get_feature_names() == ["layer_YYYY"]

    def test_create_catalog_year_plusminus(self):
        catalog = c.create_catalog(
            pd.DataFrame(
                {
                    "layer_name": ["layer_{year_minus_one}-{year_plus_one}"],
                    "path": ["path/to/param_{year_minus_one}-{year_plus_one}.tif"],
                    "type": ["temporal"],
                    "start_year": ["2014"],
                    "end_year": ["2015"],
                }
            ),
            years=[2014, 2015, 2016],
            base_path=str(Path(".").absolute()),
        )
        assert isinstance(catalog, c)
        assert catalog.data_size == 3
        assert catalog.data == {
            "2014": {
                "layer_YYMO-YYPO": {
                    "path": "path/to/param_2013-2015.tif",
                    "idx": 0,
                }
            },
            "2015": {
                "layer_YYMO-YYPO": {
                    "path": "path/to/param_2014-2016.tif",
                    "idx": 1,
                }
            },
            "2016": {
                "layer_YYMO-YYPO": {
                    "path": "path/to/param_2014-2016.tif",
                    "idx": 2,
                }
            },
        }
        assert catalog.get_groups() == ["2014", "2015", "2016"]
        assert catalog.get_otf_idx() == {}
        assert catalog.get_unrolled_catalog() == (
            [
                "layer_YYMO-YYPO",
                "layer_YYMO-YYPO",
                "layer_YYMO-YYPO",
            ],
            [
                "path/to/param_2013-2015.tif",
                "path/to/param_2014-2016.tif",
                "path/to/param_2014-2016.tif",
            ],
            [0, 1, 2],
        )
        assert catalog.get_feature_names() == ["layer_YYMO-YYPO"]

    def test_create_catalog_monthly(self):
        catalog = c.create_catalog(
            pd.DataFrame(
                {
                    "layer_name": ["layer_{year}{start_month}-{year}{end_month}"],
                    "path": ["path/to/param_{year}{start_month}-{year}{end_month}.tif"],
                    "type": ["temporal"],
                    "start_year": ["2014"],
                    "end_year": ["2015"],
                    "start_month": ["01,06"],
                    "end_month": ["05,12"],
                }
            ),
            years=[2014, 2015, 2016],
            base_path=str(Path(".").absolute()),
        )
        assert isinstance(catalog, c)
        assert catalog.data_size == 6
        assert catalog.data == {
            "2014": {
                "layer_YYYY01-YYYY05": {
                    "path": "path/to/param_201401-201405.tif",
                    "idx": 0,
                },
                "layer_YYYY06-YYYY12": {
                    "path": "path/to/param_201406-201412.tif",
                    "idx": 1,
                },
            },
            "2015": {
                "layer_YYYY01-YYYY05": {
                    "path": "path/to/param_201501-201505.tif",
                    "idx": 2,
                },
                "layer_YYYY06-YYYY12": {
                    "path": "path/to/param_201506-201512.tif",
                    "idx": 3,
                },
            },
            "2016": {
                "layer_YYYY01-YYYY05": {
                    "path": "path/to/param_201501-201505.tif",
                    "idx": 4,
                },
                "layer_YYYY06-YYYY12": {
                    "path": "path/to/param_201506-201512.tif",
                    "idx": 5,
                },
            },
        }

        assert catalog.get_groups() == ["2014", "2015", "2016"]
        assert catalog.get_otf_idx() == {}
        assert catalog.get_unrolled_catalog() == (
            [
                "layer_YYYY01-YYYY05",
                "layer_YYYY06-YYYY12",
                "layer_YYYY01-YYYY05",
                "layer_YYYY06-YYYY12",
                "layer_YYYY01-YYYY05",
                "layer_YYYY06-YYYY12",
            ],
            [
                "path/to/param_201401-201405.tif",
                "path/to/param_201406-201412.tif",
                "path/to/param_201501-201505.tif",
                "path/to/param_201506-201512.tif",
                "path/to/param_201501-201505.tif",
                "path/to/param_201506-201512.tif",
            ],
            [0, 1, 2, 3, 4, 5],
        )
        assert catalog.get_feature_names() == [
            "layer_YYYY01-YYYY05",
            "layer_YYYY06-YYYY12",
        ]

    def test_create_catalog_perc(self):
        catalog = c.create_catalog(
            catalog_def=pd.DataFrame(
                {
                    "layer_name": ["layer_{perc}"],
                    "path": ["sample/path_{perc}.tif"],
                    "type": ["common"],
                    "perc": ["sd,p25,p50,p75"],
                }
            ),
            years=[2020],
            base_path=str(Path(".").absolute()),
        )
        print(f"{catalog.data=!r}", repr(catalog), dir(catalog.data), range(2014, 2020))
        assert isinstance(catalog, c)
        assert catalog.data_size == 4
        assert catalog.data == {
            "common": {
                "layer_p25": {"path": "sample/path_p25.tif", "idx": 0},
                "layer_p50": {"path": "sample/path_p50.tif", "idx": 1},
                "layer_p75": {"path": "sample/path_p75.tif", "idx": 2},
                "layer_sd": {"path": "sample/path_sd.tif", "idx": 3},
            }
        }
        assert catalog.get_groups() == ["common"]
        assert catalog.get_otf_idx() == {}
        assert catalog.get_unrolled_catalog() == (
            ["layer_p25", "layer_p50", "layer_p75", "layer_sd"],
            [
                "sample/path_p25.tif",
                "sample/path_p50.tif",
                "sample/path_p75.tif",
                "sample/path_sd.tif",
            ],
            [0, 1, 2, 3],
        )
        assert catalog.get_feature_names() == [
            "layer_p25",
            "layer_p50",
            "layer_p75",
            "layer_sd",
        ]
