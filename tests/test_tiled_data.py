# Pure-function tests for S3 listing / percentile strings; no raster data, toy data N/A.
from skmap.tiled_data import get_percentiele_string, s3_list_files, s3_split_prefix

def test_s3_split_prefix():
    assert s3_split_prefix("bucket/path/to/data") == ("bucket", "path/to/data")
    assert s3_split_prefix("bucket") == ("bucket", "")


def test_get_percentiele_string():
    assert get_percentiele_string(0.5) == "p50"
    assert get_percentiele_string(0.025) == "p025"
    assert get_percentiele_string(0) == "p0"
    assert get_percentiele_string(1) == "p100"


class _FakeClient:
    def __init__(self, objects):
        self._objects = objects

    def list_objects(self, bucket, prefix, recursive):
        # Match MinIO semantics: only objects whose name starts with `prefix`.
        return [
            type("Obj", (), {"object_name": o})()
            for o in self._objects
            if o.startswith(prefix)
        ]


def test_s3_list_files():
    client = _FakeClient(
        [
            "prefix/tile1/a.tif",
            "prefix/tile1/b.tif",
            "prefix/tile2/c.tif",
        ]
    )
    files = s3_list_files([client], "bucket/prefix", "tile1")
    assert files == ["prefix/tile1/a.tif", "prefix/tile1/b.tif"]

    filtered = s3_list_files([client], "bucket/prefix", "tile1", file_pattern=r"a\.tif$")
    assert filtered == ["prefix/tile1/a.tif"]

    assert s3_list_files([], "bucket/prefix", "tile1") == []


def test_to_rasterdata():
    """TiledDataLoader.to_rasterdata wraps the loaded tile array into a RasterData."""
    import numpy as np
    import pandas as pd

    from skmap.catalog import DataCatalog
    from skmap.data import toy
    from skmap.tiled_data import TiledDataLoader

    elev = "elev.lowestmode_gedi.eml_mf_30m_s_20000101_20181231_nl_epsg.3035_v0.3.tif"
    slope = "slope.percent_gedi.eml_m_30m_s_20000101_20181231_nl_epsg.3035_v0.3.tif"
    toy_dir = toy.DATA_DIR
    catalog = DataCatalog.create_catalog(
        pd.DataFrame({
            "layer_name": ["elev", "slope"],
            "path": ["{base_path}/" + elev, "{base_path}/" + slope],
            "type": ["common", "common"],
        }),
        years=[2020],
        base_path=str(toy_dir / "static"),
    )

    loader = TiledDataLoader.__new__(TiledDataLoader)
    loader.catalog = catalog
    loader.x_size = 256
    loader.y_size = 256
    loader.mask_path = str(toy_dir / "static" / elev)
    loader.array = np.random.rand(2, 256 * 256).astype(np.float32)

    rdata = loader.to_rasterdata()
    assert rdata.array.shape == (2, 256 * 256)
    assert rdata.info["name"].tolist() == ["elev", "slope"]
    assert rdata._spatial_shape == (256, 256)
