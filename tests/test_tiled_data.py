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
