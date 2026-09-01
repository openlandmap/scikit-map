# Plan: `from_stac` source for `RasterData`

Goal: a lazy `RasterData.from_stac(...)` that builds a `RasterData` from a STAC
catalogue, working against `https://stac.opengeohub.org/v1/cat/ecodatacube`,
following the same `LayerSource` / `LayerSpec` classes as the YAML driver, plus
unit tests and a new notebook `docs/notebooks/05_stac_integration.ipynb`.

## Findings (ecodatacube server behaviour)

Probed live during planning:

- `/v1/cat/ecodatacube` is a **Catalog** of **67 child collections** (each one
  variable: ndvi, swir1, slope, landcover, ...). Per-collection pages live at
  `/v1/collections/<collection_id>`; items at `/v1/collections/<id>/items`.
- `pystac_client` cannot use the standard catalog `/search` out of the box:
  the server advertises **no conformance classes**, and the catalog-level
  `/cat/ecodatacube/search` **ignores the `collections` filter** (a search with
  `collections=[ndvi]` returned 2760 items across all 67 collections).
- The **per-collection `/items` endpoint is the reliable path**: it returns
  only that collection's items, ordered by descending date, supports
  `limit` and `bbox`, and paginates via a `rel=next` token link.
  - ⚠ The `/items` endpoint **rejects the `datetime` query param (HTTP 400)**,
    so date filtering must be done **client-side** (items arrive newest-first;
    stop once we go before the requested start).
- Each **item**:
  - `datetime` is `None`; the date lives in `properties.start_datetime` /
    `end_datetime` (ISO, e.g. `2022-11-01T00:00:00Z`).
  - One or more **data assets** = assets whose `roles` contains `"data"`,
    each a **COG on `https://s3.ecodatacube.eu/arco/...tif`**. (Other assets are
    `thumbnail`, `*_qml_1`, `*_sld_1` styles — skipped.)
  - `proj:epsg = 3035`, `gsd = 30`, `proj:shape`/`proj:transform` present.
- All ecodatacube collections share the **EPSG:3035 / 30 m** grid, so multiple
  collections align natively — **no `vrt_warp` reprojection needed**.
- Items are **continental** (≈153 400 × 216 700 px). A full `RasterData.read()`
  is infeasible; the intended usage is **windowed reads** (`read(extent=...)` /
  `read(window=...)`) and **point sampling via `SpaceTimeOverlay`**. Confirmed
  `rasterio` opens the remote COGs over https and windowed reads work.
- The bundled `toy.lc_samples()` are **already in EPSG:3035**, centred on
  ~5.68°E / 51.96°N (Netherlands, bbox ≈ [4020674, 3210142, 4028246, 3217730]
  EPSG:3035 / [5.632, 51.92, 5.735, 51.992] lonlat) — inside the ecodatacube
  extent, so the notebook can overlay them onto real STAC layers with **no
  reprojection**.

## Existing STAC code

`RasterData.from_stac_items(...)` (base.py:1118) takes a *pre-fetched* list of
pystac `Item`s, runs `vrt_warp` to a target CRS/res, and builds via the dict
constructor. It is **eager** (VRTs materialised) and collection-agnostic.

**Decision:** keep `from_stac_items` as-is (low-level, warp-on-load, for already
fetched items / cross-grid STAC). Add the new **`from_stac`** as the
**lazy, source-based, ecodatacube-friendly** entry point. No removals.

## Proposed API

```python
@classmethod
def from_stac(
    cls,
    url: str,                       # catalogue root, e.g.
                                    # "https://stac.opengeohub.org/v1/cat/ecodatacube"
    collections: list[str] | str,   # one or more collection ids, e.g.
                                    # "ndvi_glad.landsat.ard2.seasconv_eu_ecodatacube"
    datetime: str = None,           # "YYYY-MM-DD/YYYY-MM-DD" (client-side filtered)
    bbox: list[float] = None,       # [w,s,e,n] in EPSG:4326, passed to the items endpoint
    bands: list[str] = None,         # restrict to these data-asset keys (default: all data assets)
    max_items: int = None,           # cap total items fetched (safety)
    limit: int = 500,               # page size
    group: str = "year",            # "year" (default, like from_yaml) or "collection"
    name_template: str = None,      # default "{collection}" (year-agnostic, like YAML {band}_{season})
    date_format: str = "%Y%m%d",
    ignore_29feb: bool = True,
    backend: str = "numpy",
    verbose: bool = False,
) -> "RasterData":
    ...
```

Returns a **lazy** `RasterData` (paths = remote COG hrefs; no `.read()`),
with one `info` row per `(item, data-asset)`, carrying the standard columns
plus `collection`, `asset`, `band` (= collection short name), `year`,
`gsd`, `epsg`. `date_args` is populated per group so `filter_date` /
`timespan` work identically to the YAML path.

### `StacSource(LayerSource)` — `skmap/io/sources.py`

Mirrors `YamlSource`:

- `iter_specs()` → `Iterator[LayerSpec]`: for each requested collection, page
  the `/items` endpoint (follow `rel=next` token links), apply client-side date
  filter + `max_items`, and for each item yield one `LayerSpec` per data asset:
  - `path = asset.href` (remote COG)
  - `start_date`/`end_date` from `start_datetime`/`end_datetime`
  - `group` = year (or collection) per `group=` arg
  - `name` = `name_template.format(collection=..., asset=..., year=...)` or
    default `"{collection}"` (year-agnostic → unique within a year, like the
    YAML `{band}_{season}` design)
  - `temporal = True` (all ecodatacube items are dated)
  - `vars = {collection, asset, band, year, gsd, epsg}`
- `to_rasterdata()` inherited from `LayerSource` (builds the `info` DataFrame
  via `from_info`, same as `YamlSource`).

### HTTP / pagination layer

- Use `requests` (already a transitive dep) with a small `_get_pages(url,
  params)` generator that follows the `next` link. A single internal method
  `_fetch_item_dicts(collection_id, bbox, limit, max_items)` returns a list of
  raw item dicts — **this is the seam tests stub**, so the suite never hits the
  network. pystac is used only to *parse* an item dict into a typed object for
  easy asset/role access (`pystac.Item.from_dict(...)`).

### `from_stac` classmethod — `skmap/io/base.py`

One-liner wrapper, parallel to `from_yaml`:

```python
@classmethod
def from_stac(cls, url, collections, ...):
    from skmap.io.sources import StacSource
    return StacSource(url=url, collections=collections, ...).to_rasterdata(
        backend=backend, verbose=verbose)
```

## Implementation steps (one commit per step)

1. **`StacSource` + `from_stac`** — add `StacSource` to `skmap/io/sources.py`
   (pagination, client-side date filter, data-asset selection, grouping) and
   `RasterData.from_stac` in `skmap/io/base.py`. Export `StacSource` from the
   module docstring/`__all__`-equivalent. Add `from_stac` to the doctestable
   surface in `docs/quickstart` (see step 4).
2. **Unit tests** — `tests/io/test_stac.py`:
   - `test_iter_specs_*`: stub `_fetch_item_dicts` with canned item dicts (2
     items, one with multiple data assets); assert `LayerSpec` path/dates/
     group/vars, that style/thumbnail assets are skipped, and that
     `max_items`/date filtering behave.
   - `test_from_stac_lazy_info`: `RasterData.from_stac(...)` with the stubbed
     source returns a lazy `RasterData` (`array is None`), correct `info`
     columns (`collection`, `asset`, `band`, `year`, `epsg`, `gsd`), groups =
     years, `date_args` populated, `filter_date`/`filter("collection == ...")`
     work.
   - `test_pagination_follows_next`: assert the page follower calls the `next`
     URL (stub with two pages).
   - `test_date_filter_client_side`: items older than the requested window are
     dropped (no `datetime` param sent to the server).
   - **No live network** in the unit suite. Add an opt-in
     `@pytest.mark.network` smoke test (skipped by default) that hits the real
     ecodatacube endpoint for 1 collection / 2 items and asserts the href is a
     `.tif`. Register the marker in `pyproject.toml`.
3. **Notebook `05_stac_integration.ipynb`** — runnable, real (network) notebook:
   - Set the PROJ workaround env (like the other notebooks).
   - `from_stac` over the toy-samples AOI (EPSG:3035 bbox, ~[5.63, 51.92, 5.74,
     51.99] lonlat or the EPSG:3035 bounds) for 1–2 collections
     (e.g. `ndvi_glad.landsat.ard2.seasconv_eu_ecodatacube` + a static like
     `slope.in.degree_edtm_eu_ecodatacube`) and a short date window.
   - Show `rdata.info`, `get_groups()`, `filter_date`.
   - `SpaceTimeOverlay` with `toy.lc_samples()` (EPSG:3035) → training table
     (same workflow as notebook 03, but covariates from real STAC COGs).
   - A small **windowed** `rdata.read(extent=<AOI bbox in 3035>)` of one layer
     + `rdata.plot(img_title_text="{collection} {start_date:%Y-%m}")` to show a
     real raster (small tile, no full-continental read).
   - **Strip outputs before commit** (the README/git-size decision) so the
     notebook stays small; keep a note that it requires network access.
   - Add to `docs/tutorials.rst` and (optionally) the README's feature list.
4. **Docs** — add `from_stac` to `docs/quickstart/quickstart.rst` (a short
   section mirroring the YAML one) and ensure Sphinx autosummary picks up
   `StacSource`. Verify `make doctest` / build stays clean.

## Open decisions — please confirm

- **D1 Grouping**: default `group="year"` (parity with `from_yaml`, enables
  `get_years()` + per-year overlay/prediction), with `group="collection"` as
  an alternative. OK?
- **D2 Multiple data assets**: a collection like `ch4.vmr_s5p` has p10/p50/p90
  as three data assets per item → three `info` rows (one per asset), with an
  `asset` column to select. OK, or prefer a `bands=` default that picks only
  the first/main data asset?
- **D3 Notebook scope**: overlay + a small windowed read + plot (no
  continental prediction). OK?
- **D4 Live test**: ship a `@pytest.mark.network` test that is **skipped by
  default** (so CI stays offline), runnable via
  `pytest -m network`. OK?
- **D5 `from_stac_items`**: keep as-is (not removed). OK?
- **D6 Output stripping**: commit the new notebook with **outputs cleared**
  (consistent with the repo-size concern) and document that it needs network.
  OK?