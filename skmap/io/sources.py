"""Layer sources: populate a :class:`~skmap.io.RasterData` from external catalogues.

A *source* reads a layer catalogue (YAML, JSON, STAC, ...) and yields
:class:`LayerSpec` objects — one per expanded raster layer.  Template-based
sources (YAML/JSON) share :class:`TemplateExpander`, which expands
``{variable}`` placeholders in the path/name templates into concrete paths
and dates.

Two temporal styles are supported:

* **grid** (``bimonthly`` / ``monthly`` / ``yearly``): ``start_year`` /
  ``end_year`` -> a ``year`` axis; ``start_month`` / ``end_month``
  (comma-separated, equal length) -> zipped ``(start_month, end_month)``
  pairs; other comma-separated fields (``band``, ``perc``, ...) ->
  cross-product axes.
* **interval**: ``start_date`` / ``end_date`` / ``date_unit`` / ``date_step``
  generate ``(dt1, dt2)`` intervals via :func:`skmap.misc.date_range`; the
  ``{dt}`` placeholder expands to ``dt1_dt2``.  List variables whose length
  equals ``len(date_step)`` **cycle** with the interval index (e.g. a
  ``season`` list paired with the seasonal steps); other list variables
  cross-product.

An optional ``name`` template overrides the default file-stem name (useful
for year-agnostic names such as ``{band}_{season}`` so one model can be
trained and predicted across years).

The resulting :class:`~skmap.io.RasterData` is **lazy** (paths + dates only,
no ``.read()``).  Its ``info`` DataFrame carries the standard columns plus one
column per data variable (``band``, ``variant``, ``season``, ``perc``,
``year``, ``start_month``, ``end_month``, ...), so runners can group by
multiple columns (e.g. ``group`` and ``band``).
"""

from __future__ import annotations

import calendar
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import requests
import yaml


@dataclass
class LayerSpec:
    """One expanded raster layer (a single row of a RasterData ``info``)."""

    path: str
    group: str
    band_idx: int = 1
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    name: Optional[str] = None
    temporal: bool = False
    vars: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = Path(self.path.split("?")[0]).stem


class LayerSource(ABC):
    """Read a layer catalogue and yield expanded :class:`LayerSpec` objects."""

    date_format: str = "%Y%m%d"
    ignore_29feb: bool = True

    @abstractmethod
    def iter_specs(self) -> Iterator[LayerSpec]:
        """Yield one :class:`LayerSpec` per expanded raster layer."""

    def to_rasterdata(self, backend: str = "numpy", verbose: bool = False):
        """Build a lazy :class:`~skmap.io.RasterData` from the expanded specs."""
        from skmap.io import RasterData

        specs = list(self.iter_specs())
        if not specs:
            raise ValueError("No layers found in the source")

        # Union of variable columns across all specs (stable order).
        var_cols: List[str] = []
        for s in specs:
            for k in s.vars:
                if k not in var_cols:
                    var_cols.append(k)

        rows = []
        for s in specs:
            row = {
                RasterData.GROUP_COL: s.group,
                RasterData.PATH_COL: s.path,
                RasterData.BAND_COL: s.band_idx,
                RasterData.START_DT_COL: s.start_date,
                RasterData.END_DT_COL: s.end_date,
                RasterData.TEMPORAL_COL: s.temporal,
                RasterData.NAME_COL: s.name,
            }
            for k in var_cols:
                row[k] = s.vars.get(k)
            rows.append(row)

        import pandas as pd

        info = pd.DataFrame(rows)
        return RasterData.from_info(
            info,
            backend=backend,
            verbose=verbose,
            date_format=self.date_format,
            ignore_29feb=self.ignore_29feb,
        )


class TemplateExpander:
    """Expand ``{variable}`` placeholders in layer templates into concrete paths."""

    META_FIELDS = {"layer", "path", "temporal_resolution", "type", "leap_year"}
    YEAR_FIELDS = {"start_year", "end_year"}
    PAIRED_FIELDS = {"start_month", "end_month"}
    # Config/meta/derived fields that never become ``info`` columns.
    COLUMN_EXCLUDES = META_FIELDS | YEAR_FIELDS | {
        "name",
        "base_path",
        "group",
        "start_date",
        "end_date",
        "date_unit",
        "date_step",
        "ignore_29feb",
        "date_format",
        "dt",
    }

    def __init__(self, date_format: str = "%Y%m%d", ignore_29feb: bool = True) -> None:
        self.date_format = date_format
        self.ignore_29feb = ignore_29feb

    def expand(self, entry: Dict[str, Any], base_path: str) -> Iterator[LayerSpec]:
        """Expand one catalogue entry into :class:`LayerSpec` objects."""
        path_tmpl = entry["path"]
        name_tmpl = entry.get("name", "")
        group_tmpl = entry.get("group", "")
        temporal_resolution = entry.get("temporal_resolution", "longterm_or_static")
        type_ = entry.get("type", "temporal")

        # --- classify variables -------------------------------------------------
        scalars: Dict[str, Any] = {}
        list_axes: Dict[str, List[str]] = {}
        for key, val in entry.items():
            if key in self.COLUMN_EXCLUDES or key in self.PAIRED_FIELDS:
                continue
            if isinstance(val, str) and "," in val:
                list_axes[key] = [v.strip() for v in val.split(",")]
            else:
                scalars[key] = val

        if temporal_resolution == "interval":
            yield from self._expand_interval(
                entry, base_path, path_tmpl, name_tmpl, group_tmpl, type_, scalars, list_axes
            )
            return

        # --- grid mode: bimonthly / monthly / yearly / static -------------------
        year_range: Optional[List[int]] = None
        if "start_year" in entry:
            start_year = int(entry["start_year"])
            end_year = int(entry.get("end_year", start_year))
            year_range = list(range(start_year, end_year + 1))

        month_pairs: Optional[List[Tuple[str, Optional[str]]]] = None
        if "start_month" in entry:
            sm = self._split_list(entry["start_month"])
            em = self._split_list(entry.get("end_month", "")) or [None] * len(sm)
            if len(sm) != len(em):
                raise ValueError(
                    f"start_month and end_month must have the same length: {sm} vs {em}"
                )
            month_pairs = list(zip(sm, em))

        # active axes (only those referenced in the path template)
        axes: List[Tuple[str, List]] = []
        if year_range is not None and "{year}" in path_tmpl:
            axes.append(("year", year_range))
        for name, values in list_axes.items():
            if "{" + name + "}" in path_tmpl:
                axes.append((name, values))
        if month_pairs is not None and (
            "{start_month}" in path_tmpl or "{end_month}" in path_tmpl
        ):
            axes.append(("month", month_pairs))

        # cross product
        combos: List[Dict[str, Any]] = [{}]
        for name, values in axes:
            new_combos = []
            for combo in combos:
                for v in values:
                    c = dict(combo)
                    if name == "month":
                        c["start_month"], c["end_month"] = v
                    else:
                        c[name] = v
                    new_combos.append(c)
            combos = new_combos

        for combo in combos:
            vars_ = dict(scalars)
            vars_["base_path"] = base_path
            vars_.update(combo)
            if "start_year" in entry:
                vars_["start_year"] = entry["start_year"]
            if "end_year" in entry:
                vars_["end_year"] = entry["end_year"]

            path = self._format(path_tmpl, vars_)
            name = self._format(name_tmpl, vars_) if name_tmpl else None

            start_date, end_date = self._compute_dates(temporal_resolution, vars_)

            yield LayerSpec(
                path=path,
                group=self._group(type_, vars_, group_tmpl),
                start_date=start_date,
                end_date=end_date,
                name=name,
                temporal=start_date is not None,
                vars=self._spec_vars(vars_),
            )

    # ------------------------------------------------------------- interval mode
    def _expand_interval(
        self,
        entry: Dict[str, Any],
        base_path: str,
        path_tmpl: str,
        name_tmpl: str,
        group_tmpl: str,
        type_: str,
        scalars: Dict[str, Any],
        list_axes: Dict[str, List[str]],
    ) -> Iterator[LayerSpec]:
        from skmap.misc import date_range

        date_format = entry.get("date_format", self.date_format)
        ignore_29feb = entry.get("ignore_29feb", self.ignore_29feb)
        step_vals = [int(s) for s in self._split_list(entry.get("date_step", 1))]

        # List variables with the same length as date_step cycle with the
        # interval index (paired, not cross-product); others cross-product.
        cycling: Dict[str, List[str]] = {}
        cross_axes: Dict[str, List[str]] = {}
        for name, values in list_axes.items():
            if len(values) == len(step_vals):
                cycling[name] = values
            else:
                cross_axes[name] = values

        intervals = date_range(
            entry["start_date"],
            entry["end_date"],
            entry.get("date_unit", "days"),
            step_vals,
            date_format=date_format,
            ignore_29feb=ignore_29feb,
        )

        combos: List[Dict[str, Any]] = [{}]
        for name, values in cross_axes.items():
            if "{" + name + "}" in path_tmpl or "{" + name + "}" in name_tmpl:
                new_combos = []
                for c in combos:
                    for v in values:
                        cc = dict(c)
                        cc[name] = v
                        new_combos.append(cc)
                combos = new_combos

        for combo in combos:
            for i, (dt1, dt2) in enumerate(intervals):
                vars_ = dict(scalars)
                vars_["base_path"] = base_path
                vars_.update(combo)
                for name, values in cycling.items():
                    vars_[name] = values[i % len(values)]
                vars_["dt"] = f"{dt1.strftime(date_format)}_{dt2.strftime(date_format)}"
                vars_["year"] = dt1.year

                path = self._format(path_tmpl, vars_)
                name = self._format(name_tmpl, vars_) if name_tmpl else None

                yield LayerSpec(
                    path=path,
                    group=self._group(type_, vars_, group_tmpl),
                    start_date=dt1,
                    end_date=dt2,
                    name=name,
                    temporal=True,
                    vars=self._spec_vars(vars_),
                )

    # ------------------------------------------------------------------ helpers
    def _spec_vars(self, vars_: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in vars_.items() if k not in self.COLUMN_EXCLUDES}

    @staticmethod
    def _split_list(val) -> List[str]:
        if val is None:
            return []
        if isinstance(val, (list, tuple)):
            return [str(v).strip() for v in val]
        return [v.strip() for v in str(val).split(",") if v.strip()]

    @staticmethod
    def _format(tmpl: str, vars_: Dict[str, Any]) -> str:
        try:
            return tmpl.format_map(vars_)
        except KeyError as e:
            raise ValueError(f"Unresolved variable {e} in template: {tmpl!r}") from e

    def _group(self, type_: str, vars_: Dict[str, Any], group_tmpl: str = "") -> str:
        if group_tmpl:
            return self._format(group_tmpl, vars_)
        if type_ == "common":
            return "common"
        return str(vars_.get("year", "default"))

    def _compute_dates(
        self, temporal_resolution: str, vars_: Dict[str, Any]
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        if temporal_resolution in ("longterm_or_static", "static"):
            return None, None

        year = vars_.get("year")
        sm = vars_.get("start_month")
        em = vars_.get("end_month")

        if temporal_resolution == "bimonthly":
            if year is None or sm is None or em is None:
                raise ValueError("bimonthly requires start_year, start_month and end_month")
            sd = self._parse_month_day(year, sm)
            ed = self._parse_month_day(year, em)
            if int(em) < int(sm):  # MMDD comparison: end falls in the next year
                ed = ed.replace(year=year + 1)
            return sd, ed

        if temporal_resolution == "monthly":
            if year is None or sm is None:
                raise ValueError("monthly requires start_year and start_month")
            sd = self._parse_month_day(year, sm)
            return sd, self._last_day(sd)

        if temporal_resolution == "yearly":
            if sm is not None and em is not None:
                sd = self._parse_month_day(year, sm)
                ed = self._parse_month_day(year, em)
                if int(em) < int(sm):
                    ed = ed.replace(year=year + 1)
                return sd, ed
            return datetime(year, 1, 1), datetime(year, 12, 31)

        raise ValueError(f"Unknown temporal_resolution: {temporal_resolution!r}")

    def _parse_month_day(self, year: int, token: Any) -> datetime:
        token = str(token).strip()
        if len(token) == 4:  # MMDD
            month, day = int(token[:2]), int(token[2:4])
        elif len(token) == 2:  # MM
            month, day = int(token), 1
        else:
            raise ValueError(f"Cannot parse month token {token!r}")
        if self.ignore_29feb and month == 2 and day == 29:
            day = 28
        return datetime(year, month, day)

    @staticmethod
    def _last_day(dt: datetime) -> datetime:
        return datetime(dt.year, dt.month, calendar.monthrange(dt.year, dt.month)[1])


class YamlSource(LayerSource):
    """Read a YAML layer catalogue and expand it into :class:`LayerSpec` objects.

    :param path: Path to the YAML file.
    :param base_path: Value for the ``{base_path}`` placeholder.  Falls back to
        the ``SKMAP_BASE_PATH`` environment variable; raises ``ValueError`` if
        ``{base_path}`` is referenced but unresolved.
    :param date_format: strptime format used for the resulting RasterData's
        ``date_args`` (downstream ``timespan``/``_set_date`` consistency).
    :param ignore_29feb: Clamp Feb 29 end-dates to Feb 28 during expansion.
    """

    def __init__(
        self,
        path: str,
        base_path: str = None,
        date_format: str = "%Y%m%d",
        ignore_29feb: bool = True,
    ) -> None:
        self.path = path
        self.base_path = base_path or os.environ.get("SKMAP_BASE_PATH")
        self.date_format = date_format
        self.ignore_29feb = ignore_29feb
        self.expander = TemplateExpander(
            date_format=date_format, ignore_29feb=ignore_29feb
        )

        with open(path) as f:
            self.entries = yaml.safe_load(f) or []

        if self.base_path is None and any(
            "{base_path}" in str(e.get("path", "")) for e in self.entries
        ):
            raise ValueError(
                "base_path is required: pass base_path=... or set SKMAP_BASE_PATH"
            )

    def iter_specs(self) -> Iterator[LayerSpec]:
        for entry in self.entries:
            yield from self.expander.expand(entry, self.base_path or "")


def _parse_iso(s: Any) -> Optional[datetime]:
    """Parse an ISO-8601 datetime string (optional ``Z``) to a naive datetime."""
    if not s:
        return None
    s = str(s).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


class StacSource(LayerSource):
    """Read a STAC catalogue and expand its items into :class:`LayerSpec` objects.

    Queries the per-collection ``/items`` endpoint (collection-respected,
    token-paginated) and yields one :class:`LayerSpec` per **data asset**
    (an asset whose ``roles`` contains ``"data"``).  Dates come from the
    item's ``start_datetime`` / ``end_datetime`` properties (the ``datetime``
    field is often ``None``).  The ``datetime`` argument is filtered
    **client-side** because the ecodatacube ``/items`` endpoint rejects the
    ``datetime`` query parameter.

    :param url: Catalogue root URL, e.g.
        ``https://stac.opengeohub.org/v1/cat/ecodatacube``.
    :param collections: One collection id or a list of ids.
    :param datetime: ``"YYYY-MM-DD/YYYY-MM-DD"`` window (client-side filter).
    :param bbox: ``[west, south, east, north]`` in EPSG:4326, passed to the
        items endpoint.
    :param bands: Restrict to these data-asset keys (default: all data assets).
    :param max_items: Cap the total number of items fetched per collection.
    :param limit: Page size for the items endpoint.
    """

    def __init__(
        self,
        url: str,
        collections: Union[str, List[str]],
        datetime: str = None,
        bbox: List[float] = None,
        bands: List[str] = None,
        max_items: int = None,
        limit: int = 500,
        date_format: str = "%Y%m%d",
        ignore_29feb: bool = True,
        timeout: float = 60,
    ) -> None:
        self.url = url.rstrip("/")
        self.collections = (
            [collections] if isinstance(collections, str) else list(collections)
        )
        self.bbox = bbox
        self.bands = set(bands) if bands else None
        self.max_items = max_items
        self.limit = limit
        self.timeout = timeout
        self.date_format = date_format
        self.ignore_29feb = ignore_29feb
        self._root = None
        self._start, self._end = self._parse_range(datetime)

    # ------------------------------------------------------------- public API
    def iter_specs(self) -> Iterator[LayerSpec]:
        for cid in self.collections:
            for item in self._fetch_item_dicts(cid):
                yield from self._item_specs(cid, item)

    # ------------------------------------------------------------- item -> specs
    def _item_specs(self, cid: str, item: Dict[str, Any]) -> Iterator[LayerSpec]:
        props = item.get("properties", {})
        start = _parse_iso(props.get("start_datetime"))
        end = _parse_iso(props.get("end_datetime"))
        if start is None:
            return
        if self._start is not None and start < self._start:
            return
        if self._end is not None and start > self._end:
            return

        for key, asset in item.get("assets", {}).items():
            roles = asset.get("roles") or []
            if "data" not in roles:
                continue
            if self.bands is not None and key not in self.bands:
                continue
            yield LayerSpec(
                path=asset.get("href"),
                group=cid,
                start_date=start,
                end_date=end,
                # year-agnostic but unique within a year (overlay requirement),
                # like the YAML ``{band}_{season}`` names
                name=f"{key}_{start:%m%d}",
                temporal=True,
                vars={
                    "collection": cid,
                    "asset": key,
                    "year": start.year,
                    "gsd": props.get("gsd"),
                    "epsg": props.get("proj:epsg"),
                },
            )

    # ------------------------------------------------------------- HTTP layer
    def _fetch_item_dicts(self, collection_id: str) -> List[Dict[str, Any]]:
        """Fetch all items of a collection (token-paginated). Test seam."""
        root = self._root_url()
        url = f"{root}/collections/{collection_id}/items"
        params = {"limit": self.limit}
        if self.bbox:
            params["bbox"] = ",".join(str(b) for b in self.bbox)

        items: List[Dict[str, Any]] = []
        first = True
        while url:
            resp = requests.get(
                url, params=params if first else None, timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            items.extend(data.get("features", []))
            if self.max_items and len(items) >= self.max_items:
                return items[: self.max_items]
            nxt = next(
                (l for l in data.get("links", []) if l.get("rel") == "next"), None
            )
            url = nxt["href"] if nxt else None
            first = False
        return items

    def _root_url(self) -> str:
        """Return the catalogue root URL (from the catalog's ``root`` link)."""
        if self._root is None:
            resp = requests.get(self.url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            for link in data.get("links", []):
                if link.get("rel") == "root":
                    self._root = link["href"].rstrip("/")
                    break
            if self._root is None:
                raise ValueError(f"No 'root' link found in catalogue {self.url}")
        return self._root

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _parse_range(dt: Any) -> Tuple[Optional[datetime], Optional[datetime]]:
        if dt is None:
            return None, None
        if isinstance(dt, (list, tuple)):
            a, b = dt[0], dt[1]
        elif "/" in str(dt):
            a, b = str(dt).split("/", 1)
        else:
            a = b = dt
        start = _parse_iso(a) if a not in ("", "..", None) else None
        end = _parse_iso(b) if b not in ("", "..", None) else None
        return start, end
