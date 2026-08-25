"""Layer sources: populate a :class:`~skmap.io.RasterData` from external catalogues.

A *source* reads a layer catalogue (YAML, JSON, STAC, ...) and yields
:class:`LayerSpec` objects — one per expanded raster layer.  Template-based
sources (YAML/JSON) share :class:`TemplateExpander`, which expands
``{variable}`` placeholders in the path/name templates into concrete paths
and dates.

The YAML schema (see ``layers.yaml``)::

    - layer: '{band}_..._{year}{start_month}_{year}{end_month}_...'
      path: '{base_path}/arco/{band}_..._{year}{start_month}_{year}{end_month}_....tif'
      temporal_resolution: 'bimonthly'   # bimonthly | yearly | monthly | longterm_or_static
      type: 'temporal'                   # temporal | common
      start_year: 1997
      end_year: 2024
      band: 'blue, green, red'           # comma-separated -> cross-product axis
      start_month: '0101, 0301, 0501'    # paired with end_month (zipped)
      end_month: '0228, 0430, 0630'
      perc: 'p50'                        # any other field -> scalar or list axis

Variables are classified as:

* ``start_year``/``end_year`` (ints) -> a ``year`` iteration axis
  (``range(start_year, end_year + 1)``), also available as scalars
  ``{start_year}``/``{end_year}``.
* ``start_month``/``end_month`` (comma-separated, equal length) -> zipped
  ``(start_month, end_month)`` pairs -> a ``month`` iteration axis.
* any other comma-separated field (``band``, ``perc``, ``version``, ...) ->
  a list iteration axis (cross-product).
* single-valued fields -> scalars substituted into every combination.

Only axes whose placeholder appears in the ``path`` template are iterated.
``longterm_or_static`` layers skip the year/month axes.

The resulting :class:`~skmap.io.RasterData` is **lazy** (paths + dates only,
no ``.read()``).  Its ``info`` DataFrame carries the standard columns plus one
column per ``{variable}`` referenced in the path template (except
``base_path``), so runners can group by multiple columns (e.g. ``group`` and
``band``).
"""

from __future__ import annotations

import calendar
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

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

    def __init__(self, date_format: str = "%Y%m%d", ignore_29feb: bool = True) -> None:
        self.date_format = date_format
        self.ignore_29feb = ignore_29feb

    def expand(self, entry: Dict[str, Any], base_path: str) -> Iterator[LayerSpec]:
        """Expand one catalogue entry into :class:`LayerSpec` objects."""
        path_tmpl = entry["path"]
        layer_tmpl = entry.get("layer", "")
        temporal_resolution = entry.get("temporal_resolution", "longterm_or_static")
        type_ = entry.get("type", "temporal")

        # --- classify variables -------------------------------------------------
        scalars: Dict[str, Any] = {}
        list_axes: Dict[str, List[str]] = {}
        year_range: Optional[List[int]] = None
        month_pairs: Optional[List[Tuple[str, Optional[str]]]] = None

        for key, val in entry.items():
            if key in self.META_FIELDS or key in self.YEAR_FIELDS or key in self.PAIRED_FIELDS:
                continue
            if isinstance(val, str) and "," in val:
                list_axes[key] = [v.strip() for v in val.split(",")]
            else:
                scalars[key] = val

        if "start_year" in entry:
            start_year = int(entry["start_year"])
            end_year = int(entry.get("end_year", start_year))
            year_range = list(range(start_year, end_year + 1))

        if "start_month" in entry:
            sm = self._split_list(entry["start_month"])
            em = self._split_list(entry.get("end_month", "")) or [None] * len(sm)
            if len(sm) != len(em):
                raise ValueError(
                    f"start_month and end_month must have the same length: {sm} vs {em}"
                )
            month_pairs = list(zip(sm, em))

        # --- active axes (only those referenced in the path template) -----------
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

        # --- cross product ------------------------------------------------------
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

        # --- substitute + compute dates -----------------------------------------
        for combo in combos:
            vars_ = dict(scalars)
            vars_["base_path"] = base_path
            vars_.update(combo)
            if "start_year" in entry:
                vars_["start_year"] = entry["start_year"]
            if "end_year" in entry:
                vars_["end_year"] = entry["end_year"]

            path = self._format(path_tmpl, vars_)
            layer = self._format(layer_tmpl, vars_) if layer_tmpl else ""

            start_date, end_date = self._compute_dates(temporal_resolution, vars_)

            # one column per {variable} referenced in the path (except base_path)
            template_vars = set(re.findall(r"\{(\w+)\}", path_tmpl))
            spec_vars = {
                k: vars_[k] for k in template_vars if k != "base_path" and k in vars_
            }

            yield LayerSpec(
                path=path,
                group=self._group(type_, vars_),
                start_date=start_date,
                end_date=end_date,
                temporal=start_date is not None,
                vars=spec_vars,
            )

    # ------------------------------------------------------------------ helpers
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

    def _group(self, type_: str, vars_: Dict[str, Any]) -> str:
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
