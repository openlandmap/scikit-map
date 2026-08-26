##########
Quickstart
##########

See :ref:`installation-section` for installation instructions.

``Scikit-map`` is used in two places in a machine learning pipeline: for Space-time overlay and large-scale predictions.

.. image:: ../img/pipeline.svg

Loading rasters
===============

Every scikit-map project starts with a :class:`~skmap.io.RasterData`, which
describes a set of raster layers.  Temporal layers carry ``start_date`` /
``end_date`` columns (populated by :meth:`~skmap.io.RasterData.timespan`) and
static layers have ``None`` dates.

.. doctest::

    >>> from skmap.io import RasterData
    >>> rdata = RasterData({
    ...     "2020": ["ndvi_2020.tif", "swir1_2020.tif"],
    ...     "common": ["elev.tif", "slope.tif"],
    ... })
    >>> rdata.get_groups()
    ['2020']

The object is **lazy**: it holds only the file paths until you call
:meth:`~skmap.io.RasterData.read`. This means the space-time overlay below can
sample points directly from the files without ever loading whole rasters into
memory.

Loading rasters from YAML
=========================

A layer catalogue can be described in YAML and expanded into a lazy
:class:`~skmap.io.RasterData` via :meth:`~skmap.io.RasterData.from_yaml`.
``{variable}`` placeholders in the path template are expanded from the
``start_year``/``end_year`` range, the ``band`` list and the paired
``start_month``/``end_month`` lists (see :mod:`skmap.io.sources` for the full
schema):

.. code-block:: yaml

    - layer: '{band}_glad.landsat.ard2.swa_m_30m_s_{year}{start_month}_{year}{end_month}_go_epsg.4326_v1'
      path: '{base_path}/arco/{band}_glad.landsat.ard2.swa_m_30m_s_{year}{start_month}_{year}{end_month}_go_epsg.4326_v1.tif'
      temporal_resolution: 'bimonthly'
      type: 'temporal'
      start_year: 1997
      end_year: 2024
      band: 'blue, green, red, nir, swir1, swir2, thermal'
      start_month: '0101, 0301, 0501, 0701, 0901, 1101'
      end_month: '0228, 0430, 0630, 0831, 1031, 1231'

    - layer: 'elev'
      path: '{base_path}/elev.tif'
      temporal_resolution: 'longterm_or_static'
      type: 'common'

.. code-block:: python

    rdata = RasterData.from_yaml("layers.yaml", base_path="/data")
    rdata.info  # one row per expanded layer, plus band/year/month columns

Temporal layers are grouped by year (``group`` = ``"2015"``, ...) and static
layers under ``"common"``; every ``{variable}`` referenced in the path becomes
an extra ``info`` column, so runners can group by multiple columns (e.g.
``group`` and ``band``).  A ``group`` template overrides the default (use
``group: '{band}'`` for band-grouped time-series).

A runnable example ships with the toy data::

    >>> from skmap.data import toy
    >>> from skmap.io import RasterData
    >>> rdata = RasterData.from_yaml(str(toy.LAYERS_YAML), base_path=str(toy.DATA_DIR))
    >>> rdata.get_groups()
    ['ndvi', 'swir1']

It also demonstrates the ``interval`` temporal style, a ``name`` template
(year-agnostic ``{band}_{season}`` names) and a ``variant`` column separating
the gap-filled and gappy NDVI.

Space-Time overlay
==================

.. image:: ../img/spacetime_overlay.svg

:class:`~skmap.overlay.SpaceTimeOverlay` filters the temporal layers by date
(via :meth:`~skmap.io.RasterData.filter_date`) for each date range and always
includes the static (undated) layers.  When ``date_ranges`` is not given, one
range per unique year in the points' date column is derived automatically.

.. code-block:: python

    import geopandas as gpd
    from skmap.overlay import SpaceTimeOverlay

    samples = gpd.read_file("samples.gpkg")
    overlay = SpaceTimeOverlay(
        points=samples,
        col_date="date",
        rasterdata=rdata,
        raster_tiles=None,
    )
    train = overlay.run(max_ram_mb=512, out_file_name=None)

Large-scale predictions
=======================

.. code-block:: python

    import joblib
    from skmap.io.process import Prediction

    model = joblib.load("model.joblib")
    rdata = rdata.read()  # materialize rasters for spatial prediction
    rdata.run(Prediction(model=model))
    pred = rdata.filter('group == "prediction"').array.get()

Whales
======

Derived (on-the-fly) features are computed by :mod:`skmap.io.process` runners.
They can be applied either to a loaded :class:`~skmap.io.RasterData` or as a
``runners`` list to the overlay:

.. code-block:: python

    from skmap.io import process

    overlay = SpaceTimeOverlay(
        points=samples,
        col_date="date",
        rasterdata=rdata,
        runners=[process.NormalizedDifference("ndvi", "swir1")],
    )