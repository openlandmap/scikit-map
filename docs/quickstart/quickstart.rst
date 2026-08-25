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

    from skmap.modeler import RFRegressor

    model = RFRegressor("model.joblib")
    rdata = rdata.read()  # materialize rasters for spatial prediction
    pred = model.predict_raster(rdata)

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