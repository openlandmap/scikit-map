##############
Data structure
##############

This page explains the internal layout of a :class:`~skmap.io.RasterData`,
which is deliberately different from the raster formats most remote-sensing
and GIS users are used to.

The 4-D data cube
=================

A :class:`~skmap.io.RasterData` represents a **4-D data cube** with the
coordinates ``(x, y, time interval, feature)``.  For example:

* ``X, Y, 20010101:10:30, NDVI``
* ``X, Y, 20010101:00:00_20010131:00:00, NDVI_mean``
* ``X, Y, 20010101:00:00_20010131:00:00, NDVI_max``

The time interval is a ``(start_date, end_date)`` pair, and the feature is a
named variable (``NDVI``, ``NDVI_mean``, ``elev``, ...).  A feature that has
no time interval is **common** (static): it is replicated across every
prediction interval.

The collapsed 2-D array
=======================

To optimise memory access and parallel processing, the cube is collapsed into
a single 2-D array ``(N, H*W)`` where ``H*W`` is the flattened spatial grid
(``x`` and ``y``) and ``N`` is the number of layers:

.. code-block:: text

   array: (N, H*W)          info: N rows (the row index)

   row 0..23    ndvi   2015 winter, 2015 spring, ... 2020 fall   (chronological)
   row 24..47   swir1  2015 winter, 2015 spring, ... 2020 fall
   row 48..49   elev, slope (common: no time interval)

Rows follow the **catalogue construction order** (for a year-organised
catalogue this is year-major, band-minor, date-minor).  Irregular timesteps
are supported, and a single array can mix monthly and yearly features.

The ``info`` index
==================

The :class:`~pandas.DataFrame` ``rdata.info`` has exactly one row per array
row and is the authoritative index into the array.  Its **mandatory columns**
are:

====================  ======================================================
Column                Meaning
====================  ======================================================
``input_path``        Source raster path
``input_band``        Band index in the source file
``name``              Unique per-row label (may embed the date)
``start_date``        Start of the time interval (``NaT`` for common)
``end_date``          End of the time interval (``NaT`` for common)
``feature``           The cube's feature coordinate (shared by its timesteps)
``is_common``         ``True`` = static, replicated across predictions
====================  ======================================================

Additional columns (``year``, ``season``, ``band``, ``collection``, ...) are
**optional**: they are queryable via :meth:`~skmap.io.RasterData.select` and
:meth:`~skmap.io.RasterData.filter`, but they are **not propagated** to newly
generated features.  Mandatory columns are always propagated.

Temporal vs common in prediction
================================

During prediction, features are organised into two types:

* **temporal** (``is_common=False``) — used to run predictions for a specific
  time interval, e.g. yearly (``2020-01-01`` to ``2020-12-31``) or monthly
  (``2020-01-01`` to ``2020-01-31``);
* **common** (``is_common=True``) — replicated across all predictions
  regardless of the time interval.

The ``outname`` argument of :meth:`~skmap.io.RasterData.run` is a template
that can use any ``info`` column to populate the ``feature`` column of newly
generated features.
