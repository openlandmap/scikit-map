##########
Quickstart
##########

See :ref:`installation-section` for installation instructions.

``Scikit-map`` is used in two places in a machine learning pipeline: For Space-time overlay and large-scale predictions.

.. image:: ../img/pipeline.svg

Creating a catalog
==================

Every scikit-map project starts with a catalog. This is a Pandas Dataframe or a csv file.

.. csv-table:: Example catalog
    :file: example.csv
    :header-rows: 1

Create a catalog from it like:

.. doctest::

    >>> from pathlib import Path
    >>> from skmap.catalog import DataCatalog
    >>> catalog = DataCatalog.create_catalog(
    ...     catalog_def="docs/quickstart/example.csv",
    ...     years=list(range(2000,2022)),
    ...     base_path=str(Path(".").absolute())
    ... )
    >>> catalog.data
    {'2000': {'layer_YYYY': {'path': ' file_2000.tif', 'idx': 0}}, ... 'common': {'static': {'path': 'static.tif', 'idx': 22}}}

Space-Time overlay
==================

.. image:: ../img/spacetime_overlay.svg



Large-scale predictions
=======================

Whales
======


