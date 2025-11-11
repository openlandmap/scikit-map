##########
Quickstart
##########

See :ref:`installation-section` for installation instructions.

Creating a catalog
==================

Every scikit-map project starts with a catalog. This is a Pandas Dataframe or a csv file.

.. csv-table:: Example catalog
    :file: example.csv
    :header-rows: 1

Create a catalog from it like:

.. doctest::

    >>> from skmap.catalog import DataCatalog
    >>> catalog = DataCatalog.create_catalog("example.csv")
    >>> catalog.data

Space-Time overlay
==================

.. image:: ../img/spacetime_overlay.svg


Large-scale predictions
=======================

Whales
======


