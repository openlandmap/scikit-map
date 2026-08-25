#######
C++ API
#######

Fast code in skmap is written in C++ and exposed to Python with thin wrappers. This is a lower-level api. All Python-exposed functions handle setting up GDAL and then call a function on an array class.

Input-Output
============

.. doxygengroup:: io
   :project: skmap_bindings
   :members:

Data mangling
=============

.. doxygengroup:: mangling
   :project: skmap_bindings
   :undoc-members:

Data manipulation
=================

.. doxygengroup:: manipulation
   :project: skmap_bindings
   :undoc-members:

Data Processing
===============

.. doxygengroup:: processing
   :project: skmap_bindings
   :undoc-members:

IoArray Class
=============

This class is used by :py:class:`SpaceOverlay` and the raster I/O layer.

.. doxygenclass:: skmap::IoArray
   :members:
   :allow-dot-graphs:

TransArray Class
================

.. doxygenclass:: skmap::TransArray
   :members:
   :undoc-members:
   :allow-dot-graphs:

ParArray Class
==============

.. doxygenclass:: skmap::ParArray
    :members:
    :undoc-members:
    :allow-dot-graphs:

Bindings
========

.. doxygenfile:: skmap_bindings.cpp
   :sections: func
