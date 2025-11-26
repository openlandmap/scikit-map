#######
C++ API
#######

Fast code in skmap is written in C++ and exposed to Python with thin wrappers. This is a lower-level api. All Python-exposed functions handle setting up GDAL and then call a function on an array class.

IoArray Class
=============

This class is used by :py:class:`TiledDataLoader`, :py:class:`TiledDataExporter` and :py:class:`SpaceOverlay`

.. doxygenclass:: skmap::IoArray
   :members:
   :allow-dot-graphs:

TransArray Class
================

.. doxygenclass:: skmap::TransArray
   :members:
   :allow-dot-graphs:

ParArray Class
==============

.. doxygenclass:: skmap::ParArray
    :members:
    :allow-dot-graphs:

Bindings
========

.. doxygenfile:: skmap_bindings.cpp
   :sections: func
