..
   Note: Items in this toctree form the top-level navigation. See `api.rst` for the `autosummary` directive, and for why `api.rst` isn't called directly.

.. toctree::
   :hidden:

   Home <self>
   Tutorials <tutorials>
   API Reference <_autosummary/eumap>

##################
EUMAP LIBRARY
##################

Eumap is a library to enable easier access to several **spatial layers prepared for Continental Europe** (*Landsat and Sentinel mosaics, DTM and climate datasets, land cover, potential natural vegetation and environmental quality maps*), as well the processing workflows used to produce them. 

It implements efficient **raster access** through `rasterio <https://rasterio.readthedocs.io>`_, multiple **gapfiling** approaches, **spatial and spacetime overlay**, **training samples** preparation (LUCAS points), and **Ensemble Machine Learning** applied to spatial predictions (fully compatible with `scikit-learn <https://scikit-learn.org>`_).

The spatial layers can be accessed through `ODSE Viewer <http://maps.opendatascience.eu>`_.

.. image:: img/odse.png

|

##################
Installing
##################

Using `pip`::

   pip install -U git+https://gitlab.com/geoharmonizer_inea/eumap.git



##################
Contributing
##################
eumap has been developed and used by a group of active community members. Your help is very valuable to make the package better for everyone. Check our `contribution guidelines <https://gitlab.com/geoharmonizer_inea/eumap/-/blob/master/CONTRIBUTING.md>`_ and `open issues <https://gitlab.com/geoharmonizer_inea/eumap/-/issues?scope=all&state=opened>`_

##################
License
##################
© Contributors, 2020. Licensed under an `Apache-2 <https://gitlab.com/geoharmonizer_inea/eumap/blob/master/LICENSE>`_ license.

##################
Funding
##################

This work is co-financed under Grant Agreement Connecting Europe Facility (CEF) Telecom `project 2018-EU-IA-0095 <https://ec.europa.eu/inea/en/connecting-europe-facility/cef-telecom/2018-eu-ia-0095>`_ by the European Union.