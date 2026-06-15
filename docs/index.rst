..
   Note: Items in this toctree form the top-level navigation. See `api.rst` for the `autosummary` directive, and for why `api.rst` isn't called directly.

.. toctree::
   :hidden:

   Home <self>
   Quickstart <quickstart/quickstart>
   Tutorials <tutorials>
   In-Depth <in-depth/in-depth>
   API Reference <_autosummary/skmap>
   C++ API <cpp_api>

Scikit-Map is a library to fill gaps in the Python machine learning ecosystem when dealing with geospatial data. It offers *parallel data access* and manipulation for **space-time overlays** and **large-scale predictions**.

.. image:: img/pipeline.svg

It implements efficient **raster access** through `rasterio <https://rasterio.readthedocs.io>`_, multiple **gapfiling** approaches, **spatial and spacetime overlay**, **training samples** preparation (LUCAS points), and **Ensemble Machine Learning** applied to spatial predictions (fully compatible with `scikit-learn <https://scikit-learn.org>`_).

.. _installation-section:

##################
Installation
##################

Docker
=================

The best way to install ``skmap`` is using Docker. Check the `official documentation <https://docs.docker.com/get-docker/>`_ to get Docker running in your environment.

JupyterLab Container
--------------------

The image `opengeohub/pygeo-ide <https://hub.docker.com/r/opengeohub/pygeo-ide>`_ provides access to all ``skmap`` dependencies and to `JupyterLab <https://jupyterlab.readthedocs.io/en/stable/>`_ IDE. The follow instructions are specific for Intel CPUs, which supports `MKL <https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html>`_, for other CPUs it's recommend to use the `openblas version <https://hub.docker.com/r/opengeohub/pygeo-ide/tags?page=1&ordering=last_updated>`_.
 
First, download the image:

.. code-block:: shell-session

   docker pull opengeohub/pygeo-ide:v3.8.6-mkl-gdal314


Then create the container:

.. code-block:: shell-session

   docker run -d --restart=always --name opengeohub_pygeo_ide -v /mnt:/mnt -p 8888:8888 --user root opengeohub/pygeo-ide:v3.8.6-mkl-gdal314 jupyter lab --LabApp.token='opengeohub' --ServerApp.root_dir='/' --ip=0.0.0.0 --allow-root


As last step access the JupyterLab through http://localhost:8888 using the password **opengeohub**, open a terminal and install the last version of ``scikit-map``:

.. code-block:: shell-session
   
   pip install -e 'git+https://gitlab.com/geoharmonizer_inea/eumap.git#egg=eumap[full]'


.. image:: _static/docker_eumap.gif

Pip/`uv <https://docs.astral.sh/uv/>`_
======================================

``skmap`` can be installed with pip-compatible tooling. 

.. code-block:: shell-session

   sudo apt install libgdal-dev build-essential python3-dev git

   pip install "scikit-map[full] @ git+https://github.com/openlandmap/scikit-map@develop"


To install ``skmap`` using `conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_, first it's necessary download the file `conda_env.yml <https://gitlab.com/geoharmonizer_inea/eumap/-/raw/master/conda_env.yml>`_:

.. code-block:: shell-session

   curl https://gitlab.com/geoharmonizer_inea/eumap/-/raw/master/conda_env.yml > ./conda_env.yml

Then use this file to create new a environment and activate it:

.. code-block:: shell-session
   
   # It will take a while
   conda env create --quiet --name skmap --file conda_env.yml
   conda activate skmap

As last step install the ``skmap``:

.. code-block:: shell-session
   
   pip install -e 'git+https://gitlab.com/geoharmonizer_inea/eumap.git#egg=eumap[full]'


##################
Contributing
##################
The ``scikit-map`` library has been developed and used by a group of active community members. Your help is very valuable to make the package better for everyone. Check our `contribution guidelines <https://github.com/openlandmap/scikit-map/blob/main/CONTRIBUTING.md>`_ and `open issues <https://github.com/openlandmap/scikit-map/issues>`_

##################
License
##################
© Contributors, 2020. Licensed under an `MIT <https://github.com/openlandmap/scikit-map?tab=MIT-1-ov-file#readme>`_ license.

##################
Funding
##################

This work is co-financed under Grant Agreement Connecting Europe Facility (CEF) Telecom `project 2018-EU-IA-0095 <https://ec.europa.eu/inea/en/connecting-europe-facility/cef-telecom/2018-eu-ia-0095>`_ by the European Union.