# This file is available at the option of the licensee under:
# Public domain or licensed under MIT (LICENSE.TXT)
ARG BASE_IMAGE=intel/oneapi-basekit:2021.4-devel-ubuntu18.04
FROM $BASE_IMAGE

# Derived from jupyter/datascience-notebook
LABEL maintainer="Leandro Parente <leandro.parente@opengeohub.org>"

USER root
ENV PYTHON_DIR=/opt/intel/oneapi/intelpython/latest \
    DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash

ENV PATH=$PYTHON_DIR/bin:$PATH

# Install all OS dependencies
RUN apt-get update \
 && apt-get install -y --fix-missing --no-install-recommends \
    build-essential bzip2 ca-certificates gcc gfortran git locales \
    sudo tzdata unzip wget htop parallel locales \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Setup locales
RUN apt-get install -y --fix-missing --no-install-recommends \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen
ENV LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

# Install Conda
WORKDIR /tmp
RUN conda config --system --set auto_update_conda false && \
    conda config --system --set show_channel_urls true && \
    conda install -y mamba -n base -c conda-forge && \
    mamba clean --all -f -y

# Install Tini
RUN mamba install -c intel --no-update-deps --quiet --yes 'tini=0.18.0' && \
    mamba list tini | grep tini | tr -s ' ' | cut -d ' ' -f 1,2 >> $PYTHON_DIR/conda-meta/pinned && \
    mamba clean --all -f -y

# Install Conda packages
ARG GDAL_VERSION
RUN mamba install -c conda-forge --yes \
    "bottleneck" \
    "gdal=${GDAL_VERSION}" \
    "geos" \
    "h5py" \
    "joblib" \
    "lz4" \
    "minio" \
    "nodejs=12" \
    "owslib" \
    "proj" \
    "protobuf" \
    "rasterio" \
    "seaborn" \
    "shapely" \
    "statsmodels" \
    "xgboost"

#Encountered problems while solving:
#  - nothing provides requested auto-sklearn
#  - nothing provides requested datatable
#  - nothing provides requested pygeos
#  - nothing provides requested pyts
#  - nothing provides requested vincent
#  - nothing provides libcurl 7.83.0 h0b77cf5_0 needed by curl-7.83.0-h7f8727e_0
#  - package geopandas-0.9.0-py_1 requires fiona, but none of the providers can be installed
#  - package libarchive-3.5.2-hacfb022_0 requires openssl >=1.1.1n,<1.1.2a, but none of the providers can be installed


# Install pip packages.
# Avoid it unless it's not available in Conda
RUN pip install mlens pqdm

# Install eumap
RUN wget http://es.archive.ubuntu.com/ubuntu/pool/main/libf/libffi/libffi7_3.3-4_amd64.deb && \
     dpkg -i libffi7_3.3-4_amd64.deb && \
     python -m pip install -U git+https://gitlab.com/geoharmonizer_inea/eumap.git && \
     rm -f libffi7_3.3-4_amd64.deb 

# Install and use ipython as entrypoint
RUN mamba install -c intel --no-update-deps --quiet --yes \
    "ipython"

ENTRYPOINT ["ipython"]