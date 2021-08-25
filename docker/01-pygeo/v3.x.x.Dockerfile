# This file is available at the option of the licensee under:
# Public domain or licensed under MIT (LICENSE.TXT)
ARG BASE_IMAGE
FROM $BASE_IMAGE

# Derived from jupyter/datascience-notebook
LABEL maintainer="Leandro Parente <leandro.parente@opengeohub.org>"

# Fix DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root

# Configure environment
# To access the available blas implementations
# see https://conda-forge.org/docs/maintainer/knowledge_base.html#blas
ARG MINICONDA_SUFFIX="py38"
ARG MINICONDA_VERSION="4.8.3"
ARG MINICONDA_CHECKSUM="d63adf39f2c220950a063e0529d4ff74"
ARG BLAS_IMPLEMENTATION
ARG PYTHON_VERSION
ARG GDAL_VERSION
ARG GEOS_VERSION
ENV CONDA_DIR=/opt/conda \
    DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash

ENV PATH=$CONDA_DIR/bin:$PATH

# Install all OS dependencies for notebook server
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
    build-essential bzip2 ca-certificates cm-super dvipng emacs-nox ffmpeg fonts-dejavu fonts-liberation gcc gfortran \
    git inkscape jed libsm6 libxext-dev libxrender1 lmodern locales netcat python-dev run-one sudo texlive-fonts-recommended \
    texlive-plain-generic texlive-xetex tzdata unzip wget \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install Conda
WORKDIR /tmp
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_SUFFIX}_${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "${MINICONDA_CHECKSUM} *Miniconda3-${MINICONDA_SUFFIX}_${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash Miniconda3-${MINICONDA_SUFFIX}_${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-${MINICONDA_SUFFIX}_${MINICONDA_VERSION}-Linux-x86_64.sh && \
    # Conda configuration see https://conda.io/projects/conda/en/latest/configuration.html
    echo "conda ${CONDA_VERSION}" >> $CONDA_DIR/conda-meta/pinned && \
    conda update conda && \
    conda config --system --prepend channels conda-forge && \
    conda config --system --set auto_update_conda false && \
    conda config --system --set show_channel_urls true && \
    conda config --system --set channel_priority strict && \
    if [ ! $PYTHON_VERSION = 'default' ]; then conda install --yes "python=${PYTHON_VERSION}"; fi && \
    conda list python | grep '^python ' | tr -s ' ' | cut -d '.' -f 1,2 | sed 's/$/.*/' >> $CONDA_DIR/conda-meta/pinned && \
    conda install --quiet --yes "conda=${CONDA_VERSION}" && \
    conda install --quiet --yes pip && \
    conda update --all --quiet --yes && \
    conda update -n base -c defaults conda && \
    conda clean --all -f -y

# Install Conda packages
RUN conda install --quiet --yes \
    "beautifulsoup4=4.9.*" \
    "bokeh=2.3.*" \
    "bottleneck=1.3.*" \
    "cloudpickle=1.6.*" \
    "cython=0.29.*" \
    "daal4py=2021.3.*" \
    "dask=2021.8.*" \
    "dill=0.3.*" \
    "gdal=${GDAL_VERSION}" \
    "geopandas=0.9.*" \
    "geos=${GEOS_VERSION}" \
    "h5py=3.3.*" \
    "joblib=1.0.*" \
    "libblas=*=*${BLAS_IMPLEMENTATION}" \
    "libiconv=1.16" \
    "lz4=3.1.*" \
    "matplotlib=3.3.*" \
    "minio=7.1.*" \
    "nodejs=12" \
    "numba=0.51.*" \
    "numexpr=2.7.*" \
    "numpy=1.21.*" \
    "opencv=4.5.*" \
    "owslib=0.24.*" \
    "pandas=1.3.*" \
    "patsy=0.5.*" \
    "proj=7.1.*" \
    "protobuf=3.17.*" \
    "pygeos=0.8" \
    "pytables=3.6.*" \
    "pyts=0.11.*" \
    "rasterio=1.2.*" \
    "requests=2.26.*" \
    "scikit-image=0.18.*" \
    "scikit-learn=0.24.*" \
    "scipy=1.7.*" \
    "seaborn=0.11.*" \
    "shapely=1.7.*" \
    "sqlalchemy=1.4.*" \
    "statsmodels=0.12.*" \
    "sympy=1.8" \
    "vincent=0.4.*" \
    "xgboost=1.4.*" \
    "xlrd=2.0.*"

# Install pip packages.
# Avoid it unless it's not available in Conda
RUN pip install mlens pqdm

# FIXME: point it to master
#RUN python -m pip install -U git+https://gitlab.com/geoharmonizer_inea/eumap.git#pyeumap@documentation

# Ad-hoc packages install 
# to keep the build cache
RUN conda install --quiet --yes \
    "datatable=0.11.*" \
    "auto-sklearn=0.12.*" \
    "scipy==1.6.*"

# Install and use ipython as entrypoint
RUN conda install --quiet --yes \
    "ipython"

ENTRYPOINT ["ipython"]