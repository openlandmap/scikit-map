# This file is available at the option of the licensee under:
# Public domain or licensed under MIT (LICENSE.TXT)
ARG BASE_IMAGE
FROM $BASE_IMAGE

LABEL maintainer="Leandro Parente <leandro.parente@opengeohub.org>"

# Fix DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV VENV=/skmap
USER root
WORKDIR /

# Install all OS dependencies
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
    build-essential bzip2 ca-certificates gcc gfortran git  \
    sudo tzdata unzip wget htop parallel locales software-properties-common bash cmake \
 && add-apt-repository universe \
 && apt-get install -yq --no-install-recommends python3-pip python3-venv python3-dev      

RUN python3 -m venv $VENV \
 && . $VENV/bin/activate \
 && pip install --no-cache-dir --upgrade pip setuptools wheel numpy

RUN . $VENV/bin/activate \
# && pip install gdal==\$\(gdalinfo --version \| cut -d\\\  -f2\) \ # Installing GDAL according to OS version
 && pip install "scikit-map[full]" 'git+https://github.com/openlandmap/scikit-map@setup_cmake'

RUN sh -c "$VENV/bin/pip install gdal==$(gdalinfo --version | cut -d\  -f2)"

#RUN .skmap/bin/pip install -i https://software.repos.intel.com/python/pypi numpy scipy dpnp dpctl tbb4py \
#    && .skmap/bin/pip install smp cython numba scikit-learn-intelex

ARG EXTRA_OPT
RUN . $VENV/bin/activate \
    && if [[ "$EXTRA_OPT" = "intel" ]] ; then pip install -i https://software.repos.intel.com/python/pypi numpy scipy dpnp dpctl tbb4py  ; fi \
    && if [[ "$EXTRA_OPT" = "intel" ]] ; then pip install smp cython numba scikit-learn-intelex  ; fi

# Clean apt
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="$VENV/bin:$PATH"
ENTRYPOINT ["python3"]
