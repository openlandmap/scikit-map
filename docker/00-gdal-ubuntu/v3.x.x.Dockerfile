# This file is available at the option of the licensee under:
# Public domain or licensed under MIT (LICENSE.TXT)
ARG BASE_IMAGE=ubuntu:24.04
FROM $BASE_IMAGE

LABEL maintainer="Leandro Parente <leandro.parente@opengeohub.org>"

# Setup ubuntugis-unstable
RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --fix-missing --no-install-recommends \
        software-properties-common python3

RUN apt-get install -y gpg-agent \
    && add-apt-repository ppa:ubuntugis/ubuntugis-unstable

# Setup locales
RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --fix-missing --no-install-recommends \
        locales \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen
ENV LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

# Install gdal, geos and proj
RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --fix-missing --no-install-recommends \
        gdal-bin gdal-data geos-bin libgeos-dev proj-bin proj-data \
        libgdal-dev libgeos++-dev libproj-dev htop parallel nano

# Install oneapi-base-toolkit
RUN apt install -y gpg-agent wget sudo \
    && wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list \
    && apt update \
    && apt install -y intel-oneapi-runtime-libs

RUN apt install -y intel-oneapi-mkl

# Clean apt
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*