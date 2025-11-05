# This file is available at the option of the licensee under:
# Public domain or licensed under MIT (LICENSE.TXT)
ARG BASE_IMAGE=opengeohub/gdal-ubuntu:v3.4.3
FROM $BASE_IMAGE

LABEL maintainer="Leandro Parente <leandro.parente@opengeohub.org>"

# Setup ubuntugis-unstable
RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --fix-missing --no-install-recommends \
        grass grass-core libgdal-grass grass-dev \
        saga saga-common libsaga-dev

# Clean apt
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*