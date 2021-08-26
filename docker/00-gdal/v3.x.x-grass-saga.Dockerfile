# This file is available at the option of the licensee under:
# Public domain or licensed under MIT (LICENSE.TXT)
ARG BASE_IMAGE
FROM $BASE_IMAGE

# Derived from mundialis/grass-py3-pdal
LABEL maintainer="Leandro Parente <leandro.parente@opengeohub.org>"

# define versions to be used
ARG PDAL_VERSION
ARG LAZ_PERF_VERSION
ARG GRASS_VERSION
ARG SAGA_VERSION

SHELL ["/bin/bash", "-c"]

WORKDIR /tmp

RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --fix-missing --no-install-recommends --no-install-suggests \
        automake \
        bison \
        build-essential \
        bzip2 \
        ca-certificates \
        cmake \
        curl \
        flex \
        g++ \
        gcc \
        git \
        language-pack-en-base \
        libarmadillo-dev \
        libboost-dev \
        libbz2-dev \
        libcairo2 \
        libcairo2-dev \
        libcfitsio-dev \
        libcharls-dev \
        libcrypto++-dev \
        libcurl4-gnutls-dev \
        libdap-dev \
        libdeflate-dev \
        libepsilon-dev \
        libexpat-dev \
        libexpat1-dev \
        libfftw3-bin \
        libfftw3-dev \
        libfreetype6-dev \
        libfreexl-dev \
        libfyba-dev \
        libgeotiff-dev \
        libgeotiff5 \
        libgif-dev \
        libgsl0-dev \
        libhdf4-alt-dev \
        libhdf5-serial-dev \
        libheif-dev \
        libjpeg-dev \
        libjsoncpp-dev \
        libkml-dev \
        liblzma-dev \
        libminizip-dev \
        libwxbase3.0-0v5 \
        libmysqlclient-dev \
        libncurses5-dev \
        libnetcdf-dev \
        libogdi-dev \
        libopenblas-base \
        libopenblas-dev \
        libopenexr-dev \
        libopenjp2-7 \
        libopenjp2-7-dev \
        libpcre3-dev \
        libpng-dev \
        libpnglite-dev \
        libpoppler-dev \
        libpoppler-private-dev \
        libpq-dev \
        libpython3-all-dev \
        libsqlite3-dev \
        libssl-dev \
        libtiff-dev \
        libtiff5-dev \
        libtool \
        libwebp-dev \
        libwxbase3.0-dev \
        libwxgtk3.0-gtk3-dev \
        libxerces-c-dev \
        libxml2-dev \
        libzstd-dev \
        locales \
        make \
        mesa-common-dev \
        moreutils \
        ncurses-bin \
        netcdf-bin \
        pkg-config \
        python3 \
        python3-dateutil \
        python3-dev \
        python3-magic \
        python3-numpy \
        python3-pil \
        python3-pip \
        python3-ply \
        python3-setuptools \
        python3-venv \
        python3-wxgtk4.0 \
        software-properties-common \
        sqlite3 \
        subversion \
        unixodbc-dev \
        unzip \
        vim \
        wget \
        wx-common \
        xz-utils \
        zip \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN echo LANG="en_US.UTF-8" > /etc/default/locale
RUN echo en_US.UTF-8 UTF-8 >> /etc/locale.gen && locale-gen

## install laz-perf
WORKDIR /src
RUN wget -q https://github.com/hobu/laz-perf/archive/${LAZ_PERF_VERSION}.tar.gz -O laz-perf-${LAZ_PERF_VERSION}.tar.gz && \
    tar -zxf laz-perf-${LAZ_PERF_VERSION}.tar.gz && \
    cd laz-perf-${LAZ_PERF_VERSION} && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install

## fetch vertical datums for PDAL and store into PROJ dir
WORKDIR /src
RUN mkdir vdatum && \
    cd vdatum && \
    wget -q http://download.osgeo.org/proj/vdatum/usa_geoid2012.zip && unzip -j -u usa_geoid2012.zip -d /usr/share/proj; \
    wget -q http://download.osgeo.org/proj/vdatum/usa_geoid2009.zip && unzip -j -u usa_geoid2009.zip -d /usr/share/proj; \
    wget -q http://download.osgeo.org/proj/vdatum/usa_geoid2003.zip && unzip -j -u usa_geoid2003.zip -d /usr/share/proj; \
    wget -q http://download.osgeo.org/proj/vdatum/usa_geoid1999.zip && unzip -j -u usa_geoid1999.zip -d /usr/share/proj; \
    wget -q http://download.osgeo.org/proj/vdatum/vertcon/vertconc.gtx && mv vertconc.gtx /usr/share/proj; \
    wget -q http://download.osgeo.org/proj/vdatum/vertcon/vertcone.gtx && mv vertcone.gtx /usr/share/proj; \
    wget -q http://download.osgeo.org/proj/vdatum/vertcon/vertconw.gtx && mv vertconw.gtx /usr/share/proj; \
    wget -q http://download.osgeo.org/proj/vdatum/egm96_15/egm96_15.gtx && mv egm96_15.gtx /usr/share/proj; \
    wget -q http://download.osgeo.org/proj/vdatum/egm08_25/egm08_25.gtx && mv egm08_25.gtx /usr/share/proj; \
    cd .. && \
    rm -rf vdatum

## install pdal
WORKDIR /src
RUN wget -q \
 https://github.com/PDAL/PDAL/releases/download/${PDAL_VERSION}/PDAL-${PDAL_VERSION}-src.tar.gz && \
    tar xfz PDAL-${PDAL_VERSION}-src.tar.gz && \
    cd /src/PDAL-${PDAL_VERSION}-src && \
    mkdir build && \
    cd build && \
    cmake .. \
      -G "Unix Makefiles" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_MAKE_PROGRAM=make \
      -DBUILD_PLUGIN_PYTHON=ON \
      -DBUILD_PLUGIN_CPD=OFF \
      -DBUILD_PLUGIN_GREYHOUND=ON \
      -DBUILD_PLUGIN_HEXBIN=ON \
      -DHEXER_INCLUDE_DIR=/usr/include/ \
      -DBUILD_PLUGIN_NITF=OFF \
      -DBUILD_PLUGIN_ICEBRIDGE=ON \
      -DBUILD_PLUGIN_PGPOINTCLOUD=ON \
      -DBUILD_PGPOINTCLOUD_TESTS=OFF \
      -DBUILD_PLUGIN_SQLITE=ON \
      -DWITH_LASZIP=ON \
      -DWITH_LAZPERF=ON \
      -DWITH_TESTS=ON && \
    make "-j$(nproc)" && \
    make install

# Set environmental variables for GRASS GIS compilation, without debug symbols
# Set gcc/g++ environmental variables for GRASS GIS compilation, without debug symbols
ENV MYCFLAGS "-O2 -std=gnu99 -m64"
ENV MYLDFLAGS "-s"
# CXX stuff:
ENV LDFLAGS "$MYLDFLAGS"
ENV CFLAGS "$MYCFLAGS"
ENV CXXFLAGS "$MYCXXFLAGS"
ENV GRASS_PYTHON=/usr/bin/python3

# copy grass gis source
WORKDIR /src
RUN mkdir grass \
# TODO: to be change when 8.0.0 will be released
##    && wget -q https://github.com/OSGeo/grass/archive/refs/tags/${GRASS_VERSION}.tar.gz \
    && wget -q https://github.com/OSGeo/grass/archive/refs/heads/main.tar.gz \
    -O - | tar xz -C grass --strip-components=1 \
    && cd grass \
    && rm -f dist.*/demolocation/.grassrc7? \
    && make distclean || echo "nothing to clean"  \
    && ./configure \
      --with-cxx \
      --prefix=/usr \
      --enable-largefile \
      --with-proj --with-proj-share=/usr/share/proj \
      --with-gdal=/usr/bin/gdal-config \
      --with-geos \
      --with-sqlite \
      --with-cairo --with-cairo-ldflags=-lfontconfig \
      --with-freetype --with-freetype-includes="/usr/include/freetype2/" \
      --with-fftw \
      --with-postgres=yes --with-postgres-includes="/usr/include/postgresql" \
      --with-netcdf \
      --with-zstd \
      --with-bzlib \
      --with-pdal \
      --without-mysql \
      --without-odbc \
      --without-openmp \
      --without-ffmpeg \
      --without-opengl  \
    && make "-j$(nproc)" \
    && make install \
    && ldconfig \
#    && ln -sf /usr/bin/grass80 /usr/bin/grass \
    && mv /usr/grass80 /usr/grass \
    && cd .. \
    && rm -rf grass

# Unset environmental variables to avoid later compilation issues
ENV INTEL ""
ENV MYCFLAGS ""
ENV MYLDFLAGS ""
ENV MYCXXFLAGS ""
ENV LDFLAGS ""
ENV CFLAGS ""
ENV CXXFLAGS ""

# set SHELL var to avoid /bin/sh fallback in interactive GRASS GIS sessions
ENV SHELL /bin/bash
ENV LC_ALL "en_US.UTF-8"
ENV GRASS_SKIP_MAPSET_OWNER_CHECK 1

# show GRASS GIS, PROJ, GDAL etc versions
#RUN grass --tmp-location EPSG:4326 --exec g.version -rge && \
#    pdal --version && \
#    python3 --version

WORKDIR /scripts

# install external GRASS GIS session Python API
RUN pip3 install grass-session

# install GRASS GIS extensions
#RUN grass --tmp-location EPSG:4326 --exec g.extension extension=r.in.pdal

# add GRASS GIS envs for python usage
ENV GISBASE "/usr/grass/"
ENV GRASSBIN "/usr/bin/grass"
ENV PYTHONPATH "${PYTHONPATH}:$GISBASE/etc/python/"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:$GISBASE/lib"

WORKDIR /src
RUN mkdir saga \
    && wget https://sourceforge.net/projects/saga-gis/files/SAGA%20-%207/SAGA%20-%20${SAGA_VERSION}/saga-${SAGA_VERSION}.tar.gz/download \
      -O - | tar xz -C saga --strip-components=1 \
    && cd saga \
    && autoreconf -fi \
    && ./configure --prefix=/usr --disable-gui \
    && make "-j$(nproc)" \
    && make install \
    && cd .. \
    && rm -rf saga


# Whitebox Tools installation
WORKDIR /opt/
RUN wget https://www.uoguelph.ca/~hydrogeo/WhiteboxTools/WhiteboxTools_linux_amd64.tar.xz \
      && tar -xf WhiteboxTools_linux_amd64.tar.xz \
      && ln -s /opt/WBT/whitebox_tools /usr/bin/whitebox_tools \
      && rm WhiteboxTools_linux_amd64.tar.xz

# Remove the dev deps
RUN DEBIAN_FRONTEND=noninteractive apt-get remove -y \
        libarmadillo-dev libboost-dev libbz2-dev libcairo2-dev \
        libcfitsio-dev libcharls-dev libcrypto++-dev libcurl4-gnutls-dev \
        libdap-dev libdeflate-dev libgeotiff-dev libepsilon-dev libexpat-dev \
        libfftw3-dev libfreetype6-dev libfreexl-dev libfyba-dev libgif-dev libgsl0-dev \
        libhdf4-alt-dev libhdf5-serial-dev libheif-dev libjpeg-dev libjsoncpp-dev \
        libkml-dev liblzma-dev libminizip-dev libmysqlclient-dev libncurses5-dev libnetcdf-dev libogdi-dev \
        libopenblas-dev libopenexr-dev libopenjp2-7-dev libpcre3-dev libpng-dev libpnglite-dev libpoppler-dev \
        libpoppler-private-dev libpq-dev libpython3-all-dev libsqlite3-dev libssl-dev libtiff-dev libtiff5-dev \
        libexpat1-dev wx-common unixodbc-dev libogdi-dev libwxbase3.0-dev libwxgtk3.0-gtk3-dev \
        libwebp-dev libxerces-c-dev libxml2-dev libzstd-dev mesa-common-dev python3-dev unixodbc-dev zlib1g-dev \
    && apt-get autoremove -y \
    && apt-get clean -y