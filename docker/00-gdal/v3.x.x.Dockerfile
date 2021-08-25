# This file is available at the option of the licensee under:
# Public domain or licensed under MIT (LICENSE.TXT)
ARG BASE_IMAGE=ubuntu:20.04
FROM $BASE_IMAGE

# Derived from osgeo/gdal:ubuntu-full
LABEL maintainer="Leandro Parente <leandro.parente@opengeohub.org>"

# Setup build env for PROJ
RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --fix-missing --no-install-recommends \
            software-properties-common build-essential ca-certificates \
            git make cmake wget unzip locales libtool automake \
            zlib1g-dev libsqlite3-dev pkg-config sqlite3 libcurl4-gnutls-dev \
            libtiff5-dev \
    && rm -rf /var/lib/apt/lists/*

# Setup locales
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen
ENV LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

# Build Proj
ARG PROJ_VERSION
RUN mkdir proj \
  && wget -q "https://github.com/OSGeo/PROJ/archive/${PROJ_VERSION}.tar.gz" \
    -O - | tar xz -C proj --strip-components=1 \
  && cd proj \
  && ./autogen.sh \
  && ./configure --prefix=/usr \
  && make -j$(nproc) \
  && make install \
  && cd .. \
  && rm -rf proj

# Build Geos
ARG GEOS_VERSION
RUN mkdir geos \
    && wget -q "https://github.com/libgeos/geos/archive/${GEOS_VERSION}.tar.gz" \
      -O - | tar xz -C geos --strip-components=1 \
    && cd geos \
    && ./autogen.sh \
    && ./configure --help \
    && ./configure prefix=/usr \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf geos

# Setup build env for Spatialite
RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --fix-missing --no-install-recommends \
            libfreexl-dev libxml2-dev libminizip-dev

ARG LIBSPATIALITE_VERSION=5.0.0
RUN mkdir libspatialite \
    && wget -q "http://www.gaia-gis.it/gaia-sins/libspatialite-sources/libspatialite-${LIBSPATIALITE_VERSION}.tar.gz" \
      -O - | tar xz -C libspatialite --strip-components=1 \
    && cd libspatialite \
    && ./configure \
        prefix=/usr \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf libspatialite

# Setup build env for GDAL
ARG JAVA_VERSION=11
RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --fix-missing --no-install-recommends \
       libcharls-dev libopenjp2-7-dev libcairo2-dev \
       python3-dev python3-numpy \
       libpng-dev libjpeg-dev libgif-dev liblzma-dev \
       curl libxml2-dev libexpat-dev libxerces-c-dev \
       libnetcdf-dev libpoppler-dev libpoppler-private-dev \
       swig ant libhdf4-alt-dev libhdf5-serial-dev \
       libfreexl-dev unixodbc-dev libwebp-dev libepsilon-dev \
       liblcms2-2 libpcre3-dev libcrypto++-dev libdap-dev libfyba-dev \
       libkml-dev libmysqlclient-dev libogdi-dev \
       libcfitsio-dev openjdk-"$JAVA_VERSION"-jdk libzstd-dev \
       libpq-dev libssl-dev libboost-dev \
       autoconf automake bash-completion libarmadillo-dev \
       libopenexr-dev libheif-dev \
       libdeflate-dev \
    && rm -rf /var/lib/apt/lists/*

# Build likbkea
ARG KEA_VERSION=1.4.13
RUN wget -q https://github.com/ubarsc/kealib/archive/kealib-${KEA_VERSION}.zip \
    && unzip -q kealib-${KEA_VERSION}.zip \
    && rm -f kealib-${KEA_VERSION}.zip \
    && cd kealib-kealib-${KEA_VERSION} \
    && cmake . -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr -DHDF5_INCLUDE_DIR=/usr/include/hdf5/serial \
        -DHDF5_LIB_PATH=/usr/lib/x86_64-linux-gnu/hdf5/serial -DLIBKEA_WITH_GDAL=OFF \
    && make -j$(nproc) \
    && make install \
    && make install \
    && cd .. \
    && rm -rf kealib-kealib-${KEA_VERSION}

# Build mongo-c-driver
ARG MONGO_C_DRIVER_VERSION=1.16.2
RUN mkdir mongo-c-driver \
    && wget -q https://github.com/mongodb/mongo-c-driver/releases/download/${MONGO_C_DRIVER_VERSION}/mongo-c-driver-${MONGO_C_DRIVER_VERSION}.tar.gz -O - \
        | tar xz -C mongo-c-driver --strip-components=1 \
    && cd mongo-c-driver \
    && mkdir build_cmake \
    && cd build_cmake \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr -DENABLE_TESTS=NO -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) \
    && make install \
    && make install \
    && cd ../.. \
    && rm -rf mongo-c-driver

# Build mongocxx
ARG MONGOCXX_VERSION=3.5.0
RUN mkdir mongocxx \
    && wget -q https://github.com/mongodb/mongo-cxx-driver/archive/r${MONGOCXX_VERSION}.tar.gz -O - \
        | tar xz -C mongocxx --strip-components=1 \
    && cd mongocxx \
    && mkdir build_cmake \
    && cd build_cmake \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr -DBSONCXX_POLY_USE_BOOST=ON -DMONGOCXX_ENABLE_SLOW_TESTS=NO -DCMAKE_BUILD_TYPE=Release -DBUILD_VERSION=${MONGOCXX_VERSION} \
    && make -j$(nproc) \
    && make install \
    && make install \
    && cd ../.. \
    && rm -rf mongocxx

# Build tiledb
ARG TILEDB_VERSION=2.0.8
RUN mkdir tiledb \
    && wget -q https://github.com/TileDB-Inc/TileDB/archive/${TILEDB_VERSION}.tar.gz -O - \
        | tar xz -C tiledb --strip-components=1 \
    && cd tiledb \
    && mkdir build_cmake \
    && cd build_cmake \
    && ../bootstrap --prefix=/usr \
    && make -j$(nproc) \
    && make install-tiledb \
    && make install-tiledb \
    && cd ../.. \
    && rm -rf tiledb

# Install MDB Driver Jars
RUN wget -q https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/mdb-sqlite/mdb-sqlite-1.0.2.tar.bz2 \
  && tar -xjf mdb-sqlite-1.0.2.tar.bz2 \
  && mkdir -p /build/usr/share/java \
  && cp mdb-sqlite-1.0.2/lib/*.jar /build/usr/share/java \
  && rm -rf mdb-sqlite-1.0.2.tar.bz2 && rm -rf mdb-sqlite-1.0.2

#Set CLASSPATH so jars are found
ENV CLASSPATH="/build/usr/share/java/jackcess-1.1.14.jar:/build/usr/share/java/commons-logging-1.1.1.jar:/build/usr/share/java/commons-lang-2.4.jar"

ARG GDAL_VERSION
ARG JAVA_VERSION=11
RUN mkdir gdal \
  && wget -q "https://github.com/OSGeo/gdal/archive/${GDAL_VERSION}.tar.gz" \
    -O - | tar xz -C gdal --strip-components=1 \
  && cd gdal/gdal \
  && ./configure --prefix=/usr \
    --without-libtool \
    --with-hide-internal-symbols \
    --with-jpeg12 \
    --with-python \
    --with-poppler \
    --with-spatialite \
    --with-mysql \
    --with-liblzma \
    --with-webp \
    --with-epsilon \
    --with-proj \
    --with-poppler \
    --with-hdf5 \
    --with-dods-root=/usr \
    --with-sosi \
    --with-libtiff=internal --with-rename-internal-libtiff-symbols \
    --with-geotiff=internal --with-rename-internal-libgeotiff-symbols \
    --with-kea=/usr/bin/kea-config \
    --with-mongocxxv3 \
    --with-tiledb \
    --with-crypto \
    --with-java=/usr/lib/jvm/java-"$JAVA_VERSION"-openjdk-amd64 --with-jvm-lib=/usr/lib/jvm/java-"$JAVA_VERSION"-openjdk-amd64/lib/server --with-jvm-lib-add-rpath \
    --with-mdb \
  && make "-j$(nproc)" \
  && make install \
  && cd swig/java \
  && make "-j$(nproc)" \
  && make install \
  && cd ../../../ \
  && rm -rf gdal

# Lib and bin dependencies
RUN apt-get update \
# PROJ dependencies
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        libsqlite3-0 libtiff5 libcurl4 \
        wget curl unzip ca-certificates \
# GDAL dependencies
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        libcharls2 libopenjp2-7 libcairo2 python3-numpy \
        libpng16-16 libjpeg-turbo8 libgif7 liblzma5 \
        libxml2 libexpat1 \
        libxerces-c3.2 libnetcdf-c++4 netcdf-bin libpoppler97 gpsbabel \
        libhdf4-0-alt libhdf5-103 libhdf5-cpp-103 poppler-utils libfreexl1 unixodbc libwebp6 \
        libepsilon1 liblcms2-2 libpcre3 libcrypto++6 libdap25 libdapclient6v5 libfyba0 \
        libkmlbase1 libkmlconvenience1 libkmldom1 libkmlengine1 libkmlregionator1 libkmlxsd1 \
        libmysqlclient21 libogdi4.1 libcfitsio8 openjdk-"$JAVA_VERSION"-jre \
        libzstd1 bash bash-completion libpq5 libssl1.1 \
        libarmadillo9 libpython3.8 libopenexr24 libheif1 \
        libdeflate0 \
        python-is-python3 \
    # Workaround bug in ogdi packaging
    && ln -s /usr/lib/ogdi/libvrf.so /usr/lib \
    && rm -rf /var/lib/apt/lists/*

RUN ldconfig \
    && projsync --system-directory --all

# Additional native packages
RUN apt-get update \
    && apt install -y htop parallel

#Set CLASSPATH so jars are found
ENV CLASSPATH="/usr/share/java/jackcess-1.1.14.jar:/usr/share/java/commons-logging-1.1.1.jar:/usr/share/java/commons-lang-2.4.jar"

# Remove the dev deps
RUN DEBIAN_FRONTEND=noninteractive apt-get remove -y \
        libarmadillo-dev libopenexr-dev libboost-dev libcairo2-dev python3-dev libcharls-dev libcrypto++-dev \
        libtiff5-dev libdap-dev libepsilon-dev libexpat-dev libfreexl-dev libfyba-dev libkml-dev \
        libgif-dev libhdf4-alt-dev libhdf5-serial-dev libfreexl-dev libheif-dev libdeflate-dev libjpeg-dev liblzma-dev \
        libminizip-dev libmysqlclient-dev libogdi-dev libcfitsio-dev libopenjp2-7-dev libpcre3-dev libpoppler-dev \
        libpoppler-private-dev libsqlite3-dev libssl-dev libwebp-dev libxerces-c-dev libnetcdf-dev libxml2-dev \
        libxml2-dev libzstd-dev libpq-dev libpng-dev unixodbc-dev zlib1g-dev \
    && apt-get autoremove -y \
    && apt-get clean -y