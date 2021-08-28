# This file is available at the option of the licensee under:
# Public domain or licensed under MIT (LICENSE.TXT)
ARG BASE_IMAGE
FROM $BASE_IMAGE

# Derived from rocker/r-ubuntu and rocker/geospatial
LABEL maintainer="Leandro Parente <leandro.parente@opengeohub.org>"

# Build deps and basic packages
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    software-properties-common dirmngr ed gpg-agent less locales \
    vim wget less libgomp1 libpango-1.0-0 libxt6 libsm6 g++ ca-certificates \
    libtinfo5 build-essential pkg-config libssl-dev zlib1g-dev automake gfortran libxml2-dev make

ENV DEBIAN_FRONTEND noninteractive

## Otherwise timedatectl will get called which leads to 'no systemd' inside Docker
ENV TZ UTC

WORKDIR /tmp

# Now install R and littler, and create a link for littler in /usr/local/bin
# Default CRAN repo is now set by R itself, and littler knows about it too
# r-cran-docopt is not currently in c2d4u so we install from source
ARG UBUNTU_VERSION='focal'
ARG R_VERSION
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 \
  && add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu ${UBUNTU_VERSION}-cran40/" \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
     littler r-base=${R_VERSION} r-base-dev=${R_VERSION} r-recommended=${R_VERSION} \
  && ln -s /usr/lib/R/site-library/littler/examples/install.r /usr/bin/install.r \
  && ln -s /usr/lib/R/site-library/littler/examples/install2.r /usr/bin/install2.r \
  && ln -s /usr/lib/R/site-library/littler/examples/installGithub.r /usr/bin/installGithub.r \
  && ln -s /usr/lib/R/site-library/littler/examples/testInstalled.r /usr/bin/testInstalled.r \
  && rm -rf /tmp/downloaded_packages/ /tmp/*.rds \
  && rm -rf /var/lib/apt/lists/*

# Openblas installation and linking with R 
# (see https://github.com/eddelbuettel/mkl4deb/blob/master/script.sh)
ARG OPENBLAS_VERSION="0.3.15"
RUN  mkdir openblas \
 && wget -q "https://github.com/xianyi/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz" \
      -O - | tar xz -C openblas --strip-components=1 \
  && cd openblas \
  && make -j$(nproc) FC=gfortran TARGET=ZEN libs netlib re_lapack shared \
  && make PREFIX=/usr install \
  && update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so     libblas.so-x86_64-linux-gnu      /usr/lib/libopenblas.so 150 \
  && update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so.3   libblas.so.3-x86_64-linux-gnu    /usr/lib/libopenblas.so 150 \
  && update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so   liblapack.so-x86_64-linux-gnu    /usr/lib/libopenblas.so 150 \
  && update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so.3 liblapack.so.3-x86_64-linux-gnu  /usr/lib/libopenblas.so 150 \
  && Rscript -e 'install.packages(c( "RhpcBLASctl" ))' \
  && ldconfig

# Use libcurl for download to avoid problems with tar files
# and set 1 for number of threads for BLAS.
# Put other R default configuration here
RUN printf 'options("download.file.method" = "libcurl", Ncpus=parallel::detectCores());\
\n local({\
  \n if(require("RhpcBLASctl", quietly=TRUE)){ \
    \n if(Sys.getenv("MKL_NUM_THREADS")=="" & \
      \n Sys.getenv("OMP_NUM_THREADS")=="") \
      \n blas_set_num_threads(1) \
      \n Sys.setenv(MKL_NUM_THREADS=1) \
  \n } \
\n });\
' >> /etc/R/Rprofile.site 

# Ubuntu deps to build the R packages
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    lbzip2 libudunits2-0 libfftw3-dev libgsl0-dev libgl1-mesa-dev libglu1-mesa-dev libhdf4-alt-dev libhdf5-dev libjq-dev libpq-dev \
    libprotobuf-dev libnetcdf-dev libsqlite3-dev libssl-dev libudunits2-dev netcdf-bin protobuf-compiler sqlite3 tk-dev unixodbc-dev \
    libsodium-dev libssl-dev libsasl2-dev

# General packages
RUN Rscript -e "install.packages(c('docopt'))"
RUN install2.r --error --libloc /usr/lib/R/site-library \
    Rcpp \
    remotes \
    lattice

# Statistics, ML and spatial packages
RUN install2.r --error --libloc /usr/lib/R/site-library \
    bfast \
    caret \
    chron \
    classInt \
    config \
    deldir \
    DistributionUtils \
    doMC \
    dplyr \
    DT \
    fields \
    gdalcubes \
    geojsonsf \
    geoR \
    geosphere \
    ggnewscale \
    glmnet \
    gstat \
    hdf5r \
    Kendall \
    landmap \
    lidR \
    mapdata \
    maptools \
    mapview \
    mlr \
    mlr3 \
    mlr3learners \
    mlr3pipelines \
    mlr3filters \
    mlr3misc \
    mlr3tuning \
    mlr3verse \
    mongolite \
    ncdf4 \
    nnet \
    phenopix \
    plotrix \
    pls \
    plumber \
    proj4 \
    quantreg \
    RandomFields \
    randomForest \
    ranger \
    raster \
    RColorBrewer \
    readr \
    rgdal \
    rgeos \
    rlas \
    RNetCDF \
    Rssa \
    sf \
    sp \
    spacetime \
    spatstat \
    spdep \
    sqldf \
    SuperLearner \
    xgboost

# Packages from github
RUN export MAKEFLAGS="-j$(nproc --all)" && \
    installGithub.r \
    rspatial/terra \
    nagdevAmruthnath/minio.s3 \
    mlr-org/mlr3extralearners

# Ad-hoc packages install to keep 
# the build cache
RUN install2.r --error --libloc /usr/lib/R/site-library \
    plotKML \
    kernlab \
    deepnet

# Remove the dev deps
RUN DEBIAN_FRONTEND=noninteractive apt-get remove -y \
      libssl-dev zlib1g-dev libxml2-dev libfftw3-dev libgsl0-dev libgl1-mesa-dev \
      libglu1-mesa-dev libhdf4-alt-dev libhdf5-dev libjq-dev libpq-dev libprotobuf-dev \
      libnetcdf-dev libsqlite3-dev libssl-dev libudunits2-dev tk-dev unixodbc-dev \
      libsodium-dev libssl-dev libsasl2-dev  \
    && apt-get autoremove -y \
    && apt-get clean -y

ENTRYPOINT ["R"]