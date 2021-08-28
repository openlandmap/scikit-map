# This file is available at the option of the licensee under:
# Public domain or licensed under MIT (LICENSE.TXT)
ARG BASE_IMAGE
FROM $BASE_IMAGE

# Derived from rocker/r-ubuntu and rocker/geospatial
LABEL maintainer="Leandro Parente <leandro.parente@opengeohub.org>"

# Configure env variables
ARG RS_GID="100"
ARG RS_UID="1000"
ARG RS_USER="opengeohub"
ARG S6_VERSION="v1.21.7.0"
ARG PANDOC_TEMPLATES_VERSION="2.9"
ARG RSTUDIO_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    RSTUDIO_VERSION=$RSTUDIO_VERSION \
    S6_VERSION=$S6_VERSION \
    PANDOC_TEMPLATES_VERSION=PANDOC_TEMPLATES_VERSION \
    RS_USER=$RS_USER \
    RS_UID=$RS_UID \
    SHELL=/bin/bash \
    RS_GID=$RS_GID \
    PATH=/usr/lib/rstudio-server/bin:$PATH

ENV PATH=/usr/lib/rstudio-server/bin:$PATH \
    HOME=/home/$RS_USER

## Download and install RStudio server & dependencies
## Attempts to get detect latest version, otherwise falls back to version given in $VER
## Symlink pandoc, pandoc-citeproc so they are available system-wide
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    file \
    git \
    libapparmor1 \
    libclang-dev \
    libcurl4-openssl-dev \
    libedit2 \
    libssl-dev \
    lsb-release \
    psmisc \
    procps \
    python-setuptools \
    sudo \
    wget \
  && if [ -z "$RSTUDIO_VERSION" ]; \
    then RSTUDIO_URL="https://www.rstudio.org/download/latest/stable/server/bionic/rstudio-server-latest-amd64.deb"; \
    else RSTUDIO_URL="http://download2.rstudio.org/server/bionic/amd64/rstudio-server-${RSTUDIO_VERSION}-amd64.deb"; fi \
  && wget -q $RSTUDIO_URL \
  && dpkg -i rstudio-server-*-amd64.deb \
  && rm rstudio-server-*-amd64.deb

# S6 setup to execute rstudio on start
RUN echo '\n\
    \n# Configure httr to perform out-of-band authentication if HTTR_LOCALHOST \
    \n# is not set since a redirect to localhost may not work depending upon \
    \n# where this Docker container is running. \
    \nif(is.na(Sys.getenv("HTTR_LOCALHOST", unset=NA))) { \
    \n  options(httr_oob_default = TRUE) \
    \n}' >> /etc/Rprofile.site \
  && echo "PATH=${PATH}" >> /etc/Renviron \
  ## Need to configure non-root user for RStudio
  && useradd $RS_USER \
  && echo "$RS_USER:$RS_USER" | chpasswd \
  && mkdir $HOME \
  && chown "$RS_USER:$RS_USER" $HOME \
  && addgroup $RS_USER staff \
  ## use more robust file locking to avoid errors when using shared volumes:
  && echo 'lock-type=advisory' >> /etc/rstudio/file-locks \
  ## configure git not to request password each time
  && git config --system credential.helper 'cache --timeout=3600' \
  && git config --system push.default simple \
  && wget -P /tmp/ https://github.com/just-containers/s6-overlay/releases/download/${S6_VERSION}/s6-overlay-amd64.tar.gz \
  && tar xzf /tmp/s6-overlay-amd64.tar.gz -C / --exclude='./bin' && tar xzf /tmp/s6-overlay-amd64.tar.gz -C /usr ./bin \
  && mkdir -p /etc/services.d/rstudio \
  && echo '#!/usr/bin/with-contenv bash \
          \n## load /etc/environment vars first: \
        \n for line in $( cat /etc/environment ) ; do export $line ; done \
          \n exec /usr/lib/rstudio-server/bin/rserver --server-daemonize 0' \
          > /etc/services.d/rstudio/run \
  && echo '#!/bin/bash \
          \n rstudio-server stop' \
          > /etc/services.d/rstudio/finish \
  && mkdir -p $HOME/.rstudio/monitored/user-settings \
  && echo 'alwaysSaveHistory="0" \
          \nloadRData="0" \
          \nsaveAction="0"' \
          > $HOME/.rstudio/monitored/user-settings/user-settings \
  && chown -R $RS_USER:$RS_USER $HOME/.rstudio

COPY files/userconf.sh /etc/cont-init.d/userconf
COPY files/add_shiny.sh /etc/cont-init.d/add
COPY files/disable_auth_rserver.conf /etc/rstudio/disable_auth_rserver.conf
COPY files/pam-helper.sh /usr/lib/rstudio-server/bin/pam-helper

EXPOSE 8787

ENTRYPOINT [""]
CMD ["/init"]