# This file is available at the option of the licensee under:
# Public domain or licensed under MIT (LICENSE.TXT)
ARG BASE_IMAGE
FROM $BASE_IMAGE

# Derived from jupyter/datascience-notebook
LABEL maintainer="Leandro Parente <leandro.parente@opengeohub.org>"

# Fix DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root

# Configure env variables
ARG NB_USER="root"
ARG PYTHON_DIR
ENV PYTHON_DIR=$PYTHON_DIR \
    DEBIAN_FRONTEND=noninteractive \
    NB_USER=$NB_USER \
    NB_UID=$NB_UID \
    SHELL=/bin/bash \
    NB_GID=$NB_GID

ENV XDG_CACHE_HOME="/root/.cache/" \
    HOME=/root

WORKDIR /tmp

# Install all OS dependencies
RUN apt-get update \
 && apt-get install -yq --no-install-recommends fonts-liberation pandoc run-one sudo cm-super dvipng ffmpeg \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Copy local files as late as possible to avoid cache busting
COPY files/start.sh files/start-notebook.sh files/start-singleuser.sh /usr/local/bin/
COPY files/jupyter_notebook_config.py /etc/jupyter/

# Enable prompt color in the skeleton .bashrc before creating the default NB_USER
# hadolint ignore=SC2016
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc \
   # Add call to conda init script see https://stackoverflow.com/a/58081608/4413446
   && echo 'eval "$(command conda shell.bash hook 2> /dev/null)"' >> /etc/skel/.bashrc 

# Install Tini
RUN apt-get update \
    && apt-get install -yq tini nodejs npm

# Install JupyterLab
RUN pip install jupyterlab geemap ipywidgets ipyleaflet ipympl \
        jupyter_bokeh jupyterlab-spellchecker jupyterlab_widgets widgetsnbextension \
    && jupyter lab --generate-config

EXPOSE 8888

# Configure container startup
ENTRYPOINT ["tini", "-g", "--"]
CMD ["start-notebook.sh"]
