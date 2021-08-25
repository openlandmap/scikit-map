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
ARG NB_GID="100"
ARG NB_UID="1000"
ARG NB_USER="jupyter"
ARG PYTHON_VERSION="3.7.4"
ARG CONDA_VERSION="4.9.2"
ARG MINICONDA_VERSION="4.8.3"
ARG MINICONDA_SUFFIX="py37"
ARG MINICONDA_CHECKSUM="751786b92c00b1aeae3f017b781018df"
ENV CONDA_DIR=/opt/conda \
    DEBIAN_FRONTEND=noninteractive \
    NB_USER=$NB_USER \
    NB_UID=$NB_UID \
    SHELL=/bin/bash \
    NB_GID=$NB_GID

ENV XDG_CACHE_HOME="/home/${NB_USER}/.cache/" \
    PATH=$CONDA_DIR/bin:$PATH \
    HOME=/home/$NB_USER

# Install all OS dependencies for notebook server
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
    build-essential bzip2 ca-certificates cm-super dvipng emacs-nox ffmpeg fonts-dejavu fonts-liberation gcc gfortran \
    git inkscape jed libsm6 libxext-dev libxrender1 lmodern netcat python-dev run-one sudo texlive-fonts-recommended \
    texlive-plain-generic texlive-xetex tzdata unzip wget \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Copy a script that we will use to correct permissions after running certain commands
COPY files/fix-permissions /usr/local/bin/fix-permissions
RUN chmod a+rx /usr/local/bin/fix-permissions

# Enable prompt color in the skeleton .bashrc before creating the default NB_USER
# hadolint ignore=SC2016
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc \
   # Add call to conda init script see https://stackoverflow.com/a/58081608/4413446
   && echo 'eval "$(command conda shell.bash hook 2> /dev/null)"' >> /etc/skel/.bashrc 

# Create NB_USER with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.
RUN echo "auth requisite pam_deny.so" >> /etc/pam.d/su \
    && sed -i.bak -e 's/^%admin/#%admin/' /etc/sudoers \
    && sed -i.bak -e 's/^%sudo/#%sudo/' /etc/sudoers \
    && useradd -m -s /bin/bash -N -u $NB_UID $NB_USER \
    && mkdir -p $CONDA_DIR \
    && chown $NB_USER:$NB_GID $CONDA_DIR \
    && chmod g+w /etc/passwd \
    && fix-permissions $HOME \
    && fix-permissions $CONDA_DIR

USER $NB_UID
WORKDIR $HOME

# Setup work directory for backward-compatibility
RUN mkdir /home/$NB_USER/work \
    && fix-permissions /home/$NB_USER

# Install Conda
WORKDIR /tmp
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_SUFFIX}_${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "${MINICONDA_CHECKSUM} *Miniconda3-${MINICONDA_SUFFIX}_${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash Miniconda3-${MINICONDA_SUFFIX}_${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-${MINICONDA_SUFFIX}_${MINICONDA_VERSION}-Linux-x86_64.sh && \
    # Conda configuration see https://conda.io/projects/conda/en/latest/configuration.html
    echo "conda ${CONDA_VERSION}" >> $CONDA_DIR/conda-meta/pinned && \
    conda config --system --prepend channels conda-forge && \
    conda config --system --prepend channels intel && \
    conda config --system --set auto_update_conda false && \
    conda config --system --set show_channel_urls true && \
    conda config --system --set channel_priority strict && \
    conda config --system --set safety_checks disabled && \
    if [ ! $PYTHON_VERSION = 'default' ]; then conda install --yes python=$PYTHON_VERSION; fi && \
    conda list python | grep '^python ' | tr -s ' ' | cut -d '.' -f 1,2 | sed 's/$/.*/' >> $CONDA_DIR/conda-meta/pinned && \
    conda install --quiet --yes "conda=${CONDA_VERSION}" && \
    conda install --quiet --yes pip && \
    conda update --all --quiet --yes && \
    conda clean --all -f -y && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER && \
    conda update -n base -c defaults conda

# Install Tini
RUN conda install --quiet --yes 'tini=0.18.0' && \
    conda list tini | grep tini | tr -s ' ' | cut -d ' ' -f 1,2 >> $CONDA_DIR/conda-meta/pinned && \
    conda clean --all -f -y && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Install intl mkl
RUN conda install --quiet --yes mkl-devel

# Install packages through Conda
RUN conda install --quiet --yes \
    "beautifulsoup4" \
    "bokeh" \
    "bottleneck" \
    "cloudpickle" \
    "cython" \
    "daal4py" \
    "dask" \
    "dill" \
    "gdal=3.2.*" \
    "geopandas" \
    "geos=3.8.*" \
    "h5py" \
    "intel-aikit-tensorflow" \
    "libiconv" \
    "lz4" \
    "matplotlib=3.3.*" \
    "nodejs=12" \
    "numba" \
    "numexpr" \
    "numpy" \
    "pandas" \
    "patsy" \
    "proj=7.1.*" \
    "protobuf" \
    "pytables" \
    "rasterio" \
    "scikit-image" \
    "scikit-learn" \
    "scikit-learn-intelex" \
    "scipy" \
    "seaborn" \
    "sqlalchemy" \
    "statsmodels" \
    "sympy" \
    "vincent" \
    "widgetsnbextension"\
    "xgboost" \
    "xlrd"

RUN conda install --quiet --yes \
    "requests" \
    "joblib" \
    "shapely" \
    "owslib" \
    "pygeos" \
    "pyts" \
    "opencv_python"

# Install packages through pip
RUN pip install \
    mlens \
    pqdm

WORKDIR $HOME

EXPOSE 8888

# Configure container startup
ENTRYPOINT ["tini", "-g", "--"]
CMD ["/bin/bash"]