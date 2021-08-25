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
ARG NB_GID="100"
ARG NB_UID="1000"
ARG NB_USER="opengeohub"
ENV CONDA_DIR=/opt/conda \
    DEBIAN_FRONTEND=noninteractive \
    NB_USER=$NB_USER \
    NB_UID=$NB_UID \
    SHELL=/bin/bash \
    NB_GID=$NB_GID

ENV XDG_CACHE_HOME="/home/${NB_USER}/.cache/" \
    HOME=/home/$NB_USER

WORKDIR /tmp

# Copy local files as late as possible to avoid cache busting
COPY files/start.sh files/start-notebook.sh files/start-singleuser.sh /usr/local/bin/
COPY files/fix-permissions /usr/local/bin/fix-permissions
COPY files/jupyter_notebook_config.py /etc/jupyter/
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
    && chmod g+w /etc/passwd \
    && fix-permissions $HOME

# Setup work directory for backward-compatibility
RUN mkdir /home/$NB_USER/work \
    && fix-permissions /home/$NB_USER && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions /home/$NB_USER

# Install Tini
RUN conda install --quiet --yes 'tini=0.18.0' && \
    conda list tini | grep tini | tr -s ' ' | cut -d ' ' -f 1,2 >> $CONDA_DIR/conda-meta/pinned && \
    conda clean --all -f -y && \
    fix-permissions /home/$NB_USER

# Install Jupyter Notebook, Lab, and Hub
# Generate a notebook server config
# Cleanup temporary files
# Correct permissions
# Do all this in a single RUN command to avoid duplicating all of the
# files across image layers when the permissions change
RUN conda install --quiet --yes \
    'notebook=6.1.6' \
    'jupyterhub=1.1.0' \
    'jupyterlab=3.0.14' \
    && npm cache clean --force \
    && jupyter notebook --generate-config  \
    && rm -rf $CONDA_DIR/share/jupyter/lab/staging \
    && rm -rf /home/$NB_USER/.cache/yarn \
    && fix-permissions /home/$NB_USER

RUN conda install --quiet --yes \
    'ipyleaflet=0.14.*' \
    'ipympl=0.7.*'\
    'ipywidgets=7.6.*' \
    'jupyter_bokeh=3.0.*' \
    'jupyterlab-spellchecker=0.7.*' \
    'jupyterlab_widgets=1.0.*' \
    'widgetsnbextension=3.5.*'

# Install facets which does not have a pip or conda package at the moment
RUN git clone https://github.com/PAIR-code/facets.git && \
    jupyter nbextension install facets/facets-dist/ --sys-prefix && \
    rm -rf /tmp/facets && \
    fix-permissions "/home/${NB_USER}"

RUN conda clean --all -f -y && \
    jupyter lab build -y --log-level=DEBUG --minimize=False && \
    jupyter lab clean -y && \
    npm cache clean --force && \
    rm -rf "/home/${NB_USER}/.cache/yarn" && \
    rm -rf "/home/${NB_USER}/.node-gyp" && \
    fix-permissions "/home/${NB_USER}"

RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" \
    && fix-permissions "/home/${NB_USER}"

USER $NB_UID
WORKDIR $HOME

EXPOSE 8888

# Configure container startup
ENTRYPOINT ["tini", "-g", "--"]
CMD ["start-notebook.sh"]