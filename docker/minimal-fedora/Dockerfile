FROM fedora:44
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


# https://linuxconfig.org/how-to-skip-rpm-weak-dependencies
# This sets the not-installing-weak dependencies globally
RUN echo "install_weak_deps=false" >> /etc/dnf/dnf.conf

RUN dnf install -y\
        gdal-devel \
        python-devel \
        cmake gcc g++ \
        git

RUN GDAL_CONFIG=$(which gdal-config) \
    uv venv && \
    uv pip install --no-binary rasterio rasterio && \
    uv pip install "scikit-map @ git+https://github.com/openlandmap/scikit-map@v0.9.2"

FROM fedora:44
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --from=0 /.venv /.venv

# https://linuxconfig.org/how-to-skip-rpm-weak-dependencies
# This sets the not-installing-weak dependencies globally
# alternative is to `dnf install --setopt=install_weak_deps=false`
# to next RUN, but I think weak deps in a container are weird
RUN echo "install_weak_deps=false" >> /etc/dnf/dnf.conf

RUN dnf install -y\
        gdal \
        libgomp \
        htop \
        tini \
    && dnf clean all

ENV PATH="/.venv/bin:$PATH"
# I think this is slurm stuff?
ENTRYPOINT ["tini", "-g", "--"]
# standardize the long command, --ip=0.0.0.0 --allow-root are important, the rest prevent non-fatal errors
CMD [".venv/bin/jupyter", "lab", "--allow-root", "--no-browser", "--LabApp.token='opengeohub'", "--ServerApp.root_dir='/'", "--ip=0.0.0.0"]
