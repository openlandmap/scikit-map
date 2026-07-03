FROM debian:13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y\
        libgdal-dev \
        python3-dev \
        build-essential \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

<<<<<<< HEAD:docker/minimal/Dockerfile
RUN uv venv && uv pip install "scikit-map[full] @ git+https://github.com/openlandmap/scikit-map@ci"
=======
RUN uv venv && \
    uv pip install --no-binary rasterio rasterio && \
    uv pip install "scikit-map @ git+https://github.com/openlandmap/scikit-map@v0.9.2"
>>>>>>> ac25638 (restrict pyts to newer versions so it doesn't get stuck in old numba):docker/minimal-debian/Dockerfile

RUN uv pip install jupyter

FROM debian:13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --from=0 /.venv /.venv

RUN apt-get update && \
    apt-get install -y\
        libgdal36 \
        python3 \
        libgomp1 \
        htop \
        tini \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/.venv/bin:$PATH"
# I think this is slurm stuff?
ENTRYPOINT ["tini", "-g", "--"]
# standardize the long command, --ip=0.0.0.0 --allow-root are important, the rest prevent non-fatal errors
CMD [".venv/bin/jupyter", "lab", "--allow-root", "--no-browser", "--LabApp.token='opengeohub'", "--ServerApp.root_dir='/'", "--ip=0.0.0.0"]
