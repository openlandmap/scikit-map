# Scikit-map on Docker 

This document provides instructions on how to build and use several docker images compatible with scikit-map provided by OpenGeoHub Foundation and available in [DockerHub](https://hub.docker.com/u/opengeohub).


### Pre-requisites

Docker: Please ensure Docker is installed on your machine. You can follow the installation instructions based on your operating system from the official Docker docs: [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)

### Building images

1. **Clone the repository**
```
    git clone http://github.com/openlandmap/scikit-map
    cd scikit-map/docker
```

2. **Build docker images**
```
    sudo su
    build-all.sh
 ```

This command builds locally the following images:

- **opengeohub/gdal-ubuntu:v3.10.3**: Image based on Ubuntu 24.04 with [ubuntugis-unstable repository](https://launchpad.net/~ubuntugis/+archive/ubuntu/ubuntugis-unstable) and [GDAL 3.10.3](https://github.com/OSGeo/gdal/blob/v3.10.3/NEWS.md) pre-installed
- **opengeohub/pygeo:v3.12.3-gdal3103**: Image based on gdal-ubuntu:v3.10.3 with [Python 3.12](https://www.python.org/downloads/release/python-3120/) and scikit-map pre-installed in root [venv](https://docs.python.org/3/library/venv.html) (*no conda at all*)
- **opengeohub/pygeo:v3.12.3-intel-gdal3103**:  Image based on gdal-ubuntu:v3.10.3 with [Intel Python 3.12](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html), [scikit-learn-intelex](https://github.com/uxlfoundation/scikit-learn-intelex), scikit-map pre-installed in root [venv](https://docs.python.org/3/library/venv.html) (*no conda at all*)
- **opengeohub/pygeo-ide:v3.12.3-gdal3103**: Image based on pygeo:v3.12.3-gdal3103 with [JupyteLab](https://jupyter.org/install), [geemap](https://geemap.org/) and [ipyleaflet](https://ipyleaflet.readthedocs.io/en/latest/) pre-installed in root [venv](https://docs.python.org/3/library/venv.html) (*no conda at all*)
- **opengeohub/pygeo-ide:v3.12.3-intel-gdal3103**:  Image based on pygeo:v3.12.3-intel-gdal3103 with [JupyteLab](https://jupyter.org/install), [geemap](https://geemap.org/) and [ipyleaflet](https://ipyleaflet.readthedocs.io/en/latest/) pre-installed in root [venv](https://docs.python.org/3/library/venv.html) (*no conda at all*)


### Running images

Execute a single **GDAL command** and remove the container:

```
    docker run --rm -it opengeohub/gdal-ubuntu:v3.10.3 gdalinfo --version
```

Open an interative **Python session** and remove the container after `exit()`:
```
    docker run --rm -it -v /mnt:/mnt opengeohub/pygeo:v3.12.3-intel-gdal3103

```

Deploy a JupyterLab instance on port 8888 and share all `/mnt` folder with the container (for more options see [server options](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/common.html#jupyter-server-options)):

```
    docker run -d --restart=always --name pygeo_ide_skmap -v /mnt:/mnt -p 8888:8888 --tmpfs /tmpfs:mode=1777 opengeohub/pygeo-ide:v3.12.3-intel-gdal3103 jupyter lab --allow-root --LabApp.token='opengeohub' --ServerApp.root_dir='/' --no-browser --ip=0.0.0.0
```

JupyterLab should be accessible on http://localhost:8888 (password: `opengeohub`)

![JupyterLab login page](../docs/img/jupyterlab.png)