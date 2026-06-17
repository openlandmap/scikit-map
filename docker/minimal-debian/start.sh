#! /bin/bash
. .venv/bin/activate
uv run --with jupyter jupyter lab --allow-root

docker run -d --restart=always --name pygeo_ide_automl -v /home/fee:/home -p 8888:8888 --tmpfs /tmpfs:mode=1777 opengeohub/pygeo-ide:v3.12.3-intel-gdal3103 jupyter lab --allow-root --LabApp.token='opengeohub' --ServerApp.root_dir='/' --no-browser --ip=0.0.0.0