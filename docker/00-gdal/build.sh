#!/bin/bash

#######################################################
#### opengeohub/gdal:v3.x.x
#######################################################
echo docker build \
	--build-arg PROJ_VERSION=7.1.1 \
	--build-arg GEOS_VERSION=3.8.1 \
	--build-arg GDAL_VERSION=v3.1.4 \
	--tag opengeohub/gdal:v3.1.4 	\
	-f 00-gdal/v3.x.x.Dockerfile \
	00-gdal

echo docker build \
	--build-arg PROJ_VERSION=7.1.1 \
	--build-arg GEOS_VERSION=3.8.1 \
	--build-arg GDAL_VERSION=v3.2.2 \
	--tag opengeohub/gdal:v3.2.2 \
	-f 00-gdal/v3.x.x.Dockerfile \
	00-gdal

echo docker build \
	--build-arg PROJ_VERSION=7.1.1 \
	--build-arg GEOS_VERSION=3.8.1 \
	--build-arg GDAL_VERSION=v3.3.0 \
	--tag opengeohub/gdal:v3.3.0 \
	-f 00-gdal/v3.x.x.Dockerfile \
	00-gdal

#######################################################
#### opengeohub/gdal:v3.x.x-grassxxx-saga
#######################################################
echo docker build \
	--build-arg BASE_IMAGE=opengeohub/gdal:v3.1.4 \
	--build-arg PDAL_VERSION=2.2.0 \
	--build-arg LAZ_PERF_VERSION=1.5.0 \
	--build-arg GRASS_VERSION=8.0.dev \
	--build-arg SAGA_VERSION=7.9.0 \
	--tag opengeohub/gdal:v3.1.4-grass80dev-saga790 	\
	-f 00-gdal/v3.x.x-grass-saga.Dockerfile \
	00-gdal

echo docker build \
	--build-arg BASE_IMAGE=opengeohub/gdal:v3.2.2 \
	--build-arg PDAL_VERSION=2.2.0 \
	--build-arg LAZ_PERF_VERSION=1.5.0 \
	--build-arg GRASS_VERSION=8.0.dev \
	--build-arg SAGA_VERSION=7.9.0 \
	--tag opengeohub/gdal:v3.2.2-grass80dev-saga790 	\
	-f 00-gdal/v3.x.x-grass-saga.Dockerfile \
	00-gdal