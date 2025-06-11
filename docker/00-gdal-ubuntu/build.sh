#!/bin/bash

#######################################################
#### opengeohub/gdal-ubuntu:v3.x.x
#######################################################

echo docker build \
	--tag opengeohub/gdal-ubuntu:v3.10.3 \
	-f 00-gdal-ubuntu/v3.x.x.Dockerfile \
	00-gdal-ubuntu

#######################################################
#### opengeohub/gdal:v3.x.x-grassxxx-sagaxxx
#######################################################

#echo docker build \
#	--tag opengeohub/gdal-ubuntu:v3.4.3-grass802-saga730 	\
#	-f 00-gdal-ubuntu/v3.x.x-grass-saga.Dockerfile \
#	00-gdal-ubuntu