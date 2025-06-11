#!/bin/bash

NAME="pygeo"
CONTEXT="01-$NAME"

EXTRA_OPTS=( "intel" "default" )
GDAL_VERSIONS=( '3.10.3' )
PYTHON_VERSION="3.12.3"
BASE_IMAGE="opengeohub/gdal-ubuntu:v3.10.3"

for gdal_version in ${GDAL_VERSIONS[@]}; do
	for extra_opt in ${EXTRA_OPTS[@]}; do
		tag_suffix=gdal$(echo $gdal_version | tr -d '.' | tr -d ':')
		tag_opt=$(echo "-$extra_opt" | sed s/-default//g)
		echo docker build \
			--build-arg BASE_IMAGE=$BASE_IMAGE \
			--build-arg EXTRA_OPT=$extra_opt \
			--tag opengeohub/$NAME:v$PYTHON_VERSION${tag_opt}-$tag_suffix \
			-f $CONTEXT/v3.x.x.Dockerfile \
			$CONTEXT
	done
done

#for gdal_version in ${GDAL_VERSIONS[@]}; do
#	tag_suffix=gdal$(echo $gdal_version | tr -d '.' | tr -d ':')
#	echo docker build \
#	  --build-arg GDAL_VERSION=$gdal_version \
#		--tag opengeohub/$NAME:v3.7.11-oneapi-$tag_suffix \
#		-f $CONTEXT/v3.x.x-intel.Dockerfile \
#		$CONTEXT
#done
