#!/bin/bash

NAME="rgeo"
CONTEXT="01-$NAME"

R_VERSION="4.1.1"
MKL_VERSION="2020.0.166-1"

#FIXME OpenBlas support
#OPENBLAS_VERSION="v0.3.15"

OGH_VERSIONS=$(docker images --format "{{.Repository}}:{{.Tag}}" \
	| grep "gdal:v" | cut -d\/ -f2- | sort)

for version in ${OGH_VERSIONS[@]}; do
	tag_suffix=$(echo $version | tr -d '.' | tr -d ':' | sed s/gdalv/gdal/g)
	echo docker build \
		--build-arg BASE_IMAGE=opengeohub/$version \
		--build-arg R_VERSION=$R_VERSION-1.2004.0 \
		--build-arg MKL_VERSION=$MKL_VERSION \
		--tag opengeohub/$NAME:v$R_VERSION-mkl-$tag_suffix \
		-f $CONTEXT/v4.x.x-mkl.Dockerfile \
		$CONTEXT
	#echo docker build \
	#	--build-arg BASE_IMAGE=opengeohub/$version \
	#	--build-arg R_VERSION=$R_VERSION-1.2004.0 \
	#	--build-arg OPENBLAS_VERSION=$OPENBLAS_VERSION \
	#	--tag opengeohub/$NAME:v$R_VERSION-mkl-$tag_suffix \
	#	-f $CONTEXT/v4.x.x-openblas.Dockerfile \
	#	$CONTEXT
done