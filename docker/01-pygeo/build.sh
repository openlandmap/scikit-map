#!/bin/bash

NAME="pygeo"
CONTEXT="01-$NAME"

OGH_VERSIONS=( gdal:vgver gdal:vgver-grass785-saga790 )
BLAS_OPTS=( mkl openblas )
GDAL_VERSIONS=( '3.1.4' )

GEOS_VERSION="3.8.1"
PYTHON_VERSION="3.8.6"

for ogh_version in ${OGH_VERSIONS[@]}; do
	for gdal_version in ${GDAL_VERSIONS[@]}; do
		version=$(echo $ogh_version | sed "s/gver/$gdal_version/g")
		tag_suffix=$(echo $version | tr -d '.' | tr -d ':' | sed s/gdalv/gdal/g)
		for blas_opt in ${BLAS_OPTS[@]}; do
			echo docker build \
				--build-arg BASE_IMAGE=opengeohub/$version \
				--build-arg BLAS_IMPLEMENTATION=$blas_opt \
			  --build-arg GDAL_VERSION=$gdal_version \
			  --build-arg GEOS_VERSION=$GEOS_VERSION \
			  --build-arg PYTHON_VERSION=$PYTHON_VERSION \
				--tag opengeohub/$NAME:v$PYTHON_VERSION-$blas_opt-$tag_suffix \
				-f $CONTEXT/v3.x.x.Dockerfile \
				$CONTEXT
		done
	done
done