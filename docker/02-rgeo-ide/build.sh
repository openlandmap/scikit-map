#!/bin/bash

BASENAME="rgeo"
NAME="$BASENAME-ide"
CONTEXT="02-$NAME"

OGH_VERSIONS=$(docker images --format "{{.Repository}}:{{.Tag}}" \
	| grep $BASENAME | cut -d\/ -f2- | sort)

for version in ${OGH_VERSIONS[@]}; do
	tag_suffix=$(echo $version | cut -d\: -f2)	
	echo docker build \
		--build-arg BASE_IMAGE=opengeohub/$version \
		--tag opengeohub/$NAME:$tag_suffix \
		-f $CONTEXT/v4.Dockerfile \
		$CONTEXT
done