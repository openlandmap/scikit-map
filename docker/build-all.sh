#!/bin/bash

LEVELS=( '01' '02' )

find 00* -name "build.sh" -exec {} \; | bash

for level in ${LEVELS[@]}; do
	echo "############ Building containers - $level*"
	echo "################################################"
	find $level* -name "build.sh" -exec {} \; | parallel -j2
done
