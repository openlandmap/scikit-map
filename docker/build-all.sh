#!/bin/bash

LEVELS=( '00' '01' '02' )

for level in ${LEVELS[@]}; do
	echo "############ Building containers - $level*"
	echo "################################################"
	find $level* -name "build.sh" -exec {} \; | bash
done
