#/usr/bin/env bash

REPO_ROOT=$(dirname "$0")/..
SRC=$REPO_ROOT/docs
DST=$REPO_ROOT/tmpdocbuild

rm -r $SRC/_autosummary
rm -r $DST

sphinx-build -b html $SRC $DST
