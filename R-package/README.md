# eumap package for R

[![Build Status](https://travis-ci.org/OpenGeoHub/eumap.svg?branch=master)](https://travis-ci.org/OpenGeoHub/eumap)
[![CRAN_Status_Badge](http://www.r-pkg.org/badges/version/eumap)](https://cran.r-project.org/package=eumap)
[![Gitlab_Status_Badge](https://img.shields.io/badge/Github-0.0--1-blue.svg)](https://gitlab.com/geoharmonizer_inea/eumap)

Package provides easier access to EU environmental maps and functions to produce and improve new value-added spatial layers. Key functionality includes:

* `train.spm` --- train a spatial prediction model using [mlr3 package](https://mlr3.mlr-org.com/)) implementation with spatial coordinates and spatial cross-validation,

Warning: most of functions are optimized to run in parallel by default. This might result in high RAM and CPU usage.

## Installing

Install development versions from github:

```r
library(devtools)
install_gitlab("geoharmonizer_inea/eumap/R-package")
```

Under construction. Use for testing purposes only.

## Functionality

### Automated mapping using mlr3

The following examples demostrates spatial prediction using the meuse data set:

```r
ls <- c("rgdal", "raster", "ranger", "mlr3", 
        "xgboost", "glmnet", "matrixStats", "deepnet")
new.packages <- ls[!(ls %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
library(landmap)
library(rgdal)
library(geoR)
library(plotKML)
library(raster)
library(glmnet)
library(xgboost)
library(kernlab)
library(deepnet)
library(mlr)
demo(meuse, echo=FALSE)
```

## Contributions

* Contributions to eumap are welcome. Issues and pull requests are the preferred ways of sharing them.
* We are interested in your results and experiences with using the mlr3 functions 
  for generating spatial predictions with your own data. Share your data sets, 
  code and results either using github issues and/or R-sig-geo mailing list.
