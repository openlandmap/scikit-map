

-   [Introduction](#introduction)
-   [`train_spm`](#train_spm)
-   [`predict_spm`](#predict_spm)
-   [`plot_spm`](#plot_spm)
    -   [Required packages](#required-packages)
    -   [sic97 dataset](#sic97-dataset)
    -   [spliting the data](#spliting-the-data)
    -   [Loading required libraries:](#loading-required-libraries)
    -   [`train_spm`](#train_spm-1)
    -   [`predict_spm`](#predict_spm-1)
    -   [predicted values for the *newdata*
        set:](#predicted-values-for-the-newdata-set)
    -   [`plot_spm`](#plot_spm-1)
    -   [spatial prediction on
        *rainfall*](#spatial-prediction-on-rainfall)
-   [Croatia tile](#croatia-tile)
    -   [Overlay Demonstration](#overlay-demonstration)
    -   [reading Croatia data](#reading-croatia-data)
    -   [`stripe_years`](#stripe_years)
    -   [`extract_tif`](#extract_tif)
-   [Space-Time Overlay](#space-time-overlay)
    -   [(analysis ready) data](#analysis-ready-data)
-   [`train_spm`](#train_spm-2)
-   [`predict_spm`](#predict_spm-2)
-   [References](#references)

Follow me on [![alt
text](http://i.imgur.com/tXSoThF.png "twitter icon with padding")](https://twitter.com/sheykhmousa)


         checking for file ‘/tmp/RtmpMV2OWU/file801f05e377471/R-package/DESCRIPTION’ ...  ✓  checking for file ‘/tmp/RtmpMV2OWU/file801f05e377471/R-package/DESCRIPTION’
      ─  preparing ‘eumap’:
       checking DESCRIPTION meta-information ...  ✓  checking DESCRIPTION meta-information
      ─  checking for LF line-endings in source and make files and shell scripts
      ─  checking for empty or unneeded directories
      ─  building ‘eumap_0.0.2.tar.gz’
         

Introduction
------------

`eumap` aims at providing easier access to EU environmental maps. Basic
functions train a spatial prediction model using [mlr3
package](https://mlr3.mlr-org.com/), and related extensions in the [mlr3
ecosystem](https://github.com/mlr-org/mlr3/wiki/Extension-Packages)
(Casalicchio et al., [2017](#ref-casalicchio2017openml); Lang et al.,
[2020](#ref-MichelLang2020mlr3book)), which includes spatial prediction
using [Ensemble Machine
Learning](https://koalaverse.github.io/machine-learning-in-R/stacking.html#stacking-software-in-r)
taking spatial coordinates and spatial cross-validation into account. In
a nutshell one can `train` an arbitrary `s3` **(spatial)dataframe** in
`mlr3` ecosystem by defining *df* and *target.variable* i.e., response.
main functions are as the following:

`train_spm`
-----------

1.  `train_spm` will automatically perform `classification` or
    `regression` tasks and the output is a `train_model` which later can
    be used to predict `newdata`.It also provides *summary* of the model
    and *variable importance* and *response*. The rest of arguments can
    be either pass or default values will be passed. `train_spm`
    provides four scenarios:

1.1. `classification` task with **non spatial** resampling methods, 1.2.
`regression` task with **non spatial** resampling methods, 1.3.
`classification` task with **spatial** resampling methods, 1.4.
`regression` task with **spatial** resampling methods.

`predict_spm`
-------------

1.  Prediction on a new dataset using `train_model`,
2.  User needs to set`df.ts = test set` and also pass the `train_model`.

`plot_spm`
----------

1.  Accuracy plot in case of regression task (don’t use it for
    classification tasks for obvious reason). The following examples
    demonstrates spatial prediction using the `sic97` data set:

### Required packages

    start_time <- Sys.time()
    ls <- c("lattice", "raster", "plotKML", "ranger", "mlr3verse", "BBmisc", "knitr", "bbotk",
        "hexbin", "stringr", "magrittr", "sp", "ggplot2", "mlr3fselect", "mlr3spatiotempcv",  "tidyr", "lubridate", "R.utils", "terra","rgdal",
        "FSelectorRcpp", "future", "future.apply", "mlr3filters", "EnvStats", "grid", "mltools","gridExtra","yardstick","plotKML", "latticeExtra","devtools","progressr")
    new.packages <- ls[!(ls %in% installed.packages()[,"Package"])]
    if(length(new.packages)) install.packages(new.packages, repos="https://cran.rstudio.com", force=TRUE)

### sic97 dataset

    #sic97 source data: https://rdrr.io/github/Envirometrix/landmap/man/sic1997.html
    library("landmap")  

    version: 0.0.3

    data(sic1997) 
    sic97 <- na.omit(sic1997)
    sic97 <- sic1997$swiss1km[c("CHELSA_rainfall","DEM")]
    df <- data.frame(sic97)
    #let's create some fake cov
    df$test1 <- log10(df$DEM)*53.656 
    df$test2 <- cos(df$DEM)*-0.13 
    df$test3 <- sin(df$DEM)**31 
    df$test4 <- (df$DEM)**-5.13 
    df$test5 <- runif(1:nrow(df))
    df$test6 <- (df$DEM/2**4)**6 
    df$test7 <- (df$x)**-1
    df$test8 <- (df$y)*3
    df$test9 <- (df$CHELSA_rainfall)**-1
    df$test10 <-((df$CHELSA_rainfall)*13/34)
    df$test11 <- runif(1:nrow(df))/0.54545
    df$test12 <- sqrt(runif(1:nrow(df))) 

### spliting the data

    smp_size <- floor(0.5 * nrow(df))
    set.seed(123)
    train_ind <- sample(seq_len(nrow(df)), size = smp_size)
    df.tr <- df[train_ind,]
    df.ts <- df[-train_ind, ]
    newdata = df.ts

### Loading required libraries:

    library("mlr3verse")#
    library("mlr3spatiotempcv")#
    library("sp")#
    library("grid")#
    library("hexbin")#
    library("BBmisc")#
    library("lattice")#
    library("gridExtra")#
    library("MLmetrics")
    library("yardstick")#
    library("latticeExtra")#
    library("eumap")
    library("ppcor")
    library("progressr")
    library("checkmate")
    library("future")

### `train_spm`

`train_spm` fits multiple models/learners depending on the `class` of
the **target.variable** and returns a `trained model`, **variable
importance**, **summary** of the model, and **response** variables.
`trained model` later can predict a `newdata` set.

    tr = eumap::train_spm(df.tr, target.variable = "CHELSA_rainfall", folds = 5, n_evals = 3, crs = "+init=epsg:4326")

            Regression Task  ...TRUE

               Fitting an ensemble ML using  kknn  featureless, and Randome Forests models ncores: 32  resampling method: (spatial)repeated_cv by cooridinates ...TRUE

-   1st element contains the *trained model*,
-   2nd element contains the *variable importance*,
-   3rd element contains a summary of the *trained model*,
-   4th element contains the *predicted values* of our trained model,
-   5th element contains the ranking of the *important variables*.

<!-- -->

    train_model= tr[[1]]
    var.imp = tr[[2]]
    summary = tr[[3]]
    response = tr[[4]]
    vlp = tr[[5]]
    target = tr[[6]]

### `predict_spm`

prediction on `newdata` set

    predict.variable = eumap::predict_spm(train_model, newdata)

### predicted values for the *newdata* set:

    predict.variable = predict_spm(train_model, df.ts)
    pred.v = predict.variable[[1]]
    valu.imp= predict.variable[[2]]
    pred.v

    ...
        [1]  71.80220  72.19450  71.51627  81.67807  69.85040  71.80070  72.31600
        [8]  74.60243  83.65640  92.89200  99.65280  75.37560  67.33663  72.05823
       [15]  72.81940  89.57763  95.41123  76.39570  85.84480  79.36400  87.95857
       [22]  87.97030  86.82043 107.80677  87.84647  80.42300  73.25390  68.23027
       [29]  91.80843  73.83887  78.38040  87.88683 106.37567  82.63610  73.49277
       [36]  72.48563  94.80540  90.38707  84.00703  75.68500  75.15710  76.14463
       [43]  85.65677  92.76300  71.43830  78.56307  88.89390 101.05403 104.88523
       [50]  83.84097  87.71750  81.90317  70.24460  73.33613  84.60890  69.39127
       [57] 102.51810  95.90257  87.18923  87.08267  85.14340  85.49547  78.43923
       [64]  74.64787  76.77847  78.00963  80.06790  89.70123  76.22707  78.99527
    ...

    #valu.imp

### `plot_spm`

variable importance

    plt = plot_spm(df, gmode  = "norm" , gtype = "var.imp")

<img src="README_files/figure-markdown_strict/unnamed-chunk-12-1.png" alt="var.imp"  />
<p class="caption">
var.imp
</p>

          [,1]
     [1,]  0.7
     [2,]  1.9
     [3,]  3.1
     [4,]  4.3
     [5,]  5.5
     [6,]  6.7
     [7,]  7.9
     [8,]  9.1
     [9,] 10.3
    [10,] 11.5
    [11,] 12.7
    [12,] 13.9
    [13,] 15.1


    plt = plot_spm(df, gmode  = "norm" , gtype = "accuracy")

<img src="README_files/figure-markdown_strict/unnamed-chunk-13-1.png" alt="Accuracy plot"  />
<p class="caption">
Accuracy plot
</p>

### spatial prediction on *rainfall*

    predict.variable = predict_spm(train_model, df)
    df$rainP = predict.variable[[1]]
    coordinates(df) <- ~x+y
    proj4string(df) <- CRS("+init=epsg:28992")
    # creat raster out of output
    gridded(df) = TRUE

make a map using ensemble machine learning with spatial cross validation
for the predicted variables e.g., *rainfall* (in this case).

    plot(df[,"rainP"])

<img src="README_files/figure-markdown_strict/unnamed-chunk-15-1.png" alt="Raster grid"  />
<p class="caption">
Raster grid
</p>

    # points(sic1997, pch="+")

Croatia tile
------------

### Overlay Demonstration

we will use the eumap package to overlay all the points of a vector
layer (geopackage file) on several raster layers (geotiff files), using
the SpaceOverlay and SpaceTimeOverlay classes to handle with timeless
and temporal layers, respectively. In our dataset the elevation and
slope, based on digital terrain model, are timeless and the landsat
composites (7 spectral bands, 4 seasons and 3 percentiles) and night
light (VIIRS Night Band) layers are temporal (from 2000 to 2020).

### reading Croatia data

Our dataset refers to 1 tile, located in Croatia, extracted from a
tiling system created for European Union (7,042 tiles) by [GeoHarmonizer
Project](https://opendatascience.eu/).

    library(rgdal)

    rgdal: version: 1.5-18, (SVN revision 1082)
    Geospatial Data Abstraction Library extensions to R successfully loaded
    Loaded GDAL runtime: GDAL 3.1.3, released 2020/09/01
    Path to GDAL shared files: /usr/share/gdal
    GDAL binary built with GEOS: TRUE 
    Loaded PROJ runtime: Rel. 4.8.0, 6 March 2012, [PJ_VERSION: 480]
    Path to PROJ shared files: (autodetected)
    Linking to sp version:1.4-4

    tif1.lst = list.files("/data/eumap/sample-data/R-sample-tiles/9529", pattern=".tif", full.names=TRUE, recursive=TRUE) 
    df = readOGR("/data/eumap/sample-data/R-sample-tiles/9529_croatia_landcover_samples.gpkg")

    OGR data source with driver: GPKG 
    Source: "/data/eumap/sample-data/R-sample-tiles/9529_croatia_landcover_samples.gpkg", layer: "9529_croatia_landcover_samples"
    with 759 features
    It has 5 fields

    df <- as.data.frame(df)
    df$Date = format.Date(as.Date(paste(df$survey_date), format="%Y/%m/%d"), "%Y-%m-%d")
    df$row.id = 1:nrow(df)

### `stripe_years`

    begin.tif1.lst = sapply(tif1.lst, function(i){strip_years(i, type="begin")})
    end.tif1.lst = sapply(tif1.lst, function(i){strip_years(i, type="end")})
    unique(end.tif1.lst)

     [1] "2000-12-31" "2001-12-31" "2002-12-31" "2003-12-31" "2004-12-31"
     [6] "2005-12-31" "2006-12-31" "2007-12-31" "2008-12-31" "2009-12-31"
    [11] "2010-12-31" "2011-12-31" "2012-12-31" "2013-12-31" "2014-12-31"
    [16] "2015-12-31" "2016-12-31" "2017-12-31" "2018-12-31" "2019-12-31"
    [21] "2020-12-31"

### `extract_tif`

    cores = ifelse(parallel::detectCores()<length(tif1.lst), parallel::detectCores(), length(tif1.lst))
    ov.pnts <- parallel::mclapply(1:length(tif1.lst), function(i){ eumap::extract_tif(tif=tif1.lst[i], df, date="Date", date.tif.begin=begin.tif1.lst[i], date.tif.end=end.tif1.lst[i], coords=c("coords.x1","coords.x2")) }, mc.cores=cores)
    gc()

               used  (Mb) gc trigger  (Mb) max used  (Mb)
    Ncells  4088767 218.4    6952190 371.3  6952190 371.3
    Vcells 24699156 188.5   39811058 303.8 39782918 303.6

    ov.pnts = ov.pnts[!sapply(ov.pnts, is.null)]

    str(ov.pnts[1:3])

    List of 3
     $ :'data.frame':   180 obs. of  2 variables:
      ..$ landsat_ard_fall_blue_p25: num [1:180] 4 6 4 6 6 4 6 6 4 8 ...
      ..$ row.id                   : int [1:180] 3 4 7 15 18 20 26 28 32 36 ...
     $ :'data.frame':   180 obs. of  2 variables:
      ..$ landsat_ard_fall_blue_p50: num [1:180] 4 6 4 7 7 5 7 6 4 8 ...
      ..$ row.id                   : int [1:180] 3 4 7 15 18 20 26 28 32 36 ...
     $ :'data.frame':   180 obs. of  2 variables:
      ..$ landsat_ard_fall_blue_p75: num [1:180] 4 7 5 7 7 6 8 6 5 9 ...
      ..$ row.id                   : int [1:180] 3 4 7 15 18 20 26 28 32 36 ...

Space-Time Overlay
------------------

For the temporal layers, the points should be filtered by year and
overlayed on the right raster files. The SpaceTimeOverlay class
implements this approach using the parameter: - timeless\_data: The
result of SpaceOverlay (GeoPandas DataFrame) - col\_date: The column
that contains the date information (2018-09-13) - dir\_temporal\_layers:
The directory where the temporal raster files are stored, organized by
year.

    library(data.table)
    commcols <- Reduce(intersect, lapply(ov.pnts, names))
    L.dt <- lapply(ov.pnts, function(x) setkeyv(data.table(x), commcols))
    cmt <- do.call(cbind, L.dt) 
    uq.lst <- unique(colnames(cmt))
    cm.tif <- cmt[, .SD, .SDcols = unique(names(cmt))]
    df <- as.data.table(df)
    cm <- Reduce(merge,list(df,cm.tif))
    tt = cbind(cm,df$year)
    #saveRDS(tt, "/data/eumap/sample-data/R-sample-tiles/9529/9529_croatia_samples.rds")

    str(tt)

    Classes 'data.table' and 'data.frame':  759 obs. of  96 variables:
     $ row.id                        : int  3 3 3 3 3 4 4 4 4 4 ...
     $ lucas                         : int  0 0 0 0 0 0 0 0 0 0 ...
     $ survey_date                   : chr  "2000/06/30" "2000/06/30" "2000/06/30" "2000/06/30" ...
     $ lc_class                      : int  324 324 324 324 324 321 321 321 321 321 ...
     $ tile_id                       : int  9529 9529 9529 9529 9529 9529 9529 9529 9529 9529 ...
     $ confidence                    : num  85 85 85 85 85 85 85 85 85 85 ...
     $ coords.x1                     : num  4770204 4770204 4770204 4770204 4770204 ...
     $ coords.x2                     : num  2414683 2414683 2414683 2414683 2414683 ...
     $ Date                          : chr  "2000-06-30" "2000-06-30" "2000-06-30" "2000-06-30" ...
     $ landsat_ard_fall_blue_p25     : num  4 4 4 4 4 6 6 6 6 6 ...
     $ landsat_ard_fall_blue_p50     : num  4 4 4 4 4 6 6 6 6 6 ...
     $ landsat_ard_fall_blue_p75     : num  4 4 4 4 4 7 7 7 7 7 ...
     $ landsat_ard_fall_green_p25    : num  12 12 12 12 12 13 13 13 13 13 ...
     $ landsat_ard_fall_green_p50    : num  12 12 12 12 12 13 13 13 13 13 ...
     $ landsat_ard_fall_green_p75    : num  12 12 12 12 12 14 14 14 14 14 ...
     $ landsat_ard_fall_nir_p25      : num  53 53 53 53 53 61 61 61 61 61 ...
     $ landsat_ard_fall_nir_p50      : num  56 56 56 56 56 61 61 61 61 61 ...
     $ landsat_ard_fall_nir_p75      : num  59 59 59 59 59 62 62 62 62 62 ...
     $ landsat_ard_fall_red_p25      : num  12 12 12 12 12 14 14 14 14 14 ...
     $ landsat_ard_fall_red_p50      : num  13 13 13 13 13 14 14 14 14 14 ...
     $ landsat_ard_fall_red_p75      : num  15 15 15 15 15 15 15 15 15 15 ...
     $ landsat_ard_fall_swir1_p25    : num  49 49 49 49 49 60 60 60 60 60 ...
     $ landsat_ard_fall_swir1_p50    : num  51 51 51 51 51 60 60 60 60 60 ...
     $ landsat_ard_fall_swir1_p75    : num  52 52 52 52 52 61 61 61 61 61 ...
     $ landsat_ard_fall_swir2_p25    : num  24 24 24 24 24 33 33 33 33 33 ...
     $ landsat_ard_fall_swir2_p50    : num  26 26 26 26 26 33 33 33 33 33 ...
     $ landsat_ard_fall_swir2_p75    : num  28 28 28 28 28 34 34 34 34 34 ...
     $ landsat_ard_fall_thermal_p25  : num  185 185 185 185 185 186 186 186 186 186 ...
     $ landsat_ard_fall_thermal_p50  : num  185 185 185 185 185 186 186 186 186 186 ...
     $ landsat_ard_fall_thermal_p75  : num  185 185 185 185 185 186 186 186 186 186 ...
     $ landsat_ard_spring_blue_p25   : num  5 5 5 5 5 6 6 6 6 6 ...
     $ landsat_ard_spring_blue_p50   : num  6 6 6 6 6 6 6 6 6 6 ...
     $ landsat_ard_spring_blue_p75   : num  8 8 8 8 8 7 7 7 7 7 ...
     $ landsat_ard_spring_green_p25  : num  12 12 12 12 12 14 14 14 14 14 ...
     $ landsat_ard_spring_green_p50  : num  14 14 14 14 14 14 14 14 14 14 ...
     $ landsat_ard_spring_green_p75  : num  16 16 16 16 16 15 15 15 15 15 ...
     $ landsat_ard_spring_nir_p25    : num  67 67 67 67 67 59 59 59 59 59 ...
     $ landsat_ard_spring_nir_p50    : num  70 70 70 70 70 60 60 60 60 60 ...
     $ landsat_ard_spring_nir_p75    : num  70 70 70 70 70 60 60 60 60 60 ...
     $ landsat_ard_spring_red_p25    : num  11 11 11 11 11 15 15 15 15 15 ...
     $ landsat_ard_spring_red_p50    : num  13 13 13 13 13 16 16 16 16 16 ...
     $ landsat_ard_spring_red_p75    : num  17 17 17 17 17 17 17 17 17 17 ...
     $ landsat_ard_spring_swir1_p25  : num  50 50 50 50 50 59 59 59 59 59 ...
     $ landsat_ard_spring_swir1_p50  : num  53 53 53 53 53 60 60 60 60 60 ...
     $ landsat_ard_spring_swir1_p75  : num  59 59 59 59 59 62 62 62 62 62 ...
     $ landsat_ard_spring_swir2_p25  : num  23 23 23 23 23 30 30 30 30 30 ...
     $ landsat_ard_spring_swir2_p50  : num  25 25 25 25 25 31 31 31 31 31 ...
     $ landsat_ard_spring_swir2_p75  : num  31 31 31 31 31 33 33 33 33 33 ...
     $ landsat_ard_spring_thermal_p25: num  185 185 185 185 185 188 188 188 188 188 ...
     $ landsat_ard_spring_thermal_p50: num  185 185 185 185 185 188 188 188 188 188 ...
     $ landsat_ard_spring_thermal_p75: num  186 186 186 186 186 189 189 189 189 189 ...
     $ landsat_ard_summer_blue_p25   : num  5 5 5 5 5 6 6 6 6 6 ...
     $ landsat_ard_summer_blue_p50   : num  5 5 5 5 5 6 6 6 6 6 ...
     $ landsat_ard_summer_blue_p75   : num  5 5 5 5 5 7 7 7 7 7 ...
     $ landsat_ard_summer_green_p25  : num  12 12 12 12 12 14 14 14 14 14 ...
     $ landsat_ard_summer_green_p50  : num  12 12 12 12 12 15 15 15 15 15 ...
     $ landsat_ard_summer_green_p75  : num  13 13 13 13 13 16 16 16 16 16 ...
     $ landsat_ard_summer_nir_p25    : num  61 61 61 61 61 52 52 52 52 52 ...
     $ landsat_ard_summer_nir_p50    : num  61 61 61 61 61 54 54 54 54 54 ...
     $ landsat_ard_summer_nir_p75    : num  64 64 64 64 64 55 55 55 55 55 ...
     $ landsat_ard_summer_red_p25    : num  11 11 11 11 11 16 16 16 16 16 ...
     $ landsat_ard_summer_red_p50    : num  11 11 11 11 11 17 17 17 17 17 ...
     $ landsat_ard_summer_red_p75    : num  12 12 12 12 12 19 19 19 19 19 ...
     $ landsat_ard_summer_swir1_p25  : num  49 49 49 49 49 63 63 63 63 63 ...
     $ landsat_ard_summer_swir1_p50  : num  51 51 51 51 51 64 64 64 64 64 ...
     $ landsat_ard_summer_swir1_p75  : num  51 51 51 51 51 65 65 65 65 65 ...
     $ landsat_ard_summer_swir2_p25  : num  24 24 24 24 24 34 34 34 34 34 ...
     $ landsat_ard_summer_swir2_p50  : num  24 24 24 24 24 35 35 35 35 35 ...
     $ landsat_ard_summer_swir2_p75  : num  24 24 24 24 24 36 36 36 36 36 ...
     $ landsat_ard_summer_thermal_p25: num  186 186 186 186 186 188 188 188 188 188 ...
     $ landsat_ard_summer_thermal_p50: num  186 186 186 186 186 189 189 189 189 189 ...
     $ landsat_ard_summer_thermal_p75: num  186 186 186 186 186 190 190 190 190 190 ...
     $ landsat_ard_winter_blue_p25   : num  6 6 6 6 6 23 23 23 23 23 ...
     $ landsat_ard_winter_blue_p50   : num  6 6 6 6 6 28 28 28 28 28 ...
     $ landsat_ard_winter_blue_p75   : num  6 6 6 6 6 28 28 28 28 28 ...
     $ landsat_ard_winter_green_p25  : num  15 15 15 15 15 32 32 32 32 32 ...
     $ landsat_ard_winter_green_p50  : num  15 15 15 15 15 36 36 36 36 36 ...
     $ landsat_ard_winter_green_p75  : num  15 15 15 15 15 37 37 37 37 37 ...
     $ landsat_ard_winter_nir_p25    : num  70 70 70 70 70 75 75 75 75 75 ...
     $ landsat_ard_winter_nir_p50    : num  70 70 70 70 70 75 75 75 75 75 ...
     $ landsat_ard_winter_nir_p75    : num  70 70 70 70 70 75 75 75 75 75 ...
     $ landsat_ard_winter_red_p25    : num  15 15 15 15 15 35 35 35 35 35 ...
     $ landsat_ard_winter_red_p50    : num  15 15 15 15 15 38 38 38 38 38 ...
     $ landsat_ard_winter_red_p75    : num  15 15 15 15 15 39 39 39 39 39 ...
     $ landsat_ard_winter_swir1_p25  : num  59 59 59 59 59 42 42 42 42 42 ...
     $ landsat_ard_winter_swir1_p50  : num  59 59 59 59 59 49 49 49 49 49 ...
     $ landsat_ard_winter_swir1_p75  : num  59 59 59 59 59 56 56 56 56 56 ...
     $ landsat_ard_winter_swir2_p25  : num  31 31 31 31 31 21 21 21 21 21 ...
     $ landsat_ard_winter_swir2_p50  : num  31 31 31 31 31 25 25 25 25 25 ...
     $ landsat_ard_winter_swir2_p75  : num  31 31 31 31 31 29 29 29 29 29 ...
     $ landsat_ard_winter_thermal_p25: num  186 186 186 186 186 184 184 184 184 184 ...
     $ landsat_ard_winter_thermal_p50: num  186 186 186 186 186 185 185 185 185 185 ...
     $ landsat_ard_winter_thermal_p75: num  186 186 186 186 186 185 185 185 185 185 ...
     $ night_lights                  : num  0.125 0.125 0.125 0.125 0.125 ...
     $ dtm_elevation                 : num  721 978 698 727 450 668 626 698 833 561 ...
     $ dtm_slope                     : num  3.95 19 12.92 33.38 15.56 ...
     - attr(*, ".internal.selfref")=<externalptr> 
     - attr(*, "sorted")= chr "row.id"

### (analysis ready) data

    library(dplyr)
    cm.croatia <- readRDS("/data/eumap/sample-data/R-sample-tiles/9529/9529_croatia_samples.rds")
    # str(cm.croatia)
    df <-  cm.croatia
    df$lc_class <- as.factor(df$lc_class)
    #crs = "+init=epsg:3035"
    #target.variable = "lc_class"
    df <- df %>% group_by_if(is.character, as.factor)
    df$row.id <- NULL
    df$survey_date <- NULL
    df$lucas <- NULL
    df$Date <- NULL
    df$id <- NULL
    df$year <- NULL
    df$tile_id <- NULL
    df$confidence <- NULL
    # df$V2 <- NULL
    # colnames(df)
    colnames(df)[2] <- "x"
    colnames(df)[3] <- "y"
    #coordinate_names = c("x","y")
    df <- as.data.frame(df)
    smp_size <- floor(0.5 * nrow(df))
    set.seed(123)
    train_ind <- sample(seq_len(nrow(df)), size = smp_size)
    df.tr <- df[train_ind, ]
    df.ts <- df[ -train_ind,]
    newdata = df.ts

`train_spm`
-----------

    library(mlr3verse)
    library(future)
    library(progressr)
    library(checkmate)
    tr = eumap::train_spm(df.tr, target.variable = "lc_class" , folds = 5 , n_evals = 3)

            classification Task  ...TRUE

               Fitting an ensemble ML using  kknn  featureless, and Randome Forests models ncores: 32  resampling method: (non-spatial) repeated_cv ...TRUE

    train_model= tr[[1]]
    var.imp = tr[[2]]
    summary = tr[[3]]
    response = tr[[4]]
    vlp = tr[[5]]
    target = tr[[6]]
    summary

    Ranger result

    Call:
     ranger::ranger(dependent.variable.name = task$target_names, data = task$data(),      probability = self$predict_type == "prob", case.weights = task$weights$weight,      importance = "permutation", mtry = 2L, sample.fraction = 0.964161029900424,      num.trees = 215L) 

    Type:                             Classification 
    Number of trees:                  215 
    Sample size:                      379 
    Number of independent variables:  89 
    Mtry:                             2 
    Target node size:                 1 
    Variable importance mode:         permutation 
    Splitrule:                        gini 
    OOB prediction error:             2.11 % 

variable importance plot

    varImp = barplot(var.imp, horiz = TRUE, las = 1, col = gray.colors(10))
      title(main = "variable importance", font.main = 4)

<img src="README_files/figure-markdown_strict/unnamed-chunk-25-1.png" alt="Accuracy plot"  />
<p class="caption">
Accuracy plot
</p>

`predict_spm`
-------------

Prediction; raster map

    year.span = c(2000:2020)
    #
    aq =c("/data/eumap/sample-data/R-sample-tiles/9529/timeless/dtm_elevation.tif" ,"/data/eumap/sample-data/R-sample-tiles/9529/timeless/dtm_slope.tif")
    for (i in 1:2) {
      tif1.lst = list.files(paste0('/data/eumap/sample-data/R-sample-tiles/9529/',year.span[i]), pattern=".tif", full.names=TRUE, recursive=TRUE)
      tif1.lst <- append(tif1.lst, aq)
      out.tif <- paste0("/data/eumap/img/R/","land_cover_9529_croatia_",year.span[i],".tif")
      br01 = stack(tif1.lst)
      newdata = as(br01, "SpatialGridDataFrame")
      predict.ts = predict_spm(train_model, newdata@data)
      newdata$pred = predict.ts
      newdataa <- newdata ## copy and make new raster object
      newdataa@data$pred <- as.numeric(levels(newdataa@data$pred))[newdataa@data$pred]
      writeGDAL(newdataa["pred"], out.tif, drivername="GTiff", type="Int16", mvFlag = -32768 ,options=c("COMPRESS=DEFLATE"))
    }

-   Removing small classes in prediction optional

<!-- -->

    xg <- summary(newdata$pred, maxsum=(1+length(levels(newdata$pred))))
    str(xg)
    selg.levs <- attr(xg, "names")[xg > 5]  
    attr(xg, "names")[xg <= 5] #drop classes with less than 5 pixels
    newdata$pred[which(!newdata$pred %in% selg.levs)] <- NA
    newdata$pred <- droplevels(newdata$pred)
    str(summary(newdata$pred, maxsum=length(levels(newdata$pred))))

    plot(runif(10,1,10),runif(10,100,1e6))

<img src="README_files/figure-markdown_strict/unnamed-chunk-28-1.png" alt="LC map"  />
<p class="caption">
LC map
</p>

References
----------

Casalicchio, G., Bossek, J., Lang, M., Kirchhoff, D., Kerschke, P.,
Hofner, B., … Bischl, B. (2017). OpenML: An R package to connect to the
machine learning platform OpenML. *Computational Statistics*, 1–15.
doi:[10.1007/s00180-017-0742-2](https://doi.org/10.1007/s00180-017-0742-2)

Lang, M., Schratz, P., Binder, M., Pfisterer, F., Richter, J., Reich, N.
G., & Bischl, B. (2020, September 9). mlr3 book. Retrieved from
<https://mlr3book.mlr-org.com>
