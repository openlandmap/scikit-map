

-   [Introduction](#introduction)
-   [`train.spm()`](#train.spm)
-   [`predict.spm()`](#predict.spm)
-   [`accuracy.plot()`](#accuracy.plot)
-   [Required packages](#required-packages)
-   [meuse dataset](#meuse-dataset)
-   [split training (tr) and test (ts)
    set](#split-training-tr-and-test-ts-set)
-   [setting generic variables](#setting-generic-variables)
-   [setting generic accuracy plot
    variables](#setting-generic-accuracy-plot-variables)
-   [Loading required libraries:](#loading-required-libraries)
-   [`train.spm`](#train.spm-1)
-   [`predict.spm()`](#predict.spm-1)
    -   [predicted values for the *newdata*
        set:](#predicted-values-for-the-newdata-set)
-   [`accuracy.plot.spm`](#accuracy.plot.spm)
-   [raster grid](#raster-grid)
-   [References](#references)

Follow me on [![alt
text](http://i.imgur.com/tXSoThF.png "twitter icon with padding")](https://twitter.com/sheykhmousa)

    library(eumap)

Introduction
------------

`eumap` aims at providing easier access to EU environmental maps. Basic
functions train a spatial prediction model using [mlr3
package](https://mlr3.mlr-org.com/), (Lang et al., [2019](#ref-mlr3)),
and related extensions in the [mlr3
ecosystem](https://github.com/mlr-org/mlr3/wiki/Extension-Packages)
(Casalicchio et al., [2017](#ref-casalicchio2017openml); Lang et al.,
[2020](#ref-MichelLang2020mlr3book)), which includes spatial prediction
using [Ensemble Machine
Learning](https://koalaverse.github.io/machine-learning-in-R/stacking.html#stacking-software-in-r)
taking spatial coordinates and spatial cross-validation into account. In
a nutshell one can `train` an arbitrary `s3` **(spatial)dataframe** in
`mlr3` ecosystem by defining *df* and *target.variable* i.e., response.
main functions are as the following:

`train.spm()`
-------------

1.  `train.spm()` will automatically perform `classification` or
    `regression` tasks and the output is a `train.model` which later can
    be used to predict `newdata`.It also provides *summary* of the model
    and *variable importance* and *response*. The rest of arguments can
    be either pass or default values will be passed. `train.spm()`
    provides four scenarios:

1.1. `classification` task with **non spatial** resampling methods 1.2.
`regression` task with **non spatial** resampling methods 1.3.
`classification` task with **spatial** resampling methods 1.4.
`regression` task with **spatial** resampling methods

`predict.spm()`
---------------

1.  Prediction on a new dataset using `train.model`,
2.  User needs to set`df.ts = test set` and also pass the `train.model`.

`accuracy.plot()`
-----------------

1.  Accuracy plot in case of regression task (don’t use it for
    classification tasks for obvious reason),

**Warning:** most of functions are optimized to run in parallel by
default. This might result in high RAM and CPU usage.

The following examples demonstrates spatial prediction using the `meuse`
data set:

Required packages
-----------------

    start_time <- Sys.time()
    ls <- c("lattice", "raster", "plotKML", "ranger", "mlr3verse", "BBmisc", "knitr", "bbotk",
        "hexbin", "stringr", "magrittr", "sp", "ggplot2", "mlr3fselect", "mlr3spatiotempcv", 
        "FSelectorRcpp", "future", "future.apply", "mlr3filters", "EnvStats", "grid", "mltools","gridExtra","yardstick","plotKML", "latticeExtra","devtools")
    new.packages <- ls[!(ls %in% installed.packages()[,"Package"])]
    if(length(new.packages)) install.packages(new.packages, repos="https://cran.rstudio.com", force=TRUE)

meuse dataset
-------------

    library("sp")
    demo(meuse, echo=FALSE)
    pr.vars = c("x","y","dist","ffreq","soil","lead")
    df <- as.data.frame(meuse)
    df.grid <- as.data.frame(meuse.grid)
    # df <- df[complete.cases(df[,pr.vars]),pr.vars]
    df = na.omit(df[,])
    df.grid = na.omit(df.grid[,])
    summary(is.na(df))
    summary(is.na(df.grid))
    crs = "+init=epsg:28992"
    target.variable = "lead"

split training (tr) and test (ts) set
-------------------------------------

    smp_size <- floor(0.5 * nrow(df))
    set.seed(123)
    train_ind <- sample(seq_len(nrow(df)), size = smp_size)
    df.tr <- df[, c("x","y","dist","ffreq","soil","lead")]
    df.ts <- df.grid[, c("x","y","dist","ffreq","soil")]

setting generic variables
-------------------------

    folds = 2
    n_evals = 3
    newdata = df.ts

setting generic accuracy plot variables
---------------------------------------

    colorcut. = c(0,0.01,0.03,0.07,0.15,0.25,0.5,0.75,1)
    colramp. = colorRampPalette(c("wheat2","red3"))
    xbins. = 50

Loading required libraries:
---------------------------

    library("mlr3verse")
    library("bbotk")
    library("ggplot2")
    library("mltools")
    library("data.table")
    library("mlr3fselect")
    library("FSelectorRcpp")
    library("future")
    library("future.apply")
    library("magrittr")
    library("progress")
    library("mlr3spatiotempcv")
    library("sp")
    library("landmap")  
    library("dplyr")
    library("EnvStats")
    library("grid")
    library("hexbin")
    library("BBmisc")
    library("lattice")
    library("MASS")
    library("gridExtra")
    library("MLmetrics")
    library("yardstick")
    library("plotKML")
    library("latticeExtra")
    library("devtools")
    library("raster")

`train.spm`
-----------

`train.spm` fits multiple models/learners depending on the `class()` of
the **target.variable** and for returns a `trained model`, **var.imp**,
**summary** of the model, and **response** variables. `trained model`
later can predict a `newdata` set.

    tr = train.spm(df.tr, target.variable = target.variable , folds = folds , n_evals = n_evals , crs = "+init=epsg:28992")

             fit the regression model  (rsmp = SPCV by cooridinates) ...TRUE

`train.spm` results:

1st element is the *trained model*:

    train.model= tr[[1]]

2nd element is the *variable importance*:

    var.imp = tr[[2]]
    var.imp

        dist    ffreq     soil 
    4199.005 2418.450 1898.147 

3rd element of the summary of the *trained model*:

    summary = tr[[3]]
    summary

    Ranger result

    Call:
     ranger::ranger(dependent.variable.name = task$target_names, data = task$data(),      case.weights = task$weights$weight, importance = "permutation") 

    Type:                             Regression 
    Number of trees:                  500 
    Sample size:                      152 
    Number of independent variables:  9 
    Mtry:                             3 
    Target node size:                 5 
    Variable importance mode:         permutation 
    Splitrule:                        variance 
    OOB prediction error (MSE):       6721.765 
    R squared (OOB):                  0.4579194 

4th element is the predicted values of our trained model note: here we
just show start and the ending values

    response = tr[[4]]
    response

    ...
      [1] 268.05358 244.43426 205.18133 155.55110 119.10944 111.98254 152.94772
      [8] 221.80734 207.79976 113.21834 112.47830 194.11921 267.23847 205.43335
     [15] 210.43176 270.52192 251.40160 249.41321 272.16468 269.21231 225.45011
    ...

`predict.spm()`
---------------

prediction on `newdata` set

    predict.variable = predict.spm(train.model, newdata)

### predicted values for the *newdata* set:

note: here we just show start and the ending values

    predict.variable

    ...
       [1] 266.38482 266.38482 244.39279 249.56899 266.38482 244.39279 241.15656
       [8] 250.57212 266.22951 244.39279 241.15656 250.57212 223.56810 212.50498
    ...

`accuracy.plot.spm`
-------------------

in case of regression task, - for now we have two scenarios including: -
rng = “nat” provides visualizations with real values - rng = “norm”
provides visualizations with the normalized (0~1) values

    plt = accuracy.plot.spm(x = df.tr[,target.variable], y = response, rng = "norm")

<img src="README_files/figure-markdown_strict/unnamed-chunk-19-1.png" alt="Accuracy plot"  />
<p class="caption">
Accuracy plot
</p>

raster grid
-----------

make a map using ensemble machine learning with spatial cross validation
for the predicted variables e.g., *lead* (in this case)

    plot(df.ts[,"leadp"])
    points(meuse, pch="+")

<img src="README_files/figure-markdown_strict/unnamed-chunk-21-1.png" alt="Raster grid"  />
<p class="caption">
Raster grid
</p>

References
----------

Casalicchio, G., Bossek, J., Lang, M., Kirchhoff, D., Kerschke, P.,
Hofner, B., … Bischl, B. (2017). OpenML: An R package to connect to the
machine learning platform OpenML. *Computational Statistics*, 1–15.
doi:[10.1007/s00180-017-0742-2](https://doi.org/10.1007/s00180-017-0742-2)

Lang, M., Binder, M., Richter, J., Schratz, P., Pfisterer, F., Coors,
S., … Bischl, B. (2019). mlr3: A modern object-oriented machine learning
framework in R. *Journal of Open Source Software*.
doi:[10.21105/joss.01903](https://doi.org/10.21105/joss.01903)

Lang, M., Schratz, P., Binder, M., Pfisterer, F., Richter, J., Reich, N.
G., & Bischl, B. (2020, September 9). mlr3 book. Retrieved from
<https://mlr3book.mlr-org.com>
