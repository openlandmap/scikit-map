# eumap package for R

Follow me on [![alt text](http://i.imgur.com/tXSoThF.png "twitter icon with padding")](https://twitter.com/sheykhmousa)

Package provides easier access to EU environmental maps.
Basic functions include:

* `train.spm` --- train a spatial prediction model using [mlr3 package](https://mlr3.mlr-org.com/)) ([Lang](https://mlr3book.mlr-org.com/introduction.html#ref-mlr3) et
al. 2019)(Lang et al., [2020](#ref-MichelLang2020mlr3book)) package and
[ecosystem](https://github.com/mlr-org/mlr3/wiki/Extension-Packages) implementation with spatial coordinates and spatial cross-validation. In a nutshell one can `train` an arbitrary `s3` **(spatial) data
frame** in `mlr3` ecosystem by defining following arguments:

*df* and the *target.variable*.
    `train.spm()` will automatically perform `classification` or
    `regression` tasks.
The rest of arguments can be set or default values will be set.
If **crs** is set `train.spm()` will automatically take care of
    **spatial cross validation**,

* `predict.spm()` --- prediction on new dataset,

* `accuracy.plot()` --- Accuracy plot in case of regression task Note: don’t use it for classification tasks for obvious reasons


Warning: most of functions are optimized to run in parallel by default. This might result in high RAM and CPU usage.



The following examples demostrates spatial prediction using the meuse data set:

```r
ls <- c("mlr3verse","rgdal", "raster", "ranger", "mlr3", 
        "xgboost", "glmnet", "matrixStats", "deepnet")
new.packages <- ls[!(ls %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
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
library("latticeExtra")
library("devtools")
library("landmap")
```

<!-- -->

    train.model = train.spm(df.tr, target.variable = target.variable, folds = folds ,n_evals = n_evals, plot.workflow = TRUE, crs )

    regression Task   resampling method: non-spatialCV  ncores:  32 ...TRUE

    Using learners: method.list...TRUE

               Fitting a ensemble ML using 'mlr3::Taskregr'...TRUE

    train.model

`predict.spm()`

User needs to set`df.ts = test set` and `task = NULL`

    predict.variable = predict.spm(df.ts, task = NULL)
    predict.variable

`accuracy.plot()`

    accuracy.plot.spm(x = df.ts[,target.variable], y = predict.variable)

<img src="README_files/figure-markdown_strict/unnamed-chunk-10-1.png" alt="Accuracy plot"  />
<p class="caption">

</p>

## Contributions

* Contributions to eumap are welcome. Issues and pull requests are the preferred ways of sharing them.
* We are interested in your results and experiences with using the mlr3 functions 
  for generating spatial predictions with your own data. Share your data sets, 
  code and results either using github issues and/or R-sig-geo mailing list.

References
----------

Lang, M., Schratz, P., Binder, M., Pfisterer, F., Richter, J., Reich, N.
G., & Bischl, B. (2020, September 9). mlr3 book. Retrieved from
<https://mlr3book.mlr-org.com>
