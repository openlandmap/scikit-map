---
title: " demo: eumap Rpackage functionalities"
author: "Mohammadreza Sheykhmousa (mohammadreza.sheykhmousa@OpenGeoHub.org)"
date: "Last compiled on: 01 October, 2020"
output: 
   rmarkdown::html_document:
    keep_md: true
    md_extensions: +autolink_bare_uris+hard_line_breaks
    theme: united
    number_sections: false
    highlight: tango
    toc: true
    toc_float:
      collapsed: false
      smooth_scroll: true
    toc_depth: 3
bibliography: ./tex/refs.bib
csl: ./tex/apa.csl  
fig_caption: yes
link-citations: yes
twitter-handle: opengeohub
header-includes:
- \usepackage{caption}
---

[1.1]: http://i.imgur.com/tXSoThF.png (twitter icon with padding)
[1]: https://twitter.com/sheykhmousa
Follow me on [![alt text][1.1]][1]




```r
library(knitr)
library(eumap)
```



## Introduction
`eumap` aims at providing easier access to EU environmental maps.
Basic functions train a spatial prediction model using [mlr3 package](https://mlr3.mlr-org.com/), [@mlr3], and related extensions in the [mlr3 ecosystem](https://github.com/mlr-org/mlr3/wiki/Extension-Packages) [@casalicchio2017openml; @MichelLang2020mlr3book], 
which includes spatial prediction using [Ensemble Machine Learning](https://koalaverse.github.io/machine-learning-in-R/stacking.html#stacking-software-in-r) taking spatial coordinates and spatial cross-validation into account. 
In a nutshell one can `train` an arbitrary `s3` **(spatial)dataframe** in `mlr3` ecosystem by defining *df* and *target.variable* i.e., response.
main functions are as the following:

## `train.spm()` 

1. `train.spm()` will automatically perform `classification` or `regression` tasks and the output is a `train.model` which later can be used to predict `newdata`.It also provides *summary* of the model and *variable importance* and *response*.
The rest of arguments can be either pass or default values will be passed. `train.spm()` provides four scenarios:

  1.1. `classification` task with **non spatial** resampling methods
  1.2. `regression` task with **non spatial** resampling methods
  1.3. `classification` task with **spatial** resampling methods
  1.4. `regression` task with **spatial** resampling methods

## `predict.spm()`

1. Prediction on a new dataset using `train.model`,
2. User needs to set`df.ts = test set` and also pass the `train.model`. 

## `accuracy.plot()` 

1. Accuracy plot in case of regression task (don’t use it for classification tasks for obvious reason),
 
**Warning:** most of functions are optimized to run in parallel by default. This might result in high RAM and CPU usage.

The following examples demonstrates spatial prediction using the `meuse` data set:

## Required packages


```r
start_time <- Sys.time()
ls <- c("lattice", "raster", "plotKML", "ranger", "mlr3verse", "BBmisc", "knitr", "bbotk",
    "hexbin", "stringr", "magrittr", "sp", "ggplot2", "mlr3fselect", "mlr3spatiotempcv", 
    "FSelectorRcpp", "future", "future.apply", "mlr3filters", "EnvStats", "grid", "mltools","gridExtra","yardstick","plotKML", "latticeExtra","devtools")
new.packages <- ls[!(ls %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos="https://cran.rstudio.com", force=TRUE)
```

## meuse dataset


```r
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
```

## split training (tr) and test (ts) set  

```r
smp_size <- floor(0.5 * nrow(df))
set.seed(123)
train_ind <- sample(seq_len(nrow(df)), size = smp_size)
df.tr <- df[, c("x","y","dist","ffreq","soil","lead")]
df.ts <- df.grid[, c("x","y","dist","ffreq","soil")]
```

## setting generic variables 

```r
folds = 2
n_evals = 3
newdata = df.ts
```

## setting generic accuracy plot variables

```r
colorcut. = c(0,0.01,0.03,0.07,0.15,0.25,0.5,0.75,1)
colramp. = colorRampPalette(c("wheat2","red3"))
xbins. = 50
```

## Loading required libraries:

```r
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
```










## `train.spm`

`train.spm` fits multiple models/learners depending on the `class()` of the **target.variable** and for returns a `trained model`, **var.imp**, **summary** of the model, and **response** variables. `trained model` later can predict a `newdata` set. 


```r
tr = train.spm(df.tr, target.variable = target.variable , folds = folds , n_evals = n_evals , crs)
```

```
Regr Task   resampling method: non-spatialCV  ncores:  32 ...TRUE
```

```
Using learners: method.list...TRUE
```

```
           Fitting a ensemble ML using 'mlr3::Taskregr'...TRUE
```

`train.spm` results:

1st element is the *trained model*:


```r
train.model= tr[[1]]
```
2nd element is the *variable importance*:

```r
var.imp = tr[[2]]
var.imp
```

```
    dist        x        y    ffreq     soil 
472304.4 248899.5 226325.8 164080.2 105424.1 
```
3rd element of the summary of the *trained model*:

```r
summary = tr[[3]]
summary
```

```
Ranger result

Call:
 ranger::ranger(dependent.variable.name = task$target_names, data = task$data(),      case.weights = task$weights$weight, importance = "impurity",      mtry = 2L, sample.fraction = 0.751642505638301, num.trees = 287L) 

Type:                             Regression 
Number of trees:                  287 
Sample size:                      152 
Number of independent variables:  5 
Mtry:                             2 
Target node size:                 5 
Variable importance mode:         impurity 
Splitrule:                        variance 
OOB prediction error (MSE):       4565.397 
R squared (OOB):                  0.631821 
```

4th element is the predicted values of our trained model
note: here we just show start and the ending values

```r
response = tr[[4]]
response
```

```
...
  [1] 245.74701 198.56257 175.88260 152.64985 109.78419  89.92654 133.32049
  [8] 187.69835 152.62631  96.49871  90.27788 140.63443 230.66843 151.79353
 [15] 152.21687 244.51578 201.96843 189.52022 231.13580 196.41175 166.55726
...
```

## `predict.spm()`

prediction on `newdata` set

```r
predict.variable = predict.spm(train.model, newdata)
```
### predicted values for the *newdata* set:
note: here we just show start and the ending values

```r
predict.variable
```

```
...
   [1] 260.96457 260.96457 233.66716 234.80209 260.96457 233.66716 231.43072
   [8] 234.44279 264.24775 233.45740 231.43072 234.44279 204.73008 193.59737
...
```
## `accuracy.plot.spm` 
in case of regression task,
- for now we have two scenarios including:
 - rng = "nat" provides visualizations with real values
 - rng = "norm" provides visualizations with the normalized (0~1) values


```r
plt = accuracy.plot.spm(x = df.tr[,target.variable], y = response, rng = "norm")
```

<div class="figure" style="text-align: center">
<img src="README_files/figure-html/unnamed-chunk-19-1.png" alt="Accuracy plot"  />
<p class="caption">Accuracy plot</p>
</div>



## raster grid 
make a map using ensemble machine learning with spatial cross validation for the predicted variables e.g., *lead* (in this case) 

```r
plot(df.ts[,"leadp"])
points(meuse, pch="+")
```

<div class="figure" style="text-align: center">
<img src="README_files/figure-html/unnamed-chunk-21-1.png" alt="Raster grid"  />
<p class="caption">Raster grid</p>
</div>

## References

