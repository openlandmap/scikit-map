

-   [Aims of this tutorial](#aims-of-this-tutorial)
-   [train.spm](#train.spm)
-   [`train.spm()`](#train.spm-1)
-   [`predict.spm()`](#predict.spm)
-   [`accuracy.plot()`](#accuracy.plot)
-   [Basic requirements user needs to
    set](#basic-requirements-user-needs-to-set)
-   [References](#references)

[<img src="tex/opengeohub_logo_ml.png" alt="OpenGeoHub logo" width="350"/>](https://opengeohub.org)

Follow me on [![alt
text](http://i.imgur.com/tXSoThF.png "twitter icon with padding")](https://twitter.com/sheykhmousa)

### Aims of this tutorial

-   create an R function `train.spm` to train (classification and
    regression Spatial Prediction Model) models using Ensemble Machine
    Learning on spatial data in mlr3 ,
-   create an R function accuracy.plot.spm to visualize accuracy plots
    (regression) for models produced using `train.spm`,
-   create an R function `predict.spm` that can used the models fitted
    using `train.spm` and predict values that can be exported as a map.

### train.spm

`train.spm` sources `train.spm.fnc.R`, `predict.spm.fnc.R`, and
`accuracy.plot.spm.fnc.R` ie., for regression tasks, functions to
fulfill the aims of this tutorial using mlr3
([Lang](https://mlr3book.mlr-org.com/introduction.html#ref-mlr3) et
al. 2019)(Lang et al., [2020](#ref-MichelLang2020mlr3book)) package and
[ecosystem](https://github.com/mlr-org/mlr3/wiki/Extension-Packages).

-   In `train.spm` we need to install the required packages as
    followings:

<!-- -->

    start_time <- Sys.time()
    ls <- c("lattice", "raster", "plotKML", "ranger", "mlr3verse", "BBmisc", "knitr", "bbotk",
        "hexbin", "stringr", "magrittr", "sp", "ggplot2", "mlr3fselect", "mlr3spatiotempcv", 
        "FSelectorRcpp", "future", "future.apply", "mlr3filters", "EnvStats", "grid", "mltools","gridExtra","yardstick","plotKML", "latticeExtra","devtools")
    new.packages <- ls[!(ls %in% installed.packages()[,"Package"])]
    if(length(new.packages)) install.packages(new.packages, repos="https://cran.rstudio.com", force=TRUE)

\#\#\#meuse dataset

Splitting training (tr) and test (ts) sets and defining generic
variables - The user can modify them.

    smp_size <- floor(0.5 * nrow(df))
    set.seed(123)
    train_ind <- sample(seq_len(nrow(df)), size = smp_size)
    df.tr <- df[train_ind, ]
    df.ts <- df[-train_ind, ]
    folds = 5
    n_evals = 10
    colorcut. = c(0,0.01,0.03,0.07,0.15,0.25,0.5,0.75,1)
    colramp. = colorRampPalette(c("wheat2","red3"))
    xbins. = 30
    target.variable = "lead"

    library("mlr3verse")

    Loading required package: mlr3

    Loading required package: mlr3filters

    Loading required package: mlr3learners

    Loading required package: mlr3pipelines

    Loading required package: mlr3tuning

    Loading required package: mlr3viz

    Loading required package: paradox

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

    version: 0.0.3

    library("GSIF")

    GSIF version 0.5-5.1 (2019-01-04)

    URL: http://gsif.r-forge.r-project.org/


    Attaching package: 'GSIF'

    The following objects are masked from 'package:landmap':

        buffer.dist, fit.vgmModel, getSpatialTiles, makeTiles, spc, spfkm,
        spmultinom, spsample.prob, tile

    library("dplyr")


    Attaching package: 'dplyr'

    The following objects are masked from 'package:data.table':

        between, first, last

    The following objects are masked from 'package:stats':

        filter, lag

    The following objects are masked from 'package:base':

        intersect, setdiff, setequal, union

    library("EnvStats")


    Attaching package: 'EnvStats'

    The following object is masked from 'package:GSIF':

        predict

    The following object is masked from 'package:mltools':

        skewness

    The following objects are masked from 'package:stats':

        predict, predict.lm

    The following object is masked from 'package:base':

        print.default

    library("grid")
    library("hexbin")
    library("BBmisc")


    Attaching package: 'BBmisc'

    The following object is masked from 'package:grid':

        explode

    The following objects are masked from 'package:dplyr':

        coalesce, collapse

    The following object is masked from 'package:base':

        isFALSE

    library("lattice")
    library("MASS")


    Attaching package: 'MASS'

    The following object is masked from 'package:EnvStats':

        boxcox

    The following object is masked from 'package:dplyr':

        select

    library("gridExtra")


    Attaching package: 'gridExtra'

    The following object is masked from 'package:dplyr':

        combine

    library("MLmetrics")


    Attaching package: 'MLmetrics'

    The following object is masked from 'package:base':

        Recall

    library("yardstick")

    For binary classification, the first factor level is assumed to be the event.
    Use the argument `event_level = "second"` to alter this as needed.


    Attaching package: 'yardstick'

    The following objects are masked from 'package:mltools':

        mcc, rmse

    library("plotKML")

    plotKML version 0.6-1 (2020-03-08)

    URL: http://plotkml.r-forge.r-project.org/

    library("latticeExtra")


    Attaching package: 'latticeExtra'

    The following object is masked from 'package:ggplot2':

        layer

    library("devtools")

    Loading required package: usethis

### `train.spm()`

Here we have four scenarios:

-   `classification` task with **non spatial** resampling methods
-   `regression` task with **non spatial** resampling methods
-   `classification` task with **spatial** resampling methods
-   `regression` task with **spatial** resampling methods

<!-- -->

    train.spm = function(df.tr, target.variable, 
    parallel = TRUE, predict_type = NULL, folds = 5, method.list = NULL,  n_evals = 3, plot.workflow = FALSE, var.ens = TRUE, meta.learner = NULL, crs){
      id = deparse(substitute(df.tr))
      cv3 = rsmp("repeated_cv", folds = folds)
       if(is.factor(df.tr[,target.variable]) & missing(crs)){
        message(paste("classification Task  ","resampling method: non-spatialCV ", "ncores: ",availableCores(), "..."), immediate. = TRUE)
            message(paste0("Using learners: ", paste("method.list", collapse = ", "), "..."), immediate. = TRUE)
            tsk_clf <- mlr3::TaskClassif$new(id = id, backend = df.tr, target = target.variable)
            lrn = lrn("classif.rpart")
            gr = pipeline_robustify(tsk_clf, lrn) %>>% po("learner", lrn)
            ede = resample(tsk_clf, GraphLearner$new(gr), rsmp("holdout"))
            tsk_clasif1 = ede$task$clone()
            ranger_lrn = lrn("classif.ranger", predict_type = "response",importance ="permutation")
            ps_ranger = ParamSet$new(
               list(
                 ParamInt$new("mtry", lower = 1L, upper = 5L),
                 ParamDbl$new("sample.fraction", lower = 0.5, upper = 1),
                 ParamInt$new("num.trees", lower = 50L, upper = 500L),
                 ParamFct$new("importance", "permutation")
               ))
             at = AutoTuner$new(
               learner = ranger_lrn,
               resampling = cv3,
               measure = msr("classif.acc"),
               search_space = ps_ranger,
               terminator = trm("evals", n_evals = n_evals), 
               tuner = tnr("random_search")
             )
             at$store_tuning_instance = TRUE
            requireNamespace("lgr")
            logger = lgr::get_logger("mlr3")
            logger$set_threshold("trace")
            lgr::get_logger("mlr3")$set_threshold("warn")
            lgr::get_logger("mlr3")$set_threshold("debug")
            message("           Fitting a ensemble ML using 'mlr3::TaskClassif'...", immediate. = TRUE)
            at$train(tsk_clasif1)
            at$learner$train(tsk_clasif1)
            tr.mdl = at$learner
            train.model = tr.mdl$predict_newdata
          } else if (is.numeric(df.tr[,target.variable]) & missing(crs)) {
          message(paste("regression Task  ","resampling method: non-spatialCV ", "ncores: ",availableCores(), "..."), immediate. = TRUE)  
            if( missing(predict_type)){
              predict_type <- "response" 
            }
          message(paste0("Using learners: ", paste("method.list", collapse = ", "), "..."), immediate. = TRUE)
          tsk_rgr <- mlr3::TaskRegr$new(id = id, backend = df.tr, target = target.variable)
          lrn = lrn("regr.rpart")
          gr = pipeline_robustify(tsk_rgr, lrn) %>>% po("learner", lrn)
          ede = resample(tsk_rgr, GraphLearner$new(gr), rsmp("holdout"))
          tsk_regr1 = ede$task$clone()
          ranger_lrn = lrn("regr.ranger", predict_type = "response",importance ="permutation")
          
          ps_ranger = ParamSet$new(
            list(
              ParamInt$new("mtry", lower = 1L, upper = 5L),
              ParamDbl$new("sample.fraction", lower = 0.5, upper = 1),
              ParamInt$new("num.trees", lower = 50L, upper = 500L),
              ParamFct$new("importance", "impurity")
            ))
          at = AutoTuner$new(
            learner = ranger_lrn,
            resampling = cv3,
            measure = msr("regr.rmse"),
            search_space = ps_ranger,
            terminator = trm("evals", n_evals = n_evals), 
            tuner = tnr("random_search")
          )
          at$store_tuning_instance = TRUE
          
          requireNamespace("lgr")
          logger = lgr::get_logger("mlr3")
          logger$set_threshold("trace")
          lgr::get_logger("mlr3")$set_threshold("warn")
          lgr::get_logger("mlr3")$set_threshold("debug")
          message("           Fitting a ensemble ML using 'mlr3::Taskregr'...", immediate. = TRUE)
          at$train(tsk_regr1)
          
          at$learner$train(tsk_regr1)
          tr.mdl = at$learner
          train.model = tr.mdl$predict_newdata
        } else if (is.factor(df.tr[,target.variable]) & crs == crs){ 
            method.list <- c("classif.ranger", "classif.rpart")
            meta.learner = "classif.ranger"
            df.trf = mlr3::as_data_backend(df.tr)
            tsk_clf = TaskClassifST$new(id = id, backend = df.trf, target = target.variable, extra_args = list(
            positive = "TRUE", coordinate_names = c("x", "y"), coords_as_features = FALSE,crs = crs))
            
            g = gunion(list(
            po("learner_cv", id = "cv1", lrn("classif.ranger")),
            po("pca") %>>% po("learner_cv", id = "cv2", lrn("classif.rpart")),
            po("nop") %>>% po("encode") %>>%  po("imputemode") %>>% po("removeconstants")
            )) %>>%
            po("featureunion") %>>%
            po("learner", lrn("classif.ranger",importance ="permutation")) 

            g$param_set$values$cv1.resampling.method = "spcv_coords"
            g$param_set$values$cv2.resampling.method = "spcv_coords"
            if(plot.workflow == "TRUE"){
              plt = g$plot()
            }
            message(paste( "           fit the classif model  (rsmp = SPCV by cooridinates) ..."), immediate. = TRUE)
            g$train(tsk_clf)
            g$predict(tsk_clf)
            conf.mat = g$pipeops$classif.ranger$learner_model$model$confusion.matrix
            var.imp = g$pipeops$classif.ranger$learner_model$model$variable.importance
            summary = g$pipeops$classif.ranger$learner_model$model
            tr.model = g$pipeops$classif.ranger$learner$train(tsk_clf)
            train.model = tr.model$predict_newdata
            
      } else if(is.numeric(df.tr[,target.variable]) & crs == crs){
            if(is.null(method.list) & is.null(meta.learner)){
                       method.list <- c("regr.ranger", "regr.rpart")
                       meta.learner <- "regr.ranger"}
            df.trf = mlr3::as_data_backend(df.tr)
            tsk_regr = TaskRegrST$new(id = id, backend = df.trf, target = target.variable,
            extra_args = list( positive = "TRUE", coordinate_names = c("x", "y"), coords_as_features = FALSE,
            crs = crs))
            g = gunion(list(
            po("learner_cv", id = "cv1", lrn("regr.ranger")),
            po("pca") %>>% po("learner_cv", id = "cv2", lrn("regr.rpart")),
            po("nop") %>>% po("encode") %>>%  po("imputemode") %>>% po("removeconstants")
            )) %>>%
            po("featureunion") %>>%
            po("learner", lrn("regr.ranger")) 
              
            g$param_set$values$cv1.resampling.method = "spcv_coords"
            g$param_set$values$cv2.resampling.method = "spcv_coords"
            g$keep_results = "TRUE"
            if(plot.workflow == "TRUE"){
              plt = g$plot()
            }
            message(paste( "         fit the regression model  (rsmp = SPCV by cooridinates) ..."), immediate. = TRUE)
            g$train(tsk_regr)
            g$predict(tsk_regr)
            summary = g$pipeops$regr.ranger$learner_model$model
            tr.model = g$pipeops$regr.ranger$learner$train(tsk_regr)
            train.model = tr.model$predict_newdata
      }
      return(train.model)
    }

The above code has fitted multiple models/learners depending on the
`class()` of the **target.variable** and for now only returns a
`trained model` function so later on we could use it to train a new
dataset.

### `predict.spm()`

prediction on new dataset

    predict.spm = function (df.ts , task = NULL){
       id = deparse(substitute(df.ts))
       if(is.factor(df.ts[,target.variable])){
           tsk_clf <- mlr3::TaskClassif$new(id = id, backend = df.ts, target = target.variable)
           predict.variable = train.model(df.ts, tsk_clf) 
           y = df.ts[,target.variable]

        } else if (is.numeric(df.ts[,target.variable])) {
            task_regr <-mlr3::TaskRegr$new(id = id, backend = df.ts, target = target.variable)
            predict.variable = train.model(df.ts, task_regr)   
            y = df.ts[,target.variable]

        } else if (is.factor(df.ts[,target.variable]) ){ 
            df.tsf = mlr3::as_data_backend(df.ts)
            tsk_clf = TaskClassifST$new(id = id, backend = df.tsf, target = target.variable, extra_args = list(
            positive = "TRUE", coordinate_names = c("x", "y"), coords_as_features = FALSE,crs = crs))
            predict.variable = train.model(df.ts, tsk_clf)   
            y = df.ts[,target.variable]

        } else if (is.numeric(df.ts[,target.variable])){
          df.tsf = mlr3::as_data_backend(df.ts)
          tsk_regr = TaskRegrST$new( id = id, backend = df.tsf, target = target.variable, extra_args = list( positive = "TRUE", coordinate_names = c("x", "y"), coords_as_features = FALSE, crs = crs))
          predict.variable = train.model(df.ts, tsk_regr)   
          y = df.ts[,target.variable]
          
        }
       return(y)
    }

### `accuracy.plot()`

Accuracy plot in case of regression task Note: don’t use it for
classification tasks for obvious reasons

    pfun <- function(x,y, ...){
      panel.xyplot(x, y, ...)
      panel.hexbinplot(x,y, ...)  
      panel.abline(coef = c(0,1), col="black", size = 0.25, lwd = 2)
    }

    accuracy.plot.spm <- function(x, y, main, colramp, xbins = xbins. , rng ="nat"){
    if(rng == "norm"){
        x.= normalize(x, method = "range", range = c(0, 1))
        y. = normalize(y, method = "range", range = c(0, 1))
        plt <- hexbinplot(x. ~ y., xbins = xbins., mincnt = 1, xlab=expression(italic("0~1 measured")), ylab=expression(italic("0~1 predicted")), inner=0.2, cex.labels=1, colramp = colramp., aspect = 1, main= paste0('RMSE: ', '    RSQ: '), colorcut= colorcut., type="g",panel = pfun) 
      
      }
       if(rng == "nat"){
          plt <- hexbinplot(x ~ y, mincnt = 1, xbins=35, xlab="measured", ylab="predicted (ensemble)", inner=0.2, cex.labels=1, colramp= colramp., aspect = 1, main=paste0('RMSE: ', '    RSQ: '), colorcut=colorcut., type="g",panel = pfun) 
        plt  
       }
      print(plt)
      return(plt)
      }

### Basic requirements user needs to set

in a nutshell the user can `train` an arbitrary `s3` **(spatial) data
frame** by only defining following arguments:

`train.spm()`

    train.model = train.spm(df.tr, target.variable = target.variable, folds = folds ,n_evals = n_evals, plot.workflow = TRUE, crs )

    regression Task   resampling method: non-spatialCV  ncores:  32 ...TRUE

    Using learners: method.list...TRUE

    INFO  [06:17:53.694] Applying learner 'fixfactors.removeconstants.regr.rpart' on task 'df.tr' (iter 1/1) 

               Fitting a ensemble ML using 'mlr3::Taskregr'...TRUE

    DEBUG [06:17:54.258] Skip subsetting of task 'df.tr' 
    DEBUG [06:17:54.274] Calling train method of Learner 'regr.ranger.tuned' on task 'df.tr' with 76 observations {learner: <AutoTuner/Learner/R6>}
    INFO  [06:17:54.344] Starting to optimize 4 parameter(s) with '<OptimizerRandomSearch>' and '<TerminatorEvals>' 
    INFO  [06:17:54.397] Evaluating 1 configuration(s) 
    INFO  [06:17:54.441] Benchmark with 50 resampling iterations 
    DEBUG [06:17:54.442] Running benchmark() asynchronously with 50 iterations 
    INFO  [06:17:54.456] Applying learner 'regr.ranger' on task 'df.tr' (iter 32/50) 
    DEBUG [06:17:54.466] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 6..]}
    DEBUG [06:17:54.470] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:54.493] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:54.495] Creating Prediction for predict set 'test' 
    DEBUG [06:17:54.498] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71]}
    DEBUG [06:17:54.501] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:54.515] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:54.517] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:54.518] Applying learner 'regr.ranger' on task 'df.tr' (iter 39/50) 
    DEBUG [06:17:54.527] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:17:54.531] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:54.554] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:54.556] Creating Prediction for predict set 'test' 
    DEBUG [06:17:54.559] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 5, 14, 18, 19, 21, 28, 38, 48, 49, 51, 52, 59, 64, 65]}
    DEBUG [06:17:54.563] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:54.577] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:54.580] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:54.581] Applying learner 'regr.ranger' on task 'df.tr' (iter 16/50) 
    DEBUG [06:17:54.593] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75..]}
    DEBUG [06:17:54.597] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:54.615] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:54.616] Creating Prediction for predict set 'test' 
    DEBUG [06:17:54.620] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72]}
    DEBUG [06:17:54.623] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:54.636] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:54.638] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:54.639] Applying learner 'regr.ranger' on task 'df.tr' (iter 26/50) 
    DEBUG [06:17:54.650] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75..]}
    DEBUG [06:17:54.654] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:54.672] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:54.674] Creating Prediction for predict set 'test' 
    DEBUG [06:17:54.678] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74]}
    DEBUG [06:17:54.681] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:54.696] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:54.698] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:54.699] Applying learner 'regr.ranger' on task 'df.tr' (iter 48/50) 
    DEBUG [06:17:54.710] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:17:54.714] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:54.732] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:54.734] Creating Prediction for predict set 'test' 
    DEBUG [06:17:54.738] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72, 74]}
    DEBUG [06:17:54.742] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:54.756] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:54.758] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:54.759] Applying learner 'regr.ranger' on task 'df.tr' (iter 2/50) 
    DEBUG [06:17:54.770] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:17:54.773] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:54.790] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:54.792] Creating Prediction for predict set 'test' 
    DEBUG [06:17:54.795] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68]}
    DEBUG [06:17:54.799] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:54.812] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:54.814] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:54.815] Applying learner 'regr.ranger' on task 'df.tr' (iter 34/50) 
    DEBUG [06:17:54.827] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:17:54.832] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:54.875] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:54.878] Creating Prediction for predict set 'test' 
    DEBUG [06:17:54.881] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 7, 12, 14, 18, 20, 41, 46, 47, 53, 57, 63, 64, 72, 75]}
    DEBUG [06:17:54.884] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:54.900] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:54.903] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:54.904] Applying learner 'regr.ranger' on task 'df.tr' (iter 14/50) 
    DEBUG [06:17:54.914] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:17:54.918] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:54.957] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:54.960] Creating Prediction for predict set 'test' 
    DEBUG [06:17:54.963] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 22, 27, 36, 38, 40, 45, 46, 50, 55, 56, 60, 65, 73]}
    DEBUG [06:17:54.966] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:54.980] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:54.982] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:54.983] Applying learner 'regr.ranger' on task 'df.tr' (iter 30/50) 
    DEBUG [06:17:54.992] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:17:54.996] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.012] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:55.014] Creating Prediction for predict set 'test' 
    DEBUG [06:17:55.017] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 14, 22, 29, 34, 35, 37, 40, 43, 44, 59, 62, 64, 68, 72]}
    DEBUG [06:17:55.020] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.032] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:55.034] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:55.035] Applying learner 'regr.ranger' on task 'df.tr' (iter 28/50) 
    DEBUG [06:17:55.044] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:17:55.048] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.064] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:55.065] Creating Prediction for predict set 'test' 
    DEBUG [06:17:55.068] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75]}
    DEBUG [06:17:55.071] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.083] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:55.084] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:55.086] Applying learner 'regr.ranger' on task 'df.tr' (iter 21/50) 
    DEBUG [06:17:55.095] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66..]}
    DEBUG [06:17:55.098] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.115] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:55.117] Creating Prediction for predict set 'test' 
    DEBUG [06:17:55.120] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72]}
    DEBUG [06:17:55.123] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.136] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:55.138] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:55.139] Applying learner 'regr.ranger' on task 'df.tr' (iter 23/50) 
    DEBUG [06:17:55.149] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:17:55.153] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.168] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:55.170] Creating Prediction for predict set 'test' 
    DEBUG [06:17:55.173] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66]}
    DEBUG [06:17:55.176] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.188] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:55.190] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:55.191] Applying learner 'regr.ranger' on task 'df.tr' (iter 42/50) 
    DEBUG [06:17:55.200] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70..]}
    DEBUG [06:17:55.204] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.220] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:55.222] Creating Prediction for predict set 'test' 
    DEBUG [06:17:55.225] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66]}
    DEBUG [06:17:55.228] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.242] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:55.244] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:55.245] Applying learner 'regr.ranger' on task 'df.tr' (iter 25/50) 
    DEBUG [06:17:55.256] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:17:55.260] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.279] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:55.280] Creating Prediction for predict set 'test' 
    DEBUG [06:17:55.284] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [9, 11, 18, 21, 23, 27, 29, 43, 44, 45, 51, 55, 67, 73, 76]}
    DEBUG [06:17:55.288] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.302] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:55.304] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:55.306] Applying learner 'regr.ranger' on task 'df.tr' (iter 6/50) 
    DEBUG [06:17:55.316] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 7..]}
    DEBUG [06:17:55.320] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.338] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:55.341] Creating Prediction for predict set 'test' 
    DEBUG [06:17:55.345] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76]}
    DEBUG [06:17:55.349] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.362] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:55.363] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:55.364] Applying learner 'regr.ranger' on task 'df.tr' (iter 4/50) 
    DEBUG [06:17:55.373] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:17:55.377] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.394] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:55.401] Creating Prediction for predict set 'test' 
    DEBUG [06:17:55.405] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 3, 13, 20, 23, 25, 40, 46, 50, 53, 56, 57, 58, 63]}
    DEBUG [06:17:55.409] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.424] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:55.426] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:55.427] Applying learner 'regr.ranger' on task 'df.tr' (iter 3/50) 
    DEBUG [06:17:55.438] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:17:55.442] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.460] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:55.461] Creating Prediction for predict set 'test' 
    DEBUG [06:17:55.465] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75, 76]}
    DEBUG [06:17:55.468] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.481] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:55.483] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:55.484] Applying learner 'regr.ranger' on task 'df.tr' (iter 10/50) 
    DEBUG [06:17:55.494] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:17:55.498] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.514] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:55.516] Creating Prediction for predict set 'test' 
    DEBUG [06:17:55.520] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 17, 18, 19, 24, 26, 29, 31, 34, 36, 43, 45, 53, 66, 73]}
    DEBUG [06:17:55.524] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.537] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:55.539] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:55.540] Applying learner 'regr.ranger' on task 'df.tr' (iter 41/50) 
    DEBUG [06:17:55.551] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74..]}
    DEBUG [06:17:55.555] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.573] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:55.576] Creating Prediction for predict set 'test' 
    DEBUG [06:17:55.579] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76]}
    DEBUG [06:17:55.583] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.596] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:55.598] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:55.599] Applying learner 'regr.ranger' on task 'df.tr' (iter 12/50) 
    DEBUG [06:17:55.610] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61..]}
    DEBUG [06:17:55.613] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.629] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:55.630] Creating Prediction for predict set 'test' 
    DEBUG [06:17:55.634] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70]}
    DEBUG [06:17:55.637] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.650] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:55.652] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:55.653] Applying learner 'regr.ranger' on task 'df.tr' (iter 18/50) 
    DEBUG [06:17:55.663] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:17:55.667] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.685] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:55.687] Creating Prediction for predict set 'test' 
    DEBUG [06:17:55.690] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75]}
    DEBUG [06:17:55.694] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.707] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:55.709] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:55.710] Applying learner 'regr.ranger' on task 'df.tr' (iter 44/50) 
    DEBUG [06:17:55.720] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:17:55.724] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.743] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:55.745] Creating Prediction for predict set 'test' 
    DEBUG [06:17:55.749] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 14, 17, 19, 24, 36, 41, 47, 51, 53, 56, 62, 63, 67, 71]}
    DEBUG [06:17:55.752] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.765] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:55.767] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:55.768] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [06:17:55.778] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:17:55.782] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.798] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:55.800] Creating Prediction for predict set 'test' 
    DEBUG [06:17:55.803] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [06:17:55.807] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.821] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:55.822] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:55.824] Applying learner 'regr.ranger' on task 'df.tr' (iter 31/50) 
    DEBUG [06:17:55.834] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60..]}
    DEBUG [06:17:55.838] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.855] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:55.857] Creating Prediction for predict set 'test' 
    DEBUG [06:17:55.861] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73]}
    DEBUG [06:17:55.865] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.882] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:55.884] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:55.886] Applying learner 'regr.ranger' on task 'df.tr' (iter 17/50) 
    DEBUG [06:17:55.897] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 7..]}
    DEBUG [06:17:55.900] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.922] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:55.924] Creating Prediction for predict set 'test' 
    DEBUG [06:17:55.928] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70]}
    DEBUG [06:17:55.931] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.945] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:55.946] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:55.948] Applying learner 'regr.ranger' on task 'df.tr' (iter 50/50) 
    DEBUG [06:17:55.958] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:17:55.962] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:55.979] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:55.981] Creating Prediction for predict set 'test' 
    DEBUG [06:17:55.986] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 8, 21, 22, 30, 36, 41, 45, 49, 53, 54, 57, 68, 71]}
    DEBUG [06:17:55.989] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.002] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:56.004] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:56.005] Applying learner 'regr.ranger' on task 'df.tr' (iter 35/50) 
    DEBUG [06:17:56.016] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:17:56.019] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.039] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:56.041] Creating Prediction for predict set 'test' 
    DEBUG [06:17:56.044] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 29, 32, 33, 35, 40, 43, 44, 45, 61, 62, 70, 74]}
    DEBUG [06:17:56.048] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.062] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:56.063] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:56.065] Applying learner 'regr.ranger' on task 'df.tr' (iter 19/50) 
    DEBUG [06:17:56.116] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:17:56.120] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.139] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:56.141] Creating Prediction for predict set 'test' 
    DEBUG [06:17:56.144] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 8, 11, 14, 18, 24, 28, 33, 34, 54, 56, 62, 64, 68, 76]}
    DEBUG [06:17:56.148] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.161] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:56.163] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:56.164] Applying learner 'regr.ranger' on task 'df.tr' (iter 13/50) 
    DEBUG [06:17:56.174] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:17:56.178] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.194] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:56.196] Creating Prediction for predict set 'test' 
    DEBUG [06:17:56.199] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 72]}
    DEBUG [06:17:56.203] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.216] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:56.218] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:56.219] Applying learner 'regr.ranger' on task 'df.tr' (iter 38/50) 
    DEBUG [06:17:56.229] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:17:56.233] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.250] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:56.252] Creating Prediction for predict set 'test' 
    DEBUG [06:17:56.255] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58, 69]}
    DEBUG [06:17:56.259] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.272] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:56.273] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:56.275] Applying learner 'regr.ranger' on task 'df.tr' (iter 15/50) 
    DEBUG [06:17:56.285] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:17:56.288] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.306] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:56.309] Creating Prediction for predict set 'test' 
    DEBUG [06:17:56.312] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 19, 20, 30, 35, 42, 47, 54, 59, 64, 67, 71, 74, 75]}
    DEBUG [06:17:56.315] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.335] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:56.337] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:56.338] Applying learner 'regr.ranger' on task 'df.tr' (iter 22/50) 
    DEBUG [06:17:56.349] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57..]}
    DEBUG [06:17:56.353] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.370] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:56.372] Creating Prediction for predict set 'test' 
    DEBUG [06:17:56.375] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75]}
    DEBUG [06:17:56.379] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.391] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:56.393] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:56.394] Applying learner 'regr.ranger' on task 'df.tr' (iter 43/50) 
    DEBUG [06:17:56.405] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:17:56.408] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.427] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:56.429] Creating Prediction for predict set 'test' 
    DEBUG [06:17:56.432] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74]}
    DEBUG [06:17:56.436] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.448] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:56.450] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:56.451] Applying learner 'regr.ranger' on task 'df.tr' (iter 27/50) 
    DEBUG [06:17:56.462] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67..]}
    DEBUG [06:17:56.466] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.484] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:56.486] Creating Prediction for predict set 'test' 
    DEBUG [06:17:56.490] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76]}
    DEBUG [06:17:56.494] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.507] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:56.509] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:56.511] Applying learner 'regr.ranger' on task 'df.tr' (iter 45/50) 
    DEBUG [06:17:56.521] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:17:56.525] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.543] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:56.545] Creating Prediction for predict set 'test' 
    DEBUG [06:17:56.548] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 8, 12, 21, 22, 25, 28, 33, 37, 38, 44, 46, 48, 57, 75]}
    DEBUG [06:17:56.552] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.565] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:56.567] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:56.568] Applying learner 'regr.ranger' on task 'df.tr' (iter 7/50) 
    DEBUG [06:17:56.578] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 6..]}
    DEBUG [06:17:56.582] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.600] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:56.602] Creating Prediction for predict set 'test' 
    DEBUG [06:17:56.606] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74]}
    DEBUG [06:17:56.609] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.622] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:56.623] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:56.625] Applying learner 'regr.ranger' on task 'df.tr' (iter 33/50) 
    DEBUG [06:17:56.635] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:17:56.638] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.653] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:56.655] Creating Prediction for predict set 'test' 
    DEBUG [06:17:56.658] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60, 76]}
    DEBUG [06:17:56.661] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.673] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:56.675] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:56.676] Applying learner 'regr.ranger' on task 'df.tr' (iter 24/50) 
    DEBUG [06:17:56.685] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:17:56.688] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.705] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:56.706] Creating Prediction for predict set 'test' 
    DEBUG [06:17:56.709] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 13, 17, 25, 32, 35, 36, 37, 38, 50, 63, 65, 68, 69, 74]}
    DEBUG [06:17:56.712] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.724] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:56.726] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:56.727] Applying learner 'regr.ranger' on task 'df.tr' (iter 49/50) 
    DEBUG [06:17:56.736] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:17:56.739] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.754] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:56.756] Creating Prediction for predict set 'test' 
    DEBUG [06:17:56.758] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 7, 14, 16, 25, 26, 27, 40, 44, 56, 67, 69, 75, 76]}
    DEBUG [06:17:56.762] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.774] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:56.775] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:56.781] Applying learner 'regr.ranger' on task 'df.tr' (iter 8/50) 
    DEBUG [06:17:56.793] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:17:56.796] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.816] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:56.818] Creating Prediction for predict set 'test' 
    DEBUG [06:17:56.822] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 75]}
    DEBUG [06:17:56.826] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.839] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:56.841] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:56.842] Applying learner 'regr.ranger' on task 'df.tr' (iter 9/50) 
    DEBUG [06:17:56.852] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:17:56.856] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.873] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:56.875] Creating Prediction for predict set 'test' 
    DEBUG [06:17:56.879] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 23, 27, 40, 42, 46, 49, 56, 58, 61, 64, 67, 70, 71, 72]}
    DEBUG [06:17:56.883] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.896] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:56.897] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:56.899] Applying learner 'regr.ranger' on task 'df.tr' (iter 29/50) 
    DEBUG [06:17:56.909] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:17:56.912] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.930] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:56.931] Creating Prediction for predict set 'test' 
    DEBUG [06:17:56.935] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 18, 25, 28, 30, 39, 45, 48, 51, 54, 55, 61, 66, 71, 73]}
    DEBUG [06:17:56.938] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.951] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:56.953] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:56.954] Applying learner 'regr.ranger' on task 'df.tr' (iter 37/50) 
    DEBUG [06:17:56.964] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57..]}
    DEBUG [06:17:56.968] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:56.983] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:56.985] Creating Prediction for predict set 'test' 
    DEBUG [06:17:56.988] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76]}
    DEBUG [06:17:56.992] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:57.005] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:57.006] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:57.007] Applying learner 'regr.ranger' on task 'df.tr' (iter 40/50) 
    DEBUG [06:17:57.017] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:17:57.021] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:57.072] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:57.074] Creating Prediction for predict set 'test' 
    DEBUG [06:17:57.077] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 8, 16, 17, 24, 25, 27, 29, 41, 44, 53, 54, 67, 72]}
    DEBUG [06:17:57.080] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:57.094] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:57.096] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:57.097] Applying learner 'regr.ranger' on task 'df.tr' (iter 20/50) 
    DEBUG [06:17:57.107] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:17:57.111] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:57.128] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:57.130] Creating Prediction for predict set 'test' 
    DEBUG [06:17:57.133] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 16, 17, 32, 36, 37, 42, 43, 45, 46, 53, 61, 65, 69]}
    DEBUG [06:17:57.137] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:57.150] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:57.151] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:57.153] Applying learner 'regr.ranger' on task 'df.tr' (iter 36/50) 
    DEBUG [06:17:57.162] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58..]}
    DEBUG [06:17:57.166] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:57.184] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:57.186] Creating Prediction for predict set 'test' 
    DEBUG [06:17:57.189] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75]}
    DEBUG [06:17:57.193] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:57.205] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:57.207] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:57.208] Applying learner 'regr.ranger' on task 'df.tr' (iter 46/50) 
    DEBUG [06:17:57.218] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72..]}
    DEBUG [06:17:57.222] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:57.238] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:57.239] Creating Prediction for predict set 'test' 
    DEBUG [06:17:57.243] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73]}
    DEBUG [06:17:57.246] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:57.259] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:57.260] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:57.262] Applying learner 'regr.ranger' on task 'df.tr' (iter 5/50) 
    DEBUG [06:17:57.272] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:17:57.275] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:57.293] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:57.296] Creating Prediction for predict set 'test' 
    DEBUG [06:17:57.300] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 11, 14, 17, 27, 32, 41, 44, 45, 55, 60, 64, 65, 66, 69]}
    DEBUG [06:17:57.304] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:57.317] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:57.318] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:57.319] Applying learner 'regr.ranger' on task 'df.tr' (iter 11/50) 
    DEBUG [06:17:57.330] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 7..]}
    DEBUG [06:17:57.333] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:57.351] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:57.353] Creating Prediction for predict set 'test' 
    DEBUG [06:17:57.356] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76]}
    DEBUG [06:17:57.363] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:57.378] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:57.380] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:57.381] Applying learner 'regr.ranger' on task 'df.tr' (iter 47/50) 
    DEBUG [06:17:57.392] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64..]}
    DEBUG [06:17:57.395] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:57.412] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:57.414] Creating Prediction for predict set 'test' 
    DEBUG [06:17:57.418] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70]}
    DEBUG [06:17:57.421] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:57.434] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:57.435] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:57.442] Finished benchmark 
    INFO  [06:17:58.962] Result of batch 1: 
    INFO  [06:17:58.966]  mtry sample.fraction num.trees importance regr.rmse      resample_result 
    INFO  [06:17:58.966]     3       0.7830912       443   impurity  45.84152 <ResampleResult[19]> 
    INFO  [06:17:58.973] Evaluating 1 configuration(s) 
    INFO  [06:17:59.025] Benchmark with 50 resampling iterations 
    DEBUG [06:17:59.026] Running benchmark() asynchronously with 50 iterations 
    INFO  [06:17:59.038] Applying learner 'regr.ranger' on task 'df.tr' (iter 2/50) 
    DEBUG [06:17:59.046] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:17:59.049] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.059] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.061] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.063] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68]}
    DEBUG [06:17:59.066] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.075] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.076] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.077] Applying learner 'regr.ranger' on task 'df.tr' (iter 34/50) 
    DEBUG [06:17:59.088] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:17:59.091] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.103] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.105] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.107] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 7, 12, 14, 18, 20, 41, 46, 47, 53, 57, 63, 64, 72, 75]}
    DEBUG [06:17:59.110] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.120] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.122] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.123] Applying learner 'regr.ranger' on task 'df.tr' (iter 14/50) 
    DEBUG [06:17:59.131] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:17:59.134] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.144] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.146] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.149] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 22, 27, 36, 38, 40, 45, 46, 50, 55, 56, 60, 65, 73]}
    DEBUG [06:17:59.152] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.162] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.164] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.165] Applying learner 'regr.ranger' on task 'df.tr' (iter 30/50) 
    DEBUG [06:17:59.173] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:17:59.177] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.187] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.189] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.191] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 14, 22, 29, 34, 35, 37, 40, 43, 44, 59, 62, 64, 68, 72]}
    DEBUG [06:17:59.194] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.204] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.206] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.207] Applying learner 'regr.ranger' on task 'df.tr' (iter 28/50) 
    DEBUG [06:17:59.216] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:17:59.219] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.229] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.230] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.233] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75]}
    DEBUG [06:17:59.236] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.247] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.249] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.250] Applying learner 'regr.ranger' on task 'df.tr' (iter 21/50) 
    DEBUG [06:17:59.258] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66..]}
    DEBUG [06:17:59.261] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.272] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.273] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.276] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72]}
    DEBUG [06:17:59.279] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.289] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.290] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.291] Applying learner 'regr.ranger' on task 'df.tr' (iter 23/50) 
    DEBUG [06:17:59.300] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:17:59.303] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.313] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.315] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.317] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66]}
    DEBUG [06:17:59.320] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.330] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.331] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.332] Applying learner 'regr.ranger' on task 'df.tr' (iter 47/50) 
    DEBUG [06:17:59.342] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64..]}
    DEBUG [06:17:59.345] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.356] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.357] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.360] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70]}
    DEBUG [06:17:59.363] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.372] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.374] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.375] Applying learner 'regr.ranger' on task 'df.tr' (iter 25/50) 
    DEBUG [06:17:59.383] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:17:59.386] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.396] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.398] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.403] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [9, 11, 18, 21, 23, 27, 29, 43, 44, 45, 51, 55, 67, 73, 76]}
    DEBUG [06:17:59.405] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.416] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.417] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.418] Applying learner 'regr.ranger' on task 'df.tr' (iter 39/50) 
    DEBUG [06:17:59.439] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:17:59.442] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.452] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.453] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.455] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 5, 14, 18, 19, 21, 28, 38, 48, 49, 51, 52, 59, 64, 65]}
    DEBUG [06:17:59.458] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.467] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.468] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.469] Applying learner 'regr.ranger' on task 'df.tr' (iter 6/50) 
    DEBUG [06:17:59.476] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 7..]}
    DEBUG [06:17:59.479] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.489] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.490] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.492] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76]}
    DEBUG [06:17:59.495] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.504] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.505] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.506] Applying learner 'regr.ranger' on task 'df.tr' (iter 4/50) 
    DEBUG [06:17:59.513] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:17:59.516] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.525] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.526] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.528] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 3, 13, 20, 23, 25, 40, 46, 50, 53, 56, 57, 58, 63]}
    DEBUG [06:17:59.531] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.539] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.541] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.541] Applying learner 'regr.ranger' on task 'df.tr' (iter 3/50) 
    DEBUG [06:17:59.550] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:17:59.554] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.565] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.566] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.569] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75, 76]}
    DEBUG [06:17:59.572] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.582] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.584] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.585] Applying learner 'regr.ranger' on task 'df.tr' (iter 10/50) 
    DEBUG [06:17:59.594] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:17:59.597] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.608] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.610] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.612] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 17, 18, 19, 24, 26, 29, 31, 34, 36, 43, 45, 53, 66, 73]}
    DEBUG [06:17:59.616] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.626] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.627] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.629] Applying learner 'regr.ranger' on task 'df.tr' (iter 46/50) 
    DEBUG [06:17:59.638] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72..]}
    DEBUG [06:17:59.641] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.653] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.654] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.657] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73]}
    DEBUG [06:17:59.661] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.671] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.673] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.674] Applying learner 'regr.ranger' on task 'df.tr' (iter 12/50) 
    DEBUG [06:17:59.683] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61..]}
    DEBUG [06:17:59.686] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.697] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.699] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.702] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70]}
    DEBUG [06:17:59.705] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.717] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.718] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.719] Applying learner 'regr.ranger' on task 'df.tr' (iter 38/50) 
    DEBUG [06:17:59.728] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:17:59.732] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.743] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.744] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.747] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58, 69]}
    DEBUG [06:17:59.750] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.761] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.763] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.764] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [06:17:59.773] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:17:59.777] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.788] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.790] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.793] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [06:17:59.796] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.807] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.808] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.810] Applying learner 'regr.ranger' on task 'df.tr' (iter 35/50) 
    DEBUG [06:17:59.819] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:17:59.822] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.838] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.840] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.843] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 29, 32, 33, 35, 40, 43, 44, 45, 61, 62, 70, 74]}
    DEBUG [06:17:59.847] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.862] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.863] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.864] Applying learner 'regr.ranger' on task 'df.tr' (iter 17/50) 
    DEBUG [06:17:59.874] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 7..]}
    DEBUG [06:17:59.878] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.889] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.890] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.893] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70]}
    DEBUG [06:17:59.896] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.907] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.909] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.910] Applying learner 'regr.ranger' on task 'df.tr' (iter 33/50) 
    DEBUG [06:17:59.919] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:17:59.923] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.934] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.936] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.939] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60, 76]}
    DEBUG [06:17:59.942] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.953] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:17:59.955] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:17:59.956] Applying learner 'regr.ranger' on task 'df.tr' (iter 41/50) 
    DEBUG [06:17:59.965] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74..]}
    DEBUG [06:17:59.969] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.980] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:17:59.981] Creating Prediction for predict set 'test' 
    DEBUG [06:17:59.984] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76]}
    DEBUG [06:17:59.988] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:17:59.999] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.000] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.001] Applying learner 'regr.ranger' on task 'df.tr' (iter 19/50) 
    DEBUG [06:18:00.011] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:00.015] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.026] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:00.028] Creating Prediction for predict set 'test' 
    DEBUG [06:18:00.031] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 8, 11, 14, 18, 24, 28, 33, 34, 54, 56, 62, 64, 68, 76]}
    DEBUG [06:18:00.034] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.046] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.047] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.048] Applying learner 'regr.ranger' on task 'df.tr' (iter 13/50) 
    DEBUG [06:18:00.057] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:00.061] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.072] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:00.074] Creating Prediction for predict set 'test' 
    DEBUG [06:18:00.077] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 72]}
    DEBUG [06:18:00.080] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.091] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.093] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.094] Applying learner 'regr.ranger' on task 'df.tr' (iter 18/50) 
    DEBUG [06:18:00.103] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:00.106] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.118] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:00.119] Creating Prediction for predict set 'test' 
    DEBUG [06:18:00.122] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75]}
    DEBUG [06:18:00.125] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.136] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.137] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.138] Applying learner 'regr.ranger' on task 'df.tr' (iter 15/50) 
    DEBUG [06:18:00.148] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:00.151] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.163] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:00.164] Creating Prediction for predict set 'test' 
    DEBUG [06:18:00.167] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 19, 20, 30, 35, 42, 47, 54, 59, 64, 67, 71, 74, 75]}
    DEBUG [06:18:00.170] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.181] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.182] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.183] Applying learner 'regr.ranger' on task 'df.tr' (iter 27/50) 
    DEBUG [06:18:00.193] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67..]}
    DEBUG [06:18:00.196] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.207] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:00.209] Creating Prediction for predict set 'test' 
    DEBUG [06:18:00.212] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76]}
    DEBUG [06:18:00.215] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.227] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.229] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.230] Applying learner 'regr.ranger' on task 'df.tr' (iter 48/50) 
    DEBUG [06:18:00.239] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:00.243] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.255] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:00.256] Creating Prediction for predict set 'test' 
    DEBUG [06:18:00.259] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72, 74]}
    DEBUG [06:18:00.263] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.273] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.275] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.276] Applying learner 'regr.ranger' on task 'df.tr' (iter 26/50) 
    DEBUG [06:18:00.285] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75..]}
    DEBUG [06:18:00.288] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.300] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:00.301] Creating Prediction for predict set 'test' 
    DEBUG [06:18:00.304] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74]}
    DEBUG [06:18:00.308] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.319] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.320] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.321] Applying learner 'regr.ranger' on task 'df.tr' (iter 32/50) 
    DEBUG [06:18:00.331] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 6..]}
    DEBUG [06:18:00.334] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.345] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:00.347] Creating Prediction for predict set 'test' 
    DEBUG [06:18:00.350] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71]}
    DEBUG [06:18:00.353] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.363] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.365] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.366] Applying learner 'regr.ranger' on task 'df.tr' (iter 22/50) 
    DEBUG [06:18:00.375] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57..]}
    DEBUG [06:18:00.378] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.390] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:00.392] Creating Prediction for predict set 'test' 
    DEBUG [06:18:00.395] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75]}
    DEBUG [06:18:00.424] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.435] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.437] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.438] Applying learner 'regr.ranger' on task 'df.tr' (iter 7/50) 
    DEBUG [06:18:00.447] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 6..]}
    DEBUG [06:18:00.450] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.462] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:00.464] Creating Prediction for predict set 'test' 
    DEBUG [06:18:00.467] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74]}
    DEBUG [06:18:00.470] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.480] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.482] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.483] Applying learner 'regr.ranger' on task 'df.tr' (iter 29/50) 
    DEBUG [06:18:00.492] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:00.495] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.506] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:00.507] Creating Prediction for predict set 'test' 
    DEBUG [06:18:00.510] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 18, 25, 28, 30, 39, 45, 48, 51, 54, 55, 61, 66, 71, 73]}
    DEBUG [06:18:00.514] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.524] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.526] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.527] Applying learner 'regr.ranger' on task 'df.tr' (iter 20/50) 
    DEBUG [06:18:00.536] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:00.539] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.550] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:00.552] Creating Prediction for predict set 'test' 
    DEBUG [06:18:00.555] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 16, 17, 32, 36, 37, 42, 43, 45, 46, 53, 61, 65, 69]}
    DEBUG [06:18:00.558] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.568] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.570] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.571] Applying learner 'regr.ranger' on task 'df.tr' (iter 8/50) 
    DEBUG [06:18:00.580] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:00.588] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.603] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:00.605] Creating Prediction for predict set 'test' 
    DEBUG [06:18:00.608] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 75]}
    DEBUG [06:18:00.612] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.624] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.626] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.627] Applying learner 'regr.ranger' on task 'df.tr' (iter 9/50) 
    DEBUG [06:18:00.638] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:00.642] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.654] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:00.656] Creating Prediction for predict set 'test' 
    DEBUG [06:18:00.659] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 23, 27, 40, 42, 46, 49, 56, 58, 61, 64, 67, 70, 71, 72]}
    DEBUG [06:18:00.663] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.676] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.678] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.679] Applying learner 'regr.ranger' on task 'df.tr' (iter 49/50) 
    DEBUG [06:18:00.690] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:00.694] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.707] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:00.709] Creating Prediction for predict set 'test' 
    DEBUG [06:18:00.712] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 7, 14, 16, 25, 26, 27, 40, 44, 56, 67, 69, 75, 76]}
    DEBUG [06:18:00.716] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.728] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.730] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.731] Applying learner 'regr.ranger' on task 'df.tr' (iter 42/50) 
    DEBUG [06:18:00.742] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70..]}
    DEBUG [06:18:00.745] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.758] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:00.760] Creating Prediction for predict set 'test' 
    DEBUG [06:18:00.763] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66]}
    DEBUG [06:18:00.767] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.779] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.780] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.782] Applying learner 'regr.ranger' on task 'df.tr' (iter 11/50) 
    DEBUG [06:18:00.792] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 7..]}
    DEBUG [06:18:00.796] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.808] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:00.810] Creating Prediction for predict set 'test' 
    DEBUG [06:18:00.813] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76]}
    DEBUG [06:18:00.817] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.829] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.831] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.832] Applying learner 'regr.ranger' on task 'df.tr' (iter 44/50) 
    DEBUG [06:18:00.842] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:00.846] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.858] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:00.860] Creating Prediction for predict set 'test' 
    DEBUG [06:18:00.863] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 14, 17, 19, 24, 36, 41, 47, 51, 53, 56, 62, 63, 67, 71]}
    DEBUG [06:18:00.867] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.878] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.880] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.881] Applying learner 'regr.ranger' on task 'df.tr' (iter 40/50) 
    DEBUG [06:18:00.891] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:00.895] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.908] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:00.910] Creating Prediction for predict set 'test' 
    DEBUG [06:18:00.913] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 8, 16, 17, 24, 25, 27, 29, 41, 44, 53, 54, 67, 72]}
    DEBUG [06:18:00.917] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.928] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.930] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.931] Applying learner 'regr.ranger' on task 'df.tr' (iter 5/50) 
    DEBUG [06:18:00.942] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:00.945] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.957] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:00.959] Creating Prediction for predict set 'test' 
    DEBUG [06:18:00.962] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 11, 14, 17, 27, 32, 41, 44, 45, 55, 60, 64, 65, 66, 69]}
    DEBUG [06:18:00.966] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:00.978] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:00.980] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:00.981] Applying learner 'regr.ranger' on task 'df.tr' (iter 50/50) 
    DEBUG [06:18:00.991] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:00.995] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:01.007] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:01.009] Creating Prediction for predict set 'test' 
    DEBUG [06:18:01.012] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 8, 21, 22, 30, 36, 41, 45, 49, 53, 54, 57, 68, 71]}
    DEBUG [06:18:01.016] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:01.027] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:01.028] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:01.030] Applying learner 'regr.ranger' on task 'df.tr' (iter 37/50) 
    DEBUG [06:18:01.040] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57..]}
    DEBUG [06:18:01.043] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:01.057] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:01.059] Creating Prediction for predict set 'test' 
    DEBUG [06:18:01.063] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76]}
    DEBUG [06:18:01.066] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:01.078] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:01.080] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:01.081] Applying learner 'regr.ranger' on task 'df.tr' (iter 43/50) 
    DEBUG [06:18:01.091] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:01.095] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:01.106] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:01.108] Creating Prediction for predict set 'test' 
    DEBUG [06:18:01.111] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74]}
    DEBUG [06:18:01.115] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:01.126] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:01.128] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:01.130] Applying learner 'regr.ranger' on task 'df.tr' (iter 24/50) 
    DEBUG [06:18:01.141] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:01.145] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:01.157] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:01.159] Creating Prediction for predict set 'test' 
    DEBUG [06:18:01.162] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 13, 17, 25, 32, 35, 36, 37, 38, 50, 63, 65, 68, 69, 74]}
    DEBUG [06:18:01.166] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:01.177] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:01.179] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:01.180] Applying learner 'regr.ranger' on task 'df.tr' (iter 45/50) 
    DEBUG [06:18:01.190] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:01.194] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:01.207] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:01.208] Creating Prediction for predict set 'test' 
    DEBUG [06:18:01.212] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 8, 12, 21, 22, 25, 28, 33, 37, 38, 44, 46, 48, 57, 75]}
    DEBUG [06:18:01.215] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:01.227] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:01.229] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:01.230] Applying learner 'regr.ranger' on task 'df.tr' (iter 31/50) 
    DEBUG [06:18:01.239] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60..]}
    DEBUG [06:18:01.243] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:01.254] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:01.255] Creating Prediction for predict set 'test' 
    DEBUG [06:18:01.258] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73]}
    DEBUG [06:18:01.262] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:01.272] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:01.274] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:01.275] Applying learner 'regr.ranger' on task 'df.tr' (iter 36/50) 
    DEBUG [06:18:01.284] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58..]}
    DEBUG [06:18:01.923] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:01.935] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:01.937] Creating Prediction for predict set 'test' 
    DEBUG [06:18:01.940] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75]}
    DEBUG [06:18:01.943] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:01.955] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:01.957] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:01.958] Applying learner 'regr.ranger' on task 'df.tr' (iter 16/50) 
    DEBUG [06:18:01.967] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75..]}
    DEBUG [06:18:01.970] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:01.981] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:01.982] Creating Prediction for predict set 'test' 
    DEBUG [06:18:01.985] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72]}
    DEBUG [06:18:01.988] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:01.999] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:02.000] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:02.009] Finished benchmark 
    INFO  [06:18:03.696] Result of batch 2: 
    INFO  [06:18:03.700]  mtry sample.fraction num.trees importance regr.rmse      resample_result 
    INFO  [06:18:03.700]     2       0.6960982       141   impurity  49.20049 <ResampleResult[19]> 
    INFO  [06:18:03.709] Evaluating 1 configuration(s) 
    INFO  [06:18:03.752] Benchmark with 50 resampling iterations 
    DEBUG [06:18:03.753] Running benchmark() asynchronously with 50 iterations 
    INFO  [06:18:03.767] Applying learner 'regr.ranger' on task 'df.tr' (iter 23/50) 
    DEBUG [06:18:03.777] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:03.781] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:03.800] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:03.803] Creating Prediction for predict set 'test' 
    DEBUG [06:18:03.806] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66]}
    DEBUG [06:18:03.810] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:03.823] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:03.824] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:03.826] Applying learner 'regr.ranger' on task 'df.tr' (iter 30/50) 
    DEBUG [06:18:03.836] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:03.840] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:03.856] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:03.858] Creating Prediction for predict set 'test' 
    DEBUG [06:18:03.862] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 14, 22, 29, 34, 35, 37, 40, 43, 44, 59, 62, 64, 68, 72]}
    DEBUG [06:18:03.865] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:03.878] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:03.879] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:03.881] Applying learner 'regr.ranger' on task 'df.tr' (iter 25/50) 
    DEBUG [06:18:03.891] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:03.894] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:03.910] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:03.912] Creating Prediction for predict set 'test' 
    DEBUG [06:18:03.915] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [9, 11, 18, 21, 23, 27, 29, 43, 44, 45, 51, 55, 67, 73, 76]}
    DEBUG [06:18:03.919] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:03.931] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:03.933] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:03.935] Applying learner 'regr.ranger' on task 'df.tr' (iter 39/50) 
    DEBUG [06:18:03.944] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:03.947] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:03.962] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:03.963] Creating Prediction for predict set 'test' 
    DEBUG [06:18:03.966] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 5, 14, 18, 19, 21, 28, 38, 48, 49, 51, 52, 59, 64, 65]}
    DEBUG [06:18:03.969] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:03.981] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:03.983] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:03.984] Applying learner 'regr.ranger' on task 'df.tr' (iter 6/50) 
    DEBUG [06:18:03.994] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 7..]}
    DEBUG [06:18:03.997] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.012] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:04.014] Creating Prediction for predict set 'test' 
    DEBUG [06:18:04.017] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76]}
    DEBUG [06:18:04.021] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.033] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:04.035] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:04.036] Applying learner 'regr.ranger' on task 'df.tr' (iter 4/50) 
    DEBUG [06:18:04.046] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:04.050] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.064] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:04.066] Creating Prediction for predict set 'test' 
    DEBUG [06:18:04.069] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 3, 13, 20, 23, 25, 40, 46, 50, 53, 56, 57, 58, 63]}
    DEBUG [06:18:04.072] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.085] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:04.087] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:04.088] Applying learner 'regr.ranger' on task 'df.tr' (iter 3/50) 
    DEBUG [06:18:04.098] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:04.102] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.116] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:04.118] Creating Prediction for predict set 'test' 
    DEBUG [06:18:04.121] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75, 76]}
    DEBUG [06:18:04.124] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.136] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:04.138] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:04.139] Applying learner 'regr.ranger' on task 'df.tr' (iter 10/50) 
    DEBUG [06:18:04.149] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:04.153] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.170] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:04.172] Creating Prediction for predict set 'test' 
    DEBUG [06:18:04.175] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 17, 18, 19, 24, 26, 29, 31, 34, 36, 43, 45, 53, 66, 73]}
    DEBUG [06:18:04.178] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.190] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:04.192] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:04.193] Applying learner 'regr.ranger' on task 'df.tr' (iter 28/50) 
    DEBUG [06:18:04.203] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:04.206] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.221] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:04.223] Creating Prediction for predict set 'test' 
    DEBUG [06:18:04.226] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75]}
    DEBUG [06:18:04.229] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.241] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:04.243] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:04.244] Applying learner 'regr.ranger' on task 'df.tr' (iter 12/50) 
    DEBUG [06:18:04.254] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61..]}
    DEBUG [06:18:04.257] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.271] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:04.273] Creating Prediction for predict set 'test' 
    DEBUG [06:18:04.276] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70]}
    DEBUG [06:18:04.279] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.291] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:04.293] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:04.294] Applying learner 'regr.ranger' on task 'df.tr' (iter 44/50) 
    DEBUG [06:18:04.304] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:04.308] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.325] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:04.327] Creating Prediction for predict set 'test' 
    DEBUG [06:18:04.331] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 14, 17, 19, 24, 36, 41, 47, 51, 53, 56, 62, 63, 67, 71]}
    DEBUG [06:18:04.335] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.347] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:04.349] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:04.350] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [06:18:04.359] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:18:04.363] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.377] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:04.379] Creating Prediction for predict set 'test' 
    DEBUG [06:18:04.382] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [06:18:04.385] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.397] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:04.399] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:04.400] Applying learner 'regr.ranger' on task 'df.tr' (iter 33/50) 
    DEBUG [06:18:04.414] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:04.418] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.438] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:04.440] Creating Prediction for predict set 'test' 
    DEBUG [06:18:04.444] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60, 76]}
    DEBUG [06:18:04.447] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.460] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:04.462] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:04.463] Applying learner 'regr.ranger' on task 'df.tr' (iter 36/50) 
    DEBUG [06:18:04.473] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58..]}
    DEBUG [06:18:04.476] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.491] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:04.492] Creating Prediction for predict set 'test' 
    DEBUG [06:18:04.495] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75]}
    DEBUG [06:18:04.499] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.512] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:04.514] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:04.515] Applying learner 'regr.ranger' on task 'df.tr' (iter 13/50) 
    DEBUG [06:18:04.524] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:04.528] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.569] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:04.570] Creating Prediction for predict set 'test' 
    DEBUG [06:18:04.574] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 72]}
    DEBUG [06:18:04.577] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.591] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:04.593] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:04.594] Applying learner 'regr.ranger' on task 'df.tr' (iter 15/50) 
    DEBUG [06:18:04.603] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:04.607] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.624] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:04.626] Creating Prediction for predict set 'test' 
    DEBUG [06:18:04.630] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 19, 20, 30, 35, 42, 47, 54, 59, 64, 67, 71, 74, 75]}
    DEBUG [06:18:04.634] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.646] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:04.648] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:04.649] Applying learner 'regr.ranger' on task 'df.tr' (iter 37/50) 
    DEBUG [06:18:04.659] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57..]}
    DEBUG [06:18:04.663] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.678] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:04.680] Creating Prediction for predict set 'test' 
    DEBUG [06:18:04.683] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76]}
    DEBUG [06:18:04.686] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.699] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:04.701] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:04.702] Applying learner 'regr.ranger' on task 'df.tr' (iter 49/50) 
    DEBUG [06:18:04.711] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:04.714] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.731] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:04.733] Creating Prediction for predict set 'test' 
    DEBUG [06:18:04.736] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 7, 14, 16, 25, 26, 27, 40, 44, 56, 67, 69, 75, 76]}
    DEBUG [06:18:04.739] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.751] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:04.752] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:04.753] Applying learner 'regr.ranger' on task 'df.tr' (iter 50/50) 
    DEBUG [06:18:04.762] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:04.766] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.780] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:04.781] Creating Prediction for predict set 'test' 
    DEBUG [06:18:04.784] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 8, 21, 22, 30, 36, 41, 45, 49, 53, 54, 57, 68, 71]}
    DEBUG [06:18:04.788] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.799] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:04.801] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:04.802] Applying learner 'regr.ranger' on task 'df.tr' (iter 18/50) 
    DEBUG [06:18:04.811] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:04.814] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.829] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:04.830] Creating Prediction for predict set 'test' 
    DEBUG [06:18:04.833] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75]}
    DEBUG [06:18:04.836] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.848] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:04.850] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:04.851] Applying learner 'regr.ranger' on task 'df.tr' (iter 41/50) 
    DEBUG [06:18:04.860] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74..]}
    DEBUG [06:18:04.863] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.880] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:04.882] Creating Prediction for predict set 'test' 
    DEBUG [06:18:04.886] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76]}
    DEBUG [06:18:04.889] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.901] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:04.902] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:04.903] Applying learner 'regr.ranger' on task 'df.tr' (iter 31/50) 
    DEBUG [06:18:04.912] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60..]}
    DEBUG [06:18:04.916] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.932] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:04.934] Creating Prediction for predict set 'test' 
    DEBUG [06:18:04.938] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73]}
    DEBUG [06:18:04.941] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.953] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:04.955] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:04.956] Applying learner 'regr.ranger' on task 'df.tr' (iter 7/50) 
    DEBUG [06:18:04.965] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 6..]}
    DEBUG [06:18:04.969] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:04.983] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:04.985] Creating Prediction for predict set 'test' 
    DEBUG [06:18:04.988] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74]}
    DEBUG [06:18:04.991] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.003] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:05.004] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:05.005] Applying learner 'regr.ranger' on task 'df.tr' (iter 26/50) 
    DEBUG [06:18:05.015] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75..]}
    DEBUG [06:18:05.018] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.033] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:05.034] Creating Prediction for predict set 'test' 
    DEBUG [06:18:05.037] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74]}
    DEBUG [06:18:05.041] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.052] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:05.054] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:05.055] Applying learner 'regr.ranger' on task 'df.tr' (iter 45/50) 
    DEBUG [06:18:05.064] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:05.067] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.083] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:05.085] Creating Prediction for predict set 'test' 
    DEBUG [06:18:05.089] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 8, 12, 21, 22, 25, 28, 33, 37, 38, 44, 46, 48, 57, 75]}
    DEBUG [06:18:05.092] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.104] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:05.105] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:05.106] Applying learner 'regr.ranger' on task 'df.tr' (iter 27/50) 
    DEBUG [06:18:05.115] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67..]}
    DEBUG [06:18:05.119] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.133] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:05.134] Creating Prediction for predict set 'test' 
    DEBUG [06:18:05.137] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76]}
    DEBUG [06:18:05.140] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.155] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:05.157] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:05.158] Applying learner 'regr.ranger' on task 'df.tr' (iter 24/50) 
    DEBUG [06:18:05.167] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:05.170] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.187] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:05.189] Creating Prediction for predict set 'test' 
    DEBUG [06:18:05.192] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 13, 17, 25, 32, 35, 36, 37, 38, 50, 63, 65, 68, 69, 74]}
    DEBUG [06:18:05.195] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.208] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:05.209] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:05.210] Applying learner 'regr.ranger' on task 'df.tr' (iter 40/50) 
    DEBUG [06:18:05.220] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:05.223] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.237] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:05.238] Creating Prediction for predict set 'test' 
    DEBUG [06:18:05.241] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 8, 16, 17, 24, 25, 27, 29, 41, 44, 53, 54, 67, 72]}
    DEBUG [06:18:05.245] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.258] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:05.259] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:05.260] Applying learner 'regr.ranger' on task 'df.tr' (iter 11/50) 
    DEBUG [06:18:05.270] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 7..]}
    DEBUG [06:18:05.273] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.288] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:05.289] Creating Prediction for predict set 'test' 
    DEBUG [06:18:05.292] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76]}
    DEBUG [06:18:05.296] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.308] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:05.309] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:05.310] Applying learner 'regr.ranger' on task 'df.tr' (iter 19/50) 
    DEBUG [06:18:05.320] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:05.323] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.339] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:05.341] Creating Prediction for predict set 'test' 
    DEBUG [06:18:05.344] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 8, 11, 14, 18, 24, 28, 33, 34, 54, 56, 62, 64, 68, 76]}
    DEBUG [06:18:05.347] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.361] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:05.362] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:05.363] Applying learner 'regr.ranger' on task 'df.tr' (iter 16/50) 
    DEBUG [06:18:05.373] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75..]}
    DEBUG [06:18:05.377] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.391] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:05.393] Creating Prediction for predict set 'test' 
    DEBUG [06:18:05.396] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72]}
    DEBUG [06:18:05.399] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.413] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:05.414] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:05.416] Applying learner 'regr.ranger' on task 'df.tr' (iter 35/50) 
    DEBUG [06:18:05.425] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:05.429] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.443] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:05.445] Creating Prediction for predict set 'test' 
    DEBUG [06:18:05.448] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 29, 32, 33, 35, 40, 43, 44, 45, 61, 62, 70, 74]}
    DEBUG [06:18:05.452] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.463] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:05.465] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:05.466] Applying learner 'regr.ranger' on task 'df.tr' (iter 22/50) 
    DEBUG [06:18:05.475] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57..]}
    DEBUG [06:18:05.479] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.502] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:05.504] Creating Prediction for predict set 'test' 
    DEBUG [06:18:05.508] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75]}
    DEBUG [06:18:05.511] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.524] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:05.526] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:05.527] Applying learner 'regr.ranger' on task 'df.tr' (iter 46/50) 
    DEBUG [06:18:05.536] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72..]}
    DEBUG [06:18:05.540] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.555] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:05.557] Creating Prediction for predict set 'test' 
    DEBUG [06:18:05.561] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73]}
    DEBUG [06:18:05.565] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.577] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:05.579] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:05.580] Applying learner 'regr.ranger' on task 'df.tr' (iter 20/50) 
    DEBUG [06:18:05.590] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:05.593] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.608] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:05.609] Creating Prediction for predict set 'test' 
    DEBUG [06:18:05.612] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 16, 17, 32, 36, 37, 42, 43, 45, 46, 53, 61, 65, 69]}
    DEBUG [06:18:05.615] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.627] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:05.628] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:05.629] Applying learner 'regr.ranger' on task 'df.tr' (iter 21/50) 
    DEBUG [06:18:05.639] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66..]}
    DEBUG [06:18:05.642] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.656] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:05.658] Creating Prediction for predict set 'test' 
    DEBUG [06:18:05.660] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72]}
    DEBUG [06:18:05.664] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.675] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:05.677] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:05.678] Applying learner 'regr.ranger' on task 'df.tr' (iter 5/50) 
    DEBUG [06:18:05.687] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:05.690] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.705] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:05.706] Creating Prediction for predict set 'test' 
    DEBUG [06:18:05.709] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 11, 14, 17, 27, 32, 41, 44, 45, 55, 60, 64, 65, 66, 69]}
    DEBUG [06:18:05.712] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.724] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:05.725] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:05.727] Applying learner 'regr.ranger' on task 'df.tr' (iter 2/50) 
    DEBUG [06:18:05.736] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:18:05.739] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.780] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:05.782] Creating Prediction for predict set 'test' 
    DEBUG [06:18:05.785] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68]}
    DEBUG [06:18:05.788] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.801] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:05.802] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:05.803] Applying learner 'regr.ranger' on task 'df.tr' (iter 17/50) 
    DEBUG [06:18:05.812] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 7..]}
    DEBUG [06:18:05.815] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.832] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:05.834] Creating Prediction for predict set 'test' 
    DEBUG [06:18:05.837] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70]}
    DEBUG [06:18:05.840] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.854] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:05.855] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:05.856] Applying learner 'regr.ranger' on task 'df.tr' (iter 47/50) 
    DEBUG [06:18:05.866] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64..]}
    DEBUG [06:18:05.869] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.883] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:05.884] Creating Prediction for predict set 'test' 
    DEBUG [06:18:05.887] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70]}
    DEBUG [06:18:05.890] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.902] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:05.903] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:05.904] Applying learner 'regr.ranger' on task 'df.tr' (iter 14/50) 
    DEBUG [06:18:05.914] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:05.917] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.933] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:05.935] Creating Prediction for predict set 'test' 
    DEBUG [06:18:05.938] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 22, 27, 36, 38, 40, 45, 46, 50, 55, 56, 60, 65, 73]}
    DEBUG [06:18:05.941] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.953] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:05.955] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:05.956] Applying learner 'regr.ranger' on task 'df.tr' (iter 32/50) 
    DEBUG [06:18:05.965] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 6..]}
    DEBUG [06:18:05.968] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:05.984] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:05.986] Creating Prediction for predict set 'test' 
    DEBUG [06:18:05.990] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71]}
    DEBUG [06:18:05.994] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:06.005] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:06.007] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:06.008] Applying learner 'regr.ranger' on task 'df.tr' (iter 42/50) 
    DEBUG [06:18:06.017] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70..]}
    DEBUG [06:18:06.020] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:06.035] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:06.037] Creating Prediction for predict set 'test' 
    DEBUG [06:18:06.040] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66]}
    DEBUG [06:18:06.044] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:06.055] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:06.057] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:06.058] Applying learner 'regr.ranger' on task 'df.tr' (iter 9/50) 
    DEBUG [06:18:06.067] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:06.070] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:06.086] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:06.088] Creating Prediction for predict set 'test' 
    DEBUG [06:18:06.091] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 23, 27, 40, 42, 46, 49, 56, 58, 61, 64, 67, 70, 71, 72]}
    DEBUG [06:18:06.094] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:06.106] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:06.107] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:06.108] Applying learner 'regr.ranger' on task 'df.tr' (iter 38/50) 
    DEBUG [06:18:06.117] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:06.120] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:06.137] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:06.139] Creating Prediction for predict set 'test' 
    DEBUG [06:18:06.142] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58, 69]}
    DEBUG [06:18:06.145] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:06.157] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:06.159] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:06.160] Applying learner 'regr.ranger' on task 'df.tr' (iter 8/50) 
    DEBUG [06:18:06.169] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:06.172] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:06.188] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:06.190] Creating Prediction for predict set 'test' 
    DEBUG [06:18:06.193] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 75]}
    DEBUG [06:18:06.197] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:06.209] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:06.210] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:06.211] Applying learner 'regr.ranger' on task 'df.tr' (iter 43/50) 
    DEBUG [06:18:06.220] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:06.223] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:06.241] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:06.243] Creating Prediction for predict set 'test' 
    DEBUG [06:18:06.246] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74]}
    DEBUG [06:18:06.250] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:06.262] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:06.264] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:06.265] Applying learner 'regr.ranger' on task 'df.tr' (iter 34/50) 
    DEBUG [06:18:06.274] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:06.277] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:06.293] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:06.295] Creating Prediction for predict set 'test' 
    DEBUG [06:18:06.298] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 7, 12, 14, 18, 20, 41, 46, 47, 53, 57, 63, 64, 72, 75]}
    DEBUG [06:18:06.301] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:06.313] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:06.315] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:06.316] Applying learner 'regr.ranger' on task 'df.tr' (iter 29/50) 
    DEBUG [06:18:06.325] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:06.328] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:06.342] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:06.343] Creating Prediction for predict set 'test' 
    DEBUG [06:18:06.346] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 18, 25, 28, 30, 39, 45, 48, 51, 54, 55, 61, 66, 71, 73]}
    DEBUG [06:18:06.349] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:06.361] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:06.362] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:06.363] Applying learner 'regr.ranger' on task 'df.tr' (iter 48/50) 
    DEBUG [06:18:06.373] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:06.377] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:06.393] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:06.395] Creating Prediction for predict set 'test' 
    DEBUG [06:18:06.398] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72, 74]}
    DEBUG [06:18:06.402] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:06.414] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:06.416] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:06.429] Finished benchmark 
    INFO  [06:18:08.139] Result of batch 3: 
    INFO  [06:18:08.143]  mtry sample.fraction num.trees importance regr.rmse      resample_result 
    INFO  [06:18:08.143]     3       0.6379495       381   impurity  47.24917 <ResampleResult[19]> 
    INFO  [06:18:08.153] Evaluating 1 configuration(s) 
    INFO  [06:18:08.198] Benchmark with 50 resampling iterations 
    DEBUG [06:18:08.199] Running benchmark() asynchronously with 50 iterations 
    INFO  [06:18:08.213] Applying learner 'regr.ranger' on task 'df.tr' (iter 6/50) 
    DEBUG [06:18:08.223] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 7..]}
    DEBUG [06:18:08.226] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.239] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:08.241] Creating Prediction for predict set 'test' 
    DEBUG [06:18:08.263] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76]}
    DEBUG [06:18:08.267] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.279] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:08.280] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:08.282] Applying learner 'regr.ranger' on task 'df.tr' (iter 4/50) 
    DEBUG [06:18:08.291] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:08.295] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.306] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:08.308] Creating Prediction for predict set 'test' 
    DEBUG [06:18:08.311] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 3, 13, 20, 23, 25, 40, 46, 50, 53, 56, 57, 58, 63]}
    DEBUG [06:18:08.314] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.325] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:08.327] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:08.328] Applying learner 'regr.ranger' on task 'df.tr' (iter 3/50) 
    DEBUG [06:18:08.338] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:08.342] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.353] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:08.355] Creating Prediction for predict set 'test' 
    DEBUG [06:18:08.358] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75, 76]}
    DEBUG [06:18:08.362] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.373] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:08.374] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:08.375] Applying learner 'regr.ranger' on task 'df.tr' (iter 46/50) 
    DEBUG [06:18:08.385] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72..]}
    DEBUG [06:18:08.388] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.400] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:08.401] Creating Prediction for predict set 'test' 
    DEBUG [06:18:08.404] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73]}
    DEBUG [06:18:08.407] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.420] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:08.421] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:08.422] Applying learner 'regr.ranger' on task 'df.tr' (iter 47/50) 
    DEBUG [06:18:08.432] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64..]}
    DEBUG [06:18:08.435] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.447] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:08.448] Creating Prediction for predict set 'test' 
    DEBUG [06:18:08.451] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70]}
    DEBUG [06:18:08.454] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.465] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:08.466] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:08.467] Applying learner 'regr.ranger' on task 'df.tr' (iter 10/50) 
    DEBUG [06:18:08.477] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:08.481] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.492] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:08.493] Creating Prediction for predict set 'test' 
    DEBUG [06:18:08.497] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 17, 18, 19, 24, 26, 29, 31, 34, 36, 43, 45, 53, 66, 73]}
    DEBUG [06:18:08.500] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.510] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:08.512] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:08.513] Applying learner 'regr.ranger' on task 'df.tr' (iter 28/50) 
    DEBUG [06:18:08.522] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:08.525] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.537] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:08.538] Creating Prediction for predict set 'test' 
    DEBUG [06:18:08.541] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75]}
    DEBUG [06:18:08.544] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.555] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:08.557] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:08.558] Applying learner 'regr.ranger' on task 'df.tr' (iter 12/50) 
    DEBUG [06:18:08.567] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61..]}
    DEBUG [06:18:08.571] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.582] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:08.583] Creating Prediction for predict set 'test' 
    DEBUG [06:18:08.586] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70]}
    DEBUG [06:18:08.589] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.600] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:08.601] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:08.602] Applying learner 'regr.ranger' on task 'df.tr' (iter 48/50) 
    DEBUG [06:18:08.611] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:08.615] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.626] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:08.628] Creating Prediction for predict set 'test' 
    DEBUG [06:18:08.631] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72, 74]}
    DEBUG [06:18:08.634] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.645] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:08.647] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:08.648] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [06:18:08.657] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:18:08.660] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.671] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:08.673] Creating Prediction for predict set 'test' 
    DEBUG [06:18:08.676] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [06:18:08.679] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.689] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:08.691] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:08.692] Applying learner 'regr.ranger' on task 'df.tr' (iter 33/50) 
    DEBUG [06:18:08.701] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:08.705] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.716] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:08.718] Creating Prediction for predict set 'test' 
    DEBUG [06:18:08.721] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60, 76]}
    DEBUG [06:18:08.724] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.734] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:08.736] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:08.737] Applying learner 'regr.ranger' on task 'df.tr' (iter 36/50) 
    DEBUG [06:18:08.746] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58..]}
    DEBUG [06:18:08.749] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.760] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:08.762] Creating Prediction for predict set 'test' 
    DEBUG [06:18:08.765] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75]}
    DEBUG [06:18:08.768] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.779] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:08.781] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:08.782] Applying learner 'regr.ranger' on task 'df.tr' (iter 13/50) 
    DEBUG [06:18:08.791] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:08.794] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.806] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:08.807] Creating Prediction for predict set 'test' 
    DEBUG [06:18:08.810] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 72]}
    DEBUG [06:18:08.813] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.823] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:08.825] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:08.826] Applying learner 'regr.ranger' on task 'df.tr' (iter 15/50) 
    DEBUG [06:18:08.835] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:08.839] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.850] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:08.852] Creating Prediction for predict set 'test' 
    DEBUG [06:18:08.855] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 19, 20, 30, 35, 42, 47, 54, 59, 64, 67, 71, 74, 75]}
    DEBUG [06:18:08.858] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.869] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:08.870] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:08.871] Applying learner 'regr.ranger' on task 'df.tr' (iter 38/50) 
    DEBUG [06:18:08.881] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:08.884] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.895] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:08.897] Creating Prediction for predict set 'test' 
    DEBUG [06:18:08.900] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58, 69]}
    DEBUG [06:18:08.903] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.913] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:08.915] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:08.916] Applying learner 'regr.ranger' on task 'df.tr' (iter 30/50) 
    DEBUG [06:18:08.925] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:08.929] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.940] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:08.942] Creating Prediction for predict set 'test' 
    DEBUG [06:18:08.945] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 14, 22, 29, 34, 35, 37, 40, 43, 44, 59, 62, 64, 68, 72]}
    DEBUG [06:18:08.948] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.958] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:08.960] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:08.961] Applying learner 'regr.ranger' on task 'df.tr' (iter 18/50) 
    DEBUG [06:18:08.970] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:08.973] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:08.985] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:08.986] Creating Prediction for predict set 'test' 
    DEBUG [06:18:08.989] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75]}
    DEBUG [06:18:08.992] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.003] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.004] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.005] Applying learner 'regr.ranger' on task 'df.tr' (iter 26/50) 
    DEBUG [06:18:09.015] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75..]}
    DEBUG [06:18:09.018] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.030] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.031] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.034] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74]}
    DEBUG [06:18:09.037] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.048] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.049] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.051] Applying learner 'regr.ranger' on task 'df.tr' (iter 49/50) 
    DEBUG [06:18:09.066] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:09.070] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.086] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.088] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.091] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 7, 14, 16, 25, 26, 27, 40, 44, 56, 67, 69, 75, 76]}
    DEBUG [06:18:09.094] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.105] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.107] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.108] Applying learner 'regr.ranger' on task 'df.tr' (iter 32/50) 
    DEBUG [06:18:09.117] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 6..]}
    DEBUG [06:18:09.121] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.132] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.133] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.136] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71]}
    DEBUG [06:18:09.139] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.150] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.152] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.153] Applying learner 'regr.ranger' on task 'df.tr' (iter 24/50) 
    DEBUG [06:18:09.162] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:09.166] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.177] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.178] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.181] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 13, 17, 25, 32, 35, 36, 37, 38, 50, 63, 65, 68, 69, 74]}
    DEBUG [06:18:09.185] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.195] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.197] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.198] Applying learner 'regr.ranger' on task 'df.tr' (iter 25/50) 
    DEBUG [06:18:09.207] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:09.210] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.221] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.223] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.226] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [9, 11, 18, 21, 23, 27, 29, 43, 44, 45, 51, 55, 67, 73, 76]}
    DEBUG [06:18:09.229] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.240] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.241] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.243] Applying learner 'regr.ranger' on task 'df.tr' (iter 42/50) 
    DEBUG [06:18:09.252] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70..]}
    DEBUG [06:18:09.255] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.266] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.268] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.295] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66]}
    DEBUG [06:18:09.299] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.311] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.312] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.313] Applying learner 'regr.ranger' on task 'df.tr' (iter 29/50) 
    DEBUG [06:18:09.323] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:09.326] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.337] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.339] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.341] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 18, 25, 28, 30, 39, 45, 48, 51, 54, 55, 61, 66, 71, 73]}
    DEBUG [06:18:09.345] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.355] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.356] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.357] Applying learner 'regr.ranger' on task 'df.tr' (iter 11/50) 
    DEBUG [06:18:09.366] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 7..]}
    DEBUG [06:18:09.370] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.382] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.383] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.386] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76]}
    DEBUG [06:18:09.389] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.400] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.401] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.403] Applying learner 'regr.ranger' on task 'df.tr' (iter 19/50) 
    DEBUG [06:18:09.412] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:09.415] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.426] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.427] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.430] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 8, 11, 14, 18, 24, 28, 33, 34, 54, 56, 62, 64, 68, 76]}
    DEBUG [06:18:09.433] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.444] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.446] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.447] Applying learner 'regr.ranger' on task 'df.tr' (iter 16/50) 
    DEBUG [06:18:09.456] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75..]}
    DEBUG [06:18:09.459] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.470] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.472] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.474] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72]}
    DEBUG [06:18:09.478] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.488] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.489] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.490] Applying learner 'regr.ranger' on task 'df.tr' (iter 37/50) 
    DEBUG [06:18:09.500] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57..]}
    DEBUG [06:18:09.503] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.514] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.516] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.518] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76]}
    DEBUG [06:18:09.522] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.532] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.533] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.535] Applying learner 'regr.ranger' on task 'df.tr' (iter 40/50) 
    DEBUG [06:18:09.543] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:09.547] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.558] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.560] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.563] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 8, 16, 17, 24, 25, 27, 29, 41, 44, 53, 54, 67, 72]}
    DEBUG [06:18:09.566] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.576] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.577] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.578] Applying learner 'regr.ranger' on task 'df.tr' (iter 50/50) 
    DEBUG [06:18:09.587] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:09.591] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.605] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.606] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.609] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 8, 21, 22, 30, 36, 41, 45, 49, 53, 54, 57, 68, 71]}
    DEBUG [06:18:09.612] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.623] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.624] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.625] Applying learner 'regr.ranger' on task 'df.tr' (iter 5/50) 
    DEBUG [06:18:09.634] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:09.638] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.648] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.650] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.653] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 11, 14, 17, 27, 32, 41, 44, 45, 55, 60, 64, 65, 66, 69]}
    DEBUG [06:18:09.656] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.666] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.668] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.669] Applying learner 'regr.ranger' on task 'df.tr' (iter 2/50) 
    DEBUG [06:18:09.678] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:18:09.681] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.693] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.694] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.697] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68]}
    DEBUG [06:18:09.700] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.711] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.712] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.713] Applying learner 'regr.ranger' on task 'df.tr' (iter 17/50) 
    DEBUG [06:18:09.722] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 7..]}
    DEBUG [06:18:09.726] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.737] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.738] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.741] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70]}
    DEBUG [06:18:09.744] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.755] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.756] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.757] Applying learner 'regr.ranger' on task 'df.tr' (iter 20/50) 
    DEBUG [06:18:09.766] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:09.770] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.780] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.782] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.785] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 16, 17, 32, 36, 37, 42, 43, 45, 46, 53, 61, 65, 69]}
    DEBUG [06:18:09.789] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.800] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.802] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.803] Applying learner 'regr.ranger' on task 'df.tr' (iter 44/50) 
    DEBUG [06:18:09.813] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:09.817] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.829] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.830] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.834] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 14, 17, 19, 24, 36, 41, 47, 51, 53, 56, 62, 63, 67, 71]}
    DEBUG [06:18:09.838] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.850] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.852] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.853] Applying learner 'regr.ranger' on task 'df.tr' (iter 23/50) 
    DEBUG [06:18:09.863] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:09.867] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.878] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.880] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.883] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66]}
    DEBUG [06:18:09.886] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.897] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.899] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.900] Applying learner 'regr.ranger' on task 'df.tr' (iter 35/50) 
    DEBUG [06:18:09.909] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:09.913] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.924] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.926] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.929] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 29, 32, 33, 35, 40, 43, 44, 45, 61, 62, 70, 74]}
    DEBUG [06:18:09.932] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.942] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.943] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.945] Applying learner 'regr.ranger' on task 'df.tr' (iter 14/50) 
    DEBUG [06:18:09.954] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:09.957] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.968] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:09.970] Creating Prediction for predict set 'test' 
    DEBUG [06:18:09.973] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 22, 27, 36, 38, 40, 45, 46, 50, 55, 56, 60, 65, 73]}
    DEBUG [06:18:09.976] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:09.987] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:09.989] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:09.990] Applying learner 'regr.ranger' on task 'df.tr' (iter 22/50) 
    DEBUG [06:18:10.000] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57..]}
    DEBUG [06:18:10.003] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.015] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:10.016] Creating Prediction for predict set 'test' 
    DEBUG [06:18:10.019] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75]}
    DEBUG [06:18:10.022] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.033] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:10.034] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:10.035] Applying learner 'regr.ranger' on task 'df.tr' (iter 21/50) 
    DEBUG [06:18:10.045] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66..]}
    DEBUG [06:18:10.049] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.060] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:10.062] Creating Prediction for predict set 'test' 
    DEBUG [06:18:10.065] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72]}
    DEBUG [06:18:10.068] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.078] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:10.080] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:10.081] Applying learner 'regr.ranger' on task 'df.tr' (iter 9/50) 
    DEBUG [06:18:10.090] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:10.094] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.105] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:10.106] Creating Prediction for predict set 'test' 
    DEBUG [06:18:10.109] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 23, 27, 40, 42, 46, 49, 56, 58, 61, 64, 67, 70, 71, 72]}
    DEBUG [06:18:10.112] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.123] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:10.124] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:10.125] Applying learner 'regr.ranger' on task 'df.tr' (iter 39/50) 
    DEBUG [06:18:10.135] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:10.138] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.151] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:10.153] Creating Prediction for predict set 'test' 
    DEBUG [06:18:10.155] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 5, 14, 18, 19, 21, 28, 38, 48, 49, 51, 52, 59, 64, 65]}
    DEBUG [06:18:10.159] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.170] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:10.171] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:10.172] Applying learner 'regr.ranger' on task 'df.tr' (iter 43/50) 
    DEBUG [06:18:10.182] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:10.185] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.197] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:10.199] Creating Prediction for predict set 'test' 
    DEBUG [06:18:10.202] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74]}
    DEBUG [06:18:10.205] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.216] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:10.217] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:10.218] Applying learner 'regr.ranger' on task 'df.tr' (iter 27/50) 
    DEBUG [06:18:10.227] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67..]}
    DEBUG [06:18:10.231] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.241] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:10.243] Creating Prediction for predict set 'test' 
    DEBUG [06:18:10.246] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76]}
    DEBUG [06:18:10.249] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.260] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:10.261] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:10.262] Applying learner 'regr.ranger' on task 'df.tr' (iter 34/50) 
    DEBUG [06:18:10.271] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:10.275] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.286] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:10.335] Creating Prediction for predict set 'test' 
    DEBUG [06:18:10.339] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 7, 12, 14, 18, 20, 41, 46, 47, 53, 57, 63, 64, 72, 75]}
    DEBUG [06:18:10.342] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.372] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:10.376] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:10.378] Applying learner 'regr.ranger' on task 'df.tr' (iter 8/50) 
    DEBUG [06:18:10.395] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:10.402] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.423] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:10.425] Creating Prediction for predict set 'test' 
    DEBUG [06:18:10.431] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 75]}
    DEBUG [06:18:10.435] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.451] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:10.454] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:10.455] Applying learner 'regr.ranger' on task 'df.tr' (iter 45/50) 
    DEBUG [06:18:10.469] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:10.474] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.488] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:10.490] Creating Prediction for predict set 'test' 
    DEBUG [06:18:10.495] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 8, 12, 21, 22, 25, 28, 33, 37, 38, 44, 46, 48, 57, 75]}
    DEBUG [06:18:10.500] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.514] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:10.515] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:10.516] Applying learner 'regr.ranger' on task 'df.tr' (iter 7/50) 
    DEBUG [06:18:10.526] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 6..]}
    DEBUG [06:18:10.529] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.540] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:10.542] Creating Prediction for predict set 'test' 
    DEBUG [06:18:10.545] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74]}
    DEBUG [06:18:10.548] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.559] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:10.561] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:10.562] Applying learner 'regr.ranger' on task 'df.tr' (iter 31/50) 
    DEBUG [06:18:10.572] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60..]}
    DEBUG [06:18:10.575] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.587] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:10.589] Creating Prediction for predict set 'test' 
    DEBUG [06:18:10.592] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73]}
    DEBUG [06:18:10.595] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.606] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:10.607] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:10.608] Applying learner 'regr.ranger' on task 'df.tr' (iter 41/50) 
    DEBUG [06:18:10.617] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74..]}
    DEBUG [06:18:10.621] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.632] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:10.634] Creating Prediction for predict set 'test' 
    DEBUG [06:18:10.636] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76]}
    DEBUG [06:18:10.640] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:10.650] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:10.652] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:10.671] Finished benchmark 
    INFO  [06:18:12.368] Result of batch 4: 
    INFO  [06:18:12.371]  mtry sample.fraction num.trees importance regr.rmse      resample_result 
    INFO  [06:18:12.371]     5       0.7631363       141   impurity  43.04499 <ResampleResult[19]> 
    INFO  [06:18:12.380] Evaluating 1 configuration(s) 
    INFO  [06:18:12.420] Benchmark with 50 resampling iterations 
    DEBUG [06:18:12.421] Running benchmark() asynchronously with 50 iterations 
    INFO  [06:18:12.434] Applying learner 'regr.ranger' on task 'df.tr' (iter 46/50) 
    DEBUG [06:18:12.444] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72..]}
    DEBUG [06:18:12.447] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:12.465] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:12.467] Creating Prediction for predict set 'test' 
    DEBUG [06:18:12.470] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73]}
    DEBUG [06:18:12.474] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:12.486] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:12.487] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:12.489] Applying learner 'regr.ranger' on task 'df.tr' (iter 48/50) 
    DEBUG [06:18:12.498] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:12.501] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:12.519] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:12.522] Creating Prediction for predict set 'test' 
    DEBUG [06:18:12.525] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72, 74]}
    DEBUG [06:18:12.528] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:12.542] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:12.544] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:12.546] Applying learner 'regr.ranger' on task 'df.tr' (iter 10/50) 
    DEBUG [06:18:12.556] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:12.559] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:12.577] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:12.578] Creating Prediction for predict set 'test' 
    DEBUG [06:18:12.582] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 17, 18, 19, 24, 26, 29, 31, 34, 36, 43, 45, 53, 66, 73]}
    DEBUG [06:18:12.586] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:12.599] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:12.601] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:12.602] Applying learner 'regr.ranger' on task 'df.tr' (iter 28/50) 
    DEBUG [06:18:12.612] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:12.615] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:12.632] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:12.634] Creating Prediction for predict set 'test' 
    DEBUG [06:18:12.637] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75]}
    DEBUG [06:18:12.640] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:12.652] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:12.653] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:12.654] Applying learner 'regr.ranger' on task 'df.tr' (iter 12/50) 
    DEBUG [06:18:12.664] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61..]}
    DEBUG [06:18:12.667] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:12.685] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:12.687] Creating Prediction for predict set 'test' 
    DEBUG [06:18:12.691] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70]}
    DEBUG [06:18:12.694] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:12.706] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:12.708] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:12.709] Applying learner 'regr.ranger' on task 'df.tr' (iter 3/50) 
    DEBUG [06:18:12.718] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:12.722] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:12.738] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:12.740] Creating Prediction for predict set 'test' 
    DEBUG [06:18:12.744] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75, 76]}
    DEBUG [06:18:12.748] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:12.761] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:12.763] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:12.764] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [06:18:12.773] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:18:12.777] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:12.801] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:12.803] Creating Prediction for predict set 'test' 
    DEBUG [06:18:12.807] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [06:18:12.811] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:12.827] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:12.830] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:12.831] Applying learner 'regr.ranger' on task 'df.tr' (iter 33/50) 
    DEBUG [06:18:12.842] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:12.845] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:12.865] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:12.868] Creating Prediction for predict set 'test' 
    DEBUG [06:18:12.871] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60, 76]}
    DEBUG [06:18:12.879] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:12.897] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:12.899] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:12.900] Applying learner 'regr.ranger' on task 'df.tr' (iter 36/50) 
    DEBUG [06:18:12.909] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58..]}
    DEBUG [06:18:12.913] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:12.928] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:12.930] Creating Prediction for predict set 'test' 
    DEBUG [06:18:12.933] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75]}
    DEBUG [06:18:12.936] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:12.947] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:12.949] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:12.950] Applying learner 'regr.ranger' on task 'df.tr' (iter 13/50) 
    DEBUG [06:18:12.959] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:12.963] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:12.978] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:12.980] Creating Prediction for predict set 'test' 
    DEBUG [06:18:12.984] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 72]}
    DEBUG [06:18:12.988] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.000] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:13.002] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:13.003] Applying learner 'regr.ranger' on task 'df.tr' (iter 15/50) 
    DEBUG [06:18:13.012] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:13.015] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.049] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:13.051] Creating Prediction for predict set 'test' 
    DEBUG [06:18:13.055] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 19, 20, 30, 35, 42, 47, 54, 59, 64, 67, 71, 74, 75]}
    DEBUG [06:18:13.059] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.074] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:13.075] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:13.077] Applying learner 'regr.ranger' on task 'df.tr' (iter 41/50) 
    DEBUG [06:18:13.088] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74..]}
    DEBUG [06:18:13.092] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.109] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:13.111] Creating Prediction for predict set 'test' 
    DEBUG [06:18:13.115] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76]}
    DEBUG [06:18:13.124] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.144] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:13.146] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:13.148] Applying learner 'regr.ranger' on task 'df.tr' (iter 30/50) 
    DEBUG [06:18:13.162] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:13.167] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.191] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:13.193] Creating Prediction for predict set 'test' 
    DEBUG [06:18:13.197] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 14, 22, 29, 34, 35, 37, 40, 43, 44, 59, 62, 64, 68, 72]}
    DEBUG [06:18:13.201] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.216] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:13.218] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:13.219] Applying learner 'regr.ranger' on task 'df.tr' (iter 18/50) 
    DEBUG [06:18:13.233] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:13.237] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.255] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:13.256] Creating Prediction for predict set 'test' 
    DEBUG [06:18:13.259] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75]}
    DEBUG [06:18:13.263] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.275] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:13.277] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:13.278] Applying learner 'regr.ranger' on task 'df.tr' (iter 26/50) 
    DEBUG [06:18:13.288] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75..]}
    DEBUG [06:18:13.291] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.308] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:13.310] Creating Prediction for predict set 'test' 
    DEBUG [06:18:13.313] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74]}
    DEBUG [06:18:13.316] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.328] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:13.330] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:13.331] Applying learner 'regr.ranger' on task 'df.tr' (iter 45/50) 
    DEBUG [06:18:13.340] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:13.343] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.360] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:13.362] Creating Prediction for predict set 'test' 
    DEBUG [06:18:13.365] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 8, 12, 21, 22, 25, 28, 33, 37, 38, 44, 46, 48, 57, 75]}
    DEBUG [06:18:13.368] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.380] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:13.381] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:13.382] Applying learner 'regr.ranger' on task 'df.tr' (iter 25/50) 
    DEBUG [06:18:13.391] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:13.395] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.410] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:13.412] Creating Prediction for predict set 'test' 
    DEBUG [06:18:13.415] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [9, 11, 18, 21, 23, 27, 29, 43, 44, 45, 51, 55, 67, 73, 76]}
    DEBUG [06:18:13.419] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.431] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:13.433] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:13.434] Applying learner 'regr.ranger' on task 'df.tr' (iter 11/50) 
    DEBUG [06:18:13.443] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 7..]}
    DEBUG [06:18:13.447] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.464] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:13.466] Creating Prediction for predict set 'test' 
    DEBUG [06:18:13.475] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76]}
    DEBUG [06:18:13.479] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.494] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:13.495] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:13.497] Applying learner 'regr.ranger' on task 'df.tr' (iter 19/50) 
    DEBUG [06:18:13.508] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:13.512] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.533] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:13.534] Creating Prediction for predict set 'test' 
    DEBUG [06:18:13.538] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 8, 11, 14, 18, 24, 28, 33, 34, 54, 56, 62, 64, 68, 76]}
    DEBUG [06:18:13.541] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.556] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:13.558] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:13.559] Applying learner 'regr.ranger' on task 'df.tr' (iter 38/50) 
    DEBUG [06:18:13.570] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:13.573] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.591] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:13.593] Creating Prediction for predict set 'test' 
    DEBUG [06:18:13.596] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58, 69]}
    DEBUG [06:18:13.600] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.613] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:13.615] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:13.616] Applying learner 'regr.ranger' on task 'df.tr' (iter 16/50) 
    DEBUG [06:18:13.627] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75..]}
    DEBUG [06:18:13.630] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.649] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:13.651] Creating Prediction for predict set 'test' 
    DEBUG [06:18:13.656] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72]}
    DEBUG [06:18:13.659] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.673] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:13.675] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:13.676] Applying learner 'regr.ranger' on task 'df.tr' (iter 40/50) 
    DEBUG [06:18:13.686] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:13.690] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.709] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:13.711] Creating Prediction for predict set 'test' 
    DEBUG [06:18:13.714] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 8, 16, 17, 24, 25, 27, 29, 41, 44, 53, 54, 67, 72]}
    DEBUG [06:18:13.718] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.732] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:13.734] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:13.735] Applying learner 'regr.ranger' on task 'df.tr' (iter 43/50) 
    DEBUG [06:18:13.745] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:13.749] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.766] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:13.768] Creating Prediction for predict set 'test' 
    DEBUG [06:18:13.772] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74]}
    DEBUG [06:18:13.775] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.789] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:13.790] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:13.792] Applying learner 'regr.ranger' on task 'df.tr' (iter 6/50) 
    DEBUG [06:18:13.802] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 7..]}
    DEBUG [06:18:13.806] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.823] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:13.825] Creating Prediction for predict set 'test' 
    DEBUG [06:18:13.828] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76]}
    DEBUG [06:18:13.832] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.844] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:13.845] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:13.847] Applying learner 'regr.ranger' on task 'df.tr' (iter 5/50) 
    DEBUG [06:18:13.856] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:13.860] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.884] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:13.887] Creating Prediction for predict set 'test' 
    DEBUG [06:18:13.893] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 11, 14, 17, 27, 32, 41, 44, 45, 55, 60, 64, 65, 66, 69]}
    DEBUG [06:18:13.898] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.912] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:13.914] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:13.915] Applying learner 'regr.ranger' on task 'df.tr' (iter 2/50) 
    DEBUG [06:18:13.926] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:18:13.929] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.947] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:13.953] Creating Prediction for predict set 'test' 
    DEBUG [06:18:13.957] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68]}
    DEBUG [06:18:13.961] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:13.977] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:13.978] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:13.980] Applying learner 'regr.ranger' on task 'df.tr' (iter 22/50) 
    DEBUG [06:18:13.991] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57..]}
    DEBUG [06:18:13.995] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.015] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:14.017] Creating Prediction for predict set 'test' 
    DEBUG [06:18:14.022] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75]}
    DEBUG [06:18:14.025] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.040] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:14.042] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:14.043] Applying learner 'regr.ranger' on task 'df.tr' (iter 17/50) 
    DEBUG [06:18:14.054] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 7..]}
    DEBUG [06:18:14.058] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.075] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:14.076] Creating Prediction for predict set 'test' 
    DEBUG [06:18:14.080] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70]}
    DEBUG [06:18:14.084] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.098] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:14.100] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:14.101] Applying learner 'regr.ranger' on task 'df.tr' (iter 42/50) 
    DEBUG [06:18:14.112] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70..]}
    DEBUG [06:18:14.116] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.135] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:14.137] Creating Prediction for predict set 'test' 
    DEBUG [06:18:14.141] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66]}
    DEBUG [06:18:14.144] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.158] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:14.160] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:14.161] Applying learner 'regr.ranger' on task 'df.tr' (iter 32/50) 
    DEBUG [06:18:14.172] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 6..]}
    DEBUG [06:18:14.175] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.194] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:14.196] Creating Prediction for predict set 'test' 
    DEBUG [06:18:14.200] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71]}
    DEBUG [06:18:14.204] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.218] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:14.220] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:14.221] Applying learner 'regr.ranger' on task 'df.tr' (iter 29/50) 
    DEBUG [06:18:14.232] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:14.236] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.255] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:14.258] Creating Prediction for predict set 'test' 
    DEBUG [06:18:14.262] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 18, 25, 28, 30, 39, 45, 48, 51, 54, 55, 61, 66, 71, 73]}
    DEBUG [06:18:14.266] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.279] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:14.281] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:14.282] Applying learner 'regr.ranger' on task 'df.tr' (iter 21/50) 
    DEBUG [06:18:14.293] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66..]}
    DEBUG [06:18:14.297] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.315] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:14.317] Creating Prediction for predict set 'test' 
    DEBUG [06:18:14.320] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72]}
    DEBUG [06:18:14.324] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.338] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:14.340] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:14.341] Applying learner 'regr.ranger' on task 'df.tr' (iter 35/50) 
    DEBUG [06:18:14.351] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:14.355] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.386] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:14.390] Creating Prediction for predict set 'test' 
    DEBUG [06:18:14.397] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 29, 32, 33, 35, 40, 43, 44, 45, 61, 62, 70, 74]}
    DEBUG [06:18:14.401] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.417] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:14.418] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:14.420] Applying learner 'regr.ranger' on task 'df.tr' (iter 20/50) 
    DEBUG [06:18:14.431] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:14.474] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.497] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:14.500] Creating Prediction for predict set 'test' 
    DEBUG [06:18:14.506] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 16, 17, 32, 36, 37, 42, 43, 45, 46, 53, 61, 65, 69]}
    DEBUG [06:18:14.511] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.526] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:14.528] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:14.530] Applying learner 'regr.ranger' on task 'df.tr' (iter 47/50) 
    DEBUG [06:18:14.545] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64..]}
    DEBUG [06:18:14.553] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.576] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:14.578] Creating Prediction for predict set 'test' 
    DEBUG [06:18:14.582] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70]}
    DEBUG [06:18:14.586] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.601] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:14.603] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:14.604] Applying learner 'regr.ranger' on task 'df.tr' (iter 37/50) 
    DEBUG [06:18:14.617] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57..]}
    DEBUG [06:18:14.621] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.639] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:14.641] Creating Prediction for predict set 'test' 
    DEBUG [06:18:14.645] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76]}
    DEBUG [06:18:14.648] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.661] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:14.662] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:14.663] Applying learner 'regr.ranger' on task 'df.tr' (iter 14/50) 
    DEBUG [06:18:14.673] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:14.676] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.692] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:14.693] Creating Prediction for predict set 'test' 
    DEBUG [06:18:14.696] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 22, 27, 36, 38, 40, 45, 46, 50, 55, 56, 60, 65, 73]}
    DEBUG [06:18:14.700] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.712] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:14.713] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:14.714] Applying learner 'regr.ranger' on task 'df.tr' (iter 34/50) 
    DEBUG [06:18:14.724] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:14.727] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.746] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:14.748] Creating Prediction for predict set 'test' 
    DEBUG [06:18:14.753] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 7, 12, 14, 18, 20, 41, 46, 47, 53, 57, 63, 64, 72, 75]}
    DEBUG [06:18:14.756] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.768] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:14.770] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:14.771] Applying learner 'regr.ranger' on task 'df.tr' (iter 24/50) 
    DEBUG [06:18:14.780] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:14.789] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.811] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:14.813] Creating Prediction for predict set 'test' 
    DEBUG [06:18:14.817] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 13, 17, 25, 32, 35, 36, 37, 38, 50, 63, 65, 68, 69, 74]}
    DEBUG [06:18:14.820] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.834] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:14.835] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:14.837] Applying learner 'regr.ranger' on task 'df.tr' (iter 50/50) 
    DEBUG [06:18:14.847] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:14.851] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.870] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:14.872] Creating Prediction for predict set 'test' 
    DEBUG [06:18:14.876] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 8, 21, 22, 30, 36, 41, 45, 49, 53, 54, 57, 68, 71]}
    DEBUG [06:18:14.880] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.892] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:14.894] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:14.895] Applying learner 'regr.ranger' on task 'df.tr' (iter 7/50) 
    DEBUG [06:18:14.905] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 6..]}
    DEBUG [06:18:14.909] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.927] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:14.929] Creating Prediction for predict set 'test' 
    DEBUG [06:18:14.932] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74]}
    DEBUG [06:18:14.936] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.948] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:14.950] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:14.951] Applying learner 'regr.ranger' on task 'df.tr' (iter 39/50) 
    DEBUG [06:18:14.961] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:14.965] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:14.983] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:14.985] Creating Prediction for predict set 'test' 
    DEBUG [06:18:14.988] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 5, 14, 18, 19, 21, 28, 38, 48, 49, 51, 52, 59, 64, 65]}
    DEBUG [06:18:14.992] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:15.005] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:15.007] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:15.008] Applying learner 'regr.ranger' on task 'df.tr' (iter 8/50) 
    DEBUG [06:18:15.019] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:15.022] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:15.042] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:15.044] Creating Prediction for predict set 'test' 
    DEBUG [06:18:15.048] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 75]}
    DEBUG [06:18:15.051] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:15.064] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:15.066] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:15.067] Applying learner 'regr.ranger' on task 'df.tr' (iter 49/50) 
    DEBUG [06:18:15.078] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:15.082] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:15.102] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:15.104] Creating Prediction for predict set 'test' 
    DEBUG [06:18:15.109] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 7, 14, 16, 25, 26, 27, 40, 44, 56, 67, 69, 75, 76]}
    DEBUG [06:18:15.113] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:15.126] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:15.127] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:15.129] Applying learner 'regr.ranger' on task 'df.tr' (iter 31/50) 
    DEBUG [06:18:15.139] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60..]}
    DEBUG [06:18:15.143] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:15.160] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:15.162] Creating Prediction for predict set 'test' 
    DEBUG [06:18:15.166] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73]}
    DEBUG [06:18:15.169] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:15.183] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:15.184] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:15.186] Applying learner 'regr.ranger' on task 'df.tr' (iter 9/50) 
    DEBUG [06:18:15.196] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:15.200] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:15.218] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:15.221] Creating Prediction for predict set 'test' 
    DEBUG [06:18:15.226] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 23, 27, 40, 42, 46, 49, 56, 58, 61, 64, 67, 70, 71, 72]}
    DEBUG [06:18:15.230] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:15.244] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:15.246] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:15.247] Applying learner 'regr.ranger' on task 'df.tr' (iter 23/50) 
    DEBUG [06:18:15.256] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:15.260] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:15.284] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:15.287] Creating Prediction for predict set 'test' 
    DEBUG [06:18:15.290] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66]}
    DEBUG [06:18:15.294] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:15.307] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:15.309] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:15.310] Applying learner 'regr.ranger' on task 'df.tr' (iter 44/50) 
    DEBUG [06:18:15.320] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:15.324] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:15.342] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:15.345] Creating Prediction for predict set 'test' 
    DEBUG [06:18:15.349] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 14, 17, 19, 24, 36, 41, 47, 51, 53, 56, 62, 63, 67, 71]}
    DEBUG [06:18:15.353] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:15.366] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:15.367] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:15.369] Applying learner 'regr.ranger' on task 'df.tr' (iter 4/50) 
    DEBUG [06:18:15.380] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:15.384] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:15.403] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:15.405] Creating Prediction for predict set 'test' 
    DEBUG [06:18:15.410] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 3, 13, 20, 23, 25, 40, 46, 50, 53, 56, 57, 58, 63]}
    DEBUG [06:18:15.413] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:15.426] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:15.428] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:15.429] Applying learner 'regr.ranger' on task 'df.tr' (iter 27/50) 
    DEBUG [06:18:15.440] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67..]}
    DEBUG [06:18:15.443] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:15.462] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:15.464] Creating Prediction for predict set 'test' 
    DEBUG [06:18:15.467] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76]}
    DEBUG [06:18:15.471] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:15.484] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:15.486] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:15.513] Finished benchmark 
    INFO  [06:18:17.426] Result of batch 5: 
    INFO  [06:18:17.429]  mtry sample.fraction num.trees importance regr.rmse      resample_result 
    INFO  [06:18:17.429]     4       0.8667243       428   impurity  43.33344 <ResampleResult[19]> 
    INFO  [06:18:17.438] Evaluating 1 configuration(s) 
    INFO  [06:18:17.479] Benchmark with 50 resampling iterations 
    DEBUG [06:18:17.480] Running benchmark() asynchronously with 50 iterations 
    INFO  [06:18:17.494] Applying learner 'regr.ranger' on task 'df.tr' (iter 3/50) 
    DEBUG [06:18:17.503] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:17.507] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:17.526] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:17.528] Creating Prediction for predict set 'test' 
    DEBUG [06:18:17.531] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75, 76]}
    DEBUG [06:18:17.534] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:17.547] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:17.549] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:17.550] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [06:18:17.560] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:18:17.563] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:17.579] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:17.580] Creating Prediction for predict set 'test' 
    DEBUG [06:18:17.583] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [06:18:17.587] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:17.599] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:17.601] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:17.602] Applying learner 'regr.ranger' on task 'df.tr' (iter 44/50) 
    DEBUG [06:18:17.617] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:17.621] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:17.645] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:17.647] Creating Prediction for predict set 'test' 
    DEBUG [06:18:17.650] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 14, 17, 19, 24, 36, 41, 47, 51, 53, 56, 62, 63, 67, 71]}
    DEBUG [06:18:17.654] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:17.668] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:17.670] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:17.671] Applying learner 'regr.ranger' on task 'df.tr' (iter 33/50) 
    DEBUG [06:18:17.682] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:17.686] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:17.702] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:17.704] Creating Prediction for predict set 'test' 
    DEBUG [06:18:17.707] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60, 76]}
    DEBUG [06:18:17.711] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:17.725] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:17.727] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:17.728] Applying learner 'regr.ranger' on task 'df.tr' (iter 36/50) 
    DEBUG [06:18:17.739] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58..]}
    DEBUG [06:18:17.743] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:17.759] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:17.761] Creating Prediction for predict set 'test' 
    DEBUG [06:18:17.765] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75]}
    DEBUG [06:18:17.769] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:17.783] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:17.785] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:17.786] Applying learner 'regr.ranger' on task 'df.tr' (iter 13/50) 
    DEBUG [06:18:17.797] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:17.801] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:17.819] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:17.821] Creating Prediction for predict set 'test' 
    DEBUG [06:18:17.825] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 72]}
    DEBUG [06:18:17.828] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:17.840] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:17.842] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:17.843] Applying learner 'regr.ranger' on task 'df.tr' (iter 15/50) 
    DEBUG [06:18:17.853] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:17.857] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:17.873] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:17.875] Creating Prediction for predict set 'test' 
    DEBUG [06:18:17.878] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 19, 20, 30, 35, 42, 47, 54, 59, 64, 67, 71, 74, 75]}
    DEBUG [06:18:17.882] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:17.895] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:17.897] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:17.898] Applying learner 'regr.ranger' on task 'df.tr' (iter 45/50) 
    DEBUG [06:18:17.909] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:17.912] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:17.929] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:17.931] Creating Prediction for predict set 'test' 
    DEBUG [06:18:17.936] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 8, 12, 21, 22, 25, 28, 33, 37, 38, 44, 46, 48, 57, 75]}
    DEBUG [06:18:17.940] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:17.954] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:17.955] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:17.957] Applying learner 'regr.ranger' on task 'df.tr' (iter 30/50) 
    DEBUG [06:18:17.967] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:17.971] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:17.986] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:17.988] Creating Prediction for predict set 'test' 
    DEBUG [06:18:17.991] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 14, 22, 29, 34, 35, 37, 40, 43, 44, 59, 62, 64, 68, 72]}
    DEBUG [06:18:17.995] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.008] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:18.010] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:18.011] Applying learner 'regr.ranger' on task 'df.tr' (iter 18/50) 
    DEBUG [06:18:18.022] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:18.025] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.041] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:18.042] Creating Prediction for predict set 'test' 
    DEBUG [06:18:18.046] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75]}
    DEBUG [06:18:18.049] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.061] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:18.063] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:18.064] Applying learner 'regr.ranger' on task 'df.tr' (iter 39/50) 
    DEBUG [06:18:18.074] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:18.077] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.096] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:18.098] Creating Prediction for predict set 'test' 
    DEBUG [06:18:18.103] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 5, 14, 18, 19, 21, 28, 38, 48, 49, 51, 52, 59, 64, 65]}
    DEBUG [06:18:18.107] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.119] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:18.121] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:18.122] Applying learner 'regr.ranger' on task 'df.tr' (iter 26/50) 
    DEBUG [06:18:18.132] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75..]}
    DEBUG [06:18:18.135] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.151] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:18.152] Creating Prediction for predict set 'test' 
    DEBUG [06:18:18.156] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74]}
    DEBUG [06:18:18.160] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.172] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:18.174] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:18.175] Applying learner 'regr.ranger' on task 'df.tr' (iter 46/50) 
    DEBUG [06:18:18.185] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72..]}
    DEBUG [06:18:18.188] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.203] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:18.204] Creating Prediction for predict set 'test' 
    DEBUG [06:18:18.209] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73]}
    DEBUG [06:18:18.212] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.225] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:18.227] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:18.228] Applying learner 'regr.ranger' on task 'df.tr' (iter 38/50) 
    DEBUG [06:18:18.244] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:18.249] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.268] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:18.271] Creating Prediction for predict set 'test' 
    DEBUG [06:18:18.275] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58, 69]}
    DEBUG [06:18:18.279] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.292] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:18.294] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:18.295] Applying learner 'regr.ranger' on task 'df.tr' (iter 50/50) 
    DEBUG [06:18:18.306] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:18.310] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.328] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:18.331] Creating Prediction for predict set 'test' 
    DEBUG [06:18:18.335] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 8, 21, 22, 30, 36, 41, 45, 49, 53, 54, 57, 68, 71]}
    DEBUG [06:18:18.338] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.351] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:18.353] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:18.355] Applying learner 'regr.ranger' on task 'df.tr' (iter 25/50) 
    DEBUG [06:18:18.403] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:18.410] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.430] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:18.433] Creating Prediction for predict set 'test' 
    DEBUG [06:18:18.437] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [9, 11, 18, 21, 23, 27, 29, 43, 44, 45, 51, 55, 67, 73, 76]}
    DEBUG [06:18:18.442] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.459] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:18.462] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:18.465] Applying learner 'regr.ranger' on task 'df.tr' (iter 11/50) 
    DEBUG [06:18:18.480] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 7..]}
    DEBUG [06:18:18.484] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.501] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:18.504] Creating Prediction for predict set 'test' 
    DEBUG [06:18:18.508] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76]}
    DEBUG [06:18:18.513] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.529] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:18.531] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:18.532] Applying learner 'regr.ranger' on task 'df.tr' (iter 42/50) 
    DEBUG [06:18:18.547] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70..]}
    DEBUG [06:18:18.551] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.567] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:18.569] Creating Prediction for predict set 'test' 
    DEBUG [06:18:18.572] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66]}
    DEBUG [06:18:18.575] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.589] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:18.591] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:18.592] Applying learner 'regr.ranger' on task 'df.tr' (iter 16/50) 
    DEBUG [06:18:18.601] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75..]}
    DEBUG [06:18:18.605] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.622] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:18.623] Creating Prediction for predict set 'test' 
    DEBUG [06:18:18.626] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72]}
    DEBUG [06:18:18.629] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.641] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:18.643] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:18.644] Applying learner 'regr.ranger' on task 'df.tr' (iter 48/50) 
    DEBUG [06:18:18.653] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:18.657] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.672] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:18.673] Creating Prediction for predict set 'test' 
    DEBUG [06:18:18.676] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72, 74]}
    DEBUG [06:18:18.679] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.693] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:18.695] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:18.696] Applying learner 'regr.ranger' on task 'df.tr' (iter 34/50) 
    DEBUG [06:18:18.705] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:18.709] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.726] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:18.728] Creating Prediction for predict set 'test' 
    DEBUG [06:18:18.732] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 7, 12, 14, 18, 20, 41, 46, 47, 53, 57, 63, 64, 72, 75]}
    DEBUG [06:18:18.736] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.748] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:18.750] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:18.751] Applying learner 'regr.ranger' on task 'df.tr' (iter 6/50) 
    DEBUG [06:18:18.760] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 7..]}
    DEBUG [06:18:18.764] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.782] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:18.784] Creating Prediction for predict set 'test' 
    DEBUG [06:18:18.788] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76]}
    DEBUG [06:18:18.791] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.803] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:18.805] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:18.806] Applying learner 'regr.ranger' on task 'df.tr' (iter 5/50) 
    DEBUG [06:18:18.815] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:18.819] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.841] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:18.844] Creating Prediction for predict set 'test' 
    DEBUG [06:18:18.847] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 11, 14, 17, 27, 32, 41, 44, 45, 55, 60, 64, 65, 66, 69]}
    DEBUG [06:18:18.850] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.865] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:18.867] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:18.868] Applying learner 'regr.ranger' on task 'df.tr' (iter 2/50) 
    DEBUG [06:18:18.878] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:18:18.881] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.898] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:18.901] Creating Prediction for predict set 'test' 
    DEBUG [06:18:18.904] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68]}
    DEBUG [06:18:18.907] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.919] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:18.921] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:18.922] Applying learner 'regr.ranger' on task 'df.tr' (iter 22/50) 
    DEBUG [06:18:18.931] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57..]}
    DEBUG [06:18:18.935] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.949] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:18.951] Creating Prediction for predict set 'test' 
    DEBUG [06:18:18.954] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75]}
    DEBUG [06:18:18.957] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:18.968] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:18.970] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:18.971] Applying learner 'regr.ranger' on task 'df.tr' (iter 17/50) 
    DEBUG [06:18:18.980] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 7..]}
    DEBUG [06:18:18.983] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.000] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:19.003] Creating Prediction for predict set 'test' 
    DEBUG [06:18:19.006] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70]}
    DEBUG [06:18:19.009] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.021] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:19.023] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:19.024] Applying learner 'regr.ranger' on task 'df.tr' (iter 28/50) 
    DEBUG [06:18:19.033] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:19.036] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.053] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:19.055] Creating Prediction for predict set 'test' 
    DEBUG [06:18:19.059] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75]}
    DEBUG [06:18:19.062] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.074] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:19.076] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:19.077] Applying learner 'regr.ranger' on task 'df.tr' (iter 19/50) 
    DEBUG [06:18:19.087] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:19.091] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.107] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:19.109] Creating Prediction for predict set 'test' 
    DEBUG [06:18:19.112] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 8, 11, 14, 18, 24, 28, 33, 34, 54, 56, 62, 64, 68, 76]}
    DEBUG [06:18:19.116] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.128] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:19.130] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:19.132] Applying learner 'regr.ranger' on task 'df.tr' (iter 31/50) 
    DEBUG [06:18:19.143] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60..]}
    DEBUG [06:18:19.147] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.166] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:19.169] Creating Prediction for predict set 'test' 
    DEBUG [06:18:19.173] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73]}
    DEBUG [06:18:19.176] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.188] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:19.190] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:19.191] Applying learner 'regr.ranger' on task 'df.tr' (iter 23/50) 
    DEBUG [06:18:19.200] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:19.204] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.220] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:19.221] Creating Prediction for predict set 'test' 
    DEBUG [06:18:19.224] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66]}
    DEBUG [06:18:19.227] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.239] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:19.241] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:19.242] Applying learner 'regr.ranger' on task 'df.tr' (iter 37/50) 
    DEBUG [06:18:19.251] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57..]}
    DEBUG [06:18:19.255] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.269] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:19.272] Creating Prediction for predict set 'test' 
    DEBUG [06:18:19.276] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76]}
    DEBUG [06:18:19.279] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.292] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:19.293] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:19.294] Applying learner 'regr.ranger' on task 'df.tr' (iter 40/50) 
    DEBUG [06:18:19.304] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:19.307] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.323] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:19.324] Creating Prediction for predict set 'test' 
    DEBUG [06:18:19.327] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 8, 16, 17, 24, 25, 27, 29, 41, 44, 53, 54, 67, 72]}
    DEBUG [06:18:19.330] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.342] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:19.344] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:19.345] Applying learner 'regr.ranger' on task 'df.tr' (iter 20/50) 
    DEBUG [06:18:19.355] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:19.359] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.374] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:19.375] Creating Prediction for predict set 'test' 
    DEBUG [06:18:19.378] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 16, 17, 32, 36, 37, 42, 43, 45, 46, 53, 61, 65, 69]}
    DEBUG [06:18:19.381] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.394] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:19.395] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:19.396] Applying learner 'regr.ranger' on task 'df.tr' (iter 27/50) 
    DEBUG [06:18:19.406] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67..]}
    DEBUG [06:18:19.409] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.425] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:19.428] Creating Prediction for predict set 'test' 
    DEBUG [06:18:19.431] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76]}
    DEBUG [06:18:19.434] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.447] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:19.448] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:19.449] Applying learner 'regr.ranger' on task 'df.tr' (iter 24/50) 
    DEBUG [06:18:19.459] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:19.462] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.477] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:19.478] Creating Prediction for predict set 'test' 
    DEBUG [06:18:19.482] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 13, 17, 25, 32, 35, 36, 37, 38, 50, 63, 65, 68, 69, 74]}
    DEBUG [06:18:19.485] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.497] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:19.499] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:19.500] Applying learner 'regr.ranger' on task 'df.tr' (iter 32/50) 
    DEBUG [06:18:19.510] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 6..]}
    DEBUG [06:18:19.513] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.530] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:19.532] Creating Prediction for predict set 'test' 
    DEBUG [06:18:19.535] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71]}
    DEBUG [06:18:19.538] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.551] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:19.553] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:19.554] Applying learner 'regr.ranger' on task 'df.tr' (iter 7/50) 
    DEBUG [06:18:19.563] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 6..]}
    DEBUG [06:18:19.572] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.591] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:19.593] Creating Prediction for predict set 'test' 
    DEBUG [06:18:19.597] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74]}
    DEBUG [06:18:19.600] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.659] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:19.663] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:19.665] Applying learner 'regr.ranger' on task 'df.tr' (iter 35/50) 
    DEBUG [06:18:19.686] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:19.691] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.712] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:19.715] Creating Prediction for predict set 'test' 
    DEBUG [06:18:19.723] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 29, 32, 33, 35, 40, 43, 44, 45, 61, 62, 70, 74]}
    DEBUG [06:18:19.727] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.744] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:19.746] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:19.748] Applying learner 'regr.ranger' on task 'df.tr' (iter 8/50) 
    DEBUG [06:18:19.760] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:19.765] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.789] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:19.791] Creating Prediction for predict set 'test' 
    DEBUG [06:18:19.796] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 75]}
    DEBUG [06:18:19.801] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.818] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:19.820] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:19.821] Applying learner 'regr.ranger' on task 'df.tr' (iter 14/50) 
    DEBUG [06:18:19.830] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:19.834] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.850] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:19.852] Creating Prediction for predict set 'test' 
    DEBUG [06:18:19.855] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 22, 27, 36, 38, 40, 45, 46, 50, 55, 56, 60, 65, 73]}
    DEBUG [06:18:19.859] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.870] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:19.872] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:19.873] Applying learner 'regr.ranger' on task 'df.tr' (iter 47/50) 
    DEBUG [06:18:19.883] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64..]}
    DEBUG [06:18:19.887] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.904] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:19.906] Creating Prediction for predict set 'test' 
    DEBUG [06:18:19.909] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70]}
    DEBUG [06:18:19.913] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.925] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:19.927] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:19.928] Applying learner 'regr.ranger' on task 'df.tr' (iter 12/50) 
    DEBUG [06:18:19.937] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61..]}
    DEBUG [06:18:19.941] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.959] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:19.961] Creating Prediction for predict set 'test' 
    DEBUG [06:18:19.964] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70]}
    DEBUG [06:18:19.967] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:19.979] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:19.980] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:19.982] Applying learner 'regr.ranger' on task 'df.tr' (iter 21/50) 
    DEBUG [06:18:19.991] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66..]}
    DEBUG [06:18:19.994] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:20.009] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:20.010] Creating Prediction for predict set 'test' 
    DEBUG [06:18:20.013] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72]}
    DEBUG [06:18:20.017] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:20.028] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:20.030] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:20.031] Applying learner 'regr.ranger' on task 'df.tr' (iter 43/50) 
    DEBUG [06:18:20.041] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:20.045] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:20.060] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:20.062] Creating Prediction for predict set 'test' 
    DEBUG [06:18:20.065] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74]}
    DEBUG [06:18:20.068] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:20.080] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:20.081] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:20.082] Applying learner 'regr.ranger' on task 'df.tr' (iter 41/50) 
    DEBUG [06:18:20.092] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74..]}
    DEBUG [06:18:20.095] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:20.111] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:20.112] Creating Prediction for predict set 'test' 
    DEBUG [06:18:20.115] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76]}
    DEBUG [06:18:20.118] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:20.130] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:20.131] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:20.133] Applying learner 'regr.ranger' on task 'df.tr' (iter 49/50) 
    DEBUG [06:18:20.142] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:20.146] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:20.161] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:20.163] Creating Prediction for predict set 'test' 
    DEBUG [06:18:20.166] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 7, 14, 16, 25, 26, 27, 40, 44, 56, 67, 69, 75, 76]}
    DEBUG [06:18:20.169] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:20.181] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:20.182] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:20.183] Applying learner 'regr.ranger' on task 'df.tr' (iter 29/50) 
    DEBUG [06:18:20.193] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:20.197] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:20.213] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:20.215] Creating Prediction for predict set 'test' 
    DEBUG [06:18:20.220] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 18, 25, 28, 30, 39, 45, 48, 51, 54, 55, 61, 66, 71, 73]}
    DEBUG [06:18:20.223] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:20.235] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:20.237] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:20.238] Applying learner 'regr.ranger' on task 'df.tr' (iter 10/50) 
    DEBUG [06:18:20.248] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:20.251] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:20.268] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:20.270] Creating Prediction for predict set 'test' 
    DEBUG [06:18:20.273] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 17, 18, 19, 24, 26, 29, 31, 34, 36, 43, 45, 53, 66, 73]}
    DEBUG [06:18:20.276] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:20.288] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:20.289] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:20.290] Applying learner 'regr.ranger' on task 'df.tr' (iter 4/50) 
    DEBUG [06:18:20.300] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:20.303] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:20.318] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:20.320] Creating Prediction for predict set 'test' 
    DEBUG [06:18:20.324] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 3, 13, 20, 23, 25, 40, 46, 50, 53, 56, 57, 58, 63]}
    DEBUG [06:18:20.328] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:20.340] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:20.342] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:20.343] Applying learner 'regr.ranger' on task 'df.tr' (iter 9/50) 
    DEBUG [06:18:20.352] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:20.356] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:20.372] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:20.374] Creating Prediction for predict set 'test' 
    DEBUG [06:18:20.378] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 23, 27, 40, 42, 46, 49, 56, 58, 61, 64, 67, 70, 71, 72]}
    DEBUG [06:18:20.385] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:20.400] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:20.402] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:20.435] Finished benchmark 
    INFO  [06:18:22.304] Result of batch 6: 
    INFO  [06:18:22.308]  mtry sample.fraction num.trees importance regr.rmse      resample_result 
    INFO  [06:18:22.308]     3       0.9183301       378   impurity  44.73443 <ResampleResult[19]> 
    INFO  [06:18:22.317] Evaluating 1 configuration(s) 
    INFO  [06:18:22.359] Benchmark with 50 resampling iterations 
    DEBUG [06:18:22.360] Running benchmark() asynchronously with 50 iterations 
    INFO  [06:18:22.375] Applying learner 'regr.ranger' on task 'df.tr' (iter 13/50) 
    DEBUG [06:18:22.384] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:22.388] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.401] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:22.403] Creating Prediction for predict set 'test' 
    DEBUG [06:18:22.406] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 72]}
    DEBUG [06:18:22.409] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.420] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:22.422] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:22.423] Applying learner 'regr.ranger' on task 'df.tr' (iter 15/50) 
    DEBUG [06:18:22.432] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:22.436] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.448] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:22.450] Creating Prediction for predict set 'test' 
    DEBUG [06:18:22.453] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 19, 20, 30, 35, 42, 47, 54, 59, 64, 67, 71, 74, 75]}
    DEBUG [06:18:22.456] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.468] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:22.470] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:22.471] Applying learner 'regr.ranger' on task 'df.tr' (iter 50/50) 
    DEBUG [06:18:22.480] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:22.483] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.496] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:22.498] Creating Prediction for predict set 'test' 
    DEBUG [06:18:22.501] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 8, 21, 22, 30, 36, 41, 45, 49, 53, 54, 57, 68, 71]}
    DEBUG [06:18:22.504] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.516] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:22.517] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:22.518] Applying learner 'regr.ranger' on task 'df.tr' (iter 46/50) 
    DEBUG [06:18:22.529] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72..]}
    DEBUG [06:18:22.533] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.547] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:22.549] Creating Prediction for predict set 'test' 
    DEBUG [06:18:22.552] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73]}
    DEBUG [06:18:22.556] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.568] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:22.570] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:22.571] Applying learner 'regr.ranger' on task 'df.tr' (iter 30/50) 
    DEBUG [06:18:22.582] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:22.586] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.599] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:22.601] Creating Prediction for predict set 'test' 
    DEBUG [06:18:22.604] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 14, 22, 29, 34, 35, 37, 40, 43, 44, 59, 62, 64, 68, 72]}
    DEBUG [06:18:22.607] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.618] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:22.619] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:22.621] Applying learner 'regr.ranger' on task 'df.tr' (iter 18/50) 
    DEBUG [06:18:22.630] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:22.633] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.645] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:22.647] Creating Prediction for predict set 'test' 
    DEBUG [06:18:22.650] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75]}
    DEBUG [06:18:22.653] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.664] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:22.665] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:22.667] Applying learner 'regr.ranger' on task 'df.tr' (iter 44/50) 
    DEBUG [06:18:22.676] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:22.679] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.690] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:22.692] Creating Prediction for predict set 'test' 
    DEBUG [06:18:22.695] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 14, 17, 19, 24, 36, 41, 47, 51, 53, 56, 62, 63, 67, 71]}
    DEBUG [06:18:22.698] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.709] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:22.710] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:22.711] Applying learner 'regr.ranger' on task 'df.tr' (iter 39/50) 
    DEBUG [06:18:22.726] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:22.730] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.745] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:22.747] Creating Prediction for predict set 'test' 
    DEBUG [06:18:22.750] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 5, 14, 18, 19, 21, 28, 38, 48, 49, 51, 52, 59, 64, 65]}
    DEBUG [06:18:22.754] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.767] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:22.769] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:22.770] Applying learner 'regr.ranger' on task 'df.tr' (iter 26/50) 
    DEBUG [06:18:22.780] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75..]}
    DEBUG [06:18:22.784] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.796] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:22.798] Creating Prediction for predict set 'test' 
    DEBUG [06:18:22.801] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74]}
    DEBUG [06:18:22.805] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.818] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:22.819] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:22.821] Applying learner 'regr.ranger' on task 'df.tr' (iter 36/50) 
    DEBUG [06:18:22.831] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58..]}
    DEBUG [06:18:22.834] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.846] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:22.847] Creating Prediction for predict set 'test' 
    DEBUG [06:18:22.851] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75]}
    DEBUG [06:18:22.854] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.866] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:22.867] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:22.869] Applying learner 'regr.ranger' on task 'df.tr' (iter 41/50) 
    DEBUG [06:18:22.879] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74..]}
    DEBUG [06:18:22.883] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.895] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:22.897] Creating Prediction for predict set 'test' 
    DEBUG [06:18:22.900] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76]}
    DEBUG [06:18:22.904] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.916] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:22.918] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:22.919] Applying learner 'regr.ranger' on task 'df.tr' (iter 3/50) 
    DEBUG [06:18:22.929] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:22.933] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.945] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:22.947] Creating Prediction for predict set 'test' 
    DEBUG [06:18:22.950] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75, 76]}
    DEBUG [06:18:22.954] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:22.966] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:22.967] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:22.969] Applying learner 'regr.ranger' on task 'df.tr' (iter 25/50) 
    DEBUG [06:18:22.978] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:22.982] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.037] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:23.040] Creating Prediction for predict set 'test' 
    DEBUG [06:18:23.047] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [9, 11, 18, 21, 23, 27, 29, 43, 44, 45, 51, 55, 67, 73, 76]}
    DEBUG [06:18:23.054] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.068] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:23.070] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:23.072] Applying learner 'regr.ranger' on task 'df.tr' (iter 11/50) 
    DEBUG [06:18:23.088] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 7..]}
    DEBUG [06:18:23.094] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.108] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:23.110] Creating Prediction for predict set 'test' 
    DEBUG [06:18:23.114] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76]}
    DEBUG [06:18:23.118] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.134] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:23.136] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:23.137] Applying learner 'regr.ranger' on task 'df.tr' (iter 47/50) 
    DEBUG [06:18:23.150] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64..]}
    DEBUG [06:18:23.155] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.167] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:23.169] Creating Prediction for predict set 'test' 
    DEBUG [06:18:23.172] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70]}
    DEBUG [06:18:23.176] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.187] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:23.189] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:23.190] Applying learner 'regr.ranger' on task 'df.tr' (iter 16/50) 
    DEBUG [06:18:23.199] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75..]}
    DEBUG [06:18:23.202] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.213] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:23.215] Creating Prediction for predict set 'test' 
    DEBUG [06:18:23.218] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72]}
    DEBUG [06:18:23.221] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.231] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:23.232] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:23.234] Applying learner 'regr.ranger' on task 'df.tr' (iter 37/50) 
    DEBUG [06:18:23.243] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57..]}
    DEBUG [06:18:23.247] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.259] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:23.260] Creating Prediction for predict set 'test' 
    DEBUG [06:18:23.263] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76]}
    DEBUG [06:18:23.266] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.277] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:23.279] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:23.280] Applying learner 'regr.ranger' on task 'df.tr' (iter 32/50) 
    DEBUG [06:18:23.289] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 6..]}
    DEBUG [06:18:23.292] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.304] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:23.305] Creating Prediction for predict set 'test' 
    DEBUG [06:18:23.309] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71]}
    DEBUG [06:18:23.312] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.322] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:23.324] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:23.325] Applying learner 'regr.ranger' on task 'df.tr' (iter 31/50) 
    DEBUG [06:18:23.334] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60..]}
    DEBUG [06:18:23.341] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.356] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:23.357] Creating Prediction for predict set 'test' 
    DEBUG [06:18:23.360] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73]}
    DEBUG [06:18:23.364] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.375] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:23.377] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:23.378] Applying learner 'regr.ranger' on task 'df.tr' (iter 5/50) 
    DEBUG [06:18:23.388] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:23.391] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.403] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:23.405] Creating Prediction for predict set 'test' 
    DEBUG [06:18:23.408] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 11, 14, 17, 27, 32, 41, 44, 45, 55, 60, 64, 65, 66, 69]}
    DEBUG [06:18:23.411] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.423] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:23.425] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:23.426] Applying learner 'regr.ranger' on task 'df.tr' (iter 2/50) 
    DEBUG [06:18:23.435] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:18:23.439] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.450] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:23.452] Creating Prediction for predict set 'test' 
    DEBUG [06:18:23.455] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68]}
    DEBUG [06:18:23.458] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.469] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:23.470] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:23.472] Applying learner 'regr.ranger' on task 'df.tr' (iter 22/50) 
    DEBUG [06:18:23.482] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57..]}
    DEBUG [06:18:23.486] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.498] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:23.500] Creating Prediction for predict set 'test' 
    DEBUG [06:18:23.503] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75]}
    DEBUG [06:18:23.506] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.517] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:23.518] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:23.519] Applying learner 'regr.ranger' on task 'df.tr' (iter 17/50) 
    DEBUG [06:18:23.529] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 7..]}
    DEBUG [06:18:23.533] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.544] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:23.545] Creating Prediction for predict set 'test' 
    DEBUG [06:18:23.548] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70]}
    DEBUG [06:18:23.552] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.563] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:23.565] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:23.566] Applying learner 'regr.ranger' on task 'df.tr' (iter 33/50) 
    DEBUG [06:18:23.576] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:23.579] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.590] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:23.592] Creating Prediction for predict set 'test' 
    DEBUG [06:18:23.595] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60, 76]}
    DEBUG [06:18:23.598] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.609] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:23.611] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:23.612] Applying learner 'regr.ranger' on task 'df.tr' (iter 19/50) 
    DEBUG [06:18:23.621] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:23.624] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.636] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:23.638] Creating Prediction for predict set 'test' 
    DEBUG [06:18:23.641] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 8, 11, 14, 18, 24, 28, 33, 34, 54, 56, 62, 64, 68, 76]}
    DEBUG [06:18:23.644] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.655] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:23.656] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:23.658] Applying learner 'regr.ranger' on task 'df.tr' (iter 49/50) 
    DEBUG [06:18:23.667] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:23.671] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.683] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:23.685] Creating Prediction for predict set 'test' 
    DEBUG [06:18:23.688] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 7, 14, 16, 25, 26, 27, 40, 44, 56, 67, 69, 75, 76]}
    DEBUG [06:18:23.692] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.703] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:23.704] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:23.706] Applying learner 'regr.ranger' on task 'df.tr' (iter 42/50) 
    DEBUG [06:18:23.715] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70..]}
    DEBUG [06:18:23.719] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.732] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:23.733] Creating Prediction for predict set 'test' 
    DEBUG [06:18:23.737] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66]}
    DEBUG [06:18:23.740] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.752] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:23.754] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:23.755] Applying learner 'regr.ranger' on task 'df.tr' (iter 43/50) 
    DEBUG [06:18:23.764] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:23.768] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.780] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:23.782] Creating Prediction for predict set 'test' 
    DEBUG [06:18:23.786] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74]}
    DEBUG [06:18:23.789] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.801] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:23.803] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:23.804] Applying learner 'regr.ranger' on task 'df.tr' (iter 29/50) 
    DEBUG [06:18:23.814] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:23.818] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.829] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:23.831] Creating Prediction for predict set 'test' 
    DEBUG [06:18:23.834] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 18, 25, 28, 30, 39, 45, 48, 51, 54, 55, 61, 66, 71, 73]}
    DEBUG [06:18:23.838] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.850] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:23.852] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:23.853] Applying learner 'regr.ranger' on task 'df.tr' (iter 38/50) 
    DEBUG [06:18:23.864] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:23.867] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.878] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:23.880] Creating Prediction for predict set 'test' 
    DEBUG [06:18:23.883] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58, 69]}
    DEBUG [06:18:23.886] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.898] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:23.900] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:23.901] Applying learner 'regr.ranger' on task 'df.tr' (iter 23/50) 
    DEBUG [06:18:23.911] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:23.915] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.926] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:23.928] Creating Prediction for predict set 'test' 
    DEBUG [06:18:23.936] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66]}
    DEBUG [06:18:23.940] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.955] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:23.956] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:23.958] Applying learner 'regr.ranger' on task 'df.tr' (iter 40/50) 
    DEBUG [06:18:23.969] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:23.972] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:23.984] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:23.985] Creating Prediction for predict set 'test' 
    DEBUG [06:18:23.988] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 8, 16, 17, 24, 25, 27, 29, 41, 44, 53, 54, 67, 72]}
    DEBUG [06:18:23.992] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.004] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:24.006] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:24.007] Applying learner 'regr.ranger' on task 'df.tr' (iter 27/50) 
    DEBUG [06:18:24.017] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67..]}
    DEBUG [06:18:24.021] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.033] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:24.035] Creating Prediction for predict set 'test' 
    DEBUG [06:18:24.038] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76]}
    DEBUG [06:18:24.042] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.081] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:24.084] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:24.085] Applying learner 'regr.ranger' on task 'df.tr' (iter 45/50) 
    DEBUG [06:18:24.099] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:24.104] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.117] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:24.120] Creating Prediction for predict set 'test' 
    DEBUG [06:18:24.124] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 8, 12, 21, 22, 25, 28, 33, 37, 38, 44, 46, 48, 57, 75]}
    DEBUG [06:18:24.128] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.145] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:24.147] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:24.148] Applying learner 'regr.ranger' on task 'df.tr' (iter 7/50) 
    DEBUG [06:18:24.160] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 6..]}
    DEBUG [06:18:24.165] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.178] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:24.181] Creating Prediction for predict set 'test' 
    DEBUG [06:18:24.184] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74]}
    DEBUG [06:18:24.189] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.201] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:24.203] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:24.204] Applying learner 'regr.ranger' on task 'df.tr' (iter 24/50) 
    DEBUG [06:18:24.213] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:24.217] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.228] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:24.230] Creating Prediction for predict set 'test' 
    DEBUG [06:18:24.233] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 13, 17, 25, 32, 35, 36, 37, 38, 50, 63, 65, 68, 69, 74]}
    DEBUG [06:18:24.236] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.248] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:24.249] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:24.251] Applying learner 'regr.ranger' on task 'df.tr' (iter 8/50) 
    DEBUG [06:18:24.260] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:24.263] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.275] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:24.276] Creating Prediction for predict set 'test' 
    DEBUG [06:18:24.279] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 75]}
    DEBUG [06:18:24.282] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.294] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:24.295] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:24.296] Applying learner 'regr.ranger' on task 'df.tr' (iter 35/50) 
    DEBUG [06:18:24.306] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:24.309] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.320] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:24.322] Creating Prediction for predict set 'test' 
    DEBUG [06:18:24.325] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 29, 32, 33, 35, 40, 43, 44, 45, 61, 62, 70, 74]}
    DEBUG [06:18:24.328] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.338] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:24.340] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:24.341] Applying learner 'regr.ranger' on task 'df.tr' (iter 48/50) 
    DEBUG [06:18:24.350] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:24.353] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.364] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:24.366] Creating Prediction for predict set 'test' 
    DEBUG [06:18:24.369] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72, 74]}
    DEBUG [06:18:24.372] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.382] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:24.384] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:24.385] Applying learner 'regr.ranger' on task 'df.tr' (iter 14/50) 
    DEBUG [06:18:24.395] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:24.398] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.410] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:24.411] Creating Prediction for predict set 'test' 
    DEBUG [06:18:24.414] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 22, 27, 36, 38, 40, 45, 46, 50, 55, 56, 60, 65, 73]}
    DEBUG [06:18:24.417] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.428] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:24.429] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:24.430] Applying learner 'regr.ranger' on task 'df.tr' (iter 28/50) 
    DEBUG [06:18:24.439] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:24.443] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.455] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:24.457] Creating Prediction for predict set 'test' 
    DEBUG [06:18:24.460] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75]}
    DEBUG [06:18:24.463] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.473] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:24.475] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:24.476] Applying learner 'regr.ranger' on task 'df.tr' (iter 21/50) 
    DEBUG [06:18:24.485] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66..]}
    DEBUG [06:18:24.488] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.499] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:24.501] Creating Prediction for predict set 'test' 
    DEBUG [06:18:24.504] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72]}
    DEBUG [06:18:24.507] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.517] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:24.519] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:24.520] Applying learner 'regr.ranger' on task 'df.tr' (iter 20/50) 
    DEBUG [06:18:24.537] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:24.541] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.556] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:24.558] Creating Prediction for predict set 'test' 
    DEBUG [06:18:24.562] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 16, 17, 32, 36, 37, 42, 43, 45, 46, 53, 61, 65, 69]}
    DEBUG [06:18:24.565] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.578] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:24.580] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:24.581] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [06:18:24.592] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:18:24.596] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.609] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:24.611] Creating Prediction for predict set 'test' 
    DEBUG [06:18:24.615] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [06:18:24.618] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.630] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:24.632] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:24.633] Applying learner 'regr.ranger' on task 'df.tr' (iter 34/50) 
    DEBUG [06:18:24.644] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:24.647] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.659] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:24.661] Creating Prediction for predict set 'test' 
    DEBUG [06:18:24.664] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 7, 12, 14, 18, 20, 41, 46, 47, 53, 57, 63, 64, 72, 75]}
    DEBUG [06:18:24.667] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.678] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:24.680] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:24.681] Applying learner 'regr.ranger' on task 'df.tr' (iter 6/50) 
    DEBUG [06:18:24.691] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 7..]}
    DEBUG [06:18:24.695] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.708] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:24.709] Creating Prediction for predict set 'test' 
    DEBUG [06:18:24.713] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76]}
    DEBUG [06:18:24.717] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.729] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:24.730] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:24.732] Applying learner 'regr.ranger' on task 'df.tr' (iter 12/50) 
    DEBUG [06:18:24.741] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61..]}
    DEBUG [06:18:24.745] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.757] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:24.759] Creating Prediction for predict set 'test' 
    DEBUG [06:18:24.763] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70]}
    DEBUG [06:18:24.766] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.778] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:24.779] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:24.780] Applying learner 'regr.ranger' on task 'df.tr' (iter 4/50) 
    DEBUG [06:18:24.790] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:24.794] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.805] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:24.807] Creating Prediction for predict set 'test' 
    DEBUG [06:18:24.810] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 3, 13, 20, 23, 25, 40, 46, 50, 53, 56, 57, 58, 63]}
    DEBUG [06:18:24.814] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.825] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:24.827] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:24.828] Applying learner 'regr.ranger' on task 'df.tr' (iter 9/50) 
    DEBUG [06:18:24.838] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:24.842] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.854] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:24.856] Creating Prediction for predict set 'test' 
    DEBUG [06:18:24.859] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 23, 27, 40, 42, 46, 49, 56, 58, 61, 64, 67, 70, 71, 72]}
    DEBUG [06:18:24.862] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.874] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:24.876] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:24.877] Applying learner 'regr.ranger' on task 'df.tr' (iter 10/50) 
    DEBUG [06:18:24.887] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:24.891] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.903] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:24.905] Creating Prediction for predict set 'test' 
    DEBUG [06:18:24.908] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 17, 18, 19, 24, 26, 29, 31, 34, 36, 43, 45, 53, 66, 73]}
    DEBUG [06:18:24.912] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:24.924] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:24.925] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:24.965] Finished benchmark 
    INFO  [06:18:26.758] Result of batch 7: 
    INFO  [06:18:26.762]  mtry sample.fraction num.trees importance regr.rmse      resample_result 
    INFO  [06:18:26.762]     4        0.549811       141   impurity  46.36304 <ResampleResult[19]> 
    INFO  [06:18:26.775] Evaluating 1 configuration(s) 
    INFO  [06:18:26.828] Benchmark with 50 resampling iterations 
    DEBUG [06:18:26.829] Running benchmark() asynchronously with 50 iterations 
    INFO  [06:18:26.845] Applying learner 'regr.ranger' on task 'df.tr' (iter 30/50) 
    DEBUG [06:18:26.856] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:26.860] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:26.882] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:26.883] Creating Prediction for predict set 'test' 
    DEBUG [06:18:26.887] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 14, 22, 29, 34, 35, 37, 40, 43, 44, 59, 62, 64, 68, 72]}
    DEBUG [06:18:26.891] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:26.904] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:26.906] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:26.907] Applying learner 'regr.ranger' on task 'df.tr' (iter 18/50) 
    DEBUG [06:18:26.917] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:26.921] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:26.938] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:26.941] Creating Prediction for predict set 'test' 
    DEBUG [06:18:26.945] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75]}
    DEBUG [06:18:26.949] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:26.963] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:26.965] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:26.966] Applying learner 'regr.ranger' on task 'df.tr' (iter 44/50) 
    DEBUG [06:18:26.976] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:26.979] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:26.994] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:26.996] Creating Prediction for predict set 'test' 
    DEBUG [06:18:27.000] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 14, 17, 19, 24, 36, 41, 47, 51, 53, 56, 62, 63, 67, 71]}
    DEBUG [06:18:27.004] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.018] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:27.020] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:27.021] Applying learner 'regr.ranger' on task 'df.tr' (iter 39/50) 
    DEBUG [06:18:27.030] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:27.034] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.051] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:27.053] Creating Prediction for predict set 'test' 
    DEBUG [06:18:27.056] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 5, 14, 18, 19, 21, 28, 38, 48, 49, 51, 52, 59, 64, 65]}
    DEBUG [06:18:27.060] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.074] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:27.076] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:27.077] Applying learner 'regr.ranger' on task 'df.tr' (iter 26/50) 
    DEBUG [06:18:27.087] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75..]}
    DEBUG [06:18:27.091] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.109] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:27.111] Creating Prediction for predict set 'test' 
    DEBUG [06:18:27.115] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74]}
    DEBUG [06:18:27.119] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.132] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:27.134] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:27.135] Applying learner 'regr.ranger' on task 'df.tr' (iter 36/50) 
    DEBUG [06:18:27.145] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58..]}
    DEBUG [06:18:27.149] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.165] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:27.167] Creating Prediction for predict set 'test' 
    DEBUG [06:18:27.171] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75]}
    DEBUG [06:18:27.174] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.187] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:27.189] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:27.190] Applying learner 'regr.ranger' on task 'df.tr' (iter 45/50) 
    DEBUG [06:18:27.233] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:27.240] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.261] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:27.264] Creating Prediction for predict set 'test' 
    DEBUG [06:18:27.269] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 8, 12, 21, 22, 25, 28, 33, 37, 38, 44, 46, 48, 57, 75]}
    DEBUG [06:18:27.274] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.290] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:27.292] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:27.294] Applying learner 'regr.ranger' on task 'df.tr' (iter 3/50) 
    DEBUG [06:18:27.306] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:27.310] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.331] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:27.333] Creating Prediction for predict set 'test' 
    DEBUG [06:18:27.337] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75, 76]}
    DEBUG [06:18:27.341] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.356] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:27.357] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:27.359] Applying learner 'regr.ranger' on task 'df.tr' (iter 25/50) 
    DEBUG [06:18:27.368] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:27.371] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.387] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:27.389] Creating Prediction for predict set 'test' 
    DEBUG [06:18:27.392] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [9, 11, 18, 21, 23, 27, 29, 43, 44, 45, 51, 55, 67, 73, 76]}
    DEBUG [06:18:27.395] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.407] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:27.409] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:27.410] Applying learner 'regr.ranger' on task 'df.tr' (iter 11/50) 
    DEBUG [06:18:27.419] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 7..]}
    DEBUG [06:18:27.423] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.437] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:27.439] Creating Prediction for predict set 'test' 
    DEBUG [06:18:27.442] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76]}
    DEBUG [06:18:27.445] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.458] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:27.459] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:27.460] Applying learner 'regr.ranger' on task 'df.tr' (iter 50/50) 
    DEBUG [06:18:27.470] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:27.473] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.490] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:27.492] Creating Prediction for predict set 'test' 
    DEBUG [06:18:27.495] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 8, 21, 22, 30, 36, 41, 45, 49, 53, 54, 57, 68, 71]}
    DEBUG [06:18:27.498] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.510] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:27.511] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:27.513] Applying learner 'regr.ranger' on task 'df.tr' (iter 16/50) 
    DEBUG [06:18:27.522] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75..]}
    DEBUG [06:18:27.526] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.545] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:27.547] Creating Prediction for predict set 'test' 
    DEBUG [06:18:27.550] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72]}
    DEBUG [06:18:27.553] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.566] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:27.568] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:27.569] Applying learner 'regr.ranger' on task 'df.tr' (iter 41/50) 
    DEBUG [06:18:27.578] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74..]}
    DEBUG [06:18:27.582] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.598] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:27.600] Creating Prediction for predict set 'test' 
    DEBUG [06:18:27.603] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76]}
    DEBUG [06:18:27.606] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.619] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:27.621] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:27.622] Applying learner 'regr.ranger' on task 'df.tr' (iter 32/50) 
    DEBUG [06:18:27.631] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 6..]}
    DEBUG [06:18:27.635] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.651] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:27.653] Creating Prediction for predict set 'test' 
    DEBUG [06:18:27.656] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71]}
    DEBUG [06:18:27.659] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.671] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:27.677] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:27.678] Applying learner 'regr.ranger' on task 'df.tr' (iter 31/50) 
    DEBUG [06:18:27.689] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60..]}
    DEBUG [06:18:27.693] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.711] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:27.713] Creating Prediction for predict set 'test' 
    DEBUG [06:18:27.716] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73]}
    DEBUG [06:18:27.719] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.732] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:27.734] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:27.735] Applying learner 'regr.ranger' on task 'df.tr' (iter 2/50) 
    DEBUG [06:18:27.745] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:18:27.749] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.767] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:27.769] Creating Prediction for predict set 'test' 
    DEBUG [06:18:27.773] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68]}
    DEBUG [06:18:27.777] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.790] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:27.791] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:27.792] Applying learner 'regr.ranger' on task 'df.tr' (iter 22/50) 
    DEBUG [06:18:27.802] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57..]}
    DEBUG [06:18:27.805] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.821] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:27.822] Creating Prediction for predict set 'test' 
    DEBUG [06:18:27.825] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75]}
    DEBUG [06:18:27.829] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.842] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:27.843] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:27.844] Applying learner 'regr.ranger' on task 'df.tr' (iter 17/50) 
    DEBUG [06:18:27.854] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 7..]}
    DEBUG [06:18:27.857] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.874] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:27.876] Creating Prediction for predict set 'test' 
    DEBUG [06:18:27.879] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70]}
    DEBUG [06:18:27.882] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.895] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:27.896] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:27.897] Applying learner 'regr.ranger' on task 'df.tr' (iter 5/50) 
    DEBUG [06:18:27.907] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:27.911] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.928] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:27.929] Creating Prediction for predict set 'test' 
    DEBUG [06:18:27.933] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 11, 14, 17, 27, 32, 41, 44, 45, 55, 60, 64, 65, 66, 69]}
    DEBUG [06:18:27.936] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.948] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:27.950] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:27.951] Applying learner 'regr.ranger' on task 'df.tr' (iter 19/50) 
    DEBUG [06:18:27.961] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:27.965] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:27.981] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:27.983] Creating Prediction for predict set 'test' 
    DEBUG [06:18:27.987] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 8, 11, 14, 18, 24, 28, 33, 34, 54, 56, 62, 64, 68, 76]}
    DEBUG [06:18:27.990] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.003] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:28.005] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:28.006] Applying learner 'regr.ranger' on task 'df.tr' (iter 15/50) 
    DEBUG [06:18:28.016] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:28.020] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.037] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:28.039] Creating Prediction for predict set 'test' 
    DEBUG [06:18:28.043] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 19, 20, 30, 35, 42, 47, 54, 59, 64, 67, 71, 74, 75]}
    DEBUG [06:18:28.047] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.060] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:28.062] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:28.063] Applying learner 'regr.ranger' on task 'df.tr' (iter 48/50) 
    DEBUG [06:18:28.072] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:28.076] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.095] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:28.097] Creating Prediction for predict set 'test' 
    DEBUG [06:18:28.102] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72, 74]}
    DEBUG [06:18:28.106] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.118] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:28.120] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:28.121] Applying learner 'regr.ranger' on task 'df.tr' (iter 43/50) 
    DEBUG [06:18:28.131] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:28.134] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.152] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:28.154] Creating Prediction for predict set 'test' 
    DEBUG [06:18:28.158] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74]}
    DEBUG [06:18:28.162] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.175] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:28.177] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:28.178] Applying learner 'regr.ranger' on task 'df.tr' (iter 27/50) 
    DEBUG [06:18:28.187] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67..]}
    DEBUG [06:18:28.191] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.210] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:28.212] Creating Prediction for predict set 'test' 
    DEBUG [06:18:28.216] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76]}
    DEBUG [06:18:28.219] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.232] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:28.233] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:28.234] Applying learner 'regr.ranger' on task 'df.tr' (iter 34/50) 
    DEBUG [06:18:28.244] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:28.248] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.263] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:28.265] Creating Prediction for predict set 'test' 
    DEBUG [06:18:28.268] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 7, 12, 14, 18, 20, 41, 46, 47, 53, 57, 63, 64, 72, 75]}
    DEBUG [06:18:28.271] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.285] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:28.287] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:28.288] Applying learner 'regr.ranger' on task 'df.tr' (iter 42/50) 
    DEBUG [06:18:28.298] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70..]}
    DEBUG [06:18:28.318] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.338] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:28.341] Creating Prediction for predict set 'test' 
    DEBUG [06:18:28.346] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66]}
    DEBUG [06:18:28.350] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.365] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:28.367] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:28.368] Applying learner 'regr.ranger' on task 'df.tr' (iter 40/50) 
    DEBUG [06:18:28.383] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:28.387] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.406] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:28.408] Creating Prediction for predict set 'test' 
    DEBUG [06:18:28.411] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 8, 16, 17, 24, 25, 27, 29, 41, 44, 53, 54, 67, 72]}
    DEBUG [06:18:28.416] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.430] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:28.432] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:28.434] Applying learner 'regr.ranger' on task 'df.tr' (iter 28/50) 
    DEBUG [06:18:28.445] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:28.448] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.463] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:28.465] Creating Prediction for predict set 'test' 
    DEBUG [06:18:28.468] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75]}
    DEBUG [06:18:28.471] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.484] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:28.485] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:28.486] Applying learner 'regr.ranger' on task 'df.tr' (iter 35/50) 
    DEBUG [06:18:28.498] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:28.502] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.531] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:28.534] Creating Prediction for predict set 'test' 
    DEBUG [06:18:28.539] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 29, 32, 33, 35, 40, 43, 44, 45, 61, 62, 70, 74]}
    DEBUG [06:18:28.545] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.561] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:28.562] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:28.564] Applying learner 'regr.ranger' on task 'df.tr' (iter 37/50) 
    DEBUG [06:18:28.573] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57..]}
    DEBUG [06:18:28.577] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.592] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:28.594] Creating Prediction for predict set 'test' 
    DEBUG [06:18:28.596] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76]}
    DEBUG [06:18:28.600] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.612] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:28.613] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:28.615] Applying learner 'regr.ranger' on task 'df.tr' (iter 21/50) 
    DEBUG [06:18:28.624] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66..]}
    DEBUG [06:18:28.627] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.648] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:28.650] Creating Prediction for predict set 'test' 
    DEBUG [06:18:28.653] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72]}
    DEBUG [06:18:28.657] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.673] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:28.675] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:28.676] Applying learner 'regr.ranger' on task 'df.tr' (iter 7/50) 
    DEBUG [06:18:28.685] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 6..]}
    DEBUG [06:18:28.689] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.705] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:28.707] Creating Prediction for predict set 'test' 
    DEBUG [06:18:28.711] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74]}
    DEBUG [06:18:28.714] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.726] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:28.727] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:28.729] Applying learner 'regr.ranger' on task 'df.tr' (iter 46/50) 
    DEBUG [06:18:28.738] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72..]}
    DEBUG [06:18:28.742] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.759] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:28.761] Creating Prediction for predict set 'test' 
    DEBUG [06:18:28.764] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73]}
    DEBUG [06:18:28.767] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.780] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:28.781] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:28.782] Applying learner 'regr.ranger' on task 'df.tr' (iter 47/50) 
    DEBUG [06:18:28.791] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64..]}
    DEBUG [06:18:28.795] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.811] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:28.812] Creating Prediction for predict set 'test' 
    DEBUG [06:18:28.815] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70]}
    DEBUG [06:18:28.818] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.830] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:28.832] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:28.833] Applying learner 'regr.ranger' on task 'df.tr' (iter 8/50) 
    DEBUG [06:18:28.842] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:28.846] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.860] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:28.861] Creating Prediction for predict set 'test' 
    DEBUG [06:18:28.864] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 75]}
    DEBUG [06:18:28.867] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.879] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:28.881] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:28.882] Applying learner 'regr.ranger' on task 'df.tr' (iter 29/50) 
    DEBUG [06:18:28.891] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:28.895] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.910] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:28.911] Creating Prediction for predict set 'test' 
    DEBUG [06:18:28.914] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 18, 25, 28, 30, 39, 45, 48, 51, 54, 55, 61, 66, 71, 73]}
    DEBUG [06:18:28.917] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.930] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:28.932] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:28.934] Applying learner 'regr.ranger' on task 'df.tr' (iter 24/50) 
    DEBUG [06:18:28.944] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:28.948] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.966] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:28.968] Creating Prediction for predict set 'test' 
    DEBUG [06:18:28.971] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 13, 17, 25, 32, 35, 36, 37, 38, 50, 63, 65, 68, 69, 74]}
    DEBUG [06:18:28.975] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:28.989] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:28.990] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:28.992] Applying learner 'regr.ranger' on task 'df.tr' (iter 33/50) 
    DEBUG [06:18:29.001] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:29.005] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.019] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:29.021] Creating Prediction for predict set 'test' 
    DEBUG [06:18:29.024] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60, 76]}
    DEBUG [06:18:29.027] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.039] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:29.041] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:29.042] Applying learner 'regr.ranger' on task 'df.tr' (iter 20/50) 
    DEBUG [06:18:29.052] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:29.055] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.070] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:29.072] Creating Prediction for predict set 'test' 
    DEBUG [06:18:29.075] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 16, 17, 32, 36, 37, 42, 43, 45, 46, 53, 61, 65, 69]}
    DEBUG [06:18:29.078] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.090] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:29.092] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:29.093] Applying learner 'regr.ranger' on task 'df.tr' (iter 49/50) 
    DEBUG [06:18:29.102] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:29.106] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.121] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:29.123] Creating Prediction for predict set 'test' 
    DEBUG [06:18:29.126] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 7, 14, 16, 25, 26, 27, 40, 44, 56, 67, 69, 75, 76]}
    DEBUG [06:18:29.129] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.141] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:29.143] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:29.144] Applying learner 'regr.ranger' on task 'df.tr' (iter 23/50) 
    DEBUG [06:18:29.154] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:29.157] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.183] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:29.185] Creating Prediction for predict set 'test' 
    DEBUG [06:18:29.188] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66]}
    DEBUG [06:18:29.192] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.206] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:29.207] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:29.208] Applying learner 'regr.ranger' on task 'df.tr' (iter 9/50) 
    DEBUG [06:18:29.219] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:29.223] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.239] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:29.240] Creating Prediction for predict set 'test' 
    DEBUG [06:18:29.243] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 23, 27, 40, 42, 46, 49, 56, 58, 61, 64, 67, 70, 71, 72]}
    DEBUG [06:18:29.247] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.261] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:29.263] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:29.264] Applying learner 'regr.ranger' on task 'df.tr' (iter 10/50) 
    DEBUG [06:18:29.274] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:29.278] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.296] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:29.298] Creating Prediction for predict set 'test' 
    DEBUG [06:18:29.301] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 17, 18, 19, 24, 26, 29, 31, 34, 36, 43, 45, 53, 66, 73]}
    DEBUG [06:18:29.305] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.318] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:29.319] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:29.320] Applying learner 'regr.ranger' on task 'df.tr' (iter 6/50) 
    DEBUG [06:18:29.331] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 7..]}
    DEBUG [06:18:29.335] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.351] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:29.353] Creating Prediction for predict set 'test' 
    DEBUG [06:18:29.356] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76]}
    DEBUG [06:18:29.360] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.373] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:29.375] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:29.376] Applying learner 'regr.ranger' on task 'df.tr' (iter 14/50) 
    DEBUG [06:18:29.386] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:29.390] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.442] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:29.445] Creating Prediction for predict set 'test' 
    DEBUG [06:18:29.451] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 22, 27, 36, 38, 40, 45, 46, 50, 55, 56, 60, 65, 73]}
    DEBUG [06:18:29.457] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.478] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:29.480] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:29.482] Applying learner 'regr.ranger' on task 'df.tr' (iter 13/50) 
    DEBUG [06:18:29.502] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:29.508] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.527] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:29.529] Creating Prediction for predict set 'test' 
    DEBUG [06:18:29.532] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 72]}
    DEBUG [06:18:29.537] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.556] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:29.558] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:29.559] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [06:18:29.573] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:18:29.577] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.593] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:29.594] Creating Prediction for predict set 'test' 
    DEBUG [06:18:29.597] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [06:18:29.601] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.613] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:29.614] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:29.615] Applying learner 'regr.ranger' on task 'df.tr' (iter 4/50) 
    DEBUG [06:18:29.624] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:29.628] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.652] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:29.654] Creating Prediction for predict set 'test' 
    DEBUG [06:18:29.658] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 3, 13, 20, 23, 25, 40, 46, 50, 53, 56, 57, 58, 63]}
    DEBUG [06:18:29.662] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.677] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:29.679] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:29.680] Applying learner 'regr.ranger' on task 'df.tr' (iter 38/50) 
    DEBUG [06:18:29.692] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:29.695] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.714] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:29.716] Creating Prediction for predict set 'test' 
    DEBUG [06:18:29.720] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58, 69]}
    DEBUG [06:18:29.724] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.736] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:29.738] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:29.739] Applying learner 'regr.ranger' on task 'df.tr' (iter 12/50) 
    DEBUG [06:18:29.749] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61..]}
    DEBUG [06:18:29.752] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.767] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:29.769] Creating Prediction for predict set 'test' 
    DEBUG [06:18:29.772] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70]}
    DEBUG [06:18:29.775] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:29.788] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:29.789] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:29.834] Finished benchmark 
    INFO  [06:18:31.610] Result of batch 8: 
    INFO  [06:18:31.613]  mtry sample.fraction num.trees importance regr.rmse      resample_result 
    INFO  [06:18:31.613]     4       0.5538044       401   impurity   46.3686 <ResampleResult[19]> 
    INFO  [06:18:31.620] Evaluating 1 configuration(s) 
    INFO  [06:18:31.653] Benchmark with 50 resampling iterations 
    DEBUG [06:18:31.654] Running benchmark() asynchronously with 50 iterations 
    INFO  [06:18:31.667] Applying learner 'regr.ranger' on task 'df.tr' (iter 26/50) 
    DEBUG [06:18:31.676] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75..]}
    DEBUG [06:18:31.679] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:31.697] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:31.699] Creating Prediction for predict set 'test' 
    DEBUG [06:18:31.702] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74]}
    DEBUG [06:18:31.704] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:31.716] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:31.718] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:31.719] Applying learner 'regr.ranger' on task 'df.tr' (iter 36/50) 
    DEBUG [06:18:31.726] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58..]}
    DEBUG [06:18:31.753] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:31.770] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:31.772] Creating Prediction for predict set 'test' 
    DEBUG [06:18:31.776] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75]}
    DEBUG [06:18:31.780] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:31.795] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:31.797] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:31.798] Applying learner 'regr.ranger' on task 'df.tr' (iter 49/50) 
    DEBUG [06:18:31.810] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:31.814] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:31.834] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:31.836] Creating Prediction for predict set 'test' 
    DEBUG [06:18:31.840] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 7, 14, 16, 25, 26, 27, 40, 44, 56, 67, 69, 75, 76]}
    DEBUG [06:18:31.844] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:31.870] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:31.873] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:31.876] Applying learner 'regr.ranger' on task 'df.tr' (iter 3/50) 
    DEBUG [06:18:31.897] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:31.905] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:31.931] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:31.933] Creating Prediction for predict set 'test' 
    DEBUG [06:18:31.940] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75, 76]}
    DEBUG [06:18:31.945] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:31.962] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:31.964] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:31.965] Applying learner 'regr.ranger' on task 'df.tr' (iter 25/50) 
    DEBUG [06:18:31.979] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:31.984] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.003] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:32.005] Creating Prediction for predict set 'test' 
    DEBUG [06:18:32.009] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [9, 11, 18, 21, 23, 27, 29, 43, 44, 45, 51, 55, 67, 73, 76]}
    DEBUG [06:18:32.014] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.029] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:32.031] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:32.032] Applying learner 'regr.ranger' on task 'df.tr' (iter 11/50) 
    DEBUG [06:18:32.044] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 7..]}
    DEBUG [06:18:32.049] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.074] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:32.077] Creating Prediction for predict set 'test' 
    DEBUG [06:18:32.082] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76]}
    DEBUG [06:18:32.086] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.101] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:32.103] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:32.105] Applying learner 'regr.ranger' on task 'df.tr' (iter 30/50) 
    DEBUG [06:18:32.115] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:32.118] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.136] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:32.138] Creating Prediction for predict set 'test' 
    DEBUG [06:18:32.142] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 14, 22, 29, 34, 35, 37, 40, 43, 44, 59, 62, 64, 68, 72]}
    DEBUG [06:18:32.146] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.159] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:32.160] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:32.161] Applying learner 'regr.ranger' on task 'df.tr' (iter 16/50) 
    DEBUG [06:18:32.171] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75..]}
    DEBUG [06:18:32.174] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.190] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:32.192] Creating Prediction for predict set 'test' 
    DEBUG [06:18:32.195] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72]}
    DEBUG [06:18:32.198] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.210] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:32.212] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:32.213] Applying learner 'regr.ranger' on task 'df.tr' (iter 45/50) 
    DEBUG [06:18:32.223] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:32.226] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.245] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:32.247] Creating Prediction for predict set 'test' 
    DEBUG [06:18:32.251] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 8, 12, 21, 22, 25, 28, 33, 37, 38, 44, 46, 48, 57, 75]}
    DEBUG [06:18:32.254] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.266] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:32.268] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:32.269] Applying learner 'regr.ranger' on task 'df.tr' (iter 38/50) 
    DEBUG [06:18:32.279] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:32.282] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.301] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:32.302] Creating Prediction for predict set 'test' 
    DEBUG [06:18:32.305] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58, 69]}
    DEBUG [06:18:32.308] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.321] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:32.322] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:32.323] Applying learner 'regr.ranger' on task 'df.tr' (iter 32/50) 
    DEBUG [06:18:32.332] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 6..]}
    DEBUG [06:18:32.336] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.352] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:32.353] Creating Prediction for predict set 'test' 
    DEBUG [06:18:32.356] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71]}
    DEBUG [06:18:32.360] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.372] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:32.374] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:32.375] Applying learner 'regr.ranger' on task 'df.tr' (iter 31/50) 
    DEBUG [06:18:32.385] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60..]}
    DEBUG [06:18:32.388] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.404] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:32.406] Creating Prediction for predict set 'test' 
    DEBUG [06:18:32.409] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73]}
    DEBUG [06:18:32.412] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.425] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:32.426] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:32.427] Applying learner 'regr.ranger' on task 'df.tr' (iter 37/50) 
    DEBUG [06:18:32.437] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57..]}
    DEBUG [06:18:32.441] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.458] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:32.460] Creating Prediction for predict set 'test' 
    DEBUG [06:18:32.464] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76]}
    DEBUG [06:18:32.467] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.479] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:32.481] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:32.482] Applying learner 'regr.ranger' on task 'df.tr' (iter 2/50) 
    DEBUG [06:18:32.492] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:18:32.495] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.511] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:32.512] Creating Prediction for predict set 'test' 
    DEBUG [06:18:32.515] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68]}
    DEBUG [06:18:32.519] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.531] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:32.533] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:32.534] Applying learner 'regr.ranger' on task 'df.tr' (iter 22/50) 
    DEBUG [06:18:32.543] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57..]}
    DEBUG [06:18:32.547] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.562] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:32.564] Creating Prediction for predict set 'test' 
    DEBUG [06:18:32.567] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75]}
    DEBUG [06:18:32.570] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.583] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:32.584] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:32.585] Applying learner 'regr.ranger' on task 'df.tr' (iter 17/50) 
    DEBUG [06:18:32.595] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 7..]}
    DEBUG [06:18:32.599] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.617] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:32.618] Creating Prediction for predict set 'test' 
    DEBUG [06:18:32.622] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70]}
    DEBUG [06:18:32.625] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.637] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:32.639] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:32.640] Applying learner 'regr.ranger' on task 'df.tr' (iter 5/50) 
    DEBUG [06:18:32.650] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:32.653] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.674] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:32.676] Creating Prediction for predict set 'test' 
    DEBUG [06:18:32.679] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 11, 14, 17, 27, 32, 41, 44, 45, 55, 60, 64, 65, 66, 69]}
    DEBUG [06:18:32.683] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.700] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:32.701] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:32.703] Applying learner 'regr.ranger' on task 'df.tr' (iter 15/50) 
    DEBUG [06:18:32.713] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:32.717] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.738] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:32.740] Creating Prediction for predict set 'test' 
    DEBUG [06:18:32.744] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 19, 20, 30, 35, 42, 47, 54, 59, 64, 67, 71, 74, 75]}
    DEBUG [06:18:32.748] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.761] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:32.763] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:32.764] Applying learner 'regr.ranger' on task 'df.tr' (iter 19/50) 
    DEBUG [06:18:32.774] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:32.778] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.795] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:32.797] Creating Prediction for predict set 'test' 
    DEBUG [06:18:32.800] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 8, 11, 14, 18, 24, 28, 33, 34, 54, 56, 62, 64, 68, 76]}
    DEBUG [06:18:32.804] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.817] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:32.819] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:32.820] Applying learner 'regr.ranger' on task 'df.tr' (iter 44/50) 
    DEBUG [06:18:32.830] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:32.834] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.851] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:32.853] Creating Prediction for predict set 'test' 
    DEBUG [06:18:32.857] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 14, 17, 19, 24, 36, 41, 47, 51, 53, 56, 62, 63, 67, 71]}
    DEBUG [06:18:32.860] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.873] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:32.875] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:32.876] Applying learner 'regr.ranger' on task 'df.tr' (iter 47/50) 
    DEBUG [06:18:32.886] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64..]}
    DEBUG [06:18:32.890] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.909] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:32.911] Creating Prediction for predict set 'test' 
    DEBUG [06:18:32.915] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70]}
    DEBUG [06:18:32.919] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.932] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:32.933] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:32.935] Applying learner 'regr.ranger' on task 'df.tr' (iter 27/50) 
    DEBUG [06:18:32.945] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67..]}
    DEBUG [06:18:32.948] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.967] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:32.969] Creating Prediction for predict set 'test' 
    DEBUG [06:18:32.973] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76]}
    DEBUG [06:18:32.976] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:32.989] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:32.991] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:32.992] Applying learner 'regr.ranger' on task 'df.tr' (iter 48/50) 
    DEBUG [06:18:33.003] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:33.006] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.039] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:33.042] Creating Prediction for predict set 'test' 
    DEBUG [06:18:33.046] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72, 74]}
    DEBUG [06:18:33.049] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.065] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:33.067] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:33.068] Applying learner 'regr.ranger' on task 'df.tr' (iter 46/50) 
    DEBUG [06:18:33.079] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72..]}
    DEBUG [06:18:33.083] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.102] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:33.105] Creating Prediction for predict set 'test' 
    DEBUG [06:18:33.108] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73]}
    DEBUG [06:18:33.112] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.125] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:33.167] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:33.168] Applying learner 'regr.ranger' on task 'df.tr' (iter 33/50) 
    DEBUG [06:18:33.184] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:33.191] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.212] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:33.215] Creating Prediction for predict set 'test' 
    DEBUG [06:18:33.219] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60, 76]}
    DEBUG [06:18:33.224] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.242] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:33.244] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:33.245] Applying learner 'regr.ranger' on task 'df.tr' (iter 39/50) 
    DEBUG [06:18:33.258] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:33.263] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.281] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:33.283] Creating Prediction for predict set 'test' 
    DEBUG [06:18:33.287] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 5, 14, 18, 19, 21, 28, 38, 48, 49, 51, 52, 59, 64, 65]}
    DEBUG [06:18:33.291] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.306] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:33.308] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:33.310] Applying learner 'regr.ranger' on task 'df.tr' (iter 41/50) 
    DEBUG [06:18:33.322] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74..]}
    DEBUG [06:18:33.326] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.347] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:33.349] Creating Prediction for predict set 'test' 
    DEBUG [06:18:33.353] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76]}
    DEBUG [06:18:33.357] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.371] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:33.373] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:33.375] Applying learner 'regr.ranger' on task 'df.tr' (iter 34/50) 
    DEBUG [06:18:33.387] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:33.391] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.409] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:33.411] Creating Prediction for predict set 'test' 
    DEBUG [06:18:33.415] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 7, 12, 14, 18, 20, 41, 46, 47, 53, 57, 63, 64, 72, 75]}
    DEBUG [06:18:33.419] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.434] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:33.436] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:33.437] Applying learner 'regr.ranger' on task 'df.tr' (iter 23/50) 
    DEBUG [06:18:33.448] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:33.453] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.471] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:33.473] Creating Prediction for predict set 'test' 
    DEBUG [06:18:33.477] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66]}
    DEBUG [06:18:33.481] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.493] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:33.495] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:33.496] Applying learner 'regr.ranger' on task 'df.tr' (iter 7/50) 
    DEBUG [06:18:33.512] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 6..]}
    DEBUG [06:18:33.515] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.537] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:33.539] Creating Prediction for predict set 'test' 
    DEBUG [06:18:33.542] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74]}
    DEBUG [06:18:33.546] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.559] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:33.561] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:33.562] Applying learner 'regr.ranger' on task 'df.tr' (iter 24/50) 
    DEBUG [06:18:33.572] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:33.575] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.594] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:33.596] Creating Prediction for predict set 'test' 
    DEBUG [06:18:33.600] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 13, 17, 25, 32, 35, 36, 37, 38, 50, 63, 65, 68, 69, 74]}
    DEBUG [06:18:33.604] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.618] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:33.620] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:33.621] Applying learner 'regr.ranger' on task 'df.tr' (iter 43/50) 
    DEBUG [06:18:33.632] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:33.636] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.653] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:33.655] Creating Prediction for predict set 'test' 
    DEBUG [06:18:33.658] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74]}
    DEBUG [06:18:33.662] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.676] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:33.678] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:33.679] Applying learner 'regr.ranger' on task 'df.tr' (iter 8/50) 
    DEBUG [06:18:33.690] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:33.694] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.712] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:33.714] Creating Prediction for predict set 'test' 
    DEBUG [06:18:33.718] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 75]}
    DEBUG [06:18:33.722] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.736] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:33.737] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:33.739] Applying learner 'regr.ranger' on task 'df.tr' (iter 9/50) 
    DEBUG [06:18:33.750] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:33.754] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.778] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:33.781] Creating Prediction for predict set 'test' 
    DEBUG [06:18:33.787] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 23, 27, 40, 42, 46, 49, 56, 58, 61, 64, 67, 70, 71, 72]}
    DEBUG [06:18:33.791] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.806] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:33.808] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:33.809] Applying learner 'regr.ranger' on task 'df.tr' (iter 29/50) 
    DEBUG [06:18:33.820] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:33.824] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.845] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:33.847] Creating Prediction for predict set 'test' 
    DEBUG [06:18:33.851] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 18, 25, 28, 30, 39, 45, 48, 51, 54, 55, 61, 66, 71, 73]}
    DEBUG [06:18:33.855] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.870] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:33.872] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:33.873] Applying learner 'regr.ranger' on task 'df.tr' (iter 14/50) 
    DEBUG [06:18:33.884] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:33.888] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.909] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:33.912] Creating Prediction for predict set 'test' 
    DEBUG [06:18:33.916] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 22, 27, 36, 38, 40, 45, 46, 50, 55, 56, 60, 65, 73]}
    DEBUG [06:18:33.920] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.934] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:33.935] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:33.937] Applying learner 'regr.ranger' on task 'df.tr' (iter 50/50) 
    DEBUG [06:18:33.946] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:33.950] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.967] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:33.969] Creating Prediction for predict set 'test' 
    DEBUG [06:18:33.973] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 8, 21, 22, 30, 36, 41, 45, 49, 53, 54, 57, 68, 71]}
    DEBUG [06:18:33.980] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:33.998] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:34.000] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:34.001] Applying learner 'regr.ranger' on task 'df.tr' (iter 40/50) 
    DEBUG [06:18:34.013] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:34.017] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.036] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:34.038] Creating Prediction for predict set 'test' 
    DEBUG [06:18:34.042] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 8, 16, 17, 24, 25, 27, 29, 41, 44, 53, 54, 67, 72]}
    DEBUG [06:18:34.046] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.061] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:34.063] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:34.064] Applying learner 'regr.ranger' on task 'df.tr' (iter 28/50) 
    DEBUG [06:18:34.075] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:34.079] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.098] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:34.100] Creating Prediction for predict set 'test' 
    DEBUG [06:18:34.104] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75]}
    DEBUG [06:18:34.107] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.121] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:34.123] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:34.124] Applying learner 'regr.ranger' on task 'df.tr' (iter 21/50) 
    DEBUG [06:18:34.136] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66..]}
    DEBUG [06:18:34.140] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.159] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:34.161] Creating Prediction for predict set 'test' 
    DEBUG [06:18:34.165] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72]}
    DEBUG [06:18:34.169] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.183] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:34.185] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:34.186] Applying learner 'regr.ranger' on task 'df.tr' (iter 18/50) 
    DEBUG [06:18:34.198] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:34.202] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.222] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:34.224] Creating Prediction for predict set 'test' 
    DEBUG [06:18:34.227] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75]}
    DEBUG [06:18:34.231] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.244] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:34.246] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:34.247] Applying learner 'regr.ranger' on task 'df.tr' (iter 4/50) 
    DEBUG [06:18:34.257] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:34.261] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.279] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:34.281] Creating Prediction for predict set 'test' 
    DEBUG [06:18:34.285] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 3, 13, 20, 23, 25, 40, 46, 50, 53, 56, 57, 58, 63]}
    DEBUG [06:18:34.289] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.303] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:34.305] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:34.306] Applying learner 'regr.ranger' on task 'df.tr' (iter 13/50) 
    DEBUG [06:18:34.316] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:34.320] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.339] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:34.341] Creating Prediction for predict set 'test' 
    DEBUG [06:18:34.347] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 72]}
    DEBUG [06:18:34.351] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.365] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:34.367] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:34.368] Applying learner 'regr.ranger' on task 'df.tr' (iter 42/50) 
    DEBUG [06:18:34.378] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70..]}
    DEBUG [06:18:34.382] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.404] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:34.407] Creating Prediction for predict set 'test' 
    DEBUG [06:18:34.410] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66]}
    DEBUG [06:18:34.414] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.429] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:34.431] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:34.432] Applying learner 'regr.ranger' on task 'df.tr' (iter 12/50) 
    DEBUG [06:18:34.443] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61..]}
    DEBUG [06:18:34.447] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.465] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:34.468] Creating Prediction for predict set 'test' 
    DEBUG [06:18:34.471] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70]}
    DEBUG [06:18:34.475] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.488] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:34.490] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:34.491] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [06:18:34.501] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:18:34.505] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.524] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:34.526] Creating Prediction for predict set 'test' 
    DEBUG [06:18:34.529] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [06:18:34.533] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.548] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:34.550] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:34.551] Applying learner 'regr.ranger' on task 'df.tr' (iter 10/50) 
    DEBUG [06:18:34.561] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:34.565] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.620] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:34.623] Creating Prediction for predict set 'test' 
    DEBUG [06:18:34.628] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 17, 18, 19, 24, 26, 29, 31, 34, 36, 43, 45, 53, 66, 73]}
    DEBUG [06:18:34.633] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.651] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:34.653] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:34.655] Applying learner 'regr.ranger' on task 'df.tr' (iter 6/50) 
    DEBUG [06:18:34.670] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 7..]}
    DEBUG [06:18:34.676] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.698] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:34.701] Creating Prediction for predict set 'test' 
    DEBUG [06:18:34.705] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76]}
    DEBUG [06:18:34.709] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.725] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:34.726] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:34.728] Applying learner 'regr.ranger' on task 'df.tr' (iter 35/50) 
    DEBUG [06:18:34.740] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:34.745] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.766] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:34.768] Creating Prediction for predict set 'test' 
    DEBUG [06:18:34.772] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 29, 32, 33, 35, 40, 43, 44, 45, 61, 62, 70, 74]}
    DEBUG [06:18:34.776] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.791] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:34.793] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:34.795] Applying learner 'regr.ranger' on task 'df.tr' (iter 20/50) 
    DEBUG [06:18:34.807] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:34.811] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.832] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:34.835] Creating Prediction for predict set 'test' 
    DEBUG [06:18:34.839] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 16, 17, 32, 36, 37, 42, 43, 45, 46, 53, 61, 65, 69]}
    DEBUG [06:18:34.843] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:34.858] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:34.860] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:34.906] Finished benchmark 
    INFO  [06:18:36.835] Result of batch 9: 
    INFO  [06:18:36.839]  mtry sample.fraction num.trees importance regr.rmse      resample_result 
    INFO  [06:18:36.839]     5       0.6688764       463   impurity  43.32544 <ResampleResult[19]> 
    INFO  [06:18:36.848] Evaluating 1 configuration(s) 
    INFO  [06:18:36.891] Benchmark with 50 resampling iterations 
    DEBUG [06:18:36.893] Running benchmark() asynchronously with 50 iterations 
    INFO  [06:18:36.913] Applying learner 'regr.ranger' on task 'df.tr' (iter 25/50) 
    DEBUG [06:18:36.925] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:36.929] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:36.942] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:36.944] Creating Prediction for predict set 'test' 
    DEBUG [06:18:36.948] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [9, 11, 18, 21, 23, 27, 29, 43, 44, 45, 51, 55, 67, 73, 76]}
    DEBUG [06:18:36.952] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:36.963] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:36.965] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:36.966] Applying learner 'regr.ranger' on task 'df.tr' (iter 11/50) 
    DEBUG [06:18:36.977] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 7..]}
    DEBUG [06:18:36.981] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:36.993] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:36.995] Creating Prediction for predict set 'test' 
    DEBUG [06:18:36.998] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76]}
    DEBUG [06:18:37.002] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.013] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:37.015] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:37.016] Applying learner 'regr.ranger' on task 'df.tr' (iter 30/50) 
    DEBUG [06:18:37.027] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:37.030] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.041] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:37.043] Creating Prediction for predict set 'test' 
    DEBUG [06:18:37.046] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 14, 22, 29, 34, 35, 37, 40, 43, 44, 59, 62, 64, 68, 72]}
    DEBUG [06:18:37.050] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.062] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:37.063] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:37.065] Applying learner 'regr.ranger' on task 'df.tr' (iter 16/50) 
    DEBUG [06:18:37.075] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75..]}
    DEBUG [06:18:37.079] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.090] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:37.092] Creating Prediction for predict set 'test' 
    DEBUG [06:18:37.095] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72]}
    DEBUG [06:18:37.098] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.109] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:37.111] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:37.112] Applying learner 'regr.ranger' on task 'df.tr' (iter 49/50) 
    DEBUG [06:18:37.123] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:37.126] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.138] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:37.139] Creating Prediction for predict set 'test' 
    DEBUG [06:18:37.143] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 7, 14, 16, 25, 26, 27, 40, 44, 56, 67, 69, 75, 76]}
    DEBUG [06:18:37.147] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.159] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:37.161] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:37.162] Applying learner 'regr.ranger' on task 'df.tr' (iter 38/50) 
    DEBUG [06:18:37.173] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:37.176] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.187] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:37.189] Creating Prediction for predict set 'test' 
    DEBUG [06:18:37.192] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58, 69]}
    DEBUG [06:18:37.196] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.208] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:37.210] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:37.211] Applying learner 'regr.ranger' on task 'df.tr' (iter 32/50) 
    DEBUG [06:18:37.221] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 6..]}
    DEBUG [06:18:37.225] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.236] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:37.237] Creating Prediction for predict set 'test' 
    DEBUG [06:18:37.241] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71]}
    DEBUG [06:18:37.244] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.255] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:37.257] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:37.258] Applying learner 'regr.ranger' on task 'df.tr' (iter 31/50) 
    DEBUG [06:18:37.300] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60..]}
    DEBUG [06:18:37.304] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.317] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:37.319] Creating Prediction for predict set 'test' 
    DEBUG [06:18:37.324] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73]}
    DEBUG [06:18:37.328] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.342] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:37.344] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:37.346] Applying learner 'regr.ranger' on task 'df.tr' (iter 37/50) 
    DEBUG [06:18:37.358] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57..]}
    DEBUG [06:18:37.362] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.375] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:37.376] Creating Prediction for predict set 'test' 
    DEBUG [06:18:37.380] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76]}
    DEBUG [06:18:37.385] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.405] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:37.407] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:37.409] Applying learner 'regr.ranger' on task 'df.tr' (iter 2/50) 
    DEBUG [06:18:37.422] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:18:37.427] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.441] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:37.443] Creating Prediction for predict set 'test' 
    DEBUG [06:18:37.447] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68]}
    DEBUG [06:18:37.452] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.465] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:37.467] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:37.468] Applying learner 'regr.ranger' on task 'df.tr' (iter 22/50) 
    DEBUG [06:18:37.480] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57..]}
    DEBUG [06:18:37.484] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.496] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:37.498] Creating Prediction for predict set 'test' 
    DEBUG [06:18:37.502] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75]}
    DEBUG [06:18:37.506] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.519] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:37.521] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:37.522] Applying learner 'regr.ranger' on task 'df.tr' (iter 17/50) 
    DEBUG [06:18:37.534] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 7..]}
    DEBUG [06:18:37.538] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.551] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:37.553] Creating Prediction for predict set 'test' 
    DEBUG [06:18:37.556] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70]}
    DEBUG [06:18:37.560] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.570] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:37.572] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:37.573] Applying learner 'regr.ranger' on task 'df.tr' (iter 5/50) 
    DEBUG [06:18:37.583] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:37.586] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.597] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:37.599] Creating Prediction for predict set 'test' 
    DEBUG [06:18:37.602] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 11, 14, 17, 27, 32, 41, 44, 45, 55, 60, 64, 65, 66, 69]}
    DEBUG [06:18:37.605] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.615] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:37.617] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:37.618] Applying learner 'regr.ranger' on task 'df.tr' (iter 15/50) 
    DEBUG [06:18:37.627] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:37.631] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.641] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:37.643] Creating Prediction for predict set 'test' 
    DEBUG [06:18:37.646] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 19, 20, 30, 35, 42, 47, 54, 59, 64, 67, 71, 74, 75]}
    DEBUG [06:18:37.650] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.660] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:37.662] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:37.663] Applying learner 'regr.ranger' on task 'df.tr' (iter 44/50) 
    DEBUG [06:18:37.673] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:37.676] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.686] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:37.688] Creating Prediction for predict set 'test' 
    DEBUG [06:18:37.691] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 14, 17, 19, 24, 36, 41, 47, 51, 53, 56, 62, 63, 67, 71]}
    DEBUG [06:18:37.694] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.704] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:37.706] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:37.707] Applying learner 'regr.ranger' on task 'df.tr' (iter 35/50) 
    DEBUG [06:18:37.717] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:37.720] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.730] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:37.731] Creating Prediction for predict set 'test' 
    DEBUG [06:18:37.734] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 29, 32, 33, 35, 40, 43, 44, 45, 61, 62, 70, 74]}
    DEBUG [06:18:37.738] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.749] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:37.751] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:37.752] Applying learner 'regr.ranger' on task 'df.tr' (iter 27/50) 
    DEBUG [06:18:37.762] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67..]}
    DEBUG [06:18:37.765] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.775] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:37.777] Creating Prediction for predict set 'test' 
    DEBUG [06:18:37.780] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76]}
    DEBUG [06:18:37.783] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.793] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:37.795] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:37.796] Applying learner 'regr.ranger' on task 'df.tr' (iter 40/50) 
    DEBUG [06:18:37.806] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:37.809] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.820] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:37.821] Creating Prediction for predict set 'test' 
    DEBUG [06:18:37.824] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 8, 16, 17, 24, 25, 27, 29, 41, 44, 53, 54, 67, 72]}
    DEBUG [06:18:37.828] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.838] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:37.839] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:37.841] Applying learner 'regr.ranger' on task 'df.tr' (iter 43/50) 
    DEBUG [06:18:37.850] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:37.853] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.864] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:37.865] Creating Prediction for predict set 'test' 
    DEBUG [06:18:37.868] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74]}
    DEBUG [06:18:37.872] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.882] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:37.883] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:37.889] Applying learner 'regr.ranger' on task 'df.tr' (iter 50/50) 
    DEBUG [06:18:37.901] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:37.906] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.919] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:37.921] Creating Prediction for predict set 'test' 
    DEBUG [06:18:37.925] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 8, 21, 22, 30, 36, 41, 45, 49, 53, 54, 57, 68, 71]}
    DEBUG [06:18:37.928] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.939] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:37.941] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:37.942] Applying learner 'regr.ranger' on task 'df.tr' (iter 42/50) 
    DEBUG [06:18:37.954] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70..]}
    DEBUG [06:18:37.957] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.968] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:37.970] Creating Prediction for predict set 'test' 
    DEBUG [06:18:37.973] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66]}
    DEBUG [06:18:37.977] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:37.989] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:37.991] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:37.992] Applying learner 'regr.ranger' on task 'df.tr' (iter 34/50) 
    DEBUG [06:18:38.002] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:38.006] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.017] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:38.018] Creating Prediction for predict set 'test' 
    DEBUG [06:18:38.022] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 7, 12, 14, 18, 20, 41, 46, 47, 53, 57, 63, 64, 72, 75]}
    DEBUG [06:18:38.025] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.037] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:38.039] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:38.040] Applying learner 'regr.ranger' on task 'df.tr' (iter 3/50) 
    DEBUG [06:18:38.051] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:38.055] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.066] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:38.068] Creating Prediction for predict set 'test' 
    DEBUG [06:18:38.071] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75, 76]}
    DEBUG [06:18:38.075] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.086] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:38.088] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:38.089] Applying learner 'regr.ranger' on task 'df.tr' (iter 41/50) 
    DEBUG [06:18:38.100] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74..]}
    DEBUG [06:18:38.104] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.116] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:38.117] Creating Prediction for predict set 'test' 
    DEBUG [06:18:38.121] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76]}
    DEBUG [06:18:38.124] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.135] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:38.136] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:38.138] Applying learner 'regr.ranger' on task 'df.tr' (iter 45/50) 
    DEBUG [06:18:38.147] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [06:18:38.151] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.162] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:38.164] Creating Prediction for predict set 'test' 
    DEBUG [06:18:38.167] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 8, 12, 21, 22, 25, 28, 33, 37, 38, 44, 46, 48, 57, 75]}
    DEBUG [06:18:38.171] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.182] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:38.183] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:38.184] Applying learner 'regr.ranger' on task 'df.tr' (iter 26/50) 
    DEBUG [06:18:38.195] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75..]}
    DEBUG [06:18:38.199] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.210] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:38.211] Creating Prediction for predict set 'test' 
    DEBUG [06:18:38.215] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74]}
    DEBUG [06:18:38.218] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.229] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:38.231] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:38.232] Applying learner 'regr.ranger' on task 'df.tr' (iter 7/50) 
    DEBUG [06:18:38.242] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 6..]}
    DEBUG [06:18:38.246] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.257] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:38.259] Creating Prediction for predict set 'test' 
    DEBUG [06:18:38.262] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74]}
    DEBUG [06:18:38.266] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.276] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:38.278] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:38.279] Applying learner 'regr.ranger' on task 'df.tr' (iter 29/50) 
    DEBUG [06:18:38.289] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:38.293] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.303] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:38.305] Creating Prediction for predict set 'test' 
    DEBUG [06:18:38.308] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 18, 25, 28, 30, 39, 45, 48, 51, 54, 55, 61, 66, 71, 73]}
    DEBUG [06:18:38.311] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.324] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:38.326] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:38.327] Applying learner 'regr.ranger' on task 'df.tr' (iter 47/50) 
    DEBUG [06:18:38.338] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64..]}
    DEBUG [06:18:38.342] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.353] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:38.354] Creating Prediction for predict set 'test' 
    DEBUG [06:18:38.358] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70]}
    DEBUG [06:18:38.361] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.409] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:38.413] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:38.415] Applying learner 'regr.ranger' on task 'df.tr' (iter 8/50) 
    DEBUG [06:18:38.431] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:38.438] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.453] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:38.456] Creating Prediction for predict set 'test' 
    DEBUG [06:18:38.460] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 75]}
    DEBUG [06:18:38.464] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.478] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:38.480] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:38.482] Applying learner 'regr.ranger' on task 'df.tr' (iter 18/50) 
    DEBUG [06:18:38.495] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:38.500] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.522] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:38.524] Creating Prediction for predict set 'test' 
    DEBUG [06:18:38.529] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75]}
    DEBUG [06:18:38.536] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.553] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:38.555] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:38.557] Applying learner 'regr.ranger' on task 'df.tr' (iter 19/50) 
    DEBUG [06:18:38.573] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:38.578] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.591] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:38.593] Creating Prediction for predict set 'test' 
    DEBUG [06:18:38.597] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 8, 11, 14, 18, 24, 28, 33, 34, 54, 56, 62, 64, 68, 76]}
    DEBUG [06:18:38.602] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.615] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:38.617] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:38.618] Applying learner 'regr.ranger' on task 'df.tr' (iter 9/50) 
    DEBUG [06:18:38.631] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:38.635] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.648] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:38.650] Creating Prediction for predict set 'test' 
    DEBUG [06:18:38.654] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 23, 27, 40, 42, 46, 49, 56, 58, 61, 64, 67, 70, 71, 72]}
    DEBUG [06:18:38.658] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.671] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:38.673] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:38.674] Applying learner 'regr.ranger' on task 'df.tr' (iter 14/50) 
    DEBUG [06:18:38.686] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:38.690] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.702] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:38.703] Creating Prediction for predict set 'test' 
    DEBUG [06:18:38.707] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 22, 27, 36, 38, 40, 45, 46, 50, 55, 56, 60, 65, 73]}
    DEBUG [06:18:38.710] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.720] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:38.722] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:38.723] Applying learner 'regr.ranger' on task 'df.tr' (iter 28/50) 
    DEBUG [06:18:38.733] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [06:18:38.737] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.747] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:38.749] Creating Prediction for predict set 'test' 
    DEBUG [06:18:38.752] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75]}
    DEBUG [06:18:38.755] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.765] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:38.767] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:38.768] Applying learner 'regr.ranger' on task 'df.tr' (iter 36/50) 
    DEBUG [06:18:38.778] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58..]}
    DEBUG [06:18:38.781] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.792] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:38.794] Creating Prediction for predict set 'test' 
    DEBUG [06:18:38.797] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75]}
    DEBUG [06:18:38.800] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.811] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:38.812] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:38.814] Applying learner 'regr.ranger' on task 'df.tr' (iter 13/50) 
    DEBUG [06:18:38.823] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [06:18:38.827] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.837] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:38.838] Creating Prediction for predict set 'test' 
    DEBUG [06:18:38.841] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 72]}
    DEBUG [06:18:38.845] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.856] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:38.857] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:38.859] Applying learner 'regr.ranger' on task 'df.tr' (iter 24/50) 
    DEBUG [06:18:38.868] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:38.872] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.882] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:38.884] Creating Prediction for predict set 'test' 
    DEBUG [06:18:38.887] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 13, 17, 25, 32, 35, 36, 37, 38, 50, 63, 65, 68, 69, 74]}
    DEBUG [06:18:38.890] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.901] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:38.902] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:38.903] Applying learner 'regr.ranger' on task 'df.tr' (iter 21/50) 
    DEBUG [06:18:38.913] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66..]}
    DEBUG [06:18:38.917] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.927] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:38.928] Creating Prediction for predict set 'test' 
    DEBUG [06:18:38.931] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72]}
    DEBUG [06:18:38.935] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.946] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:38.948] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:38.949] Applying learner 'regr.ranger' on task 'df.tr' (iter 10/50) 
    DEBUG [06:18:38.958] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [06:18:38.962] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.972] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:38.974] Creating Prediction for predict set 'test' 
    DEBUG [06:18:38.977] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 17, 18, 19, 24, 26, 29, 31, 34, 36, 43, 45, 53, 66, 73]}
    DEBUG [06:18:38.980] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:38.990] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:38.992] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:38.993] Applying learner 'regr.ranger' on task 'df.tr' (iter 4/50) 
    DEBUG [06:18:39.003] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [06:18:39.007] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.017] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:39.019] Creating Prediction for predict set 'test' 
    DEBUG [06:18:39.022] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 3, 13, 20, 23, 25, 40, 46, 50, 53, 56, 57, 58, 63]}
    DEBUG [06:18:39.025] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.035] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:39.037] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:39.038] Applying learner 'regr.ranger' on task 'df.tr' (iter 33/50) 
    DEBUG [06:18:39.047] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [06:18:39.051] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.061] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:39.063] Creating Prediction for predict set 'test' 
    DEBUG [06:18:39.066] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60, 76]}
    DEBUG [06:18:39.069] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.079] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:39.081] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:39.082] Applying learner 'regr.ranger' on task 'df.tr' (iter 39/50) 
    DEBUG [06:18:39.092] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [06:18:39.102] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.115] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:39.117] Creating Prediction for predict set 'test' 
    DEBUG [06:18:39.120] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 5, 14, 18, 19, 21, 28, 38, 48, 49, 51, 52, 59, 64, 65]}
    DEBUG [06:18:39.123] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.134] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:39.135] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:39.136] Applying learner 'regr.ranger' on task 'df.tr' (iter 12/50) 
    DEBUG [06:18:39.147] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61..]}
    DEBUG [06:18:39.151] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.162] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:39.163] Creating Prediction for predict set 'test' 
    DEBUG [06:18:39.166] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70]}
    DEBUG [06:18:39.170] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.180] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:39.182] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:39.183] Applying learner 'regr.ranger' on task 'df.tr' (iter 48/50) 
    DEBUG [06:18:39.192] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [06:18:39.196] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.206] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:39.208] Creating Prediction for predict set 'test' 
    DEBUG [06:18:39.211] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72, 74]}
    DEBUG [06:18:39.214] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.224] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:39.226] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:39.227] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [06:18:39.237] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:18:39.240] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.251] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:39.253] Creating Prediction for predict set 'test' 
    DEBUG [06:18:39.256] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [06:18:39.259] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.270] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:39.271] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:39.272] Applying learner 'regr.ranger' on task 'df.tr' (iter 20/50) 
    DEBUG [06:18:39.283] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [06:18:39.286] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.298] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:39.299] Creating Prediction for predict set 'test' 
    DEBUG [06:18:39.303] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 16, 17, 32, 36, 37, 42, 43, 45, 46, 53, 61, 65, 69]}
    DEBUG [06:18:39.306] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.317] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:39.318] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:39.319] Applying learner 'regr.ranger' on task 'df.tr' (iter 6/50) 
    DEBUG [06:18:39.329] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 7..]}
    DEBUG [06:18:39.333] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.343] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:39.344] Creating Prediction for predict set 'test' 
    DEBUG [06:18:39.347] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76]}
    DEBUG [06:18:39.351] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.361] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:39.363] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:39.364] Applying learner 'regr.ranger' on task 'df.tr' (iter 46/50) 
    DEBUG [06:18:39.374] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72..]}
    DEBUG [06:18:39.377] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.389] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:39.390] Creating Prediction for predict set 'test' 
    DEBUG [06:18:39.393] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73]}
    DEBUG [06:18:39.397] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.408] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:39.409] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:39.411] Applying learner 'regr.ranger' on task 'df.tr' (iter 23/50) 
    DEBUG [06:18:39.421] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [06:18:39.425] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.435] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:39.437] Creating Prediction for predict set 'test' 
    DEBUG [06:18:39.440] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66]}
    DEBUG [06:18:39.443] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.457] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:39.458] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:39.533] Finished benchmark 
    INFO  [06:18:41.443] Result of batch 10: 
    INFO  [06:18:41.447]  mtry sample.fraction num.trees importance regr.rmse      resample_result 
    INFO  [06:18:41.447]     3       0.7453881        55   impurity  46.77007 <ResampleResult[19]> 
    INFO  [06:18:41.464] Finished optimizing after 10 evaluation(s) 
    INFO  [06:18:41.465] Result: 
    INFO  [06:18:41.467]  mtry sample.fraction num.trees importance learner_param_vals  x_domain 
    INFO  [06:18:41.467]     5       0.7631363       141   impurity          <list[4]> <list[4]> 
    INFO  [06:18:41.467]  regr.rmse 
    INFO  [06:18:41.467]   43.04499 
    DEBUG [06:18:41.505] Skip subsetting of task 'df.tr' 
    DEBUG [06:18:41.508] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 76 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:41.523] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:41.534] Learner 'regr.ranger.tuned' on task 'df.tr' succeeded to fit a model {learner: <AutoTuner/Learner/R6>, result: <list>, messages: }
    DEBUG [06:18:41.544] Skip subsetting of task 'df.tr' 
    DEBUG [06:18:41.547] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 76 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:41.560] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }

    train.model

    function (newdata, task = NULL) 
    .__Learner__predict_newdata(self = self, private = private, super = super, 
        newdata = newdata, task = task)
    <environment: 0x556b5c6382b8>

-   User only **must** define a `df` and the `target.variable` and
    `train.spm()` will automatically perform `classification` or
    `regression` tasks.
-   The rest of arguments can be set or default values will be set.
-   If **crs** is set `train.spm()` will automatically take care of
    **spatial cross validation**.

`predict.spm()`

    predict.variable = predict.spm(df.ts, task = NULL)

    DEBUG [06:18:41.721] Skip subsetting of task 'df.ts' 
    DEBUG [06:18:41.723] Calling predict method of Learner 'regr.ranger' on task 'df.ts' with 76 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:41.742] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }

    predict.variable

     [1] 299 277 199 117 150  80  86  97 130 133 148 207 194  75  87  65  78  80  97
    [20] 166 217 102 281 211 135 112  94 191 654 180 207 541 310 210 158 144 181 399
    [39]  45 325 268 252 260 237 241 204 462 159 131 216 145 148  66  63  48  51  64
    [58]  84  48  84 250  94 244  49  50 162  80  42  48  41 146  51 136 179  68  49

User needs to set only `df.ts = test set` and put the `task = NULL`

`accuracy.plot()`

    accuracy.plot.spm(x = df.ts[,target.variable], y = predict.variable)

![](README_files/figure-markdown_strict/unnamed-chunk-10-1.png)![](README_files/figure-markdown_strict/unnamed-chunk-10-2.png)

References
----------

Lang, M., Schratz, P., Binder, M., Pfisterer, F., Richter, J., Reich, N.
G., & Bischl, B. (2020, September 9). mlr3 book. Retrieved from
<https://mlr3book.mlr-org.com>
