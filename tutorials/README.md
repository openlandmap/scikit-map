

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

<PredictionDataRegr/PredictionData>, messages: }
    DEBUG [06:18:39.226] Erasing stored model for learner 'regr.ranger' 
    INFO  [06:18:39.227] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [06:18:39.237] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [06:18:39.240] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.251] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [06:18:39.253] Creating Prediction for predict set 'test' 
    DEBUG [06:18:39.256] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [06:18:39.259] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [06:18:39.270] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
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
