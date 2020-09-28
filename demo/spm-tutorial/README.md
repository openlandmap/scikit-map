

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

### `train.spm()`

Here we have four scenarios:

-   `classification` task with **non spatial** resampling methods
-   `regression` task with **non spatial** resampling methods
-   `classification` task with **spatial** resampling methods
-   `regression` task with **spatial** resampling methods

The above code has fitted multiple models/learners depending on the
`class()` of the **target.variable** and for now only returns a
`trained model` function so later on we could use it to train a new
dataset.

### `predict.spm()`

prediction on new dataset

### `accuracy.plot()`

Accuracy plot in case of regression task Note: don’t use it for
classification tasks for obvious reasons

### Basic requirements user needs to set

in a nutshell the user can `train` an arbitrary `s3` **(spatial) data
frame** by only defining following arguments:

`train.spm()`

    train.model = train.spm(df.tr, target.variable = target.variable, folds = folds ,n_evals = n_evals, plot.workflow = TRUE, crs )

    regression Task   resampling method: non-spatialCV  ncores:  32 ...TRUE

    Using learners: method.list...TRUE

    INFO  [09:09:30.131] Applying learner 'fixfactors.removeconstants.regr.rpart' on task 'df.tr' (iter 1/1) 

               Fitting a ensemble ML using 'mlr3::Taskregr'...TRUE

    DEBUG [09:09:30.644] Skip subsetting of task 'df.tr' 
    DEBUG [09:09:30.657] Calling train method of Learner 'regr.ranger.tuned' on task 'df.tr' with 76 observations {learner: <AutoTuner/Learner/R6>}
    INFO  [09:09:31.061] Starting to optimize 4 parameter(s) with '<OptimizerRandomSearch>' and '<TerminatorEvals>' 
    INFO  [09:09:31.102] Evaluating 1 configuration(s) 
    INFO  [09:09:31.137] Benchmark with 50 resampling iterations 
    DEBUG [09:09:31.138] Running benchmark() asynchronously with 50 iterations 
    INFO  [09:09:31.151] Applying learner 'regr.ranger' on task 'df.tr' (iter 32/50) 
    DEBUG [09:09:31.161] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 6..]}
    DEBUG [09:09:31.164] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.183] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:31.185] Creating Prediction for predict set 'test' 
    DEBUG [09:09:31.188] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71]}
    DEBUG [09:09:31.190] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.202] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:31.203] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:31.204] Applying learner 'regr.ranger' on task 'df.tr' (iter 39/50) 
    DEBUG [09:09:31.211] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:31.214] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.228] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:31.230] Creating Prediction for predict set 'test' 
    DEBUG [09:09:31.233] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 5, 14, 18, 19, 21, 28, 38, 48, 49, 51, 52, 59, 64, 65]}
    DEBUG [09:09:31.235] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.244] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:31.246] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:31.247] Applying learner 'regr.ranger' on task 'df.tr' (iter 16/50) 
    DEBUG [09:09:31.254] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75..]}
    DEBUG [09:09:31.256] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.270] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:31.272] Creating Prediction for predict set 'test' 
    DEBUG [09:09:31.275] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72]}
    DEBUG [09:09:31.279] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.290] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:31.292] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:31.293] Applying learner 'regr.ranger' on task 'df.tr' (iter 26/50) 
    DEBUG [09:09:31.302] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75..]}
    DEBUG [09:09:31.305] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.330] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:31.332] Creating Prediction for predict set 'test' 
    DEBUG [09:09:31.335] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74]}
    DEBUG [09:09:31.339] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.354] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:31.356] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:31.357] Applying learner 'regr.ranger' on task 'df.tr' (iter 48/50) 
    DEBUG [09:09:31.367] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:31.371] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.390] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:31.392] Creating Prediction for predict set 'test' 
    DEBUG [09:09:31.395] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72, 74]}
    DEBUG [09:09:31.399] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.412] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:31.413] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:31.414] Applying learner 'regr.ranger' on task 'df.tr' (iter 2/50) 
    DEBUG [09:09:31.424] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [09:09:31.428] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.446] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:31.448] Creating Prediction for predict set 'test' 
    DEBUG [09:09:31.452] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68]}
    DEBUG [09:09:31.455] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.468] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:31.470] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:31.471] Applying learner 'regr.ranger' on task 'df.tr' (iter 34/50) 
    DEBUG [09:09:31.481] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:31.484] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.501] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:31.503] Creating Prediction for predict set 'test' 
    DEBUG [09:09:31.507] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 7, 12, 14, 18, 20, 41, 46, 47, 53, 57, 63, 64, 72, 75]}
    DEBUG [09:09:31.511] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.523] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:31.525] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:31.526] Applying learner 'regr.ranger' on task 'df.tr' (iter 14/50) 
    DEBUG [09:09:31.536] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:31.539] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.557] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:31.559] Creating Prediction for predict set 'test' 
    DEBUG [09:09:31.563] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 22, 27, 36, 38, 40, 45, 46, 50, 55, 56, 60, 65, 73]}
    DEBUG [09:09:31.566] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.579] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:31.580] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:31.581] Applying learner 'regr.ranger' on task 'df.tr' (iter 30/50) 
    DEBUG [09:09:31.591] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:31.595] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.611] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:31.612] Creating Prediction for predict set 'test' 
    DEBUG [09:09:31.615] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 14, 22, 29, 34, 35, 37, 40, 43, 44, 59, 62, 64, 68, 72]}
    DEBUG [09:09:31.619] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.632] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:31.633] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:31.634] Applying learner 'regr.ranger' on task 'df.tr' (iter 28/50) 
    DEBUG [09:09:31.644] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:31.648] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.663] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:31.665] Creating Prediction for predict set 'test' 
    DEBUG [09:09:31.668] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75]}
    DEBUG [09:09:31.672] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.684] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:31.686] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:31.687] Applying learner 'regr.ranger' on task 'df.tr' (iter 21/50) 
    DEBUG [09:09:31.697] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66..]}
    DEBUG [09:09:31.701] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.718] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:31.720] Creating Prediction for predict set 'test' 
    DEBUG [09:09:31.725] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72]}
    DEBUG [09:09:31.729] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.741] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:31.743] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:31.744] Applying learner 'regr.ranger' on task 'df.tr' (iter 23/50) 
    DEBUG [09:09:31.754] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:31.757] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.775] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:31.777] Creating Prediction for predict set 'test' 
    DEBUG [09:09:31.780] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66]}
    DEBUG [09:09:31.784] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.797] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:31.798] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:31.799] Applying learner 'regr.ranger' on task 'df.tr' (iter 42/50) 
    DEBUG [09:09:31.809] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70..]}
    DEBUG [09:09:31.813] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.829] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:31.830] Creating Prediction for predict set 'test' 
    DEBUG [09:09:31.834] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66]}
    DEBUG [09:09:31.838] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.851] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:31.853] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:31.854] Applying learner 'regr.ranger' on task 'df.tr' (iter 25/50) 
    DEBUG [09:09:31.864] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:31.868] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.885] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:31.887] Creating Prediction for predict set 'test' 
    DEBUG [09:09:31.891] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [9, 11, 18, 21, 23, 27, 29, 43, 44, 45, 51, 55, 67, 73, 76]}
    DEBUG [09:09:31.894] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.907] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:31.909] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:31.910] Applying learner 'regr.ranger' on task 'df.tr' (iter 6/50) 
    DEBUG [09:09:31.920] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 7..]}
    DEBUG [09:09:31.924] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.942] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:31.944] Creating Prediction for predict set 'test' 
    DEBUG [09:09:31.947] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76]}
    DEBUG [09:09:31.951] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.965] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:31.966] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:31.967] Applying learner 'regr.ranger' on task 'df.tr' (iter 4/50) 
    DEBUG [09:09:31.977] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:31.981] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:31.997] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:31.999] Creating Prediction for predict set 'test' 
    DEBUG [09:09:32.002] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 3, 13, 20, 23, 25, 40, 46, 50, 53, 56, 57, 58, 63]}
    DEBUG [09:09:32.006] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.019] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:32.021] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:32.022] Applying learner 'regr.ranger' on task 'df.tr' (iter 3/50) 
    DEBUG [09:09:32.032] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:32.036] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.054] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:32.056] Creating Prediction for predict set 'test' 
    DEBUG [09:09:32.059] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75, 76]}
    DEBUG [09:09:32.063] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.076] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:32.077] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:32.079] Applying learner 'regr.ranger' on task 'df.tr' (iter 10/50) 
    DEBUG [09:09:32.088] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:32.092] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.111] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:32.113] Creating Prediction for predict set 'test' 
    DEBUG [09:09:32.118] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 17, 18, 19, 24, 26, 29, 31, 34, 36, 43, 45, 53, 66, 73]}
    DEBUG [09:09:32.121] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.134] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:32.136] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:32.137] Applying learner 'regr.ranger' on task 'df.tr' (iter 41/50) 
    DEBUG [09:09:32.147] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74..]}
    DEBUG [09:09:32.150] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.166] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:32.168] Creating Prediction for predict set 'test' 
    DEBUG [09:09:32.171] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76]}
    DEBUG [09:09:32.174] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.187] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:32.189] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:32.190] Applying learner 'regr.ranger' on task 'df.tr' (iter 12/50) 
    DEBUG [09:09:32.200] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61..]}
    DEBUG [09:09:32.204] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.219] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:32.221] Creating Prediction for predict set 'test' 
    DEBUG [09:09:32.226] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70]}
    DEBUG [09:09:32.230] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.243] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:32.244] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:32.245] Applying learner 'regr.ranger' on task 'df.tr' (iter 18/50) 
    DEBUG [09:09:32.255] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:32.259] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.276] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:32.278] Creating Prediction for predict set 'test' 
    DEBUG [09:09:32.282] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75]}
    DEBUG [09:09:32.286] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.298] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:32.300] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:32.301] Applying learner 'regr.ranger' on task 'df.tr' (iter 44/50) 
    DEBUG [09:09:32.311] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:32.314] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.338] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:32.340] Creating Prediction for predict set 'test' 
    DEBUG [09:09:32.343] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 14, 17, 19, 24, 36, 41, 47, 51, 53, 56, 62, 63, 67, 71]}
    DEBUG [09:09:32.347] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.361] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:32.362] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:32.363] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [09:09:32.374] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [09:09:32.429] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.445] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:32.446] Creating Prediction for predict set 'test' 
    DEBUG [09:09:32.449] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [09:09:32.453] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.465] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:32.466] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:32.467] Applying learner 'regr.ranger' on task 'df.tr' (iter 31/50) 
    DEBUG [09:09:32.477] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60..]}
    DEBUG [09:09:32.480] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.498] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:32.500] Creating Prediction for predict set 'test' 
    DEBUG [09:09:32.503] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73]}
    DEBUG [09:09:32.506] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.519] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:32.521] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:32.522] Applying learner 'regr.ranger' on task 'df.tr' (iter 17/50) 
    DEBUG [09:09:32.532] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 7..]}
    DEBUG [09:09:32.536] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.553] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:32.555] Creating Prediction for predict set 'test' 
    DEBUG [09:09:32.559] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70]}
    DEBUG [09:09:32.562] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.575] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:32.576] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:32.577] Applying learner 'regr.ranger' on task 'df.tr' (iter 50/50) 
    DEBUG [09:09:32.587] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:32.590] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.606] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:32.607] Creating Prediction for predict set 'test' 
    DEBUG [09:09:32.610] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 8, 21, 22, 30, 36, 41, 45, 49, 53, 54, 57, 68, 71]}
    DEBUG [09:09:32.614] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.627] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:32.628] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:32.629] Applying learner 'regr.ranger' on task 'df.tr' (iter 35/50) 
    DEBUG [09:09:32.639] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:32.643] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.658] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:32.660] Creating Prediction for predict set 'test' 
    DEBUG [09:09:32.664] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 29, 32, 33, 35, 40, 43, 44, 45, 61, 62, 70, 74]}
    DEBUG [09:09:32.667] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.680] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:32.681] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:32.683] Applying learner 'regr.ranger' on task 'df.tr' (iter 19/50) 
    DEBUG [09:09:32.693] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:32.696] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.712] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:32.714] Creating Prediction for predict set 'test' 
    DEBUG [09:09:32.718] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 8, 11, 14, 18, 24, 28, 33, 34, 54, 56, 62, 64, 68, 76]}
    DEBUG [09:09:32.721] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.734] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:32.736] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:32.737] Applying learner 'regr.ranger' on task 'df.tr' (iter 13/50) 
    DEBUG [09:09:32.747] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:32.750] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.766] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:32.767] Creating Prediction for predict set 'test' 
    DEBUG [09:09:32.771] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 72]}
    DEBUG [09:09:32.774] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.787] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:32.788] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:32.789] Applying learner 'regr.ranger' on task 'df.tr' (iter 38/50) 
    DEBUG [09:09:32.799] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:32.803] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.819] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:32.821] Creating Prediction for predict set 'test' 
    DEBUG [09:09:32.824] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58, 69]}
    DEBUG [09:09:32.828] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.840] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:32.842] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:32.843] Applying learner 'regr.ranger' on task 'df.tr' (iter 15/50) 
    DEBUG [09:09:32.853] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:32.857] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.874] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:32.875] Creating Prediction for predict set 'test' 
    DEBUG [09:09:32.879] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 19, 20, 30, 35, 42, 47, 54, 59, 64, 67, 71, 74, 75]}
    DEBUG [09:09:32.882] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.894] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:32.896] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:32.897] Applying learner 'regr.ranger' on task 'df.tr' (iter 22/50) 
    DEBUG [09:09:32.907] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57..]}
    DEBUG [09:09:32.911] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.927] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:32.930] Creating Prediction for predict set 'test' 
    DEBUG [09:09:32.934] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75]}
    DEBUG [09:09:32.937] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.949] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:32.951] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:32.952] Applying learner 'regr.ranger' on task 'df.tr' (iter 43/50) 
    DEBUG [09:09:32.962] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:32.965] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:32.981] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:32.983] Creating Prediction for predict set 'test' 
    DEBUG [09:09:32.986] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74]}
    DEBUG [09:09:32.989] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.002] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:33.003] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:33.004] Applying learner 'regr.ranger' on task 'df.tr' (iter 27/50) 
    DEBUG [09:09:33.014] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67..]}
    DEBUG [09:09:33.018] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.033] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:33.035] Creating Prediction for predict set 'test' 
    DEBUG [09:09:33.038] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76]}
    DEBUG [09:09:33.041] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.053] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:33.055] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:33.056] Applying learner 'regr.ranger' on task 'df.tr' (iter 45/50) 
    DEBUG [09:09:33.066] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:33.070] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.085] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:33.086] Creating Prediction for predict set 'test' 
    DEBUG [09:09:33.090] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 8, 12, 21, 22, 25, 28, 33, 37, 38, 44, 46, 48, 57, 75]}
    DEBUG [09:09:33.093] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.105] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:33.107] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:33.108] Applying learner 'regr.ranger' on task 'df.tr' (iter 7/50) 
    DEBUG [09:09:33.119] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 6..]}
    DEBUG [09:09:33.122] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.144] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:33.146] Creating Prediction for predict set 'test' 
    DEBUG [09:09:33.149] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74]}
    DEBUG [09:09:33.153] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.165] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:33.167] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:33.168] Applying learner 'regr.ranger' on task 'df.tr' (iter 33/50) 
    DEBUG [09:09:33.178] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:33.182] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.197] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:33.199] Creating Prediction for predict set 'test' 
    DEBUG [09:09:33.202] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60, 76]}
    DEBUG [09:09:33.206] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.218] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:33.220] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:33.221] Applying learner 'regr.ranger' on task 'df.tr' (iter 24/50) 
    DEBUG [09:09:33.231] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:33.234] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.250] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:33.251] Creating Prediction for predict set 'test' 
    DEBUG [09:09:33.254] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 13, 17, 25, 32, 35, 36, 37, 38, 50, 63, 65, 68, 69, 74]}
    DEBUG [09:09:33.258] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.270] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:33.272] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:33.273] Applying learner 'regr.ranger' on task 'df.tr' (iter 49/50) 
    DEBUG [09:09:33.283] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:33.287] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.302] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:33.304] Creating Prediction for predict set 'test' 
    DEBUG [09:09:33.307] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 7, 14, 16, 25, 26, 27, 40, 44, 56, 67, 69, 75, 76]}
    DEBUG [09:09:33.310] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.323] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:33.324] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:33.326] Applying learner 'regr.ranger' on task 'df.tr' (iter 8/50) 
    DEBUG [09:09:33.335] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:33.339] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.356] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:33.358] Creating Prediction for predict set 'test' 
    DEBUG [09:09:33.362] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 75]}
    DEBUG [09:09:33.366] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.378] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:33.380] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:33.381] Applying learner 'regr.ranger' on task 'df.tr' (iter 9/50) 
    DEBUG [09:09:33.391] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:33.395] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.412] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:33.415] Creating Prediction for predict set 'test' 
    DEBUG [09:09:33.418] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 23, 27, 40, 42, 46, 49, 56, 58, 61, 64, 67, 70, 71, 72]}
    DEBUG [09:09:33.422] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.434] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:33.436] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:33.437] Applying learner 'regr.ranger' on task 'df.tr' (iter 29/50) 
    DEBUG [09:09:33.447] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:33.451] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.466] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:33.468] Creating Prediction for predict set 'test' 
    DEBUG [09:09:33.472] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 18, 25, 28, 30, 39, 45, 48, 51, 54, 55, 61, 66, 71, 73]}
    DEBUG [09:09:33.476] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.488] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:33.490] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:33.491] Applying learner 'regr.ranger' on task 'df.tr' (iter 37/50) 
    DEBUG [09:09:33.501] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57..]}
    DEBUG [09:09:33.505] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.520] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:33.522] Creating Prediction for predict set 'test' 
    DEBUG [09:09:33.540] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76]}
    DEBUG [09:09:33.543] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.556] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:33.557] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:33.558] Applying learner 'regr.ranger' on task 'df.tr' (iter 40/50) 
    DEBUG [09:09:33.567] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:33.570] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.586] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:33.588] Creating Prediction for predict set 'test' 
    DEBUG [09:09:33.591] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 8, 16, 17, 24, 25, 27, 29, 41, 44, 53, 54, 67, 72]}
    DEBUG [09:09:33.594] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.606] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:33.608] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:33.609] Applying learner 'regr.ranger' on task 'df.tr' (iter 20/50) 
    DEBUG [09:09:33.617] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:33.621] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.636] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:33.637] Creating Prediction for predict set 'test' 
    DEBUG [09:09:33.640] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 16, 17, 32, 36, 37, 42, 43, 45, 46, 53, 61, 65, 69]}
    DEBUG [09:09:33.643] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.655] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:33.656] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:33.657] Applying learner 'regr.ranger' on task 'df.tr' (iter 36/50) 
    DEBUG [09:09:33.666] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58..]}
    DEBUG [09:09:33.669] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.684] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:33.685] Creating Prediction for predict set 'test' 
    DEBUG [09:09:33.688] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75]}
    DEBUG [09:09:33.693] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.704] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:33.705] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:33.707] Applying learner 'regr.ranger' on task 'df.tr' (iter 46/50) 
    DEBUG [09:09:33.716] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72..]}
    DEBUG [09:09:33.719] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.736] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:33.738] Creating Prediction for predict set 'test' 
    DEBUG [09:09:33.742] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73]}
    DEBUG [09:09:33.745] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.756] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:33.758] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:33.759] Applying learner 'regr.ranger' on task 'df.tr' (iter 5/50) 
    DEBUG [09:09:33.767] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:33.771] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.786] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:33.788] Creating Prediction for predict set 'test' 
    DEBUG [09:09:33.792] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 11, 14, 17, 27, 32, 41, 44, 45, 55, 60, 64, 65, 66, 69]}
    DEBUG [09:09:33.795] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.807] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:33.808] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:33.809] Applying learner 'regr.ranger' on task 'df.tr' (iter 11/50) 
    DEBUG [09:09:33.818] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 7..]}
    DEBUG [09:09:33.821] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.837] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:33.839] Creating Prediction for predict set 'test' 
    DEBUG [09:09:33.843] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76]}
    DEBUG [09:09:33.846] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.857] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:33.859] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:33.860] Applying learner 'regr.ranger' on task 'df.tr' (iter 47/50) 
    DEBUG [09:09:33.869] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64..]}
    DEBUG [09:09:33.872] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.889] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:33.891] Creating Prediction for predict set 'test' 
    DEBUG [09:09:33.894] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70]}
    DEBUG [09:09:33.897] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:33.909] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:33.910] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:33.917] Finished benchmark 
    INFO  [09:09:35.409] Result of batch 1: 
    INFO  [09:09:35.413]  mtry sample.fraction num.trees importance regr.rmse      resample_result 
    INFO  [09:09:35.413]     3       0.7830912       443   impurity  45.84152 <ResampleResult[19]> 
    INFO  [09:09:35.420] Evaluating 1 configuration(s) 
    INFO  [09:09:35.499] Benchmark with 50 resampling iterations 
    DEBUG [09:09:35.500] Running benchmark() asynchronously with 50 iterations 
    INFO  [09:09:35.512] Applying learner 'regr.ranger' on task 'df.tr' (iter 2/50) 
    DEBUG [09:09:35.520] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [09:09:35.523] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.533] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:35.535] Creating Prediction for predict set 'test' 
    DEBUG [09:09:35.537] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68]}
    DEBUG [09:09:35.540] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.549] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:35.551] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:35.551] Applying learner 'regr.ranger' on task 'df.tr' (iter 34/50) 
    DEBUG [09:09:35.559] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:35.562] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.572] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:35.573] Creating Prediction for predict set 'test' 
    DEBUG [09:09:35.576] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 7, 12, 14, 18, 20, 41, 46, 47, 53, 57, 63, 64, 72, 75]}
    DEBUG [09:09:35.579] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.588] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:35.589] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:35.590] Applying learner 'regr.ranger' on task 'df.tr' (iter 14/50) 
    DEBUG [09:09:35.598] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:35.601] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.610] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:35.612] Creating Prediction for predict set 'test' 
    DEBUG [09:09:35.614] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 22, 27, 36, 38, 40, 45, 46, 50, 55, 56, 60, 65, 73]}
    DEBUG [09:09:35.616] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.625] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:35.626] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:35.627] Applying learner 'regr.ranger' on task 'df.tr' (iter 30/50) 
    DEBUG [09:09:35.635] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:35.637] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.646] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:35.647] Creating Prediction for predict set 'test' 
    DEBUG [09:09:35.650] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 14, 22, 29, 34, 35, 37, 40, 43, 44, 59, 62, 64, 68, 72]}
    DEBUG [09:09:35.652] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.660] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:35.661] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:35.662] Applying learner 'regr.ranger' on task 'df.tr' (iter 28/50) 
    DEBUG [09:09:35.669] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:35.672] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.681] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:35.682] Creating Prediction for predict set 'test' 
    DEBUG [09:09:35.685] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75]}
    DEBUG [09:09:35.687] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.695] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:35.697] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:35.697] Applying learner 'regr.ranger' on task 'df.tr' (iter 21/50) 
    DEBUG [09:09:35.705] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66..]}
    DEBUG [09:09:35.708] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.717] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:35.718] Creating Prediction for predict set 'test' 
    DEBUG [09:09:35.721] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72]}
    DEBUG [09:09:35.723] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.732] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:35.733] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:35.734] Applying learner 'regr.ranger' on task 'df.tr' (iter 23/50) 
    DEBUG [09:09:35.742] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:35.744] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.754] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:35.755] Creating Prediction for predict set 'test' 
    DEBUG [09:09:35.758] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66]}
    DEBUG [09:09:35.760] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.769] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:35.770] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:35.771] Applying learner 'regr.ranger' on task 'df.tr' (iter 47/50) 
    DEBUG [09:09:35.779] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64..]}
    DEBUG [09:09:35.782] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.791] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:35.792] Creating Prediction for predict set 'test' 
    DEBUG [09:09:35.794] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70]}
    DEBUG [09:09:35.797] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.806] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:35.807] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:35.808] Applying learner 'regr.ranger' on task 'df.tr' (iter 25/50) 
    DEBUG [09:09:35.816] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:35.818] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.828] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:35.829] Creating Prediction for predict set 'test' 
    DEBUG [09:09:35.832] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [9, 11, 18, 21, 23, 27, 29, 43, 44, 45, 51, 55, 67, 73, 76]}
    DEBUG [09:09:35.834] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.843] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:35.844] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:35.845] Applying learner 'regr.ranger' on task 'df.tr' (iter 39/50) 
    DEBUG [09:09:35.853] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:35.855] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.864] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:35.866] Creating Prediction for predict set 'test' 
    DEBUG [09:09:35.868] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 5, 14, 18, 19, 21, 28, 38, 48, 49, 51, 52, 59, 64, 65]}
    DEBUG [09:09:35.871] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.879] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:35.880] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:35.881] Applying learner 'regr.ranger' on task 'df.tr' (iter 6/50) 
    DEBUG [09:09:35.889] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 7..]}
    DEBUG [09:09:35.891] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.900] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:35.901] Creating Prediction for predict set 'test' 
    DEBUG [09:09:35.903] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76]}
    DEBUG [09:09:35.906] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.914] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:35.916] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:35.916] Applying learner 'regr.ranger' on task 'df.tr' (iter 4/50) 
    DEBUG [09:09:35.924] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:35.926] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.935] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:35.936] Creating Prediction for predict set 'test' 
    DEBUG [09:09:35.939] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 3, 13, 20, 23, 25, 40, 46, 50, 53, 56, 57, 58, 63]}
    DEBUG [09:09:35.941] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.950] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:35.951] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:35.952] Applying learner 'regr.ranger' on task 'df.tr' (iter 3/50) 
    DEBUG [09:09:35.962] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:35.965] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.974] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:35.975] Creating Prediction for predict set 'test' 
    DEBUG [09:09:35.978] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75, 76]}
    DEBUG [09:09:35.980] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:35.989] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:35.990] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:35.991] Applying learner 'regr.ranger' on task 'df.tr' (iter 10/50) 
    DEBUG [09:09:35.998] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:36.001] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.010] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.011] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.013] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 17, 18, 19, 24, 26, 29, 31, 34, 36, 43, 45, 53, 66, 73]}
    DEBUG [09:09:36.015] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.024] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.025] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.026] Applying learner 'regr.ranger' on task 'df.tr' (iter 46/50) 
    DEBUG [09:09:36.033] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72..]}
    DEBUG [09:09:36.036] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.045] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.046] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.049] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73]}
    DEBUG [09:09:36.051] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.060] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.061] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.062] Applying learner 'regr.ranger' on task 'df.tr' (iter 12/50) 
    DEBUG [09:09:36.070] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61..]}
    DEBUG [09:09:36.073] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.083] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.084] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.087] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70]}
    DEBUG [09:09:36.089] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.098] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.100] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.101] Applying learner 'regr.ranger' on task 'df.tr' (iter 38/50) 
    DEBUG [09:09:36.109] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:36.112] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.121] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.123] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.125] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58, 69]}
    DEBUG [09:09:36.128] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.137] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.139] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.140] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [09:09:36.148] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [09:09:36.151] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.160] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.161] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.164] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [09:09:36.167] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.176] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.177] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.178] Applying learner 'regr.ranger' on task 'df.tr' (iter 35/50) 
    DEBUG [09:09:36.186] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:36.189] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.199] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.200] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.203] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 29, 32, 33, 35, 40, 43, 44, 45, 61, 62, 70, 74]}
    DEBUG [09:09:36.206] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.215] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.216] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.217] Applying learner 'regr.ranger' on task 'df.tr' (iter 17/50) 
    DEBUG [09:09:36.225] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 7..]}
    DEBUG [09:09:36.228] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.238] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.239] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.242] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70]}
    DEBUG [09:09:36.245] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.254] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.255] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.256] Applying learner 'regr.ranger' on task 'df.tr' (iter 33/50) 
    DEBUG [09:09:36.264] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:36.267] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.277] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.279] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.282] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60, 76]}
    DEBUG [09:09:36.286] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.297] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.299] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.300] Applying learner 'regr.ranger' on task 'df.tr' (iter 41/50) 
    DEBUG [09:09:36.310] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74..]}
    DEBUG [09:09:36.314] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.326] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.327] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.331] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76]}
    DEBUG [09:09:36.334] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.345] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.347] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.348] Applying learner 'regr.ranger' on task 'df.tr' (iter 19/50) 
    DEBUG [09:09:36.358] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:36.362] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.374] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.375] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.379] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 8, 11, 14, 18, 24, 28, 33, 34, 54, 56, 62, 64, 68, 76]}
    DEBUG [09:09:36.382] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.393] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.395] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.396] Applying learner 'regr.ranger' on task 'df.tr' (iter 13/50) 
    DEBUG [09:09:36.406] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:36.410] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.421] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.423] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.426] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 72]}
    DEBUG [09:09:36.429] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.440] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.442] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.443] Applying learner 'regr.ranger' on task 'df.tr' (iter 18/50) 
    DEBUG [09:09:36.453] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:36.456] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.468] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.469] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.473] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75]}
    DEBUG [09:09:36.476] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.487] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.489] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.490] Applying learner 'regr.ranger' on task 'df.tr' (iter 15/50) 
    DEBUG [09:09:36.500] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:36.504] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.516] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.517] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.520] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 19, 20, 30, 35, 42, 47, 54, 59, 64, 67, 71, 74, 75]}
    DEBUG [09:09:36.524] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.535] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.537] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.538] Applying learner 'regr.ranger' on task 'df.tr' (iter 27/50) 
    DEBUG [09:09:36.548] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67..]}
    DEBUG [09:09:36.552] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.563] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.565] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.568] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76]}
    DEBUG [09:09:36.571] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.582] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.584] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.585] Applying learner 'regr.ranger' on task 'df.tr' (iter 48/50) 
    DEBUG [09:09:36.595] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:36.599] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.612] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.614] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.617] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72, 74]}
    DEBUG [09:09:36.621] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.643] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.645] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.646] Applying learner 'regr.ranger' on task 'df.tr' (iter 26/50) 
    DEBUG [09:09:36.655] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75..]}
    DEBUG [09:09:36.658] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.669] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.670] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.673] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74]}
    DEBUG [09:09:36.676] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.686] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.688] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.689] Applying learner 'regr.ranger' on task 'df.tr' (iter 32/50) 
    DEBUG [09:09:36.698] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 6..]}
    DEBUG [09:09:36.701] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.711] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.713] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.716] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71]}
    DEBUG [09:09:36.719] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.729] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.731] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.732] Applying learner 'regr.ranger' on task 'df.tr' (iter 22/50) 
    DEBUG [09:09:36.741] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57..]}
    DEBUG [09:09:36.744] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.754] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.756] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.759] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75]}
    DEBUG [09:09:36.762] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.772] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.773] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.774] Applying learner 'regr.ranger' on task 'df.tr' (iter 7/50) 
    DEBUG [09:09:36.783] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 6..]}
    DEBUG [09:09:36.786] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.797] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.799] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.801] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74]}
    DEBUG [09:09:36.804] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.815] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.816] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.817] Applying learner 'regr.ranger' on task 'df.tr' (iter 29/50) 
    DEBUG [09:09:36.826] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:36.829] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.840] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.841] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.844] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 18, 25, 28, 30, 39, 45, 48, 51, 54, 55, 61, 66, 71, 73]}
    DEBUG [09:09:36.847] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.857] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.859] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.860] Applying learner 'regr.ranger' on task 'df.tr' (iter 20/50) 
    DEBUG [09:09:36.868] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:36.872] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.882] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.884] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.887] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 16, 17, 32, 36, 37, 42, 43, 45, 46, 53, 61, 65, 69]}
    DEBUG [09:09:36.890] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.900] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.901] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.903] Applying learner 'regr.ranger' on task 'df.tr' (iter 8/50) 
    DEBUG [09:09:36.911] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:36.915] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.925] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.927] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.930] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 75]}
    DEBUG [09:09:36.933] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.943] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.945] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.946] Applying learner 'regr.ranger' on task 'df.tr' (iter 9/50) 
    DEBUG [09:09:36.955] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:36.958] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.969] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:36.970] Creating Prediction for predict set 'test' 
    DEBUG [09:09:36.973] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 23, 27, 40, 42, 46, 49, 56, 58, 61, 64, 67, 70, 71, 72]}
    DEBUG [09:09:36.976] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:36.994] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:36.996] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:36.997] Applying learner 'regr.ranger' on task 'df.tr' (iter 49/50) 
    DEBUG [09:09:37.007] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:37.010] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.023] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:37.024] Creating Prediction for predict set 'test' 
    DEBUG [09:09:37.027] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 7, 14, 16, 25, 26, 27, 40, 44, 56, 67, 69, 75, 76]}
    DEBUG [09:09:37.030] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.040] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:37.042] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:37.043] Applying learner 'regr.ranger' on task 'df.tr' (iter 42/50) 
    DEBUG [09:09:37.052] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70..]}
    DEBUG [09:09:37.055] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.066] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:37.067] Creating Prediction for predict set 'test' 
    DEBUG [09:09:37.070] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66]}
    DEBUG [09:09:37.073] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.084] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:37.085] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:37.086] Applying learner 'regr.ranger' on task 'df.tr' (iter 11/50) 
    DEBUG [09:09:37.095] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 7..]}
    DEBUG [09:09:37.099] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.109] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:37.111] Creating Prediction for predict set 'test' 
    DEBUG [09:09:37.113] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76]}
    DEBUG [09:09:37.116] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.127] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:37.128] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:37.129] Applying learner 'regr.ranger' on task 'df.tr' (iter 44/50) 
    DEBUG [09:09:37.138] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:37.142] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.153] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:37.154] Creating Prediction for predict set 'test' 
    DEBUG [09:09:37.157] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 14, 17, 19, 24, 36, 41, 47, 51, 53, 56, 62, 63, 67, 71]}
    DEBUG [09:09:37.160] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.171] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:37.172] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:37.173] Applying learner 'regr.ranger' on task 'df.tr' (iter 40/50) 
    DEBUG [09:09:37.182] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:37.185] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.196] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:37.197] Creating Prediction for predict set 'test' 
    DEBUG [09:09:37.200] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 8, 16, 17, 24, 25, 27, 29, 41, 44, 53, 54, 67, 72]}
    DEBUG [09:09:37.203] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.213] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:37.215] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:37.216] Applying learner 'regr.ranger' on task 'df.tr' (iter 5/50) 
    DEBUG [09:09:37.225] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:37.228] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.239] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:37.240] Creating Prediction for predict set 'test' 
    DEBUG [09:09:37.243] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 11, 14, 17, 27, 32, 41, 44, 45, 55, 60, 64, 65, 66, 69]}
    DEBUG [09:09:37.246] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.256] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:37.258] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:37.259] Applying learner 'regr.ranger' on task 'df.tr' (iter 50/50) 
    DEBUG [09:09:37.268] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:37.271] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.281] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:37.283] Creating Prediction for predict set 'test' 
    DEBUG [09:09:37.286] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 8, 21, 22, 30, 36, 41, 45, 49, 53, 54, 57, 68, 71]}
    DEBUG [09:09:37.289] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.299] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:37.301] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:37.302] Applying learner 'regr.ranger' on task 'df.tr' (iter 37/50) 
    DEBUG [09:09:37.311] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57..]}
    DEBUG [09:09:37.314] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.324] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:37.326] Creating Prediction for predict set 'test' 
    DEBUG [09:09:37.329] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76]}
    DEBUG [09:09:37.332] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.342] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:37.343] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:37.344] Applying learner 'regr.ranger' on task 'df.tr' (iter 43/50) 
    DEBUG [09:09:37.353] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:37.357] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.367] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:37.369] Creating Prediction for predict set 'test' 
    DEBUG [09:09:37.371] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74]}
    DEBUG [09:09:37.375] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.385] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:37.386] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:37.387] Applying learner 'regr.ranger' on task 'df.tr' (iter 24/50) 
    DEBUG [09:09:37.396] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:37.399] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.410] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:37.411] Creating Prediction for predict set 'test' 
    DEBUG [09:09:37.414] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 13, 17, 25, 32, 35, 36, 37, 38, 50, 63, 65, 68, 69, 74]}
    DEBUG [09:09:37.417] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.427] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:37.429] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:37.430] Applying learner 'regr.ranger' on task 'df.tr' (iter 45/50) 
    DEBUG [09:09:37.439] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:37.442] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.453] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:37.454] Creating Prediction for predict set 'test' 
    DEBUG [09:09:37.457] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 8, 12, 21, 22, 25, 28, 33, 37, 38, 44, 46, 48, 57, 75]}
    DEBUG [09:09:37.460] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.470] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:37.472] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:37.473] Applying learner 'regr.ranger' on task 'df.tr' (iter 31/50) 
    DEBUG [09:09:37.482] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60..]}
    DEBUG [09:09:37.485] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.496] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:37.497] Creating Prediction for predict set 'test' 
    DEBUG [09:09:37.500] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73]}
    DEBUG [09:09:37.503] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.513] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:37.515] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:37.516] Applying learner 'regr.ranger' on task 'df.tr' (iter 36/50) 
    DEBUG [09:09:37.525] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58..]}
    DEBUG [09:09:37.528] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.539] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:37.540] Creating Prediction for predict set 'test' 
    DEBUG [09:09:37.543] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75]}
    DEBUG [09:09:37.546] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.556] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:37.558] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:37.559] Applying learner 'regr.ranger' on task 'df.tr' (iter 16/50) 
    DEBUG [09:09:37.592] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75..]}
    DEBUG [09:09:37.596] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.607] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:37.608] Creating Prediction for predict set 'test' 
    DEBUG [09:09:37.611] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72]}
    DEBUG [09:09:37.614] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:37.625] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:37.626] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:37.635] Finished benchmark 
    INFO  [09:09:39.050] Result of batch 2: 
    INFO  [09:09:39.053]  mtry sample.fraction num.trees importance regr.rmse      resample_result 
    INFO  [09:09:39.053]     2       0.6960982       141   impurity  49.20049 <ResampleResult[19]> 
    INFO  [09:09:39.060] Evaluating 1 configuration(s) 
    INFO  [09:09:39.091] Benchmark with 50 resampling iterations 
    DEBUG [09:09:39.092] Running benchmark() asynchronously with 50 iterations 
    INFO  [09:09:39.103] Applying learner 'regr.ranger' on task 'df.tr' (iter 23/50) 
    DEBUG [09:09:39.110] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:39.113] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.125] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:39.126] Creating Prediction for predict set 'test' 
    DEBUG [09:09:39.129] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66]}
    DEBUG [09:09:39.131] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.141] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:39.142] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:39.143] Applying learner 'regr.ranger' on task 'df.tr' (iter 30/50) 
    DEBUG [09:09:39.150] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:39.153] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.165] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:39.167] Creating Prediction for predict set 'test' 
    DEBUG [09:09:39.170] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 14, 22, 29, 34, 35, 37, 40, 43, 44, 59, 62, 64, 68, 72]}
    DEBUG [09:09:39.172] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.182] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:39.183] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:39.184] Applying learner 'regr.ranger' on task 'df.tr' (iter 25/50) 
    DEBUG [09:09:39.191] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:39.194] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.221] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:39.223] Creating Prediction for predict set 'test' 
    DEBUG [09:09:39.225] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [9, 11, 18, 21, 23, 27, 29, 43, 44, 45, 51, 55, 67, 73, 76]}
    DEBUG [09:09:39.228] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.238] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:39.240] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:39.241] Applying learner 'regr.ranger' on task 'df.tr' (iter 39/50) 
    DEBUG [09:09:39.248] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:39.250] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.261] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:39.263] Creating Prediction for predict set 'test' 
    DEBUG [09:09:39.265] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 5, 14, 18, 19, 21, 28, 38, 48, 49, 51, 52, 59, 64, 65]}
    DEBUG [09:09:39.267] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.277] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:39.278] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:39.279] Applying learner 'regr.ranger' on task 'df.tr' (iter 6/50) 
    DEBUG [09:09:39.290] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 7..]}
    DEBUG [09:09:39.293] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.306] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:39.307] Creating Prediction for predict set 'test' 
    DEBUG [09:09:39.310] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76]}
    DEBUG [09:09:39.312] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.322] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:39.323] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:39.324] Applying learner 'regr.ranger' on task 'df.tr' (iter 4/50) 
    DEBUG [09:09:39.331] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:39.334] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.346] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:39.347] Creating Prediction for predict set 'test' 
    DEBUG [09:09:39.349] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 3, 13, 20, 23, 25, 40, 46, 50, 53, 56, 57, 58, 63]}
    DEBUG [09:09:39.352] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.361] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:39.363] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:39.363] Applying learner 'regr.ranger' on task 'df.tr' (iter 3/50) 
    DEBUG [09:09:39.372] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:39.375] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.391] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:39.393] Creating Prediction for predict set 'test' 
    DEBUG [09:09:39.397] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75, 76]}
    DEBUG [09:09:39.400] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.412] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:39.413] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:39.414] Applying learner 'regr.ranger' on task 'df.tr' (iter 10/50) 
    DEBUG [09:09:39.423] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:39.427] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.442] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:39.444] Creating Prediction for predict set 'test' 
    DEBUG [09:09:39.447] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 17, 18, 19, 24, 26, 29, 31, 34, 36, 43, 45, 53, 66, 73]}
    DEBUG [09:09:39.451] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.462] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:39.464] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:39.465] Applying learner 'regr.ranger' on task 'df.tr' (iter 28/50) 
    DEBUG [09:09:39.474] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:39.477] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.491] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:39.492] Creating Prediction for predict set 'test' 
    DEBUG [09:09:39.495] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75]}
    DEBUG [09:09:39.498] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.509] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:39.511] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:39.512] Applying learner 'regr.ranger' on task 'df.tr' (iter 12/50) 
    DEBUG [09:09:39.521] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61..]}
    DEBUG [09:09:39.524] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.540] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:39.542] Creating Prediction for predict set 'test' 
    DEBUG [09:09:39.546] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70]}
    DEBUG [09:09:39.550] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.562] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:39.563] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:39.564] Applying learner 'regr.ranger' on task 'df.tr' (iter 44/50) 
    DEBUG [09:09:39.573] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:39.577] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.590] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:39.592] Creating Prediction for predict set 'test' 
    DEBUG [09:09:39.595] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 14, 17, 19, 24, 36, 41, 47, 51, 53, 56, 62, 63, 67, 71]}
    DEBUG [09:09:39.599] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.610] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:39.612] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:39.613] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [09:09:39.622] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [09:09:39.625] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.639] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:39.640] Creating Prediction for predict set 'test' 
    DEBUG [09:09:39.643] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [09:09:39.646] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.657] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:39.659] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:39.660] Applying learner 'regr.ranger' on task 'df.tr' (iter 33/50) 
    DEBUG [09:09:39.669] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:39.672] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.686] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:39.687] Creating Prediction for predict set 'test' 
    DEBUG [09:09:39.690] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60, 76]}
    DEBUG [09:09:39.693] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.705] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:39.706] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:39.707] Applying learner 'regr.ranger' on task 'df.tr' (iter 36/50) 
    DEBUG [09:09:39.716] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58..]}
    DEBUG [09:09:39.720] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.735] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:39.737] Creating Prediction for predict set 'test' 
    DEBUG [09:09:39.740] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75]}
    DEBUG [09:09:39.744] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.756] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:39.758] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:39.759] Applying learner 'regr.ranger' on task 'df.tr' (iter 13/50) 
    DEBUG [09:09:39.768] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:39.771] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.787] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:39.789] Creating Prediction for predict set 'test' 
    DEBUG [09:09:39.791] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 72]}
    DEBUG [09:09:39.795] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.806] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:39.808] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:39.809] Applying learner 'regr.ranger' on task 'df.tr' (iter 15/50) 
    DEBUG [09:09:39.818] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:39.821] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.837] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:39.839] Creating Prediction for predict set 'test' 
    DEBUG [09:09:39.843] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 19, 20, 30, 35, 42, 47, 54, 59, 64, 67, 71, 74, 75]}
    DEBUG [09:09:39.847] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.858] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:39.860] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:39.861] Applying learner 'regr.ranger' on task 'df.tr' (iter 37/50) 
    DEBUG [09:09:39.870] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57..]}
    DEBUG [09:09:39.873] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.890] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:39.892] Creating Prediction for predict set 'test' 
    DEBUG [09:09:39.897] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76]}
    DEBUG [09:09:39.900] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.911] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:39.913] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:39.914] Applying learner 'regr.ranger' on task 'df.tr' (iter 49/50) 
    DEBUG [09:09:39.923] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:39.926] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.942] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:39.943] Creating Prediction for predict set 'test' 
    DEBUG [09:09:39.947] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 7, 14, 16, 25, 26, 27, 40, 44, 56, 67, 69, 75, 76]}
    DEBUG [09:09:39.950] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.961] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:39.963] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:39.964] Applying learner 'regr.ranger' on task 'df.tr' (iter 50/50) 
    DEBUG [09:09:39.973] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:39.976] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:39.992] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:39.995] Creating Prediction for predict set 'test' 
    DEBUG [09:09:39.998] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 8, 21, 22, 30, 36, 41, 45, 49, 53, 54, 57, 68, 71]}
    DEBUG [09:09:40.002] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.015] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:40.016] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:40.017] Applying learner 'regr.ranger' on task 'df.tr' (iter 18/50) 
    DEBUG [09:09:40.027] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:40.031] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.046] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:40.047] Creating Prediction for predict set 'test' 
    DEBUG [09:09:40.051] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75]}
    DEBUG [09:09:40.054] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.067] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:40.069] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:40.070] Applying learner 'regr.ranger' on task 'df.tr' (iter 41/50) 
    DEBUG [09:09:40.080] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74..]}
    DEBUG [09:09:40.084] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.098] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:40.100] Creating Prediction for predict set 'test' 
    DEBUG [09:09:40.103] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76]}
    DEBUG [09:09:40.107] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.119] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:40.121] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:40.122] Applying learner 'regr.ranger' on task 'df.tr' (iter 31/50) 
    DEBUG [09:09:40.132] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60..]}
    DEBUG [09:09:40.136] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.152] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:40.155] Creating Prediction for predict set 'test' 
    DEBUG [09:09:40.159] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73]}
    DEBUG [09:09:40.163] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.176] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:40.177] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:40.179] Applying learner 'regr.ranger' on task 'df.tr' (iter 7/50) 
    DEBUG [09:09:40.189] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 6..]}
    DEBUG [09:09:40.192] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.207] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:40.209] Creating Prediction for predict set 'test' 
    DEBUG [09:09:40.212] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74]}
    DEBUG [09:09:40.216] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.228] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:40.230] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:40.231] Applying learner 'regr.ranger' on task 'df.tr' (iter 26/50) 
    DEBUG [09:09:40.241] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75..]}
    DEBUG [09:09:40.245] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.259] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:40.261] Creating Prediction for predict set 'test' 
    DEBUG [09:09:40.264] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74]}
    DEBUG [09:09:40.268] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.280] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:40.282] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:40.283] Applying learner 'regr.ranger' on task 'df.tr' (iter 45/50) 
    DEBUG [09:09:40.293] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:40.297] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.312] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:40.314] Creating Prediction for predict set 'test' 
    DEBUG [09:09:40.317] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 8, 12, 21, 22, 25, 28, 33, 37, 38, 44, 46, 48, 57, 75]}
    DEBUG [09:09:40.321] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.333] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:40.335] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:40.336] Applying learner 'regr.ranger' on task 'df.tr' (iter 27/50) 
    DEBUG [09:09:40.346] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67..]}
    DEBUG [09:09:40.350] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.367] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:40.370] Creating Prediction for predict set 'test' 
    DEBUG [09:09:40.374] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76]}
    DEBUG [09:09:40.377] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.390] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:40.391] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:40.393] Applying learner 'regr.ranger' on task 'df.tr' (iter 24/50) 
    DEBUG [09:09:40.415] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:40.418] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.433] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:40.434] Creating Prediction for predict set 'test' 
    DEBUG [09:09:40.437] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 13, 17, 25, 32, 35, 36, 37, 38, 50, 63, 65, 68, 69, 74]}
    DEBUG [09:09:40.440] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.452] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:40.453] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:40.454] Applying learner 'regr.ranger' on task 'df.tr' (iter 40/50) 
    DEBUG [09:09:40.463] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:40.466] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.482] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:40.484] Creating Prediction for predict set 'test' 
    DEBUG [09:09:40.487] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 8, 16, 17, 24, 25, 27, 29, 41, 44, 53, 54, 67, 72]}
    DEBUG [09:09:40.490] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.501] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:40.503] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:40.504] Applying learner 'regr.ranger' on task 'df.tr' (iter 11/50) 
    DEBUG [09:09:40.513] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 7..]}
    DEBUG [09:09:40.516] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.535] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:40.538] Creating Prediction for predict set 'test' 
    DEBUG [09:09:40.541] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76]}
    DEBUG [09:09:40.544] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.559] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:40.561] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:40.562] Applying learner 'regr.ranger' on task 'df.tr' (iter 19/50) 
    DEBUG [09:09:40.571] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:40.575] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.591] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:40.593] Creating Prediction for predict set 'test' 
    DEBUG [09:09:40.596] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 8, 11, 14, 18, 24, 28, 33, 34, 54, 56, 62, 64, 68, 76]}
    DEBUG [09:09:40.599] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.610] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:40.611] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:40.613] Applying learner 'regr.ranger' on task 'df.tr' (iter 16/50) 
    DEBUG [09:09:40.623] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75..]}
    DEBUG [09:09:40.627] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.642] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:40.643] Creating Prediction for predict set 'test' 
    DEBUG [09:09:40.647] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72]}
    DEBUG [09:09:40.650] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.662] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:40.664] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:40.665] Applying learner 'regr.ranger' on task 'df.tr' (iter 35/50) 
    DEBUG [09:09:40.675] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:40.679] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.694] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:40.695] Creating Prediction for predict set 'test' 
    DEBUG [09:09:40.699] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 29, 32, 33, 35, 40, 43, 44, 45, 61, 62, 70, 74]}
    DEBUG [09:09:40.702] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.715] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:40.717] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:40.718] Applying learner 'regr.ranger' on task 'df.tr' (iter 22/50) 
    DEBUG [09:09:40.729] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57..]}
    DEBUG [09:09:40.733] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.750] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:40.753] Creating Prediction for predict set 'test' 
    DEBUG [09:09:40.757] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75]}
    DEBUG [09:09:40.761] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.774] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:40.775] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:40.777] Applying learner 'regr.ranger' on task 'df.tr' (iter 46/50) 
    DEBUG [09:09:40.786] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72..]}
    DEBUG [09:09:40.790] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.804] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:40.805] Creating Prediction for predict set 'test' 
    DEBUG [09:09:40.808] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73]}
    DEBUG [09:09:40.812] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.823] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:40.825] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:40.826] Applying learner 'regr.ranger' on task 'df.tr' (iter 20/50) 
    DEBUG [09:09:40.835] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:40.838] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.852] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:40.853] Creating Prediction for predict set 'test' 
    DEBUG [09:09:40.856] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 16, 17, 32, 36, 37, 42, 43, 45, 46, 53, 61, 65, 69]}
    DEBUG [09:09:40.859] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.871] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:40.873] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:40.874] Applying learner 'regr.ranger' on task 'df.tr' (iter 21/50) 
    DEBUG [09:09:40.883] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66..]}
    DEBUG [09:09:40.886] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.902] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:40.904] Creating Prediction for predict set 'test' 
    DEBUG [09:09:40.907] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72]}
    DEBUG [09:09:40.910] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.922] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:40.923] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:40.924] Applying learner 'regr.ranger' on task 'df.tr' (iter 5/50) 
    DEBUG [09:09:40.933] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:40.937] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.950] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:40.952] Creating Prediction for predict set 'test' 
    DEBUG [09:09:40.955] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 11, 14, 17, 27, 32, 41, 44, 45, 55, 60, 64, 65, 66, 69]}
    DEBUG [09:09:40.958] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.969] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:40.970] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:40.971] Applying learner 'regr.ranger' on task 'df.tr' (iter 2/50) 
    DEBUG [09:09:40.980] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [09:09:40.984] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:40.997] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:40.999] Creating Prediction for predict set 'test' 
    DEBUG [09:09:41.002] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68]}
    DEBUG [09:09:41.005] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.016] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:41.018] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:41.019] Applying learner 'regr.ranger' on task 'df.tr' (iter 17/50) 
    DEBUG [09:09:41.028] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 7..]}
    DEBUG [09:09:41.031] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.044] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:41.046] Creating Prediction for predict set 'test' 
    DEBUG [09:09:41.049] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70]}
    DEBUG [09:09:41.052] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.064] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:41.065] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:41.066] Applying learner 'regr.ranger' on task 'df.tr' (iter 47/50) 
    DEBUG [09:09:41.075] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64..]}
    DEBUG [09:09:41.079] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.092] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:41.094] Creating Prediction for predict set 'test' 
    DEBUG [09:09:41.098] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70]}
    DEBUG [09:09:41.101] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.112] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:41.114] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:41.115] Applying learner 'regr.ranger' on task 'df.tr' (iter 14/50) 
    DEBUG [09:09:41.124] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:41.127] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.140] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:41.142] Creating Prediction for predict set 'test' 
    DEBUG [09:09:41.145] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 22, 27, 36, 38, 40, 45, 46, 50, 55, 56, 60, 65, 73]}
    DEBUG [09:09:41.148] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.159] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:41.161] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:41.162] Applying learner 'regr.ranger' on task 'df.tr' (iter 32/50) 
    DEBUG [09:09:41.171] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 6..]}
    DEBUG [09:09:41.174] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.188] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:41.189] Creating Prediction for predict set 'test' 
    DEBUG [09:09:41.192] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71]}
    DEBUG [09:09:41.195] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.206] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:41.208] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:41.209] Applying learner 'regr.ranger' on task 'df.tr' (iter 42/50) 
    DEBUG [09:09:41.218] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70..]}
    DEBUG [09:09:41.221] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.236] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:41.238] Creating Prediction for predict set 'test' 
    DEBUG [09:09:41.242] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66]}
    DEBUG [09:09:41.245] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.256] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:41.258] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:41.259] Applying learner 'regr.ranger' on task 'df.tr' (iter 9/50) 
    DEBUG [09:09:41.268] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:41.271] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.308] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:41.310] Creating Prediction for predict set 'test' 
    DEBUG [09:09:41.313] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 23, 27, 40, 42, 46, 49, 56, 58, 61, 64, 67, 70, 71, 72]}
    DEBUG [09:09:41.316] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.328] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:41.330] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:41.331] Applying learner 'regr.ranger' on task 'df.tr' (iter 38/50) 
    DEBUG [09:09:41.340] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:41.343] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.356] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:41.358] Creating Prediction for predict set 'test' 
    DEBUG [09:09:41.361] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58, 69]}
    DEBUG [09:09:41.364] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.375] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:41.377] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:41.378] Applying learner 'regr.ranger' on task 'df.tr' (iter 8/50) 
    DEBUG [09:09:41.387] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:41.390] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.406] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:41.408] Creating Prediction for predict set 'test' 
    DEBUG [09:09:41.411] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 75]}
    DEBUG [09:09:41.414] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.425] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:41.427] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:41.428] Applying learner 'regr.ranger' on task 'df.tr' (iter 43/50) 
    DEBUG [09:09:41.437] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:41.440] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.454] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:41.455] Creating Prediction for predict set 'test' 
    DEBUG [09:09:41.458] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74]}
    DEBUG [09:09:41.461] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.472] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:41.474] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:41.475] Applying learner 'regr.ranger' on task 'df.tr' (iter 34/50) 
    DEBUG [09:09:41.484] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:41.487] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.503] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:41.505] Creating Prediction for predict set 'test' 
    DEBUG [09:09:41.509] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 7, 12, 14, 18, 20, 41, 46, 47, 53, 57, 63, 64, 72, 75]}
    DEBUG [09:09:41.512] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.523] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:41.525] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:41.526] Applying learner 'regr.ranger' on task 'df.tr' (iter 29/50) 
    DEBUG [09:09:41.535] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:41.538] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.551] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:41.552] Creating Prediction for predict set 'test' 
    DEBUG [09:09:41.555] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 18, 25, 28, 30, 39, 45, 48, 51, 54, 55, 61, 66, 71, 73]}
    DEBUG [09:09:41.558] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.570] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:41.571] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:41.572] Applying learner 'regr.ranger' on task 'df.tr' (iter 48/50) 
    DEBUG [09:09:41.581] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:41.584] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.598] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:41.599] Creating Prediction for predict set 'test' 
    DEBUG [09:09:41.602] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72, 74]}
    DEBUG [09:09:41.605] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:41.616] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:41.618] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:41.631] Finished benchmark 
    INFO  [09:09:43.076] Result of batch 3: 
    INFO  [09:09:43.078]  mtry sample.fraction num.trees importance regr.rmse      resample_result 
    INFO  [09:09:43.078]     3       0.6379495       381   impurity  47.24917 <ResampleResult[19]> 
    INFO  [09:09:43.085] Evaluating 1 configuration(s) 
    INFO  [09:09:43.117] Benchmark with 50 resampling iterations 
    DEBUG [09:09:43.118] Running benchmark() asynchronously with 50 iterations 
    INFO  [09:09:43.129] Applying learner 'regr.ranger' on task 'df.tr' (iter 6/50) 
    DEBUG [09:09:43.136] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 7..]}
    DEBUG [09:09:43.139] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.149] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:43.150] Creating Prediction for predict set 'test' 
    DEBUG [09:09:43.152] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76]}
    DEBUG [09:09:43.155] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.164] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:43.165] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:43.166] Applying learner 'regr.ranger' on task 'df.tr' (iter 4/50) 
    DEBUG [09:09:43.173] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:43.176] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.185] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:43.186] Creating Prediction for predict set 'test' 
    DEBUG [09:09:43.188] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 3, 13, 20, 23, 25, 40, 46, 50, 53, 56, 57, 58, 63]}
    DEBUG [09:09:43.191] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.199] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:43.201] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:43.201] Applying learner 'regr.ranger' on task 'df.tr' (iter 3/50) 
    DEBUG [09:09:43.209] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:43.211] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.220] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:43.222] Creating Prediction for predict set 'test' 
    DEBUG [09:09:43.224] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75, 76]}
    DEBUG [09:09:43.226] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.235] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:43.236] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:43.237] Applying learner 'regr.ranger' on task 'df.tr' (iter 46/50) 
    DEBUG [09:09:43.244] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72..]}
    DEBUG [09:09:43.247] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.261] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:43.262] Creating Prediction for predict set 'test' 
    DEBUG [09:09:43.265] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73]}
    DEBUG [09:09:43.267] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.277] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:43.278] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:43.279] Applying learner 'regr.ranger' on task 'df.tr' (iter 47/50) 
    DEBUG [09:09:43.287] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64..]}
    DEBUG [09:09:43.290] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.303] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:43.304] Creating Prediction for predict set 'test' 
    DEBUG [09:09:43.307] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70]}
    DEBUG [09:09:43.310] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.319] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:43.321] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:43.322] Applying learner 'regr.ranger' on task 'df.tr' (iter 10/50) 
    DEBUG [09:09:43.330] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:43.333] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.343] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:43.344] Creating Prediction for predict set 'test' 
    DEBUG [09:09:43.347] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 17, 18, 19, 24, 26, 29, 31, 34, 36, 43, 45, 53, 66, 73]}
    DEBUG [09:09:43.350] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.359] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:43.360] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:43.361] Applying learner 'regr.ranger' on task 'df.tr' (iter 28/50) 
    DEBUG [09:09:43.370] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:43.373] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.382] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:43.384] Creating Prediction for predict set 'test' 
    DEBUG [09:09:43.387] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75]}
    DEBUG [09:09:43.389] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.399] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:43.400] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:43.401] Applying learner 'regr.ranger' on task 'df.tr' (iter 12/50) 
    DEBUG [09:09:43.409] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61..]}
    DEBUG [09:09:43.412] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.423] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:43.424] Creating Prediction for predict set 'test' 
    DEBUG [09:09:43.427] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70]}
    DEBUG [09:09:43.429] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.439] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:43.440] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:43.441] Applying learner 'regr.ranger' on task 'df.tr' (iter 48/50) 
    DEBUG [09:09:43.449] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:43.452] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.462] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:43.464] Creating Prediction for predict set 'test' 
    DEBUG [09:09:43.466] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72, 74]}
    DEBUG [09:09:43.469] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.479] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:43.480] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:43.481] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [09:09:43.489] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [09:09:43.492] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.502] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:43.503] Creating Prediction for predict set 'test' 
    DEBUG [09:09:43.506] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [09:09:43.509] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.518] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:43.519] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:43.520] Applying learner 'regr.ranger' on task 'df.tr' (iter 33/50) 
    DEBUG [09:09:43.528] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:43.531] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.541] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:43.542] Creating Prediction for predict set 'test' 
    DEBUG [09:09:43.545] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60, 76]}
    DEBUG [09:09:43.547] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.557] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:43.558] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:43.559] Applying learner 'regr.ranger' on task 'df.tr' (iter 36/50) 
    DEBUG [09:09:43.567] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58..]}
    DEBUG [09:09:43.570] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.580] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:43.581] Creating Prediction for predict set 'test' 
    DEBUG [09:09:43.584] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75]}
    DEBUG [09:09:43.587] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.595] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:43.597] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:43.598] Applying learner 'regr.ranger' on task 'df.tr' (iter 13/50) 
    DEBUG [09:09:43.606] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:43.609] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.618] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:43.620] Creating Prediction for predict set 'test' 
    DEBUG [09:09:43.622] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 72]}
    DEBUG [09:09:43.625] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.635] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:43.637] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:43.638] Applying learner 'regr.ranger' on task 'df.tr' (iter 15/50) 
    DEBUG [09:09:43.648] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:43.652] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.664] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:43.666] Creating Prediction for predict set 'test' 
    DEBUG [09:09:43.669] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 19, 20, 30, 35, 42, 47, 54, 59, 64, 67, 71, 74, 75]}
    DEBUG [09:09:43.673] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.684] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:43.686] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:43.687] Applying learner 'regr.ranger' on task 'df.tr' (iter 38/50) 
    DEBUG [09:09:43.697] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:43.701] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.713] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:43.715] Creating Prediction for predict set 'test' 
    DEBUG [09:09:43.718] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58, 69]}
    DEBUG [09:09:43.722] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.733] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:43.735] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:43.736] Applying learner 'regr.ranger' on task 'df.tr' (iter 30/50) 
    DEBUG [09:09:43.747] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:43.750] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:43.762] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:43.764] Creating Prediction for predict set 'test' 
    DEBUG [09:09:43.767] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 14, 22, 29, 34, 35, 37, 40, 43, 44, 59, 62, 64, 68, 72]}
    DEBUG [09:09:44.346] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.358] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:44.359] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:44.360] Applying learner 'regr.ranger' on task 'df.tr' (iter 18/50) 
    DEBUG [09:09:44.369] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:44.373] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.383] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:44.385] Creating Prediction for predict set 'test' 
    DEBUG [09:09:44.388] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75]}
    DEBUG [09:09:44.391] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.401] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:44.402] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:44.403] Applying learner 'regr.ranger' on task 'df.tr' (iter 26/50) 
    DEBUG [09:09:44.412] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75..]}
    DEBUG [09:09:44.415] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.426] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:44.427] Creating Prediction for predict set 'test' 
    DEBUG [09:09:44.430] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74]}
    DEBUG [09:09:44.433] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.443] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:44.445] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:44.446] Applying learner 'regr.ranger' on task 'df.tr' (iter 49/50) 
    DEBUG [09:09:44.454] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:44.457] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.468] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:44.470] Creating Prediction for predict set 'test' 
    DEBUG [09:09:44.472] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 7, 14, 16, 25, 26, 27, 40, 44, 56, 67, 69, 75, 76]}
    DEBUG [09:09:44.475] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.485] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:44.487] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:44.488] Applying learner 'regr.ranger' on task 'df.tr' (iter 32/50) 
    DEBUG [09:09:44.497] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 6..]}
    DEBUG [09:09:44.500] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.511] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:44.512] Creating Prediction for predict set 'test' 
    DEBUG [09:09:44.515] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71]}
    DEBUG [09:09:44.518] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.528] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:44.529] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:44.530] Applying learner 'regr.ranger' on task 'df.tr' (iter 24/50) 
    DEBUG [09:09:44.539] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:44.542] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.553] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:44.554] Creating Prediction for predict set 'test' 
    DEBUG [09:09:44.557] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 13, 17, 25, 32, 35, 36, 37, 38, 50, 63, 65, 68, 69, 74]}
    DEBUG [09:09:44.560] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.570] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:44.572] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:44.573] Applying learner 'regr.ranger' on task 'df.tr' (iter 25/50) 
    DEBUG [09:09:44.581] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:44.584] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.595] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:44.596] Creating Prediction for predict set 'test' 
    DEBUG [09:09:44.599] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [9, 11, 18, 21, 23, 27, 29, 43, 44, 45, 51, 55, 67, 73, 76]}
    DEBUG [09:09:44.602] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.612] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:44.614] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:44.615] Applying learner 'regr.ranger' on task 'df.tr' (iter 42/50) 
    DEBUG [09:09:44.623] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70..]}
    DEBUG [09:09:44.627] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.637] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:44.639] Creating Prediction for predict set 'test' 
    DEBUG [09:09:44.642] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66]}
    DEBUG [09:09:44.645] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.654] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:44.656] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:44.657] Applying learner 'regr.ranger' on task 'df.tr' (iter 29/50) 
    DEBUG [09:09:44.666] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:44.669] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.680] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:44.681] Creating Prediction for predict set 'test' 
    DEBUG [09:09:44.684] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 18, 25, 28, 30, 39, 45, 48, 51, 54, 55, 61, 66, 71, 73]}
    DEBUG [09:09:44.687] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.697] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:44.699] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:44.700] Applying learner 'regr.ranger' on task 'df.tr' (iter 11/50) 
    DEBUG [09:09:44.709] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 7..]}
    DEBUG [09:09:44.712] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.723] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:44.724] Creating Prediction for predict set 'test' 
    DEBUG [09:09:44.727] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76]}
    DEBUG [09:09:44.737] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.751] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:44.752] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:44.753] Applying learner 'regr.ranger' on task 'df.tr' (iter 19/50) 
    DEBUG [09:09:44.763] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:44.766] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.778] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:44.779] Creating Prediction for predict set 'test' 
    DEBUG [09:09:44.782] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 8, 11, 14, 18, 24, 28, 33, 34, 54, 56, 62, 64, 68, 76]}
    DEBUG [09:09:44.785] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.797] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:44.799] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:44.800] Applying learner 'regr.ranger' on task 'df.tr' (iter 16/50) 
    DEBUG [09:09:44.812] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75..]}
    DEBUG [09:09:44.815] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.829] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:44.832] Creating Prediction for predict set 'test' 
    DEBUG [09:09:44.837] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72]}
    DEBUG [09:09:44.840] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.851] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:44.853] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:44.854] Applying learner 'regr.ranger' on task 'df.tr' (iter 37/50) 
    DEBUG [09:09:44.864] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57..]}
    DEBUG [09:09:44.867] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.879] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:44.880] Creating Prediction for predict set 'test' 
    DEBUG [09:09:44.884] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76]}
    DEBUG [09:09:44.887] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.898] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:44.900] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:44.901] Applying learner 'regr.ranger' on task 'df.tr' (iter 40/50) 
    DEBUG [09:09:44.911] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:44.915] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.927] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:44.929] Creating Prediction for predict set 'test' 
    DEBUG [09:09:44.932] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 8, 16, 17, 24, 25, 27, 29, 41, 44, 53, 54, 67, 72]}
    DEBUG [09:09:44.936] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.946] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:44.948] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:44.949] Applying learner 'regr.ranger' on task 'df.tr' (iter 50/50) 
    DEBUG [09:09:44.958] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:44.962] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.973] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:44.975] Creating Prediction for predict set 'test' 
    DEBUG [09:09:44.978] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 8, 21, 22, 30, 36, 41, 45, 49, 53, 54, 57, 68, 71]}
    DEBUG [09:09:44.982] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:44.993] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:44.994] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:44.995] Applying learner 'regr.ranger' on task 'df.tr' (iter 5/50) 
    DEBUG [09:09:45.005] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:45.009] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.020] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:45.021] Creating Prediction for predict set 'test' 
    DEBUG [09:09:45.024] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 11, 14, 17, 27, 32, 41, 44, 45, 55, 60, 64, 65, 66, 69]}
    DEBUG [09:09:45.028] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.038] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:45.040] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:45.041] Applying learner 'regr.ranger' on task 'df.tr' (iter 2/50) 
    DEBUG [09:09:45.051] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [09:09:45.055] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.066] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:45.068] Creating Prediction for predict set 'test' 
    DEBUG [09:09:45.071] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68]}
    DEBUG [09:09:45.074] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.085] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:45.086] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:45.087] Applying learner 'regr.ranger' on task 'df.tr' (iter 17/50) 
    DEBUG [09:09:45.097] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 7..]}
    DEBUG [09:09:45.100] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.112] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:45.113] Creating Prediction for predict set 'test' 
    DEBUG [09:09:45.117] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70]}
    DEBUG [09:09:45.120] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.131] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:45.133] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:45.134] Applying learner 'regr.ranger' on task 'df.tr' (iter 20/50) 
    DEBUG [09:09:45.143] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:45.147] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.158] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:45.159] Creating Prediction for predict set 'test' 
    DEBUG [09:09:45.162] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 16, 17, 32, 36, 37, 42, 43, 45, 46, 53, 61, 65, 69]}
    DEBUG [09:09:45.166] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.176] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:45.178] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:45.179] Applying learner 'regr.ranger' on task 'df.tr' (iter 44/50) 
    DEBUG [09:09:45.189] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:45.193] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.204] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:45.205] Creating Prediction for predict set 'test' 
    DEBUG [09:09:45.208] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 14, 17, 19, 24, 36, 41, 47, 51, 53, 56, 62, 63, 67, 71]}
    DEBUG [09:09:45.212] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.222] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:45.224] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:45.225] Applying learner 'regr.ranger' on task 'df.tr' (iter 23/50) 
    DEBUG [09:09:45.234] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:45.238] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.249] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:45.251] Creating Prediction for predict set 'test' 
    DEBUG [09:09:45.254] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66]}
    DEBUG [09:09:45.258] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.268] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:45.270] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:45.271] Applying learner 'regr.ranger' on task 'df.tr' (iter 35/50) 
    DEBUG [09:09:45.280] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:45.284] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.295] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:45.296] Creating Prediction for predict set 'test' 
    DEBUG [09:09:45.300] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 29, 32, 33, 35, 40, 43, 44, 45, 61, 62, 70, 74]}
    DEBUG [09:09:45.303] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.313] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:45.315] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:45.316] Applying learner 'regr.ranger' on task 'df.tr' (iter 14/50) 
    DEBUG [09:09:45.326] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:45.330] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.341] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:45.342] Creating Prediction for predict set 'test' 
    DEBUG [09:09:45.345] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 22, 27, 36, 38, 40, 45, 46, 50, 55, 56, 60, 65, 73]}
    DEBUG [09:09:45.349] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.359] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:45.361] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:45.362] Applying learner 'regr.ranger' on task 'df.tr' (iter 22/50) 
    DEBUG [09:09:45.371] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57..]}
    DEBUG [09:09:45.375] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.386] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:45.388] Creating Prediction for predict set 'test' 
    DEBUG [09:09:45.391] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75]}
    DEBUG [09:09:45.395] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.406] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:45.407] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:45.408] Applying learner 'regr.ranger' on task 'df.tr' (iter 21/50) 
    DEBUG [09:09:45.418] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66..]}
    DEBUG [09:09:45.422] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.433] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:45.434] Creating Prediction for predict set 'test' 
    DEBUG [09:09:45.438] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72]}
    DEBUG [09:09:45.441] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.452] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:45.453] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:45.454] Applying learner 'regr.ranger' on task 'df.tr' (iter 9/50) 
    DEBUG [09:09:45.465] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:45.468] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.480] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:45.481] Creating Prediction for predict set 'test' 
    DEBUG [09:09:45.484] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 23, 27, 40, 42, 46, 49, 56, 58, 61, 64, 67, 70, 71, 72]}
    DEBUG [09:09:45.488] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.498] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:45.500] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:45.501] Applying learner 'regr.ranger' on task 'df.tr' (iter 39/50) 
    DEBUG [09:09:45.510] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:45.514] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.525] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:45.527] Creating Prediction for predict set 'test' 
    DEBUG [09:09:45.530] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 5, 14, 18, 19, 21, 28, 38, 48, 49, 51, 52, 59, 64, 65]}
    DEBUG [09:09:45.534] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.544] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:45.546] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:45.547] Applying learner 'regr.ranger' on task 'df.tr' (iter 43/50) 
    DEBUG [09:09:45.556] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:45.560] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.571] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:45.572] Creating Prediction for predict set 'test' 
    DEBUG [09:09:45.576] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74]}
    DEBUG [09:09:45.579] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.590] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:45.591] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:45.593] Applying learner 'regr.ranger' on task 'df.tr' (iter 27/50) 
    DEBUG [09:09:45.602] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67..]}
    DEBUG [09:09:45.605] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.617] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:45.619] Creating Prediction for predict set 'test' 
    DEBUG [09:09:45.622] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76]}
    DEBUG [09:09:45.625] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.636] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:45.637] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:45.638] Applying learner 'regr.ranger' on task 'df.tr' (iter 34/50) 
    DEBUG [09:09:45.648] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:45.651] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.662] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:45.664] Creating Prediction for predict set 'test' 
    DEBUG [09:09:45.667] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 7, 12, 14, 18, 20, 41, 46, 47, 53, 57, 63, 64, 72, 75]}
    DEBUG [09:09:45.671] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.681] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:45.683] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:45.684] Applying learner 'regr.ranger' on task 'df.tr' (iter 8/50) 
    DEBUG [09:09:45.693] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:45.697] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.708] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:45.710] Creating Prediction for predict set 'test' 
    DEBUG [09:09:45.713] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 75]}
    DEBUG [09:09:45.716] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.727] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:45.728] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:45.729] Applying learner 'regr.ranger' on task 'df.tr' (iter 45/50) 
    DEBUG [09:09:45.739] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:45.743] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.777] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:45.779] Creating Prediction for predict set 'test' 
    DEBUG [09:09:45.781] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 8, 12, 21, 22, 25, 28, 33, 37, 38, 44, 46, 48, 57, 75]}
    DEBUG [09:09:45.785] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.795] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:45.797] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:45.798] Applying learner 'regr.ranger' on task 'df.tr' (iter 7/50) 
    DEBUG [09:09:45.807] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 6..]}
    DEBUG [09:09:45.810] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.821] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:45.823] Creating Prediction for predict set 'test' 
    DEBUG [09:09:45.825] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74]}
    DEBUG [09:09:45.829] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.839] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:45.840] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:45.842] Applying learner 'regr.ranger' on task 'df.tr' (iter 31/50) 
    DEBUG [09:09:45.850] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60..]}
    DEBUG [09:09:45.854] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.865] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:45.866] Creating Prediction for predict set 'test' 
    DEBUG [09:09:45.869] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73]}
    DEBUG [09:09:45.872] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.882] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:45.884] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:45.885] Applying learner 'regr.ranger' on task 'df.tr' (iter 41/50) 
    DEBUG [09:09:45.894] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74..]}
    DEBUG [09:09:45.897] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.908] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:45.909] Creating Prediction for predict set 'test' 
    DEBUG [09:09:45.912] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76]}
    DEBUG [09:09:45.915] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:45.925] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:45.927] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:45.944] Finished benchmark 
    INFO  [09:09:47.340] Result of batch 4: 
    INFO  [09:09:47.343]  mtry sample.fraction num.trees importance regr.rmse      resample_result 
    INFO  [09:09:47.343]     5       0.7631363       141   impurity  43.04499 <ResampleResult[19]> 
    INFO  [09:09:47.350] Evaluating 1 configuration(s) 
    INFO  [09:09:47.381] Benchmark with 50 resampling iterations 
    DEBUG [09:09:47.382] Running benchmark() asynchronously with 50 iterations 
    INFO  [09:09:47.393] Applying learner 'regr.ranger' on task 'df.tr' (iter 46/50) 
    DEBUG [09:09:47.400] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72..]}
    DEBUG [09:09:47.403] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.417] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:47.418] Creating Prediction for predict set 'test' 
    DEBUG [09:09:47.421] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73]}
    DEBUG [09:09:47.424] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.434] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:47.435] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:47.436] Applying learner 'regr.ranger' on task 'df.tr' (iter 48/50) 
    DEBUG [09:09:47.443] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:47.446] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.462] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:47.464] Creating Prediction for predict set 'test' 
    DEBUG [09:09:47.467] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72, 74]}
    DEBUG [09:09:47.470] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.481] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:47.482] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:47.483] Applying learner 'regr.ranger' on task 'df.tr' (iter 10/50) 
    DEBUG [09:09:47.504] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:47.508] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.524] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:47.526] Creating Prediction for predict set 'test' 
    DEBUG [09:09:47.529] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 17, 18, 19, 24, 26, 29, 31, 34, 36, 43, 45, 53, 66, 73]}
    DEBUG [09:09:47.532] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.545] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:47.547] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:47.548] Applying learner 'regr.ranger' on task 'df.tr' (iter 28/50) 
    DEBUG [09:09:47.557] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:47.560] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.575] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:47.576] Creating Prediction for predict set 'test' 
    DEBUG [09:09:47.579] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75]}
    DEBUG [09:09:47.582] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.594] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:47.596] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:47.597] Applying learner 'regr.ranger' on task 'df.tr' (iter 12/50) 
    DEBUG [09:09:47.606] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61..]}
    DEBUG [09:09:47.609] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.626] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:47.628] Creating Prediction for predict set 'test' 
    DEBUG [09:09:47.633] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70]}
    DEBUG [09:09:47.636] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.647] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:47.649] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:47.650] Applying learner 'regr.ranger' on task 'df.tr' (iter 3/50) 
    DEBUG [09:09:47.658] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:47.662] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.678] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:47.680] Creating Prediction for predict set 'test' 
    DEBUG [09:09:47.683] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75, 76]}
    DEBUG [09:09:47.686] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.698] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:47.699] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:47.700] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [09:09:47.709] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [09:09:47.712] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.729] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:47.731] Creating Prediction for predict set 'test' 
    DEBUG [09:09:47.734] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [09:09:47.737] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.749] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:47.750] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:47.751] Applying learner 'regr.ranger' on task 'df.tr' (iter 33/50) 
    DEBUG [09:09:47.760] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:47.763] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.780] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:47.782] Creating Prediction for predict set 'test' 
    DEBUG [09:09:47.785] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60, 76]}
    DEBUG [09:09:47.789] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.800] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:47.802] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:47.803] Applying learner 'regr.ranger' on task 'df.tr' (iter 36/50) 
    DEBUG [09:09:47.811] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58..]}
    DEBUG [09:09:47.815] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.831] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:47.833] Creating Prediction for predict set 'test' 
    DEBUG [09:09:47.836] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75]}
    DEBUG [09:09:47.839] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.851] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:47.852] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:47.853] Applying learner 'regr.ranger' on task 'df.tr' (iter 13/50) 
    DEBUG [09:09:47.862] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:47.865] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.882] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:47.884] Creating Prediction for predict set 'test' 
    DEBUG [09:09:47.887] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 72]}
    DEBUG [09:09:47.890] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.901] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:47.903] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:47.904] Applying learner 'regr.ranger' on task 'df.tr' (iter 15/50) 
    DEBUG [09:09:47.913] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:47.916] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.933] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:47.935] Creating Prediction for predict set 'test' 
    DEBUG [09:09:47.938] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 19, 20, 30, 35, 42, 47, 54, 59, 64, 67, 71, 74, 75]}
    DEBUG [09:09:47.941] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.959] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:47.960] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:47.961] Applying learner 'regr.ranger' on task 'df.tr' (iter 41/50) 
    DEBUG [09:09:47.971] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74..]}
    DEBUG [09:09:47.974] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:47.991] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:47.993] Creating Prediction for predict set 'test' 
    DEBUG [09:09:47.995] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76]}
    DEBUG [09:09:47.999] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.010] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:48.012] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:48.013] Applying learner 'regr.ranger' on task 'df.tr' (iter 30/50) 
    DEBUG [09:09:48.022] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:48.025] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.041] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:48.043] Creating Prediction for predict set 'test' 
    DEBUG [09:09:48.046] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 14, 22, 29, 34, 35, 37, 40, 43, 44, 59, 62, 64, 68, 72]}
    DEBUG [09:09:48.050] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.061] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:48.063] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:48.064] Applying learner 'regr.ranger' on task 'df.tr' (iter 18/50) 
    DEBUG [09:09:48.073] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:48.076] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.093] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:48.095] Creating Prediction for predict set 'test' 
    DEBUG [09:09:48.097] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75]}
    DEBUG [09:09:48.100] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.112] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:48.114] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:48.115] Applying learner 'regr.ranger' on task 'df.tr' (iter 26/50) 
    DEBUG [09:09:48.123] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75..]}
    DEBUG [09:09:48.127] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.144] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:48.146] Creating Prediction for predict set 'test' 
    DEBUG [09:09:48.149] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74]}
    DEBUG [09:09:48.152] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.164] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:48.166] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:48.167] Applying learner 'regr.ranger' on task 'df.tr' (iter 45/50) 
    DEBUG [09:09:48.177] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:48.180] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.198] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:48.200] Creating Prediction for predict set 'test' 
    DEBUG [09:09:48.203] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 8, 12, 21, 22, 25, 28, 33, 37, 38, 44, 46, 48, 57, 75]}
    DEBUG [09:09:48.207] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.219] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:48.221] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:48.222] Applying learner 'regr.ranger' on task 'df.tr' (iter 25/50) 
    DEBUG [09:09:48.232] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:48.235] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.251] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:48.252] Creating Prediction for predict set 'test' 
    DEBUG [09:09:48.255] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [9, 11, 18, 21, 23, 27, 29, 43, 44, 45, 51, 55, 67, 73, 76]}
    DEBUG [09:09:48.258] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.271] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:48.272] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:48.273] Applying learner 'regr.ranger' on task 'df.tr' (iter 11/50) 
    DEBUG [09:09:48.283] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 7..]}
    DEBUG [09:09:48.286] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.302] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:48.304] Creating Prediction for predict set 'test' 
    DEBUG [09:09:48.308] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76]}
    DEBUG [09:09:48.312] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.325] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:48.326] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:48.327] Applying learner 'regr.ranger' on task 'df.tr' (iter 19/50) 
    DEBUG [09:09:48.336] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:48.340] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.358] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:48.360] Creating Prediction for predict set 'test' 
    DEBUG [09:09:48.364] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 8, 11, 14, 18, 24, 28, 33, 34, 54, 56, 62, 64, 68, 76]}
    DEBUG [09:09:48.368] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.380] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:48.382] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:48.383] Applying learner 'regr.ranger' on task 'df.tr' (iter 38/50) 
    DEBUG [09:09:48.393] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:48.396] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.413] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:48.415] Creating Prediction for predict set 'test' 
    DEBUG [09:09:48.420] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58, 69]}
    DEBUG [09:09:48.423] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.436] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:48.437] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:48.439] Applying learner 'regr.ranger' on task 'df.tr' (iter 16/50) 
    DEBUG [09:09:48.448] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75..]}
    DEBUG [09:09:48.452] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.468] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:48.469] Creating Prediction for predict set 'test' 
    DEBUG [09:09:48.472] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72]}
    DEBUG [09:09:48.476] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.488] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:48.489] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:48.490] Applying learner 'regr.ranger' on task 'df.tr' (iter 40/50) 
    DEBUG [09:09:48.500] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:48.503] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.521] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:48.523] Creating Prediction for predict set 'test' 
    DEBUG [09:09:48.527] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 8, 16, 17, 24, 25, 27, 29, 41, 44, 53, 54, 67, 72]}
    DEBUG [09:09:48.530] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.543] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:48.544] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:48.545] Applying learner 'regr.ranger' on task 'df.tr' (iter 43/50) 
    DEBUG [09:09:48.554] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:48.558] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.573] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:48.575] Creating Prediction for predict set 'test' 
    DEBUG [09:09:48.580] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74]}
    DEBUG [09:09:48.583] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.595] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:48.597] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:48.598] Applying learner 'regr.ranger' on task 'df.tr' (iter 6/50) 
    DEBUG [09:09:48.608] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 7..]}
    DEBUG [09:09:48.612] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.627] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:48.629] Creating Prediction for predict set 'test' 
    DEBUG [09:09:48.632] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76]}
    DEBUG [09:09:48.635] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.648] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:48.649] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:48.650] Applying learner 'regr.ranger' on task 'df.tr' (iter 5/50) 
    DEBUG [09:09:48.660] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:48.664] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.679] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:48.681] Creating Prediction for predict set 'test' 
    DEBUG [09:09:48.684] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 11, 14, 17, 27, 32, 41, 44, 45, 55, 60, 64, 65, 66, 69]}
    DEBUG [09:09:48.688] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.700] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:48.701] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:48.702] Applying learner 'regr.ranger' on task 'df.tr' (iter 2/50) 
    DEBUG [09:09:48.712] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [09:09:48.716] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.733] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:48.736] Creating Prediction for predict set 'test' 
    DEBUG [09:09:48.739] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68]}
    DEBUG [09:09:48.742] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.755] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:48.756] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:48.758] Applying learner 'regr.ranger' on task 'df.tr' (iter 22/50) 
    DEBUG [09:09:48.767] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57..]}
    DEBUG [09:09:48.771] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.788] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:48.790] Creating Prediction for predict set 'test' 
    DEBUG [09:09:48.793] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75]}
    DEBUG [09:09:48.796] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.809] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:48.810] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:48.811] Applying learner 'regr.ranger' on task 'df.tr' (iter 17/50) 
    DEBUG [09:09:48.821] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 7..]}
    DEBUG [09:09:48.825] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.842] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:48.843] Creating Prediction for predict set 'test' 
    DEBUG [09:09:48.847] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70]}
    DEBUG [09:09:48.850] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.862] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:48.864] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:48.865] Applying learner 'regr.ranger' on task 'df.tr' (iter 42/50) 
    DEBUG [09:09:48.875] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70..]}
    DEBUG [09:09:48.878] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.895] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:48.897] Creating Prediction for predict set 'test' 
    DEBUG [09:09:48.900] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66]}
    DEBUG [09:09:48.904] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.916] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:48.918] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:48.919] Applying learner 'regr.ranger' on task 'df.tr' (iter 32/50) 
    DEBUG [09:09:48.929] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 6..]}
    DEBUG [09:09:48.932] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.950] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:48.953] Creating Prediction for predict set 'test' 
    DEBUG [09:09:48.957] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71]}
    DEBUG [09:09:48.960] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:48.973] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:48.974] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:48.976] Applying learner 'regr.ranger' on task 'df.tr' (iter 29/50) 
    DEBUG [09:09:48.985] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:48.989] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.004] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:49.006] Creating Prediction for predict set 'test' 
    DEBUG [09:09:49.009] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 18, 25, 28, 30, 39, 45, 48, 51, 54, 55, 61, 66, 71, 73]}
    DEBUG [09:09:49.012] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.025] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:49.026] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:49.027] Applying learner 'regr.ranger' on task 'df.tr' (iter 21/50) 
    DEBUG [09:09:49.037] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66..]}
    DEBUG [09:09:49.041] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.057] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:49.058] Creating Prediction for predict set 'test' 
    DEBUG [09:09:49.079] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72]}
    DEBUG [09:09:49.082] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.096] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:49.098] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:49.099] Applying learner 'regr.ranger' on task 'df.tr' (iter 35/50) 
    DEBUG [09:09:49.108] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:49.112] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.128] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:49.130] Creating Prediction for predict set 'test' 
    DEBUG [09:09:49.132] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 29, 32, 33, 35, 40, 43, 44, 45, 61, 62, 70, 74]}
    DEBUG [09:09:49.136] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.148] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:49.150] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:49.151] Applying learner 'regr.ranger' on task 'df.tr' (iter 20/50) 
    DEBUG [09:09:49.160] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:49.163] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.180] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:49.182] Creating Prediction for predict set 'test' 
    DEBUG [09:09:49.186] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 16, 17, 32, 36, 37, 42, 43, 45, 46, 53, 61, 65, 69]}
    DEBUG [09:09:49.190] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.201] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:49.203] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:49.204] Applying learner 'regr.ranger' on task 'df.tr' (iter 47/50) 
    DEBUG [09:09:49.213] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64..]}
    DEBUG [09:09:49.216] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.231] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:49.233] Creating Prediction for predict set 'test' 
    DEBUG [09:09:49.236] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70]}
    DEBUG [09:09:49.239] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.250] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:49.252] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:49.253] Applying learner 'regr.ranger' on task 'df.tr' (iter 37/50) 
    DEBUG [09:09:49.262] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57..]}
    DEBUG [09:09:49.265] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.282] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:49.283] Creating Prediction for predict set 'test' 
    DEBUG [09:09:49.286] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76]}
    DEBUG [09:09:49.290] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.301] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:49.303] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:49.304] Applying learner 'regr.ranger' on task 'df.tr' (iter 14/50) 
    DEBUG [09:09:49.313] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:49.316] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.331] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:49.332] Creating Prediction for predict set 'test' 
    DEBUG [09:09:49.335] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 22, 27, 36, 38, 40, 45, 46, 50, 55, 56, 60, 65, 73]}
    DEBUG [09:09:49.338] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.350] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:49.351] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:49.352] Applying learner 'regr.ranger' on task 'df.tr' (iter 34/50) 
    DEBUG [09:09:49.361] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:49.364] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.379] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:49.381] Creating Prediction for predict set 'test' 
    DEBUG [09:09:49.385] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 7, 12, 14, 18, 20, 41, 46, 47, 53, 57, 63, 64, 72, 75]}
    DEBUG [09:09:49.388] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.400] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:49.401] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:49.402] Applying learner 'regr.ranger' on task 'df.tr' (iter 24/50) 
    DEBUG [09:09:49.411] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:49.414] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.431] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:49.433] Creating Prediction for predict set 'test' 
    DEBUG [09:09:49.436] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 13, 17, 25, 32, 35, 36, 37, 38, 50, 63, 65, 68, 69, 74]}
    DEBUG [09:09:49.439] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.450] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:49.452] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:49.453] Applying learner 'regr.ranger' on task 'df.tr' (iter 50/50) 
    DEBUG [09:09:49.462] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:49.465] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.480] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:49.481] Creating Prediction for predict set 'test' 
    DEBUG [09:09:49.484] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 8, 21, 22, 30, 36, 41, 45, 49, 53, 54, 57, 68, 71]}
    DEBUG [09:09:49.487] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.499] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:49.500] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:49.501] Applying learner 'regr.ranger' on task 'df.tr' (iter 7/50) 
    DEBUG [09:09:49.510] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 6..]}
    DEBUG [09:09:49.514] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.531] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:49.533] Creating Prediction for predict set 'test' 
    DEBUG [09:09:49.535] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74]}
    DEBUG [09:09:49.539] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.550] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:49.552] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:49.553] Applying learner 'regr.ranger' on task 'df.tr' (iter 39/50) 
    DEBUG [09:09:49.562] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:49.565] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.580] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:49.581] Creating Prediction for predict set 'test' 
    DEBUG [09:09:49.584] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 5, 14, 18, 19, 21, 28, 38, 48, 49, 51, 52, 59, 64, 65]}
    DEBUG [09:09:49.587] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.599] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:49.600] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:49.601] Applying learner 'regr.ranger' on task 'df.tr' (iter 8/50) 
    DEBUG [09:09:49.610] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:49.613] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.630] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:49.631] Creating Prediction for predict set 'test' 
    DEBUG [09:09:49.635] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 75]}
    DEBUG [09:09:49.638] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.649] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:49.651] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:49.652] Applying learner 'regr.ranger' on task 'df.tr' (iter 49/50) 
    DEBUG [09:09:49.661] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:49.664] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.680] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:49.682] Creating Prediction for predict set 'test' 
    DEBUG [09:09:49.685] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 7, 14, 16, 25, 26, 27, 40, 44, 56, 67, 69, 75, 76]}
    DEBUG [09:09:49.688] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.699] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:49.701] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:49.702] Applying learner 'regr.ranger' on task 'df.tr' (iter 31/50) 
    DEBUG [09:09:49.711] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60..]}
    DEBUG [09:09:49.714] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.731] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:49.733] Creating Prediction for predict set 'test' 
    DEBUG [09:09:49.735] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73]}
    DEBUG [09:09:49.739] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.750] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:49.751] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:49.753] Applying learner 'regr.ranger' on task 'df.tr' (iter 9/50) 
    DEBUG [09:09:49.761] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:49.770] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.792] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:49.794] Creating Prediction for predict set 'test' 
    DEBUG [09:09:49.798] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 23, 27, 40, 42, 46, 49, 56, 58, 61, 64, 67, 70, 71, 72]}
    DEBUG [09:09:49.801] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.813] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:49.814] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:49.815] Applying learner 'regr.ranger' on task 'df.tr' (iter 23/50) 
    DEBUG [09:09:49.825] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:49.828] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.848] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:49.850] Creating Prediction for predict set 'test' 
    DEBUG [09:09:49.854] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66]}
    DEBUG [09:09:49.857] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.871] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:49.873] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:49.874] Applying learner 'regr.ranger' on task 'df.tr' (iter 44/50) 
    DEBUG [09:09:49.887] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:49.894] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.913] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:49.915] Creating Prediction for predict set 'test' 
    DEBUG [09:09:49.919] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 14, 17, 19, 24, 36, 41, 47, 51, 53, 56, 62, 63, 67, 71]}
    DEBUG [09:09:49.923] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.934] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:49.936] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:49.937] Applying learner 'regr.ranger' on task 'df.tr' (iter 4/50) 
    DEBUG [09:09:49.946] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:49.949] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.965] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:49.967] Creating Prediction for predict set 'test' 
    DEBUG [09:09:49.970] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 3, 13, 20, 23, 25, 40, 46, 50, 53, 56, 57, 58, 63]}
    DEBUG [09:09:49.973] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:49.985] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:49.986] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:49.987] Applying learner 'regr.ranger' on task 'df.tr' (iter 27/50) 
    DEBUG [09:09:49.996] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67..]}
    DEBUG [09:09:50.000] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:50.019] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:50.021] Creating Prediction for predict set 'test' 
    DEBUG [09:09:50.025] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76]}
    DEBUG [09:09:50.029] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:50.046] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:50.047] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:50.070] Finished benchmark 
    INFO  [09:09:51.442] Result of batch 5: 
    INFO  [09:09:51.445]  mtry sample.fraction num.trees importance regr.rmse      resample_result 
    INFO  [09:09:51.445]     4       0.8667243       428   impurity  43.33344 <ResampleResult[19]> 
    INFO  [09:09:51.451] Evaluating 1 configuration(s) 
    INFO  [09:09:51.483] Benchmark with 50 resampling iterations 
    DEBUG [09:09:51.484] Running benchmark() asynchronously with 50 iterations 
    INFO  [09:09:51.494] Applying learner 'regr.ranger' on task 'df.tr' (iter 3/50) 
    DEBUG [09:09:51.501] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:51.504] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:51.517] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:51.518] Creating Prediction for predict set 'test' 
    DEBUG [09:09:51.520] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75, 76]}
    DEBUG [09:09:51.523] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:51.533] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:51.534] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:51.535] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [09:09:51.542] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [09:09:51.545] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:51.559] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:51.561] Creating Prediction for predict set 'test' 
    DEBUG [09:09:51.563] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [09:09:51.565] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:51.581] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:51.583] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:51.584] Applying learner 'regr.ranger' on task 'df.tr' (iter 44/50) 
    DEBUG [09:09:51.592] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:51.595] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:51.609] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:51.610] Creating Prediction for predict set 'test' 
    DEBUG [09:09:51.613] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 14, 17, 19, 24, 36, 41, 47, 51, 53, 56, 62, 63, 67, 71]}
    DEBUG [09:09:51.615] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:51.625] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:51.627] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:51.627] Applying learner 'regr.ranger' on task 'df.tr' (iter 33/50) 
    DEBUG [09:09:51.635] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:51.638] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:51.652] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:51.654] Creating Prediction for predict set 'test' 
    DEBUG [09:09:51.657] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60, 76]}
    DEBUG [09:09:51.660] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:51.670] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:51.671] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:51.672] Applying learner 'regr.ranger' on task 'df.tr' (iter 36/50) 
    DEBUG [09:09:51.680] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58..]}
    DEBUG [09:09:51.683] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:51.697] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:51.699] Creating Prediction for predict set 'test' 
    DEBUG [09:09:51.703] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75]}
    DEBUG [09:09:51.707] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:51.719] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:51.720] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:51.721] Applying learner 'regr.ranger' on task 'df.tr' (iter 13/50) 
    DEBUG [09:09:51.731] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:51.735] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:51.749] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:51.751] Creating Prediction for predict set 'test' 
    DEBUG [09:09:51.754] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 72]}
    DEBUG [09:09:51.757] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:51.769] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:51.771] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:51.772] Applying learner 'regr.ranger' on task 'df.tr' (iter 15/50) 
    DEBUG [09:09:51.782] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:51.785] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:51.801] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:51.803] Creating Prediction for predict set 'test' 
    DEBUG [09:09:51.806] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 19, 20, 30, 35, 42, 47, 54, 59, 64, 67, 71, 74, 75]}
    DEBUG [09:09:51.810] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:51.822] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:51.824] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:51.825] Applying learner 'regr.ranger' on task 'df.tr' (iter 45/50) 
    DEBUG [09:09:51.834] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:51.838] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:51.895] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:51.897] Creating Prediction for predict set 'test' 
    DEBUG [09:09:51.900] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 8, 12, 21, 22, 25, 28, 33, 37, 38, 44, 46, 48, 57, 75]}
    DEBUG [09:09:51.903] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:51.915] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:51.917] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:51.918] Applying learner 'regr.ranger' on task 'df.tr' (iter 30/50) 
    DEBUG [09:09:51.927] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:51.930] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:51.945] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:51.947] Creating Prediction for predict set 'test' 
    DEBUG [09:09:51.950] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 14, 22, 29, 34, 35, 37, 40, 43, 44, 59, 62, 64, 68, 72]}
    DEBUG [09:09:51.953] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:51.968] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:51.970] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:51.971] Applying learner 'regr.ranger' on task 'df.tr' (iter 18/50) 
    DEBUG [09:09:51.980] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:51.984] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:51.999] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:52.000] Creating Prediction for predict set 'test' 
    DEBUG [09:09:52.003] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75]}
    DEBUG [09:09:52.007] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.019] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:52.021] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:52.022] Applying learner 'regr.ranger' on task 'df.tr' (iter 39/50) 
    DEBUG [09:09:52.031] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:52.035] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.050] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:52.051] Creating Prediction for predict set 'test' 
    DEBUG [09:09:52.059] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 5, 14, 18, 19, 21, 28, 38, 48, 49, 51, 52, 59, 64, 65]}
    DEBUG [09:09:52.062] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.078] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:52.081] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:52.083] Applying learner 'regr.ranger' on task 'df.tr' (iter 26/50) 
    DEBUG [09:09:52.094] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75..]}
    DEBUG [09:09:52.098] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.121] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:52.123] Creating Prediction for predict set 'test' 
    DEBUG [09:09:52.126] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74]}
    DEBUG [09:09:52.130] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.144] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:52.146] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:52.147] Applying learner 'regr.ranger' on task 'df.tr' (iter 46/50) 
    DEBUG [09:09:52.160] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72..]}
    DEBUG [09:09:52.165] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.183] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:52.186] Creating Prediction for predict set 'test' 
    DEBUG [09:09:52.189] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73]}
    DEBUG [09:09:52.193] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.208] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:52.210] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:52.211] Applying learner 'regr.ranger' on task 'df.tr' (iter 38/50) 
    DEBUG [09:09:52.222] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:52.226] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.245] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:52.247] Creating Prediction for predict set 'test' 
    DEBUG [09:09:52.251] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58, 69]}
    DEBUG [09:09:52.254] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.268] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:52.270] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:52.271] Applying learner 'regr.ranger' on task 'df.tr' (iter 50/50) 
    DEBUG [09:09:52.282] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:52.286] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.309] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:52.311] Creating Prediction for predict set 'test' 
    DEBUG [09:09:52.315] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 8, 21, 22, 30, 36, 41, 45, 49, 53, 54, 57, 68, 71]}
    DEBUG [09:09:52.321] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.337] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:52.339] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:52.341] Applying learner 'regr.ranger' on task 'df.tr' (iter 25/50) 
    DEBUG [09:09:52.353] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:52.357] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.376] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:52.378] Creating Prediction for predict set 'test' 
    DEBUG [09:09:52.382] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [9, 11, 18, 21, 23, 27, 29, 43, 44, 45, 51, 55, 67, 73, 76]}
    DEBUG [09:09:52.386] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.400] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:52.402] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:52.403] Applying learner 'regr.ranger' on task 'df.tr' (iter 11/50) 
    DEBUG [09:09:52.414] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 7..]}
    DEBUG [09:09:52.418] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.434] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:52.436] Creating Prediction for predict set 'test' 
    DEBUG [09:09:52.439] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76]}
    DEBUG [09:09:52.443] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.456] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:52.458] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:52.459] Applying learner 'regr.ranger' on task 'df.tr' (iter 42/50) 
    DEBUG [09:09:52.469] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70..]}
    DEBUG [09:09:52.472] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.490] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:52.492] Creating Prediction for predict set 'test' 
    DEBUG [09:09:52.495] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66]}
    DEBUG [09:09:52.498] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.512] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:52.513] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:52.514] Applying learner 'regr.ranger' on task 'df.tr' (iter 16/50) 
    DEBUG [09:09:52.525] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75..]}
    DEBUG [09:09:52.533] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.555] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:52.557] Creating Prediction for predict set 'test' 
    DEBUG [09:09:52.562] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72]}
    DEBUG [09:09:52.565] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.579] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:52.580] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:52.582] Applying learner 'regr.ranger' on task 'df.tr' (iter 48/50) 
    DEBUG [09:09:52.593] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:52.597] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.614] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:52.616] Creating Prediction for predict set 'test' 
    DEBUG [09:09:52.619] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72, 74]}
    DEBUG [09:09:52.623] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.635] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:52.637] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:52.638] Applying learner 'regr.ranger' on task 'df.tr' (iter 34/50) 
    DEBUG [09:09:52.648] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:52.651] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.666] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:52.668] Creating Prediction for predict set 'test' 
    DEBUG [09:09:52.671] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 7, 12, 14, 18, 20, 41, 46, 47, 53, 57, 63, 64, 72, 75]}
    DEBUG [09:09:52.675] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.687] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:52.689] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:52.690] Applying learner 'regr.ranger' on task 'df.tr' (iter 6/50) 
    DEBUG [09:09:52.700] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 7..]}
    DEBUG [09:09:52.704] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.720] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:52.721] Creating Prediction for predict set 'test' 
    DEBUG [09:09:52.725] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76]}
    DEBUG [09:09:52.728] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.741] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:52.743] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:52.744] Applying learner 'regr.ranger' on task 'df.tr' (iter 5/50) 
    DEBUG [09:09:52.754] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:52.758] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.773] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:52.774] Creating Prediction for predict set 'test' 
    DEBUG [09:09:52.777] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 11, 14, 17, 27, 32, 41, 44, 45, 55, 60, 64, 65, 66, 69]}
    DEBUG [09:09:52.781] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.793] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:52.795] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:52.796] Applying learner 'regr.ranger' on task 'df.tr' (iter 2/50) 
    DEBUG [09:09:52.806] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [09:09:52.810] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.825] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:52.827] Creating Prediction for predict set 'test' 
    DEBUG [09:09:52.831] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68]}
    DEBUG [09:09:52.834] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.847] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:52.848] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:52.850] Applying learner 'regr.ranger' on task 'df.tr' (iter 22/50) 
    DEBUG [09:09:52.860] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57..]}
    DEBUG [09:09:52.863] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.878] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:52.880] Creating Prediction for predict set 'test' 
    DEBUG [09:09:52.883] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75]}
    DEBUG [09:09:52.887] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.899] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:52.901] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:52.902] Applying learner 'regr.ranger' on task 'df.tr' (iter 17/50) 
    DEBUG [09:09:52.912] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 7..]}
    DEBUG [09:09:52.916] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.934] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:52.936] Creating Prediction for predict set 'test' 
    DEBUG [09:09:52.939] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70]}
    DEBUG [09:09:52.943] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.956] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:52.957] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:52.959] Applying learner 'regr.ranger' on task 'df.tr' (iter 28/50) 
    DEBUG [09:09:52.969] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:52.973] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:52.988] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:52.990] Creating Prediction for predict set 'test' 
    DEBUG [09:09:52.993] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75]}
    DEBUG [09:09:52.997] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.014] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:53.016] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:53.017] Applying learner 'regr.ranger' on task 'df.tr' (iter 19/50) 
    DEBUG [09:09:53.029] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:53.032] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.050] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:53.051] Creating Prediction for predict set 'test' 
    DEBUG [09:09:53.055] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 8, 11, 14, 18, 24, 28, 33, 34, 54, 56, 62, 64, 68, 76]}
    DEBUG [09:09:53.058] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.072] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:53.073] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:53.075] Applying learner 'regr.ranger' on task 'df.tr' (iter 31/50) 
    DEBUG [09:09:53.085] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60..]}
    DEBUG [09:09:53.089] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.105] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:53.106] Creating Prediction for predict set 'test' 
    DEBUG [09:09:53.110] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73]}
    DEBUG [09:09:53.113] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.126] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:53.128] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:53.129] Applying learner 'regr.ranger' on task 'df.tr' (iter 23/50) 
    DEBUG [09:09:53.139] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:53.143] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.161] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:53.163] Creating Prediction for predict set 'test' 
    DEBUG [09:09:53.166] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66]}
    DEBUG [09:09:53.170] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.182] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:53.184] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:53.185] Applying learner 'regr.ranger' on task 'df.tr' (iter 37/50) 
    DEBUG [09:09:53.196] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57..]}
    DEBUG [09:09:53.200] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.218] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:53.221] Creating Prediction for predict set 'test' 
    DEBUG [09:09:53.225] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76]}
    DEBUG [09:09:53.229] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.241] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:53.243] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:53.244] Applying learner 'regr.ranger' on task 'df.tr' (iter 40/50) 
    DEBUG [09:09:53.255] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:53.258] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.274] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:53.276] Creating Prediction for predict set 'test' 
    DEBUG [09:09:53.279] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 8, 16, 17, 24, 25, 27, 29, 41, 44, 53, 54, 67, 72]}
    DEBUG [09:09:53.283] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.296] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:53.298] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:53.299] Applying learner 'regr.ranger' on task 'df.tr' (iter 20/50) 
    DEBUG [09:09:53.309] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:53.313] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.330] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:53.333] Creating Prediction for predict set 'test' 
    DEBUG [09:09:53.336] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 16, 17, 32, 36, 37, 42, 43, 45, 46, 53, 61, 65, 69]}
    DEBUG [09:09:53.340] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.353] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:53.354] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:53.355] Applying learner 'regr.ranger' on task 'df.tr' (iter 27/50) 
    DEBUG [09:09:53.366] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67..]}
    DEBUG [09:09:53.369] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.385] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:53.386] Creating Prediction for predict set 'test' 
    DEBUG [09:09:53.389] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76]}
    DEBUG [09:09:53.393] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.406] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:53.407] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:53.409] Applying learner 'regr.ranger' on task 'df.tr' (iter 24/50) 
    DEBUG [09:09:53.419] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:53.422] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.440] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:53.442] Creating Prediction for predict set 'test' 
    DEBUG [09:09:53.447] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 13, 17, 25, 32, 35, 36, 37, 38, 50, 63, 65, 68, 69, 74]}
    DEBUG [09:09:53.451] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.463] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:53.465] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:53.466] Applying learner 'regr.ranger' on task 'df.tr' (iter 32/50) 
    DEBUG [09:09:53.476] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 6..]}
    DEBUG [09:09:53.480] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.504] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:53.505] Creating Prediction for predict set 'test' 
    DEBUG [09:09:53.509] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71]}
    DEBUG [09:09:53.560] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.575] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:53.577] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:53.578] Applying learner 'regr.ranger' on task 'df.tr' (iter 7/50) 
    DEBUG [09:09:53.590] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 6..]}
    DEBUG [09:09:53.594] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.610] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:53.612] Creating Prediction for predict set 'test' 
    DEBUG [09:09:53.615] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74]}
    DEBUG [09:09:53.619] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.632] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:53.634] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:53.635] Applying learner 'regr.ranger' on task 'df.tr' (iter 35/50) 
    DEBUG [09:09:53.645] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:53.649] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.667] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:53.669] Creating Prediction for predict set 'test' 
    DEBUG [09:09:53.674] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 29, 32, 33, 35, 40, 43, 44, 45, 61, 62, 70, 74]}
    DEBUG [09:09:53.677] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.689] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:53.691] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:53.692] Applying learner 'regr.ranger' on task 'df.tr' (iter 8/50) 
    DEBUG [09:09:53.702] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:53.707] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.723] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:53.725] Creating Prediction for predict set 'test' 
    DEBUG [09:09:53.729] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 75]}
    DEBUG [09:09:53.732] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.746] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:53.748] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:53.749] Applying learner 'regr.ranger' on task 'df.tr' (iter 14/50) 
    DEBUG [09:09:53.759] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:53.763] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.781] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:53.783] Creating Prediction for predict set 'test' 
    DEBUG [09:09:53.787] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 22, 27, 36, 38, 40, 45, 46, 50, 55, 56, 60, 65, 73]}
    DEBUG [09:09:53.790] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.803] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:53.805] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:53.806] Applying learner 'regr.ranger' on task 'df.tr' (iter 47/50) 
    DEBUG [09:09:53.816] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64..]}
    DEBUG [09:09:53.820] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.836] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:53.838] Creating Prediction for predict set 'test' 
    DEBUG [09:09:53.841] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70]}
    DEBUG [09:09:53.846] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.860] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:53.861] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:53.863] Applying learner 'regr.ranger' on task 'df.tr' (iter 12/50) 
    DEBUG [09:09:53.873] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61..]}
    DEBUG [09:09:53.876] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.894] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:53.896] Creating Prediction for predict set 'test' 
    DEBUG [09:09:53.899] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70]}
    DEBUG [09:09:53.903] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.916] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:53.917] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:53.918] Applying learner 'regr.ranger' on task 'df.tr' (iter 21/50) 
    DEBUG [09:09:53.928] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66..]}
    DEBUG [09:09:53.932] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.950] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:53.953] Creating Prediction for predict set 'test' 
    DEBUG [09:09:53.957] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72]}
    DEBUG [09:09:53.960] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:53.973] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:53.974] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:53.976] Applying learner 'regr.ranger' on task 'df.tr' (iter 43/50) 
    DEBUG [09:09:53.986] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:53.990] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:54.007] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:54.009] Creating Prediction for predict set 'test' 
    DEBUG [09:09:54.013] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74]}
    DEBUG [09:09:54.017] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:54.030] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:54.031] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:54.038] Applying learner 'regr.ranger' on task 'df.tr' (iter 41/50) 
    DEBUG [09:09:54.052] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74..]}
    DEBUG [09:09:54.056] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:54.076] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:54.078] Creating Prediction for predict set 'test' 
    DEBUG [09:09:54.082] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76]}
    DEBUG [09:09:54.085] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:54.098] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:54.100] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:54.101] Applying learner 'regr.ranger' on task 'df.tr' (iter 49/50) 
    DEBUG [09:09:54.112] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:54.116] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:54.131] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:54.132] Creating Prediction for predict set 'test' 
    DEBUG [09:09:54.136] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 7, 14, 16, 25, 26, 27, 40, 44, 56, 67, 69, 75, 76]}
    DEBUG [09:09:54.139] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:54.152] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:54.153] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:54.155] Applying learner 'regr.ranger' on task 'df.tr' (iter 29/50) 
    DEBUG [09:09:54.165] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:54.169] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:54.186] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:54.187] Creating Prediction for predict set 'test' 
    DEBUG [09:09:54.191] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 18, 25, 28, 30, 39, 45, 48, 51, 54, 55, 61, 66, 71, 73]}
    DEBUG [09:09:54.195] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:54.207] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:54.209] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:54.210] Applying learner 'regr.ranger' on task 'df.tr' (iter 10/50) 
    DEBUG [09:09:54.221] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:54.225] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:54.242] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:54.243] Creating Prediction for predict set 'test' 
    DEBUG [09:09:54.246] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 17, 18, 19, 24, 26, 29, 31, 34, 36, 43, 45, 53, 66, 73]}
    DEBUG [09:09:54.250] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:54.262] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:54.264] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:54.265] Applying learner 'regr.ranger' on task 'df.tr' (iter 4/50) 
    DEBUG [09:09:54.275] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:54.279] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:54.296] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:54.297] Creating Prediction for predict set 'test' 
    DEBUG [09:09:54.301] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 3, 13, 20, 23, 25, 40, 46, 50, 53, 56, 57, 58, 63]}
    DEBUG [09:09:54.304] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:54.317] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:54.319] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:54.320] Applying learner 'regr.ranger' on task 'df.tr' (iter 9/50) 
    DEBUG [09:09:54.330] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:54.334] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:54.349] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:54.351] Creating Prediction for predict set 'test' 
    DEBUG [09:09:54.354] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 23, 27, 40, 42, 46, 49, 56, 58, 61, 64, 67, 70, 71, 72]}
    DEBUG [09:09:54.357] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:54.370] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:54.372] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:54.400] Finished benchmark 
    INFO  [09:09:56.173] Result of batch 6: 
    INFO  [09:09:56.177]  mtry sample.fraction num.trees importance regr.rmse      resample_result 
    INFO  [09:09:56.177]     3       0.9183301       378   impurity  44.73443 <ResampleResult[19]> 
    INFO  [09:09:56.186] Evaluating 1 configuration(s) 
    INFO  [09:09:56.233] Benchmark with 50 resampling iterations 
    DEBUG [09:09:56.236] Running benchmark() asynchronously with 50 iterations 
    INFO  [09:09:56.254] Applying learner 'regr.ranger' on task 'df.tr' (iter 13/50) 
    DEBUG [09:09:56.264] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:56.268] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.283] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:56.285] Creating Prediction for predict set 'test' 
    DEBUG [09:09:56.288] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 72]}
    DEBUG [09:09:56.292] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.304] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:56.306] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:56.307] Applying learner 'regr.ranger' on task 'df.tr' (iter 15/50) 
    DEBUG [09:09:56.321] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:56.326] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.338] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:56.340] Creating Prediction for predict set 'test' 
    DEBUG [09:09:56.343] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 19, 20, 30, 35, 42, 47, 54, 59, 64, 67, 71, 74, 75]}
    DEBUG [09:09:56.347] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.358] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:56.359] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:56.360] Applying learner 'regr.ranger' on task 'df.tr' (iter 50/50) 
    DEBUG [09:09:56.370] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:56.374] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.387] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:56.388] Creating Prediction for predict set 'test' 
    DEBUG [09:09:56.392] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 8, 21, 22, 30, 36, 41, 45, 49, 53, 54, 57, 68, 71]}
    DEBUG [09:09:56.395] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.407] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:56.408] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:56.410] Applying learner 'regr.ranger' on task 'df.tr' (iter 46/50) 
    DEBUG [09:09:56.420] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72..]}
    DEBUG [09:09:56.424] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.435] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:56.437] Creating Prediction for predict set 'test' 
    DEBUG [09:09:56.440] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73]}
    DEBUG [09:09:56.444] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.454] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:56.457] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:56.459] Applying learner 'regr.ranger' on task 'df.tr' (iter 30/50) 
    DEBUG [09:09:56.470] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:56.474] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.486] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:56.488] Creating Prediction for predict set 'test' 
    DEBUG [09:09:56.491] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 14, 22, 29, 34, 35, 37, 40, 43, 44, 59, 62, 64, 68, 72]}
    DEBUG [09:09:56.494] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.505] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:56.507] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:56.508] Applying learner 'regr.ranger' on task 'df.tr' (iter 18/50) 
    DEBUG [09:09:56.518] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:56.521] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.533] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:56.534] Creating Prediction for predict set 'test' 
    DEBUG [09:09:56.537] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75]}
    DEBUG [09:09:56.541] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.551] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:56.553] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:56.554] Applying learner 'regr.ranger' on task 'df.tr' (iter 44/50) 
    DEBUG [09:09:56.564] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:56.567] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.626] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:56.628] Creating Prediction for predict set 'test' 
    DEBUG [09:09:56.631] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 14, 17, 19, 24, 36, 41, 47, 51, 53, 56, 62, 63, 67, 71]}
    DEBUG [09:09:56.634] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.644] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:56.646] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:56.647] Applying learner 'regr.ranger' on task 'df.tr' (iter 39/50) 
    DEBUG [09:09:56.656] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:56.660] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.671] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:56.673] Creating Prediction for predict set 'test' 
    DEBUG [09:09:56.676] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 5, 14, 18, 19, 21, 28, 38, 48, 49, 51, 52, 59, 64, 65]}
    DEBUG [09:09:56.679] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.690] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:56.691] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:56.693] Applying learner 'regr.ranger' on task 'df.tr' (iter 26/50) 
    DEBUG [09:09:56.702] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75..]}
    DEBUG [09:09:56.706] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.717] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:56.719] Creating Prediction for predict set 'test' 
    DEBUG [09:09:56.722] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74]}
    DEBUG [09:09:56.725] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.736] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:56.737] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:56.739] Applying learner 'regr.ranger' on task 'df.tr' (iter 36/50) 
    DEBUG [09:09:56.748] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58..]}
    DEBUG [09:09:56.752] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.763] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:56.765] Creating Prediction for predict set 'test' 
    DEBUG [09:09:56.768] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75]}
    DEBUG [09:09:56.771] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.782] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:56.784] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:56.785] Applying learner 'regr.ranger' on task 'df.tr' (iter 41/50) 
    DEBUG [09:09:56.794] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74..]}
    DEBUG [09:09:56.798] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.810] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:56.812] Creating Prediction for predict set 'test' 
    DEBUG [09:09:56.815] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76]}
    DEBUG [09:09:56.818] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.829] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:56.831] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:56.832] Applying learner 'regr.ranger' on task 'df.tr' (iter 3/50) 
    DEBUG [09:09:56.842] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:56.850] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.864] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:56.866] Creating Prediction for predict set 'test' 
    DEBUG [09:09:56.869] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75, 76]}
    DEBUG [09:09:56.873] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.886] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:56.888] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:56.890] Applying learner 'regr.ranger' on task 'df.tr' (iter 25/50) 
    DEBUG [09:09:56.900] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:56.904] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.916] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:56.917] Creating Prediction for predict set 'test' 
    DEBUG [09:09:56.921] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [9, 11, 18, 21, 23, 27, 29, 43, 44, 45, 51, 55, 67, 73, 76]}
    DEBUG [09:09:56.924] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.935] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:56.937] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:56.939] Applying learner 'regr.ranger' on task 'df.tr' (iter 11/50) 
    DEBUG [09:09:56.949] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 7..]}
    DEBUG [09:09:56.953] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.965] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:56.966] Creating Prediction for predict set 'test' 
    DEBUG [09:09:56.970] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76]}
    DEBUG [09:09:56.973] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:56.985] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:56.987] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:56.988] Applying learner 'regr.ranger' on task 'df.tr' (iter 47/50) 
    DEBUG [09:09:56.998] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64..]}
    DEBUG [09:09:57.001] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.014] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:57.016] Creating Prediction for predict set 'test' 
    DEBUG [09:09:57.019] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70]}
    DEBUG [09:09:57.023] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.036] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:57.037] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:57.038] Applying learner 'regr.ranger' on task 'df.tr' (iter 16/50) 
    DEBUG [09:09:57.048] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75..]}
    DEBUG [09:09:57.051] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.062] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:57.064] Creating Prediction for predict set 'test' 
    DEBUG [09:09:57.067] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72]}
    DEBUG [09:09:57.070] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.081] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:57.082] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:57.083] Applying learner 'regr.ranger' on task 'df.tr' (iter 37/50) 
    DEBUG [09:09:57.093] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57..]}
    DEBUG [09:09:57.096] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.108] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:57.110] Creating Prediction for predict set 'test' 
    DEBUG [09:09:57.113] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76]}
    DEBUG [09:09:57.116] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.128] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:57.130] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:57.131] Applying learner 'regr.ranger' on task 'df.tr' (iter 32/50) 
    DEBUG [09:09:57.141] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 6..]}
    DEBUG [09:09:57.145] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.157] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:57.159] Creating Prediction for predict set 'test' 
    DEBUG [09:09:57.162] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71]}
    DEBUG [09:09:57.166] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.178] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:57.180] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:57.181] Applying learner 'regr.ranger' on task 'df.tr' (iter 31/50) 
    DEBUG [09:09:57.192] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60..]}
    DEBUG [09:09:57.195] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.208] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:57.210] Creating Prediction for predict set 'test' 
    DEBUG [09:09:57.213] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73]}
    DEBUG [09:09:57.217] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.229] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:57.231] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:57.233] Applying learner 'regr.ranger' on task 'df.tr' (iter 5/50) 
    DEBUG [09:09:57.243] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:57.247] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.259] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:57.261] Creating Prediction for predict set 'test' 
    DEBUG [09:09:57.264] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 11, 14, 17, 27, 32, 41, 44, 45, 55, 60, 64, 65, 66, 69]}
    DEBUG [09:09:57.268] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.279] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:57.281] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:57.282] Applying learner 'regr.ranger' on task 'df.tr' (iter 2/50) 
    DEBUG [09:09:57.292] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [09:09:57.296] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.308] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:57.310] Creating Prediction for predict set 'test' 
    DEBUG [09:09:57.313] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68]}
    DEBUG [09:09:57.317] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.328] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:57.329] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:57.331] Applying learner 'regr.ranger' on task 'df.tr' (iter 22/50) 
    DEBUG [09:09:57.341] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57..]}
    DEBUG [09:09:57.344] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.356] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:57.358] Creating Prediction for predict set 'test' 
    DEBUG [09:09:57.361] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75]}
    DEBUG [09:09:57.364] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.375] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:57.377] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:57.378] Applying learner 'regr.ranger' on task 'df.tr' (iter 17/50) 
    DEBUG [09:09:57.388] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 7..]}
    DEBUG [09:09:57.392] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.403] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:57.405] Creating Prediction for predict set 'test' 
    DEBUG [09:09:57.408] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70]}
    DEBUG [09:09:57.412] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.423] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:57.425] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:57.426] Applying learner 'regr.ranger' on task 'df.tr' (iter 33/50) 
    DEBUG [09:09:57.442] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:57.447] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.461] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:57.463] Creating Prediction for predict set 'test' 
    DEBUG [09:09:57.466] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60, 76]}
    DEBUG [09:09:57.470] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.482] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:57.484] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:57.485] Applying learner 'regr.ranger' on task 'df.tr' (iter 19/50) 
    DEBUG [09:09:57.495] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:57.499] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.512] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:57.513] Creating Prediction for predict set 'test' 
    DEBUG [09:09:57.517] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 8, 11, 14, 18, 24, 28, 33, 34, 54, 56, 62, 64, 68, 76]}
    DEBUG [09:09:57.520] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.532] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:57.534] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:57.535] Applying learner 'regr.ranger' on task 'df.tr' (iter 49/50) 
    DEBUG [09:09:57.545] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:57.549] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.561] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:57.563] Creating Prediction for predict set 'test' 
    DEBUG [09:09:57.566] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 7, 14, 16, 25, 26, 27, 40, 44, 56, 67, 69, 75, 76]}
    DEBUG [09:09:57.569] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.580] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:57.582] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:57.583] Applying learner 'regr.ranger' on task 'df.tr' (iter 42/50) 
    DEBUG [09:09:57.594] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70..]}
    DEBUG [09:09:57.597] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.609] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:57.611] Creating Prediction for predict set 'test' 
    DEBUG [09:09:57.614] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66]}
    DEBUG [09:09:57.617] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.629] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:57.631] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:57.632] Applying learner 'regr.ranger' on task 'df.tr' (iter 43/50) 
    DEBUG [09:09:57.641] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:57.645] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.657] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:57.658] Creating Prediction for predict set 'test' 
    DEBUG [09:09:57.661] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74]}
    DEBUG [09:09:57.665] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.676] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:57.678] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:57.679] Applying learner 'regr.ranger' on task 'df.tr' (iter 29/50) 
    DEBUG [09:09:57.689] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:57.693] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.704] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:57.706] Creating Prediction for predict set 'test' 
    DEBUG [09:09:57.709] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 18, 25, 28, 30, 39, 45, 48, 51, 54, 55, 61, 66, 71, 73]}
    DEBUG [09:09:57.713] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.724] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:57.725] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:57.727] Applying learner 'regr.ranger' on task 'df.tr' (iter 38/50) 
    DEBUG [09:09:57.737] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:57.740] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.752] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:57.754] Creating Prediction for predict set 'test' 
    DEBUG [09:09:57.757] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58, 69]}
    DEBUG [09:09:57.760] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.772] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:57.773] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:57.774] Applying learner 'regr.ranger' on task 'df.tr' (iter 23/50) 
    DEBUG [09:09:57.785] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:57.788] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.801] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:57.803] Creating Prediction for predict set 'test' 
    DEBUG [09:09:57.807] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66]}
    DEBUG [09:09:57.810] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.822] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:57.824] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:57.825] Applying learner 'regr.ranger' on task 'df.tr' (iter 40/50) 
    DEBUG [09:09:57.835] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:09:57.838] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.850] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:57.852] Creating Prediction for predict set 'test' 
    DEBUG [09:09:57.855] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 8, 16, 17, 24, 25, 27, 29, 41, 44, 53, 54, 67, 72]}
    DEBUG [09:09:57.859] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.870] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:57.872] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:57.873] Applying learner 'regr.ranger' on task 'df.tr' (iter 27/50) 
    DEBUG [09:09:57.883] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67..]}
    DEBUG [09:09:57.887] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.899] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:57.900] Creating Prediction for predict set 'test' 
    DEBUG [09:09:57.903] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76]}
    DEBUG [09:09:57.907] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.918] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:57.920] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:57.921] Applying learner 'regr.ranger' on task 'df.tr' (iter 45/50) 
    DEBUG [09:09:57.931] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:09:57.934] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.946] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:57.948] Creating Prediction for predict set 'test' 
    DEBUG [09:09:57.977] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 8, 12, 21, 22, 25, 28, 33, 37, 38, 44, 46, 48, 57, 75]}
    DEBUG [09:09:57.980] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:57.992] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:57.993] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:57.995] Applying learner 'regr.ranger' on task 'df.tr' (iter 7/50) 
    DEBUG [09:09:58.004] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 6..]}
    DEBUG [09:09:58.007] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.018] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:58.020] Creating Prediction for predict set 'test' 
    DEBUG [09:09:58.023] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74]}
    DEBUG [09:09:58.026] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.036] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:58.038] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:58.039] Applying learner 'regr.ranger' on task 'df.tr' (iter 24/50) 
    DEBUG [09:09:58.048] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:09:58.051] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.062] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:58.064] Creating Prediction for predict set 'test' 
    DEBUG [09:09:58.066] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 13, 17, 25, 32, 35, 36, 37, 38, 50, 63, 65, 68, 69, 74]}
    DEBUG [09:09:58.070] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.080] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:58.081] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:58.082] Applying learner 'regr.ranger' on task 'df.tr' (iter 8/50) 
    DEBUG [09:09:58.091] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:58.094] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.105] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:58.107] Creating Prediction for predict set 'test' 
    DEBUG [09:09:58.111] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 75]}
    DEBUG [09:09:58.114] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.125] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:58.126] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:58.127] Applying learner 'regr.ranger' on task 'df.tr' (iter 35/50) 
    DEBUG [09:09:58.136] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:58.139] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.150] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:58.151] Creating Prediction for predict set 'test' 
    DEBUG [09:09:58.160] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 29, 32, 33, 35, 40, 43, 44, 45, 61, 62, 70, 74]}
    DEBUG [09:09:58.164] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.178] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:58.179] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:58.181] Applying learner 'regr.ranger' on task 'df.tr' (iter 48/50) 
    DEBUG [09:09:58.191] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:09:58.195] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.207] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:58.208] Creating Prediction for predict set 'test' 
    DEBUG [09:09:58.212] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72, 74]}
    DEBUG [09:09:58.215] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.227] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:58.228] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:58.229] Applying learner 'regr.ranger' on task 'df.tr' (iter 14/50) 
    DEBUG [09:09:58.239] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:09:58.243] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.255] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:58.256] Creating Prediction for predict set 'test' 
    DEBUG [09:09:58.259] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 22, 27, 36, 38, 40, 45, 46, 50, 55, 56, 60, 65, 73]}
    DEBUG [09:09:58.263] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.274] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:58.277] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:58.279] Applying learner 'regr.ranger' on task 'df.tr' (iter 28/50) 
    DEBUG [09:09:58.290] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:09:58.294] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.307] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:58.309] Creating Prediction for predict set 'test' 
    DEBUG [09:09:58.312] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75]}
    DEBUG [09:09:58.316] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.327] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:58.329] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:58.330] Applying learner 'regr.ranger' on task 'df.tr' (iter 21/50) 
    DEBUG [09:09:58.341] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66..]}
    DEBUG [09:09:58.345] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.357] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:58.359] Creating Prediction for predict set 'test' 
    DEBUG [09:09:58.364] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72]}
    DEBUG [09:09:58.368] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.380] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:58.381] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:58.383] Applying learner 'regr.ranger' on task 'df.tr' (iter 20/50) 
    DEBUG [09:09:58.393] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:09:58.397] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.409] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:58.410] Creating Prediction for predict set 'test' 
    DEBUG [09:09:58.414] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 16, 17, 32, 36, 37, 42, 43, 45, 46, 53, 61, 65, 69]}
    DEBUG [09:09:58.417] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.429] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:58.430] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:58.432] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [09:09:58.442] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [09:09:58.445] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.457] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:58.459] Creating Prediction for predict set 'test' 
    DEBUG [09:09:58.462] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [09:09:58.465] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.477] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:58.479] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:58.480] Applying learner 'regr.ranger' on task 'df.tr' (iter 34/50) 
    DEBUG [09:09:58.490] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:09:58.495] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.508] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:58.509] Creating Prediction for predict set 'test' 
    DEBUG [09:09:58.513] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 7, 12, 14, 18, 20, 41, 46, 47, 53, 57, 63, 64, 72, 75]}
    DEBUG [09:09:58.516] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.528] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:58.530] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:58.531] Applying learner 'regr.ranger' on task 'df.tr' (iter 6/50) 
    DEBUG [09:09:58.540] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 7..]}
    DEBUG [09:09:58.544] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.556] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:58.558] Creating Prediction for predict set 'test' 
    DEBUG [09:09:58.561] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76]}
    DEBUG [09:09:58.565] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.576] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:58.577] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:58.579] Applying learner 'regr.ranger' on task 'df.tr' (iter 12/50) 
    DEBUG [09:09:58.588] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61..]}
    DEBUG [09:09:58.591] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.603] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:58.604] Creating Prediction for predict set 'test' 
    DEBUG [09:09:58.608] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70]}
    DEBUG [09:09:58.611] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.622] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:58.624] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:58.625] Applying learner 'regr.ranger' on task 'df.tr' (iter 4/50) 
    DEBUG [09:09:58.634] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:09:58.638] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.649] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:58.650] Creating Prediction for predict set 'test' 
    DEBUG [09:09:58.653] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 3, 13, 20, 23, 25, 40, 46, 50, 53, 56, 57, 58, 63]}
    DEBUG [09:09:58.657] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.668] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:58.670] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:58.671] Applying learner 'regr.ranger' on task 'df.tr' (iter 9/50) 
    DEBUG [09:09:58.680] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:58.684] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.695] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:58.697] Creating Prediction for predict set 'test' 
    DEBUG [09:09:58.700] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 23, 27, 40, 42, 46, 49, 56, 58, 61, 64, 67, 70, 71, 72]}
    DEBUG [09:09:58.703] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.714] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:58.716] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:58.717] Applying learner 'regr.ranger' on task 'df.tr' (iter 10/50) 
    DEBUG [09:09:58.729] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:09:58.733] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.744] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:09:58.745] Creating Prediction for predict set 'test' 
    DEBUG [09:09:58.749] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 17, 18, 19, 24, 26, 29, 31, 34, 36, 43, 45, 53, 66, 73]}
    DEBUG [09:09:58.752] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:09:58.763] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:09:58.764] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:09:58.803] Finished benchmark 
    INFO  [09:10:00.502] Result of batch 7: 
    INFO  [09:10:00.506]  mtry sample.fraction num.trees importance regr.rmse      resample_result 
    INFO  [09:10:00.506]     4        0.549811       141   impurity  46.36304 <ResampleResult[19]> 
    INFO  [09:10:00.514] Evaluating 1 configuration(s) 
    INFO  [09:10:00.563] Benchmark with 50 resampling iterations 
    DEBUG [09:10:00.566] Running benchmark() asynchronously with 50 iterations 
    INFO  [09:10:00.581] Applying learner 'regr.ranger' on task 'df.tr' (iter 30/50) 
    DEBUG [09:10:00.592] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:10:00.595] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:00.617] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:00.618] Creating Prediction for predict set 'test' 
    DEBUG [09:10:00.621] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 14, 22, 29, 34, 35, 37, 40, 43, 44, 59, 62, 64, 68, 72]}
    DEBUG [09:10:00.625] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:00.638] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:00.640] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:00.641] Applying learner 'regr.ranger' on task 'df.tr' (iter 18/50) 
    DEBUG [09:10:00.651] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:10:00.654] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:00.671] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:00.674] Creating Prediction for predict set 'test' 
    DEBUG [09:10:00.677] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75]}
    DEBUG [09:10:00.721] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:00.734] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:00.736] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:00.737] Applying learner 'regr.ranger' on task 'df.tr' (iter 44/50) 
    DEBUG [09:10:00.746] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:10:00.749] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:00.765] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:00.766] Creating Prediction for predict set 'test' 
    DEBUG [09:10:00.769] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 14, 17, 19, 24, 36, 41, 47, 51, 53, 56, 62, 63, 67, 71]}
    DEBUG [09:10:00.773] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:00.785] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:00.787] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:00.788] Applying learner 'regr.ranger' on task 'df.tr' (iter 39/50) 
    DEBUG [09:10:00.797] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:10:00.801] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:00.817] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:00.819] Creating Prediction for predict set 'test' 
    DEBUG [09:10:00.823] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 5, 14, 18, 19, 21, 28, 38, 48, 49, 51, 52, 59, 64, 65]}
    DEBUG [09:10:00.826] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:00.838] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:00.840] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:00.841] Applying learner 'regr.ranger' on task 'df.tr' (iter 26/50) 
    DEBUG [09:10:00.851] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75..]}
    DEBUG [09:10:00.854] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:00.869] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:00.870] Creating Prediction for predict set 'test' 
    DEBUG [09:10:00.873] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74]}
    DEBUG [09:10:00.877] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:00.889] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:00.890] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:00.892] Applying learner 'regr.ranger' on task 'df.tr' (iter 36/50) 
    DEBUG [09:10:00.901] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58..]}
    DEBUG [09:10:00.905] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:00.919] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:00.920] Creating Prediction for predict set 'test' 
    DEBUG [09:10:00.923] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75]}
    DEBUG [09:10:00.927] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:00.939] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:00.940] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:00.941] Applying learner 'regr.ranger' on task 'df.tr' (iter 45/50) 
    DEBUG [09:10:00.951] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:10:00.954] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:00.969] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:00.970] Creating Prediction for predict set 'test' 
    DEBUG [09:10:00.973] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 8, 12, 21, 22, 25, 28, 33, 37, 38, 44, 46, 48, 57, 75]}
    DEBUG [09:10:00.976] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:00.989] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:00.990] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:00.991] Applying learner 'regr.ranger' on task 'df.tr' (iter 3/50) 
    DEBUG [09:10:01.001] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:10:01.004] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.021] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:01.023] Creating Prediction for predict set 'test' 
    DEBUG [09:10:01.026] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75, 76]}
    DEBUG [09:10:01.030] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.042] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:01.044] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:01.045] Applying learner 'regr.ranger' on task 'df.tr' (iter 25/50) 
    DEBUG [09:10:01.055] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:10:01.059] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.073] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:01.075] Creating Prediction for predict set 'test' 
    DEBUG [09:10:01.078] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [9, 11, 18, 21, 23, 27, 29, 43, 44, 45, 51, 55, 67, 73, 76]}
    DEBUG [09:10:01.082] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.094] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:01.096] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:01.097] Applying learner 'regr.ranger' on task 'df.tr' (iter 11/50) 
    DEBUG [09:10:01.106] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 7..]}
    DEBUG [09:10:01.110] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.127] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:01.130] Creating Prediction for predict set 'test' 
    DEBUG [09:10:01.134] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76]}
    DEBUG [09:10:01.138] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.151] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:01.153] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:01.154] Applying learner 'regr.ranger' on task 'df.tr' (iter 50/50) 
    DEBUG [09:10:01.164] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:10:01.167] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.183] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:01.186] Creating Prediction for predict set 'test' 
    DEBUG [09:10:01.190] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 8, 21, 22, 30, 36, 41, 45, 49, 53, 54, 57, 68, 71]}
    DEBUG [09:10:01.194] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.206] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:01.207] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:01.208] Applying learner 'regr.ranger' on task 'df.tr' (iter 16/50) 
    DEBUG [09:10:01.218] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75..]}
    DEBUG [09:10:01.221] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.235] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:01.236] Creating Prediction for predict set 'test' 
    DEBUG [09:10:01.239] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72]}
    DEBUG [09:10:01.242] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.254] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:01.256] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:01.257] Applying learner 'regr.ranger' on task 'df.tr' (iter 41/50) 
    DEBUG [09:10:01.266] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74..]}
    DEBUG [09:10:01.269] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.284] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:01.285] Creating Prediction for predict set 'test' 
    DEBUG [09:10:01.288] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76]}
    DEBUG [09:10:01.292] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.304] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:01.305] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:01.306] Applying learner 'regr.ranger' on task 'df.tr' (iter 32/50) 
    DEBUG [09:10:01.316] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 6..]}
    DEBUG [09:10:01.319] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.336] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:01.338] Creating Prediction for predict set 'test' 
    DEBUG [09:10:01.341] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71]}
    DEBUG [09:10:01.344] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.356] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:01.357] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:01.358] Applying learner 'regr.ranger' on task 'df.tr' (iter 31/50) 
    DEBUG [09:10:01.368] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60..]}
    DEBUG [09:10:01.372] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.390] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:01.392] Creating Prediction for predict set 'test' 
    DEBUG [09:10:01.396] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73]}
    DEBUG [09:10:01.399] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.415] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:01.416] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:01.418] Applying learner 'regr.ranger' on task 'df.tr' (iter 2/50) 
    DEBUG [09:10:01.428] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [09:10:01.432] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.450] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:01.452] Creating Prediction for predict set 'test' 
    DEBUG [09:10:01.457] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68]}
    DEBUG [09:10:01.460] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.473] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:01.475] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:01.476] Applying learner 'regr.ranger' on task 'df.tr' (iter 22/50) 
    DEBUG [09:10:01.486] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57..]}
    DEBUG [09:10:01.490] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.506] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:01.507] Creating Prediction for predict set 'test' 
    DEBUG [09:10:01.511] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75]}
    DEBUG [09:10:01.515] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.528] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:01.530] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:01.531] Applying learner 'regr.ranger' on task 'df.tr' (iter 17/50) 
    DEBUG [09:10:01.541] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 7..]}
    DEBUG [09:10:01.545] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.562] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:01.564] Creating Prediction for predict set 'test' 
    DEBUG [09:10:01.569] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70]}
    DEBUG [09:10:01.572] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.585] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:01.586] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:01.588] Applying learner 'regr.ranger' on task 'df.tr' (iter 5/50) 
    DEBUG [09:10:01.597] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:10:01.601] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.619] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:01.621] Creating Prediction for predict set 'test' 
    DEBUG [09:10:01.624] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 11, 14, 17, 27, 32, 41, 44, 45, 55, 60, 64, 65, 66, 69]}
    DEBUG [09:10:01.628] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.640] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:01.642] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:01.643] Applying learner 'regr.ranger' on task 'df.tr' (iter 19/50) 
    DEBUG [09:10:01.653] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:10:01.657] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.674] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:01.675] Creating Prediction for predict set 'test' 
    DEBUG [09:10:01.679] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 8, 11, 14, 18, 24, 28, 33, 34, 54, 56, 62, 64, 68, 76]}
    DEBUG [09:10:01.682] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.695] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:01.697] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:01.698] Applying learner 'regr.ranger' on task 'df.tr' (iter 15/50) 
    DEBUG [09:10:01.708] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:10:01.711] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.729] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:01.731] Creating Prediction for predict set 'test' 
    DEBUG [09:10:01.735] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 19, 20, 30, 35, 42, 47, 54, 59, 64, 67, 71, 74, 75]}
    DEBUG [09:10:01.739] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.751] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:01.753] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:01.754] Applying learner 'regr.ranger' on task 'df.tr' (iter 48/50) 
    DEBUG [09:10:01.765] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:10:01.768] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.785] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:01.788] Creating Prediction for predict set 'test' 
    DEBUG [09:10:01.791] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72, 74]}
    DEBUG [09:10:01.794] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.807] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:01.809] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:01.810] Applying learner 'regr.ranger' on task 'df.tr' (iter 43/50) 
    DEBUG [09:10:01.820] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:10:01.824] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.839] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:01.841] Creating Prediction for predict set 'test' 
    DEBUG [09:10:01.844] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74]}
    DEBUG [09:10:01.848] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.861] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:01.862] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:01.864] Applying learner 'regr.ranger' on task 'df.tr' (iter 27/50) 
    DEBUG [09:10:01.874] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67..]}
    DEBUG [09:10:01.878] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.895] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:01.897] Creating Prediction for predict set 'test' 
    DEBUG [09:10:01.900] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76]}
    DEBUG [09:10:01.904] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.917] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:01.918] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:01.920] Applying learner 'regr.ranger' on task 'df.tr' (iter 34/50) 
    DEBUG [09:10:01.930] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:10:01.933] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.950] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:01.952] Creating Prediction for predict set 'test' 
    DEBUG [09:10:01.955] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 7, 12, 14, 18, 20, 41, 46, 47, 53, 57, 63, 64, 72, 75]}
    DEBUG [09:10:01.959] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:01.972] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:01.973] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:01.974] Applying learner 'regr.ranger' on task 'df.tr' (iter 42/50) 
    DEBUG [09:10:01.984] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70..]}
    DEBUG [09:10:01.988] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.002] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:02.004] Creating Prediction for predict set 'test' 
    DEBUG [09:10:02.007] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66]}
    DEBUG [09:10:02.010] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.023] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:02.025] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:02.026] Applying learner 'regr.ranger' on task 'df.tr' (iter 40/50) 
    DEBUG [09:10:02.036] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:10:02.039] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.056] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:02.058] Creating Prediction for predict set 'test' 
    DEBUG [09:10:02.063] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 8, 16, 17, 24, 25, 27, 29, 41, 44, 53, 54, 67, 72]}
    DEBUG [09:10:02.066] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.079] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:02.081] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:02.082] Applying learner 'regr.ranger' on task 'df.tr' (iter 28/50) 
    DEBUG [09:10:02.092] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:10:02.095] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.129] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:02.131] Creating Prediction for predict set 'test' 
    DEBUG [09:10:02.134] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75]}
    DEBUG [09:10:02.138] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.152] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:02.153] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:02.155] Applying learner 'regr.ranger' on task 'df.tr' (iter 35/50) 
    DEBUG [09:10:02.164] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:10:02.167] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.182] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:02.183] Creating Prediction for predict set 'test' 
    DEBUG [09:10:02.186] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 29, 32, 33, 35, 40, 43, 44, 45, 61, 62, 70, 74]}
    DEBUG [09:10:02.189] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.201] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:02.203] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:02.204] Applying learner 'regr.ranger' on task 'df.tr' (iter 37/50) 
    DEBUG [09:10:02.213] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57..]}
    DEBUG [09:10:02.216] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.232] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:02.234] Creating Prediction for predict set 'test' 
    DEBUG [09:10:02.237] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76]}
    DEBUG [09:10:02.240] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.252] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:02.253] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:02.254] Applying learner 'regr.ranger' on task 'df.tr' (iter 21/50) 
    DEBUG [09:10:02.263] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66..]}
    DEBUG [09:10:02.267] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.281] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:02.282] Creating Prediction for predict set 'test' 
    DEBUG [09:10:02.285] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72]}
    DEBUG [09:10:02.289] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.301] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:02.302] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:02.303] Applying learner 'regr.ranger' on task 'df.tr' (iter 7/50) 
    DEBUG [09:10:02.312] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 6..]}
    DEBUG [09:10:02.315] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.332] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:02.334] Creating Prediction for predict set 'test' 
    DEBUG [09:10:02.337] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74]}
    DEBUG [09:10:02.340] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.351] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:02.353] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:02.354] Applying learner 'regr.ranger' on task 'df.tr' (iter 46/50) 
    DEBUG [09:10:02.363] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72..]}
    DEBUG [09:10:02.367] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.393] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:02.395] Creating Prediction for predict set 'test' 
    DEBUG [09:10:02.398] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73]}
    DEBUG [09:10:02.401] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.414] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:02.415] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:02.416] Applying learner 'regr.ranger' on task 'df.tr' (iter 47/50) 
    DEBUG [09:10:02.426] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64..]}
    DEBUG [09:10:02.429] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.446] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:02.448] Creating Prediction for predict set 'test' 
    DEBUG [09:10:02.453] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70]}
    DEBUG [09:10:02.456] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.468] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:02.469] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:02.470] Applying learner 'regr.ranger' on task 'df.tr' (iter 8/50) 
    DEBUG [09:10:02.479] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:10:02.483] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.499] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:02.501] Creating Prediction for predict set 'test' 
    DEBUG [09:10:02.505] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 75]}
    DEBUG [09:10:02.508] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.519] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:02.521] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:02.522] Applying learner 'regr.ranger' on task 'df.tr' (iter 29/50) 
    DEBUG [09:10:02.531] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:10:02.534] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.550] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:02.552] Creating Prediction for predict set 'test' 
    DEBUG [09:10:02.555] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 18, 25, 28, 30, 39, 45, 48, 51, 54, 55, 61, 66, 71, 73]}
    DEBUG [09:10:02.558] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.571] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:02.573] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:02.574] Applying learner 'regr.ranger' on task 'df.tr' (iter 24/50) 
    DEBUG [09:10:02.585] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:10:02.589] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.605] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:02.607] Creating Prediction for predict set 'test' 
    DEBUG [09:10:02.610] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 13, 17, 25, 32, 35, 36, 37, 38, 50, 63, 65, 68, 69, 74]}
    DEBUG [09:10:02.614] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.627] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:02.628] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:02.630] Applying learner 'regr.ranger' on task 'df.tr' (iter 33/50) 
    DEBUG [09:10:02.640] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:10:02.644] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.661] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:02.663] Creating Prediction for predict set 'test' 
    DEBUG [09:10:02.666] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60, 76]}
    DEBUG [09:10:02.670] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.683] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:02.685] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:02.686] Applying learner 'regr.ranger' on task 'df.tr' (iter 20/50) 
    DEBUG [09:10:02.697] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:10:02.700] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.718] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:02.720] Creating Prediction for predict set 'test' 
    DEBUG [09:10:02.723] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 16, 17, 32, 36, 37, 42, 43, 45, 46, 53, 61, 65, 69]}
    DEBUG [09:10:02.726] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.738] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:02.740] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:02.741] Applying learner 'regr.ranger' on task 'df.tr' (iter 49/50) 
    DEBUG [09:10:02.751] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:10:02.754] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.771] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:02.773] Creating Prediction for predict set 'test' 
    DEBUG [09:10:02.776] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 7, 14, 16, 25, 26, 27, 40, 44, 56, 67, 69, 75, 76]}
    DEBUG [09:10:02.780] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.791] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:02.793] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:02.794] Applying learner 'regr.ranger' on task 'df.tr' (iter 23/50) 
    DEBUG [09:10:02.804] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:10:02.807] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.821] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:02.823] Creating Prediction for predict set 'test' 
    DEBUG [09:10:02.827] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66]}
    DEBUG [09:10:02.830] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.841] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:02.843] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:02.844] Applying learner 'regr.ranger' on task 'df.tr' (iter 9/50) 
    DEBUG [09:10:02.853] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:10:02.856] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.870] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:02.871] Creating Prediction for predict set 'test' 
    DEBUG [09:10:02.874] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 23, 27, 40, 42, 46, 49, 56, 58, 61, 64, 67, 70, 71, 72]}
    DEBUG [09:10:02.877] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.889] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:02.891] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:02.892] Applying learner 'regr.ranger' on task 'df.tr' (iter 10/50) 
    DEBUG [09:10:02.901] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:10:02.904] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.919] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:02.920] Creating Prediction for predict set 'test' 
    DEBUG [09:10:02.923] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 17, 18, 19, 24, 26, 29, 31, 34, 36, 43, 45, 53, 66, 73]}
    DEBUG [09:10:02.926] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.938] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:02.939] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:02.941] Applying learner 'regr.ranger' on task 'df.tr' (iter 6/50) 
    DEBUG [09:10:02.950] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 7..]}
    DEBUG [09:10:02.953] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.967] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:02.968] Creating Prediction for predict set 'test' 
    DEBUG [09:10:02.971] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76]}
    DEBUG [09:10:02.975] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:02.986] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:02.988] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:02.989] Applying learner 'regr.ranger' on task 'df.tr' (iter 14/50) 
    DEBUG [09:10:02.998] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:10:03.001] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:03.017] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:03.019] Creating Prediction for predict set 'test' 
    DEBUG [09:10:03.022] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 22, 27, 36, 38, 40, 45, 46, 50, 55, 56, 60, 65, 73]}
    DEBUG [09:10:03.026] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:03.037] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:03.039] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:03.040] Applying learner 'regr.ranger' on task 'df.tr' (iter 13/50) 
    DEBUG [09:10:03.049] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:10:03.052] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:03.068] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:03.070] Creating Prediction for predict set 'test' 
    DEBUG [09:10:03.074] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 72]}
    DEBUG [09:10:03.077] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:03.088] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:03.090] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:03.091] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [09:10:03.100] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [09:10:03.103] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:03.117] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:03.119] Creating Prediction for predict set 'test' 
    DEBUG [09:10:03.122] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [09:10:03.125] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:03.137] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:03.138] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:03.139] Applying learner 'regr.ranger' on task 'df.tr' (iter 4/50) 
    DEBUG [09:10:03.148] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:10:03.152] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:03.168] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:03.169] Creating Prediction for predict set 'test' 
    DEBUG [09:10:03.172] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 3, 13, 20, 23, 25, 40, 46, 50, 53, 56, 57, 58, 63]}
    DEBUG [09:10:03.176] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:03.187] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:03.189] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:03.190] Applying learner 'regr.ranger' on task 'df.tr' (iter 38/50) 
    DEBUG [09:10:03.199] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:10:03.203] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:03.217] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:03.218] Creating Prediction for predict set 'test' 
    DEBUG [09:10:03.221] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58, 69]}
    DEBUG [09:10:03.224] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:03.236] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:03.237] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:03.239] Applying learner 'regr.ranger' on task 'df.tr' (iter 12/50) 
    DEBUG [09:10:03.248] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61..]}
    DEBUG [09:10:03.251] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:03.266] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:03.267] Creating Prediction for predict set 'test' 
    DEBUG [09:10:03.270] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70]}
    DEBUG [09:10:03.274] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:03.285] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:03.287] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:03.358] Finished benchmark 
    INFO  [09:10:04.863] Result of batch 8: 
    INFO  [09:10:04.866]  mtry sample.fraction num.trees importance regr.rmse      resample_result 
    INFO  [09:10:04.866]     4       0.5538044       401   impurity   46.3686 <ResampleResult[19]> 
    INFO  [09:10:04.873] Evaluating 1 configuration(s) 
    INFO  [09:10:04.905] Benchmark with 50 resampling iterations 
    DEBUG [09:10:04.906] Running benchmark() asynchronously with 50 iterations 
    INFO  [09:10:04.917] Applying learner 'regr.ranger' on task 'df.tr' (iter 26/50) 
    DEBUG [09:10:04.924] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75..]}
    DEBUG [09:10:04.927] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:04.945] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:04.947] Creating Prediction for predict set 'test' 
    DEBUG [09:10:04.950] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74]}
    DEBUG [09:10:04.952] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:04.963] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:04.964] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:04.965] Applying learner 'regr.ranger' on task 'df.tr' (iter 36/50) 
    DEBUG [09:10:04.972] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58..]}
    DEBUG [09:10:04.975] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:04.990] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:04.992] Creating Prediction for predict set 'test' 
    DEBUG [09:10:04.994] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75]}
    DEBUG [09:10:04.997] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.007] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:05.008] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:05.009] Applying learner 'regr.ranger' on task 'df.tr' (iter 49/50) 
    DEBUG [09:10:05.017] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:10:05.019] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.035] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:05.037] Creating Prediction for predict set 'test' 
    DEBUG [09:10:05.041] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 7, 14, 16, 25, 26, 27, 40, 44, 56, 67, 69, 75, 76]}
    DEBUG [09:10:05.043] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.053] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:05.054] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:05.055] Applying learner 'regr.ranger' on task 'df.tr' (iter 3/50) 
    DEBUG [09:10:05.062] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:10:05.065] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.080] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:05.082] Creating Prediction for predict set 'test' 
    DEBUG [09:10:05.086] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75, 76]}
    DEBUG [09:10:05.090] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.102] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:05.104] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:05.105] Applying learner 'regr.ranger' on task 'df.tr' (iter 25/50) 
    DEBUG [09:10:05.114] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:10:05.117] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.135] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:05.137] Creating Prediction for predict set 'test' 
    DEBUG [09:10:05.141] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [9, 11, 18, 21, 23, 27, 29, 43, 44, 45, 51, 55, 67, 73, 76]}
    DEBUG [09:10:05.144] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.156] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:05.158] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:05.159] Applying learner 'regr.ranger' on task 'df.tr' (iter 11/50) 
    DEBUG [09:10:05.168] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 7..]}
    DEBUG [09:10:05.171] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.188] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:05.190] Creating Prediction for predict set 'test' 
    DEBUG [09:10:05.193] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76]}
    DEBUG [09:10:05.196] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.208] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:05.210] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:05.211] Applying learner 'regr.ranger' on task 'df.tr' (iter 30/50) 
    DEBUG [09:10:05.220] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:10:05.224] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.242] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:05.244] Creating Prediction for predict set 'test' 
    DEBUG [09:10:05.247] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 14, 22, 29, 34, 35, 37, 40, 43, 44, 59, 62, 64, 68, 72]}
    DEBUG [09:10:05.251] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.263] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:05.264] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:05.265] Applying learner 'regr.ranger' on task 'df.tr' (iter 16/50) 
    DEBUG [09:10:05.275] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75..]}
    DEBUG [09:10:05.278] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.294] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:05.295] Creating Prediction for predict set 'test' 
    DEBUG [09:10:05.298] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72]}
    DEBUG [09:10:05.301] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.313] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:05.315] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:05.316] Applying learner 'regr.ranger' on task 'df.tr' (iter 45/50) 
    DEBUG [09:10:05.325] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:10:05.329] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.345] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:05.347] Creating Prediction for predict set 'test' 
    DEBUG [09:10:05.351] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 8, 12, 21, 22, 25, 28, 33, 37, 38, 44, 46, 48, 57, 75]}
    DEBUG [09:10:05.359] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.376] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:05.377] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:05.378] Applying learner 'regr.ranger' on task 'df.tr' (iter 38/50) 
    DEBUG [09:10:05.388] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:10:05.392] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.408] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:05.409] Creating Prediction for predict set 'test' 
    DEBUG [09:10:05.412] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58, 69]}
    DEBUG [09:10:05.415] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.427] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:05.429] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:05.430] Applying learner 'regr.ranger' on task 'df.tr' (iter 32/50) 
    DEBUG [09:10:05.439] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 6..]}
    DEBUG [09:10:05.443] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.461] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:05.463] Creating Prediction for predict set 'test' 
    DEBUG [09:10:05.466] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71]}
    DEBUG [09:10:05.470] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.487] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:05.490] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:05.492] Applying learner 'regr.ranger' on task 'df.tr' (iter 31/50) 
    DEBUG [09:10:05.505] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60..]}
    DEBUG [09:10:05.509] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.527] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:05.529] Creating Prediction for predict set 'test' 
    DEBUG [09:10:05.532] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73]}
    DEBUG [09:10:05.535] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.547] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:05.548] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:05.549] Applying learner 'regr.ranger' on task 'df.tr' (iter 37/50) 
    DEBUG [09:10:05.558] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57..]}
    DEBUG [09:10:05.562] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.577] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:05.579] Creating Prediction for predict set 'test' 
    DEBUG [09:10:05.582] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76]}
    DEBUG [09:10:05.585] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.598] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:05.599] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:05.600] Applying learner 'regr.ranger' on task 'df.tr' (iter 2/50) 
    DEBUG [09:10:05.609] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [09:10:05.613] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.630] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:05.632] Creating Prediction for predict set 'test' 
    DEBUG [09:10:05.635] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68]}
    DEBUG [09:10:05.639] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.651] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:05.653] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:05.654] Applying learner 'regr.ranger' on task 'df.tr' (iter 22/50) 
    DEBUG [09:10:05.664] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57..]}
    DEBUG [09:10:05.667] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.685] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:05.687] Creating Prediction for predict set 'test' 
    DEBUG [09:10:05.690] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75]}
    DEBUG [09:10:05.693] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.706] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:05.707] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:05.709] Applying learner 'regr.ranger' on task 'df.tr' (iter 17/50) 
    DEBUG [09:10:05.718] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 7..]}
    DEBUG [09:10:05.721] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.738] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:05.740] Creating Prediction for predict set 'test' 
    DEBUG [09:10:05.744] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70]}
    DEBUG [09:10:05.747] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.760] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:05.762] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:05.763] Applying learner 'regr.ranger' on task 'df.tr' (iter 5/50) 
    DEBUG [09:10:05.772] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:10:05.775] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.792] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:05.794] Creating Prediction for predict set 'test' 
    DEBUG [09:10:05.797] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 11, 14, 17, 27, 32, 41, 44, 45, 55, 60, 64, 65, 66, 69]}
    DEBUG [09:10:05.800] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.812] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:05.814] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:05.815] Applying learner 'regr.ranger' on task 'df.tr' (iter 15/50) 
    DEBUG [09:10:05.824] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:10:05.828] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.843] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:05.844] Creating Prediction for predict set 'test' 
    DEBUG [09:10:05.848] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 19, 20, 30, 35, 42, 47, 54, 59, 64, 67, 71, 74, 75]}
    DEBUG [09:10:05.851] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.863] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:05.865] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:05.866] Applying learner 'regr.ranger' on task 'df.tr' (iter 19/50) 
    DEBUG [09:10:05.876] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:10:05.879] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.897] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:05.899] Creating Prediction for predict set 'test' 
    DEBUG [09:10:05.903] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 8, 11, 14, 18, 24, 28, 33, 34, 54, 56, 62, 64, 68, 76]}
    DEBUG [09:10:05.906] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.919] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:05.921] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:05.922] Applying learner 'regr.ranger' on task 'df.tr' (iter 44/50) 
    DEBUG [09:10:05.931] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:10:05.935] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.952] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:05.953] Creating Prediction for predict set 'test' 
    DEBUG [09:10:05.957] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 14, 17, 19, 24, 36, 41, 47, 51, 53, 56, 62, 63, 67, 71]}
    DEBUG [09:10:05.960] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:05.973] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:05.974] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:05.975] Applying learner 'regr.ranger' on task 'df.tr' (iter 47/50) 
    DEBUG [09:10:05.985] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64..]}
    DEBUG [09:10:05.988] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.004] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:06.006] Creating Prediction for predict set 'test' 
    DEBUG [09:10:06.009] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70]}
    DEBUG [09:10:06.039] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.053] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:06.055] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:06.056] Applying learner 'regr.ranger' on task 'df.tr' (iter 27/50) 
    DEBUG [09:10:06.065] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67..]}
    DEBUG [09:10:06.068] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.085] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:06.087] Creating Prediction for predict set 'test' 
    DEBUG [09:10:06.090] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76]}
    DEBUG [09:10:06.093] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.106] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:06.107] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:06.108] Applying learner 'regr.ranger' on task 'df.tr' (iter 48/50) 
    DEBUG [09:10:06.117] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:10:06.121] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.136] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:06.138] Creating Prediction for predict set 'test' 
    DEBUG [09:10:06.141] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72, 74]}
    DEBUG [09:10:06.144] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.156] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:06.157] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:06.158] Applying learner 'regr.ranger' on task 'df.tr' (iter 46/50) 
    DEBUG [09:10:06.167] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72..]}
    DEBUG [09:10:06.171] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.187] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:06.188] Creating Prediction for predict set 'test' 
    DEBUG [09:10:06.191] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73]}
    DEBUG [09:10:06.194] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.206] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:06.207] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:06.209] Applying learner 'regr.ranger' on task 'df.tr' (iter 33/50) 
    DEBUG [09:10:06.217] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:10:06.221] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.237] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:06.238] Creating Prediction for predict set 'test' 
    DEBUG [09:10:06.241] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60, 76]}
    DEBUG [09:10:06.244] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.256] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:06.257] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:06.258] Applying learner 'regr.ranger' on task 'df.tr' (iter 39/50) 
    DEBUG [09:10:06.267] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:10:06.271] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.289] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:06.290] Creating Prediction for predict set 'test' 
    DEBUG [09:10:06.293] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 5, 14, 18, 19, 21, 28, 38, 48, 49, 51, 52, 59, 64, 65]}
    DEBUG [09:10:06.296] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.308] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:06.309] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:06.310] Applying learner 'regr.ranger' on task 'df.tr' (iter 41/50) 
    DEBUG [09:10:06.319] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74..]}
    DEBUG [09:10:06.323] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.338] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:06.340] Creating Prediction for predict set 'test' 
    DEBUG [09:10:06.343] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76]}
    DEBUG [09:10:06.346] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.358] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:06.360] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:06.361] Applying learner 'regr.ranger' on task 'df.tr' (iter 34/50) 
    DEBUG [09:10:06.370] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:10:06.373] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.391] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:06.393] Creating Prediction for predict set 'test' 
    DEBUG [09:10:06.397] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 7, 12, 14, 18, 20, 41, 46, 47, 53, 57, 63, 64, 72, 75]}
    DEBUG [09:10:06.400] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.412] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:06.414] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:06.415] Applying learner 'regr.ranger' on task 'df.tr' (iter 23/50) 
    DEBUG [09:10:06.424] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:10:06.427] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.442] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:06.444] Creating Prediction for predict set 'test' 
    DEBUG [09:10:06.447] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66]}
    DEBUG [09:10:06.451] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.465] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:06.466] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:06.467] Applying learner 'regr.ranger' on task 'df.tr' (iter 7/50) 
    DEBUG [09:10:06.478] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 6..]}
    DEBUG [09:10:06.481] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.499] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:06.501] Creating Prediction for predict set 'test' 
    DEBUG [09:10:06.505] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74]}
    DEBUG [09:10:06.509] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.529] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:06.531] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:06.532] Applying learner 'regr.ranger' on task 'df.tr' (iter 24/50) 
    DEBUG [09:10:06.544] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:10:06.548] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.569] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:06.570] Creating Prediction for predict set 'test' 
    DEBUG [09:10:06.574] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 13, 17, 25, 32, 35, 36, 37, 38, 50, 63, 65, 68, 69, 74]}
    DEBUG [09:10:06.578] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.591] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:06.593] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:06.594] Applying learner 'regr.ranger' on task 'df.tr' (iter 43/50) 
    DEBUG [09:10:06.604] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:10:06.608] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.626] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:06.627] Creating Prediction for predict set 'test' 
    DEBUG [09:10:06.631] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74]}
    DEBUG [09:10:06.634] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.647] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:06.649] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:06.650] Applying learner 'regr.ranger' on task 'df.tr' (iter 8/50) 
    DEBUG [09:10:06.660] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:10:06.663] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.679] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:06.681] Creating Prediction for predict set 'test' 
    DEBUG [09:10:06.684] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 75]}
    DEBUG [09:10:06.688] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.700] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:06.702] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:06.703] Applying learner 'regr.ranger' on task 'df.tr' (iter 9/50) 
    DEBUG [09:10:06.713] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:10:06.716] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.734] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:06.736] Creating Prediction for predict set 'test' 
    DEBUG [09:10:06.739] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 23, 27, 40, 42, 46, 49, 56, 58, 61, 64, 67, 70, 71, 72]}
    DEBUG [09:10:06.743] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.756] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:06.757] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:06.758] Applying learner 'regr.ranger' on task 'df.tr' (iter 29/50) 
    DEBUG [09:10:06.768] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:10:06.772] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.790] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:06.791] Creating Prediction for predict set 'test' 
    DEBUG [09:10:06.795] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 18, 25, 28, 30, 39, 45, 48, 51, 54, 55, 61, 66, 71, 73]}
    DEBUG [09:10:06.799] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.811] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:06.813] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:06.814] Applying learner 'regr.ranger' on task 'df.tr' (iter 14/50) 
    DEBUG [09:10:06.824] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:10:06.828] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.845] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:06.847] Creating Prediction for predict set 'test' 
    DEBUG [09:10:06.850] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 22, 27, 36, 38, 40, 45, 46, 50, 55, 56, 60, 65, 73]}
    DEBUG [09:10:06.853] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.866] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:06.868] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:06.869] Applying learner 'regr.ranger' on task 'df.tr' (iter 50/50) 
    DEBUG [09:10:06.879] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:10:06.883] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.900] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:06.902] Creating Prediction for predict set 'test' 
    DEBUG [09:10:06.906] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 8, 21, 22, 30, 36, 41, 45, 49, 53, 54, 57, 68, 71]}
    DEBUG [09:10:06.909] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.929] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:06.931] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:06.932] Applying learner 'regr.ranger' on task 'df.tr' (iter 40/50) 
    DEBUG [09:10:06.943] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:10:06.947] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.968] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:06.971] Creating Prediction for predict set 'test' 
    DEBUG [09:10:06.975] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 8, 16, 17, 24, 25, 27, 29, 41, 44, 53, 54, 67, 72]}
    DEBUG [09:10:06.979] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:06.993] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:06.994] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:06.996] Applying learner 'regr.ranger' on task 'df.tr' (iter 28/50) 
    DEBUG [09:10:07.006] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:10:07.010] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.029] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:07.031] Creating Prediction for predict set 'test' 
    DEBUG [09:10:07.035] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75]}
    DEBUG [09:10:07.038] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.052] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:07.054] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:07.055] Applying learner 'regr.ranger' on task 'df.tr' (iter 21/50) 
    DEBUG [09:10:07.065] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66..]}
    DEBUG [09:10:07.069] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.089] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:07.091] Creating Prediction for predict set 'test' 
    DEBUG [09:10:07.094] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72]}
    DEBUG [09:10:07.098] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.111] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:07.113] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:07.114] Applying learner 'regr.ranger' on task 'df.tr' (iter 18/50) 
    DEBUG [09:10:07.125] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:10:07.128] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.147] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:07.149] Creating Prediction for predict set 'test' 
    DEBUG [09:10:07.152] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75]}
    DEBUG [09:10:07.156] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.169] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:07.171] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:07.172] Applying learner 'regr.ranger' on task 'df.tr' (iter 4/50) 
    DEBUG [09:10:07.183] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:10:07.187] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.209] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:07.211] Creating Prediction for predict set 'test' 
    DEBUG [09:10:07.389] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 3, 13, 20, 23, 25, 40, 46, 50, 53, 56, 57, 58, 63]}
    DEBUG [09:10:07.393] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.408] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:07.410] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:07.412] Applying learner 'regr.ranger' on task 'df.tr' (iter 13/50) 
    DEBUG [09:10:07.423] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:10:07.427] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.445] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:07.447] Creating Prediction for predict set 'test' 
    DEBUG [09:10:07.451] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 72]}
    DEBUG [09:10:07.455] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.469] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:07.471] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:07.472] Applying learner 'regr.ranger' on task 'df.tr' (iter 42/50) 
    DEBUG [09:10:07.483] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70..]}
    DEBUG [09:10:07.487] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.507] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:07.509] Creating Prediction for predict set 'test' 
    DEBUG [09:10:07.513] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66]}
    DEBUG [09:10:07.517] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.531] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:07.533] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:07.534] Applying learner 'regr.ranger' on task 'df.tr' (iter 12/50) 
    DEBUG [09:10:07.545] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61..]}
    DEBUG [09:10:07.549] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.567] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:07.569] Creating Prediction for predict set 'test' 
    DEBUG [09:10:07.573] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70]}
    DEBUG [09:10:07.576] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.590] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:07.592] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:07.594] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [09:10:07.605] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [09:10:07.609] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.636] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:07.639] Creating Prediction for predict set 'test' 
    DEBUG [09:10:07.643] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [09:10:07.650] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.668] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:07.670] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:07.671] Applying learner 'regr.ranger' on task 'df.tr' (iter 10/50) 
    DEBUG [09:10:07.683] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:10:07.690] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.709] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:07.711] Creating Prediction for predict set 'test' 
    DEBUG [09:10:07.715] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 17, 18, 19, 24, 26, 29, 31, 34, 36, 43, 45, 53, 66, 73]}
    DEBUG [09:10:07.719] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.734] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:07.736] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:07.738] Applying learner 'regr.ranger' on task 'df.tr' (iter 6/50) 
    DEBUG [09:10:07.749] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 7..]}
    DEBUG [09:10:07.753] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.774] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:07.776] Creating Prediction for predict set 'test' 
    DEBUG [09:10:07.780] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76]}
    DEBUG [09:10:07.784] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.796] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:07.798] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:07.799] Applying learner 'regr.ranger' on task 'df.tr' (iter 35/50) 
    DEBUG [09:10:07.808] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:10:07.811] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.826] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:07.828] Creating Prediction for predict set 'test' 
    DEBUG [09:10:07.831] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 29, 32, 33, 35, 40, 43, 44, 45, 61, 62, 70, 74]}
    DEBUG [09:10:07.834] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.845] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:07.847] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:07.848] Applying learner 'regr.ranger' on task 'df.tr' (iter 20/50) 
    DEBUG [09:10:07.857] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:10:07.861] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.876] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:07.877] Creating Prediction for predict set 'test' 
    DEBUG [09:10:07.880] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 16, 17, 32, 36, 37, 42, 43, 45, 46, 53, 61, 65, 69]}
    DEBUG [09:10:07.884] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:07.895] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:07.897] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:07.942] Finished benchmark 
    INFO  [09:10:09.556] Result of batch 9: 
    INFO  [09:10:09.560]  mtry sample.fraction num.trees importance regr.rmse      resample_result 
    INFO  [09:10:09.560]     5       0.6688764       463   impurity  43.32544 <ResampleResult[19]> 
    INFO  [09:10:09.567] Evaluating 1 configuration(s) 
    INFO  [09:10:09.601] Benchmark with 50 resampling iterations 
    DEBUG [09:10:09.602] Running benchmark() asynchronously with 50 iterations 
    INFO  [09:10:09.616] Applying learner 'regr.ranger' on task 'df.tr' (iter 25/50) 
    DEBUG [09:10:09.624] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:10:09.627] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:09.637] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:09.639] Creating Prediction for predict set 'test' 
    DEBUG [09:10:09.641] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [9, 11, 18, 21, 23, 27, 29, 43, 44, 45, 51, 55, 67, 73, 76]}
    DEBUG [09:10:09.644] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:09.654] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:09.655] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:09.656] Applying learner 'regr.ranger' on task 'df.tr' (iter 11/50) 
    DEBUG [09:10:09.665] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 7..]}
    DEBUG [09:10:09.668] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:09.685] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:09.686] Creating Prediction for predict set 'test' 
    DEBUG [09:10:09.689] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76]}
    DEBUG [09:10:09.692] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:09.702] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:09.704] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:09.705] Applying learner 'regr.ranger' on task 'df.tr' (iter 30/50) 
    DEBUG [09:10:09.713] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:10:09.716] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:09.725] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:09.727] Creating Prediction for predict set 'test' 
    DEBUG [09:10:09.729] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 14, 22, 29, 34, 35, 37, 40, 43, 44, 59, 62, 64, 68, 72]}
    DEBUG [09:10:09.732] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:09.741] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:09.742] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:09.743] Applying learner 'regr.ranger' on task 'df.tr' (iter 16/50) 
    DEBUG [09:10:09.751] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75..]}
    DEBUG [09:10:09.754] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:09.763] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:09.764] Creating Prediction for predict set 'test' 
    DEBUG [09:10:09.767] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72]}
    DEBUG [09:10:09.770] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:09.778] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:09.779] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:09.781] Applying learner 'regr.ranger' on task 'df.tr' (iter 49/50) 
    DEBUG [09:10:09.789] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:10:09.792] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:09.801] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:09.802] Creating Prediction for predict set 'test' 
    DEBUG [09:10:09.805] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 7, 14, 16, 25, 26, 27, 40, 44, 56, 67, 69, 75, 76]}
    DEBUG [09:10:09.807] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:09.816] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:09.817] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:09.818] Applying learner 'regr.ranger' on task 'df.tr' (iter 38/50) 
    DEBUG [09:10:09.827] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:10:09.829] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:09.838] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:09.840] Creating Prediction for predict set 'test' 
    DEBUG [09:10:09.842] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58, 69]}
    DEBUG [09:10:09.845] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:09.853] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:09.855] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:09.855] Applying learner 'regr.ranger' on task 'df.tr' (iter 32/50) 
    DEBUG [09:10:09.864] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 6..]}
    DEBUG [09:10:09.867] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:09.876] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:09.877] Creating Prediction for predict set 'test' 
    DEBUG [09:10:09.880] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71]}
    DEBUG [09:10:09.882] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:09.891] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:09.892] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:09.893] Applying learner 'regr.ranger' on task 'df.tr' (iter 31/50) 
    DEBUG [09:10:09.902] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66, 67, 71, 1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60..]}
    DEBUG [09:10:09.905] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:09.913] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:09.915] Creating Prediction for predict set 'test' 
    DEBUG [09:10:09.917] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73]}
    DEBUG [09:10:09.920] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:09.929] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:09.931] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:09.932] Applying learner 'regr.ranger' on task 'df.tr' (iter 37/50) 
    DEBUG [09:10:09.939] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57..]}
    DEBUG [09:10:09.942] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:09.952] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:09.953] Creating Prediction for predict set 'test' 
    DEBUG [09:10:09.955] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76]}
    DEBUG [09:10:09.958] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:09.967] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:09.968] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:09.969] Applying learner 'regr.ranger' on task 'df.tr' (iter 2/50) 
    DEBUG [09:10:09.977] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [09:10:09.979] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:09.988] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:09.990] Creating Prediction for predict set 'test' 
    DEBUG [09:10:09.992] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68]}
    DEBUG [09:10:09.995] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.004] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:10.005] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:10.006] Applying learner 'regr.ranger' on task 'df.tr' (iter 22/50) 
    DEBUG [09:10:10.014] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57..]}
    DEBUG [09:10:10.017] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.026] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:10.027] Creating Prediction for predict set 'test' 
    DEBUG [09:10:10.029] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75]}
    DEBUG [09:10:10.032] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.040] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:10.042] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:10.043] Applying learner 'regr.ranger' on task 'df.tr' (iter 17/50) 
    DEBUG [09:10:10.050] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 7..]}
    DEBUG [09:10:10.053] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.062] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:10.063] Creating Prediction for predict set 'test' 
    DEBUG [09:10:10.066] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57, 59, 70]}
    DEBUG [09:10:10.068] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.077] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:10.078] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:10.079] Applying learner 'regr.ranger' on task 'df.tr' (iter 5/50) 
    DEBUG [09:10:10.087] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:10:10.090] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.099] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:10.100] Creating Prediction for predict set 'test' 
    DEBUG [09:10:10.103] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 11, 14, 17, 27, 32, 41, 44, 45, 55, 60, 64, 65, 66, 69]}
    DEBUG [09:10:10.106] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.114] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:10.116] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:10.116] Applying learner 'regr.ranger' on task 'df.tr' (iter 15/50) 
    DEBUG [09:10:10.124] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:10:10.127] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.136] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:10.137] Creating Prediction for predict set 'test' 
    DEBUG [09:10:10.140] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 17, 19, 20, 30, 35, 42, 47, 54, 59, 64, 67, 71, 74, 75]}
    DEBUG [09:10:10.143] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.151] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:10.153] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:10.154] Applying learner 'regr.ranger' on task 'df.tr' (iter 44/50) 
    DEBUG [09:10:10.163] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:10:10.172] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.184] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:10.185] Creating Prediction for predict set 'test' 
    DEBUG [09:10:10.189] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 14, 17, 19, 24, 36, 41, 47, 51, 53, 56, 62, 63, 67, 71]}
    DEBUG [09:10:10.193] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.204] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:10.206] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:10.207] Applying learner 'regr.ranger' on task 'df.tr' (iter 35/50) 
    DEBUG [09:10:10.218] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:10:10.284] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.302] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:10.305] Creating Prediction for predict set 'test' 
    DEBUG [09:10:10.311] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 29, 32, 33, 35, 40, 43, 44, 45, 61, 62, 70, 74]}
    DEBUG [09:10:10.317] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.332] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:10.334] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:10.336] Applying learner 'regr.ranger' on task 'df.tr' (iter 27/50) 
    DEBUG [09:10:10.350] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67..]}
    DEBUG [09:10:10.356] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.370] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:10.372] Creating Prediction for predict set 'test' 
    DEBUG [09:10:10.377] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76]}
    DEBUG [09:10:10.382] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.394] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:10.396] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:10.398] Applying learner 'regr.ranger' on task 'df.tr' (iter 40/50) 
    DEBUG [09:10:10.411] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:10:10.416] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.428] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:10.430] Creating Prediction for predict set 'test' 
    DEBUG [09:10:10.434] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 8, 16, 17, 24, 25, 27, 29, 41, 44, 53, 54, 67, 72]}
    DEBUG [09:10:10.438] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.451] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:10.453] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:10.455] Applying learner 'regr.ranger' on task 'df.tr' (iter 43/50) 
    DEBUG [09:10:10.467] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:10:10.471] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.483] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:10.485] Creating Prediction for predict set 'test' 
    DEBUG [09:10:10.489] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74]}
    DEBUG [09:10:10.494] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.506] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:10.508] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:10.509] Applying learner 'regr.ranger' on task 'df.tr' (iter 50/50) 
    DEBUG [09:10:10.521] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:10:10.526] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.538] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:10.540] Creating Prediction for predict set 'test' 
    DEBUG [09:10:10.544] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 8, 21, 22, 30, 36, 41, 45, 49, 53, 54, 57, 68, 71]}
    DEBUG [09:10:10.547] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.560] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:10.562] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:10.563] Applying learner 'regr.ranger' on task 'df.tr' (iter 42/50) 
    DEBUG [09:10:10.574] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70..]}
    DEBUG [09:10:10.578] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.590] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:10.592] Creating Prediction for predict set 'test' 
    DEBUG [09:10:10.596] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66]}
    DEBUG [09:10:10.600] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.612] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:10.614] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:10.615] Applying learner 'regr.ranger' on task 'df.tr' (iter 34/50) 
    DEBUG [09:10:10.625] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:10:10.629] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.638] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:10.640] Creating Prediction for predict set 'test' 
    DEBUG [09:10:10.643] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 7, 12, 14, 18, 20, 41, 46, 47, 53, 57, 63, 64, 72, 75]}
    DEBUG [09:10:10.646] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.656] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:10.657] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:10.658] Applying learner 'regr.ranger' on task 'df.tr' (iter 3/50) 
    DEBUG [09:10:10.668] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:10:10.671] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.681] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:10.683] Creating Prediction for predict set 'test' 
    DEBUG [09:10:10.685] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75, 76]}
    DEBUG [09:10:10.689] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.699] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:10.700] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:10.701] Applying learner 'regr.ranger' on task 'df.tr' (iter 41/50) 
    DEBUG [09:10:10.710] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 65, 66, 3, 7, 10, 11, 15, 29, 31, 43, 54, 59, 68, 69, 70, 72, 74..]}
    DEBUG [09:10:10.714] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.723] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:10.725] Creating Prediction for predict set 'test' 
    DEBUG [09:10:10.728] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76]}
    DEBUG [09:10:10.731] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.741] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:10.742] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:10.743] Applying learner 'regr.ranger' on task 'df.tr' (iter 45/50) 
    DEBUG [09:10:10.752] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 18, 23, 27, 30, 32, 34, 35, 39, 42, 45, 52, 58, 61, 73, 76, 5, 6, 9, 13, 16, 20, 26, 40, 49, 50, 55, 60, 64, 6..]}
    DEBUG [09:10:10.755] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.766] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:10.767] Creating Prediction for predict set 'test' 
    DEBUG [09:10:10.770] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 8, 12, 21, 22, 25, 28, 33, 37, 38, 44, 46, 48, 57, 75]}
    DEBUG [09:10:10.773] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.783] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:10.785] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:10.786] Applying learner 'regr.ranger' on task 'df.tr' (iter 26/50) 
    DEBUG [09:10:10.796] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69, 70, 76, 4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75..]}
    DEBUG [09:10:10.799] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.809] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:10.811] Creating Prediction for predict set 'test' 
    DEBUG [09:10:10.813] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74]}
    DEBUG [09:10:10.817] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.827] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:10.828] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:10.829] Applying learner 'regr.ranger' on task 'df.tr' (iter 7/50) 
    DEBUG [09:10:10.838] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 6..]}
    DEBUG [09:10:10.842] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.853] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:10.854] Creating Prediction for predict set 'test' 
    DEBUG [09:10:10.857] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74]}
    DEBUG [09:10:10.860] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.871] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:10.872] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:10.873] Applying learner 'regr.ranger' on task 'df.tr' (iter 29/50) 
    DEBUG [09:10:10.883] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:10:10.886] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.896] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:10.898] Creating Prediction for predict set 'test' 
    DEBUG [09:10:10.901] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 18, 25, 28, 30, 39, 45, 48, 51, 54, 55, 61, 66, 71, 73]}
    DEBUG [09:10:10.904] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.914] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:10.915] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:10.917] Applying learner 'regr.ranger' on task 'df.tr' (iter 47/50) 
    DEBUG [09:10:10.926] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64..]}
    DEBUG [09:10:10.930] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.941] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:10.942] Creating Prediction for predict set 'test' 
    DEBUG [09:10:10.945] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70]}
    DEBUG [09:10:10.948] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.959] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:10.960] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:10.961] Applying learner 'regr.ranger' on task 'df.tr' (iter 8/50) 
    DEBUG [09:10:10.971] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:10:10.974] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:10.984] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:10.986] Creating Prediction for predict set 'test' 
    DEBUG [09:10:10.989] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 75]}
    DEBUG [09:10:10.992] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.001] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:11.003] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:11.004] Applying learner 'regr.ranger' on task 'df.tr' (iter 18/50) 
    DEBUG [09:10:11.013] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:10:11.016] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.026] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:11.028] Creating Prediction for predict set 'test' 
    DEBUG [09:10:11.031] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 4, 9, 12, 15, 31, 35, 44, 50, 63, 66, 73, 74, 75]}
    DEBUG [09:10:11.034] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.044] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:11.045] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:11.047] Applying learner 'regr.ranger' on task 'df.tr' (iter 19/50) 
    DEBUG [09:10:11.056] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:10:11.059] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.071] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:11.072] Creating Prediction for predict set 'test' 
    DEBUG [09:10:11.075] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 8, 11, 14, 18, 24, 28, 33, 34, 54, 56, 62, 64, 68, 76]}
    DEBUG [09:10:11.078] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.088] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:11.090] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:11.091] Applying learner 'regr.ranger' on task 'df.tr' (iter 9/50) 
    DEBUG [09:10:11.105] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:10:11.110] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.123] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:11.125] Creating Prediction for predict set 'test' 
    DEBUG [09:10:11.128] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 23, 27, 40, 42, 46, 49, 56, 58, 61, 64, 67, 70, 71, 72]}
    DEBUG [09:10:11.132] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.142] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:11.144] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:11.145] Applying learner 'regr.ranger' on task 'df.tr' (iter 14/50) 
    DEBUG [09:10:11.155] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:10:11.158] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.169] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:11.170] Creating Prediction for predict set 'test' 
    DEBUG [09:10:11.173] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 8, 22, 27, 36, 38, 40, 45, 46, 50, 55, 56, 60, 65, 73]}
    DEBUG [09:10:11.177] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.187] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:11.188] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:11.190] Applying learner 'regr.ranger' on task 'df.tr' (iter 28/50) 
    DEBUG [09:10:11.199] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 11, 13, 24, 31, 32, 33, 42, 47, 52, 53, 57, 58, 65, 74, 2, 12, 17, 19, 23, 26, 36, 46, 49, 56, 60, 63, 69..]}
    DEBUG [09:10:11.203] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.213] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:11.214] Creating Prediction for predict set 'test' 
    DEBUG [09:10:11.217] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 5, 6, 7, 10, 15, 16, 20, 21, 27, 38, 41, 50, 67, 75]}
    DEBUG [09:10:11.221] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.230] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:11.232] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:11.233] Applying learner 'regr.ranger' on task 'df.tr' (iter 36/50) 
    DEBUG [09:10:11.242] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71, 74, 76, 6, 10, 11, 20, 22, 23, 26, 30, 31, 32, 47, 56, 57, 58..]}
    DEBUG [09:10:11.246] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.256] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:11.257] Creating Prediction for predict set 'test' 
    DEBUG [09:10:11.260] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75]}
    DEBUG [09:10:11.263] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.273] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:11.275] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:11.276] Applying learner 'regr.ranger' on task 'df.tr' (iter 13/50) 
    DEBUG [09:10:11.285] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69..]}
    DEBUG [09:10:11.288] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.298] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:11.300] Creating Prediction for predict set 'test' 
    DEBUG [09:10:11.303] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61, 72]}
    DEBUG [09:10:11.306] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.316] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:11.317] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:11.318] Applying learner 'regr.ranger' on task 'df.tr' (iter 24/50) 
    DEBUG [09:10:11.328] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:10:11.331] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.341] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:11.342] Creating Prediction for predict set 'test' 
    DEBUG [09:10:11.345] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [7, 13, 17, 25, 32, 35, 36, 37, 38, 50, 63, 65, 68, 69, 74]}
    DEBUG [09:10:11.349] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.359] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:11.361] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:11.362] Applying learner 'regr.ranger' on task 'df.tr' (iter 21/50) 
    DEBUG [09:10:11.371] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70, 75, 1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66..]}
    DEBUG [09:10:11.375] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.385] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:11.387] Creating Prediction for predict set 'test' 
    DEBUG [09:10:11.390] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72]}
    DEBUG [09:10:11.393] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.403] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:11.405] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:11.406] Applying learner 'regr.ranger' on task 'df.tr' (iter 10/50) 
    DEBUG [09:10:11.415] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76, 6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62..]}
    DEBUG [09:10:11.419] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.429] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:11.431] Creating Prediction for predict set 'test' 
    DEBUG [09:10:11.434] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 17, 18, 19, 24, 26, 29, 31, 34, 36, 43, 45, 53, 66, 73]}
    DEBUG [09:10:11.437] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.448] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:11.449] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:11.450] Applying learner 'regr.ranger' on task 'df.tr' (iter 4/50) 
    DEBUG [09:10:11.459] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73, 16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 6..]}
    DEBUG [09:10:11.463] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.473] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:11.474] Creating Prediction for predict set 'test' 
    DEBUG [09:10:11.477] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 3, 13, 20, 23, 25, 40, 46, 50, 53, 56, 57, 58, 63]}
    DEBUG [09:10:11.480] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.490] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:11.492] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:11.493] Applying learner 'regr.ranger' on task 'df.tr' (iter 33/50) 
    DEBUG [09:10:11.502] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [8, 10, 13, 15, 23, 25, 26, 28, 36, 48, 50, 51, 65, 68, 69, 73, 16, 17, 21, 22, 27, 31, 34, 38, 39, 52, 55, 59, 66..]}
    DEBUG [09:10:11.505] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.515] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:11.517] Creating Prediction for predict set 'test' 
    DEBUG [09:10:11.520] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 6, 9, 11, 19, 24, 30, 37, 42, 49, 54, 56, 58, 60, 76]}
    DEBUG [09:10:11.523] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.533] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:11.534] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:11.535] Applying learner 'regr.ranger' on task 'df.tr' (iter 39/50) 
    DEBUG [09:10:11.545] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 9, 12, 13, 34, 35, 40, 42, 43, 50, 55, 61, 62, 66, 73, 75, 4, 15, 33, 36, 37, 39, 45, 46, 60, 63, 68, 70, 71..]}
    DEBUG [09:10:11.548] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.558] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:11.559] Creating Prediction for predict set 'test' 
    DEBUG [09:10:11.562] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 5, 14, 18, 19, 21, 28, 38, 48, 49, 51, 52, 59, 64, 65]}
    DEBUG [09:10:11.565] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.575] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:11.577] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:11.578] Applying learner 'regr.ranger' on task 'df.tr' (iter 12/50) 
    DEBUG [09:10:11.587] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 3, 6, 12, 15, 18, 24, 33, 37, 39, 49, 53, 58, 62, 68, 76, 2, 7, 10, 14, 23, 25, 28, 31, 32, 34, 41, 48, 57, 61..]}
    DEBUG [09:10:11.591] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.600] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:11.602] Creating Prediction for predict set 'test' 
    DEBUG [09:10:11.605] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 9, 11, 13, 21, 26, 29, 43, 44, 51, 52, 63, 66, 69, 70]}
    DEBUG [09:10:11.608] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.618] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:11.619] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:11.620] Applying learner 'regr.ranger' on task 'df.tr' (iter 48/50) 
    DEBUG [09:10:11.630] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73, 6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61..]}
    DEBUG [09:10:11.633] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.643] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:11.645] Creating Prediction for predict set 'test' 
    DEBUG [09:10:11.648] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72, 74]}
    DEBUG [09:10:11.651] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.711] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:11.713] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:11.716] Applying learner 'regr.ranger' on task 'df.tr' (iter 1/50) 
    DEBUG [09:10:11.733] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [16, 18, 22, 24, 26, 28, 33, 38, 48, 49, 51, 61, 62, 67, 68, 5, 8, 19, 29, 34, 35, 39, 42, 43, 47, 54, 70, 74, 75..]}
    DEBUG [09:10:11.743] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.757] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:11.760] Creating Prediction for predict set 'test' 
    DEBUG [09:10:11.765] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 7, 9, 12, 15, 21, 30, 31, 36, 37, 52, 59, 71, 72, 73]}
    DEBUG [09:10:11.771] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.793] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:11.795] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:11.796] Applying learner 'regr.ranger' on task 'df.tr' (iter 20/50) 
    DEBUG [09:10:11.809] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [10, 13, 19, 20, 21, 22, 23, 30, 39, 41, 48, 58, 60, 67, 71, 72, 7, 25, 26, 27, 29, 38, 40, 47, 49, 51, 52, 55, 57..]}
    DEBUG [09:10:11.813] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.829] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:11.831] Creating Prediction for predict set 'test' 
    DEBUG [09:10:11.837] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [5, 6, 16, 17, 32, 36, 37, 42, 43, 45, 46, 53, 61, 65, 69]}
    DEBUG [09:10:11.841] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.854] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:11.857] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:11.859] Applying learner 'regr.ranger' on task 'df.tr' (iter 6/50) 
    DEBUG [09:10:11.871] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 7, 9, 14, 15, 16, 22, 33, 39, 44, 47, 55, 59, 62, 74, 3, 13, 21, 28, 32, 37, 38, 48, 50, 51, 57, 60, 63, 68, 7..]}
    DEBUG [09:10:11.875] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.887] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:11.889] Creating Prediction for predict set 'test' 
    DEBUG [09:10:11.894] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 2, 5, 10, 11, 12, 20, 25, 30, 35, 41, 52, 54, 65, 69, 76]}
    DEBUG [09:10:11.898] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.910] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:11.912] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:11.914] Applying learner 'regr.ranger' on task 'df.tr' (iter 46/50) 
    DEBUG [09:10:11.927] Subsetting task 'df.tr' to 60 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [6, 10, 12, 20, 32, 34, 35, 46, 47, 48, 51, 52, 61, 63, 70, 3, 9, 15, 18, 23, 31, 33, 37, 38, 55, 60, 62, 64, 72..]}
    DEBUG [09:10:11.931] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 60 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.943] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:11.945] Creating Prediction for predict set 'test' 
    DEBUG [09:10:11.949] Subsetting task 'df.tr' to 16 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [11, 13, 17, 19, 24, 28, 29, 39, 42, 43, 50, 58, 59, 65, 66, 73]}
    DEBUG [09:10:11.953] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 16 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.966] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:11.967] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:11.969] Applying learner 'regr.ranger' on task 'df.tr' (iter 23/50) 
    DEBUG [09:10:11.980] Subsetting task 'df.tr' to 61 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [4, 6, 10, 22, 24, 26, 33, 46, 48, 49, 56, 59, 61, 64, 71, 72, 2, 3, 5, 8, 12, 14, 16, 19, 30, 53, 54, 58, 60, 70..]}
    DEBUG [09:10:11.985] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 61 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:11.996] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:11.998] Creating Prediction for predict set 'test' 
    DEBUG [09:10:12.002] Subsetting task 'df.tr' to 15 rows {task: <TaskRegr/TaskSupervised/Task/R6>, row_ids: [1, 15, 20, 28, 31, 34, 39, 40, 41, 42, 47, 52, 57, 62, 66]}
    DEBUG [09:10:12.006] Calling predict method of Learner 'regr.ranger' on task 'df.tr' with 15 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:12.018] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }
    DEBUG [09:10:12.020] Erasing stored model for learner 'regr.ranger' 
    INFO  [09:10:12.072] Finished benchmark 
    INFO  [09:10:13.864] Result of batch 10: 
    INFO  [09:10:13.867]  mtry sample.fraction num.trees importance regr.rmse      resample_result 
    INFO  [09:10:13.867]     3       0.7453881        55   impurity  46.77007 <ResampleResult[19]> 
    INFO  [09:10:13.880] Finished optimizing after 10 evaluation(s) 
    INFO  [09:10:13.881] Result: 
    INFO  [09:10:13.882]  mtry sample.fraction num.trees importance learner_param_vals  x_domain 
    INFO  [09:10:13.882]     5       0.7631363       141   impurity          <list[4]> <list[4]> 
    INFO  [09:10:13.882]  regr.rmse 
    INFO  [09:10:13.882]   43.04499 
    DEBUG [09:10:13.910] Skip subsetting of task 'df.tr' 
    DEBUG [09:10:13.912] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 76 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:13.922] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }
    DEBUG [09:10:13.929] Learner 'regr.ranger.tuned' on task 'df.tr' succeeded to fit a model {learner: <AutoTuner/Learner/R6>, result: <list>, messages: }
    DEBUG [09:10:13.937] Skip subsetting of task 'df.tr' 
    DEBUG [09:10:13.939] Calling train method of Learner 'regr.ranger' on task 'df.tr' with 76 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:13.949] Learner 'regr.ranger' on task 'df.tr' succeeded to fit a model {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, result: <ranger>, messages: }

    train.model

    function (newdata, task = NULL) 
    .__Learner__predict_newdata(self = self, private = private, super = super, 
        newdata = newdata, task = task)
    <environment: 0x55a8b7b00950>

-   User only **must** define a `df` and the `target.variable` and
    `train.spm()` will automatically perform `classification` or
    `regression` tasks.
-   The rest of arguments can be set or default values will be set.
-   If **crs** is set `train.spm()` will automatically take care of
    **spatial cross validation**.

`predict.spm()`

    predict.variable = predict.spm(df.ts, task = NULL)

    DEBUG [09:10:14.087] Skip subsetting of task 'df.ts' 
    DEBUG [09:10:14.090] Calling predict method of Learner 'regr.ranger' on task 'df.ts' with 76 observations {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>}
    DEBUG [09:10:14.107] Learner 'regr.ranger' returned an object of class 'PredictionDataRegr' {learner: <LearnerRegrRanger/LearnerRegr/Learner/R6>, prediction: <PredictionDataRegr/PredictionData>, messages: }

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
