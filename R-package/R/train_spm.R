#' Train a spatial prediction model, from a (spatial) matrix, using ensemble machine learning,
#' @description
#' This is a function to train (spatial) dataframe   \href{https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0169748}{using Ensemble Machine Learning} and \href{https://mlr3.mlr-org.com/}{mlr3} ecosystem,
#' @param df.tr Observation data frame,
#' @param target.variable Target variable or response variable,
#' @param parallel Parallel processing mode,
#' @param predict_type Prediction type 'prob' or 'response',
#' @param folds Sub-item for spcv (see mlr3spatiotempcv),
#' @param n_evals Number of evaluation process,
#' @param method.list Learning methods (see mlr3learners),
#' @param var.imp Variable importance,
#' @param super.learner Ensemble stacking model usually \code{regr.ranger} for regression tasks,
#' @param crs Coordinate reference system, necessary for spcv (see mlr3spatiotempcv),
#' @param coordinate_names Columns names for X and Y coordinates,
#' @param ... other arguments that can be passed on to \code{TaskSupervised},
#'
#' @return List of objects of class \code{mlr3},
#' @export 
#'@author  \href{https://opengeohub.org/people/mohammadreza-sheykhmousa}{Mohammadreza Sheykhmousa} and  \href{https://opengeohub.org/people/tom-hengl}{Tom Hengl}
#' @examples
#' ## Meuse Demo
#' library(sp)
#' library(mlr3verse)
#' library(mlr3spatiotempcv)
#' library(checkmate)
#' library(future)
#' library(progress)
#' library(scales)
#' library(eumap)
#' demo(meuse, echo=FALSE)
#' df <- as.data.frame(meuse)
#' df.grid <- as.data.frame(meuse.grid)
#' df = na.omit(df[,])
#' df.grid = na.omit(df.grid[,])
#' smp_size <- floor(0.8 * nrow(df))
#' set.seed(123)
#' train_ind <- sample(seq_len(nrow(df)), size = smp_size)
#' df.tr <- df[, c("x","y","dist","ffreq","soil","lead")]
#' df.ts <- df.grid[, c("x","y","dist","ffreq","soil")]
#' newdata <-df.ts
#' tr = eumap::train_spm(df.tr, target.variable = "lead",crs )
#' train_model= tr[[1]]
#' #var.imp = tr[[2]]
#' summary = tr[[3]]
#' response = tr[[4]]
#' vlp = tr[[5]]
#' target = tr[[6]]
#' predict.variable = eumap::predict_spm(train_model, newdata)
#' pred.v = predict.variable[[1]]
#' valu.imp= predict.variable[[2]]
#' plt = eumap::plot_spm(df, gmode  = "norm" , gtype = "var.imp")
#' df.ts$leadp = predict.variable
#' coordinates(df.ts) <- ~x+y
#' proj4string(df.ts) <- CRS("+init=epsg:28992")
#' gridded(df.ts) = TRUE
#' ## regression grid 
#' #make a spatial prediction map 
#' plot(df.ts[,"leadp"])
#' points(meuse, pch="+")
train_spm = function(df.tr, target.variable, parallel = TRUE, predict_type = NULL, folds = NULL, n_evals = NULL, method.list = NULL, var.imp = NULL, super.learner = NULL, crs = NULL, coordinate_names = c("x","y"), ...){
  target = target.variable
  if ( is.null(predict_type)) {
    predict_type <- "response"
  }
  id = deparse(substitute(df.tr))
  cv = rsmp("repeated_cv", folds = folds)
  task_type = c("classification Task", "Regression Task")
  ml_method = c("kknn", "featureless")
  meta_learner = "ranger"
  resample_method = c("Using resampling method: (non-spatial) repeated_cv...", "Using resampling method: (spatial)repeated_cv by cooridinates...")
  number_cores = paste0(" ncores: ", availableCores())
  message(paste0("Fitting an ensemble ML using ", ml_method[1], " ", ml_method[2], ", and ", meta_learner," models", number_cores), immediate. = TRUE)
  
  ## start running ensemble
  
  ##  classif CV ----
  if (is.factor(df.tr[,target]) & is.null(crs)) {
    
    message(paste0(task_type[1],"..."), immediate. = TRUE)
    
    tsk_clf <- TaskClassif$new(
    id = id, backend = df.tr, target = target.variable
    )
    
    ranger_lrn = lrn("classif.ranger", predict_type = "response", importance = "permutation")
    ps_ranger = 
      ParamSet$new(
      list(ParamInt$new("mtry", lower = 1L, upper = 5L),
      ParamDbl$new("sample.fraction", lower = 0.5, upper = 1),
      ParamInt$new("num.trees", lower = 50L, upper = 500L),
      ParamFct$new("importance", "permutation"))
      )
    at = AutoTuner$new(
      learner = ranger_lrn,
      resampling = cv,
      measure =  msr("classif.acc"),
      search_space = ps_ranger,
      terminator = trm("evals", n_evals = n_evals), 
      tuner = tnr("random_search")
      )
      message(resample_method[1], immediate. = TRUE)
      # at$train(tsk_clf)
      at$
        train(tsk_clf)$
        predict(tsk_clf)$
        score(msr('classif.acc'))
      at$learner$train(tsk_clf)
      best.model = at$archive$best()
      var.imp = at$learner$importance()
      vlp = names(var.imp[1:(round(length(var.imp)*0.1) + 1)])
      summary = at$learner$state$model
      tr.model = at$learner
      train_model = tr.model$predict_newdata
      response = tr.model$model$predictions
  }
  ## regr CV ----
  else if (is.numeric(df.tr[,target.variable]) & is.null(crs)) {
      
    message( 
    paste0(task_type[2],"...", immediate. = TRUE)
    )   
    
    tsk_regr <- TaskRegr$new(id = id, backend = df.tr, target = target.variable)
    ranger_lrn = lrn("regr.ranger", predict_type = "response",importance = "permutation")
    ps_ranger = ParamSet$new(
      list(
        ParamInt$new("mtry", lower = 1L, upper = 5L),
        ParamDbl$new("sample.fraction", lower = 0.5, upper = 1),
        ParamInt$new("num.trees", lower = 50L, upper = 500L),
        ParamFct$new("importance", "impurity")
        )
      )
    at = AutoTuner$new(
      learner = ranger_lrn,
      resampling = cv,
      measure = msr("regr.rmse"),
      search_space = ps_ranger,
      terminator =  trm("evals", n_evals = n_evals), 
      tuner =  tnr("random_search")
      )
    #at$store_tuning_instance = TRUE
    message(resample_method[1], immediate. = TRUE)
    # at$train(tsk_regr)
    at$
      train(tsk_regr)$
      predict(tsk_regr)$
      score(msr('regr.rmse'))
    at$learner$train(tsk_regr)
    tr.model = at$learner
    summary = tr.model$model
    var.imp = tr.model$importance()
    vlp = names(var.imp[1:(round(length(var.imp)*0.1) + 1)])
    train_model = tr.model$predict_newdata
    response = tr.model$model$predictions
  }
  ## classif spcv ----
  else if (is.factor(df.tr[,target.variable]) & crs == crs) { 
    
    message( 
      paste0(task_type[1],"...", immediate. = TRUE)
    ) 
    if (is.null(method.list) & is.null(super.learner)) {
    method.list <- c("classif.kknn", "classif.featureless", "classif.rpart")
    super.learner = "classif.ranger"
    }
    # task$set_col_role('tile_id', 'group') later for whole eu should be used
    # task$set_col_role('confidence', 'weight')
    df.trf = as_data_backend(df.tr)
    tsk_clf = TaskClassifST$new(id = id, backend = df.trf, target = target.variable,
    extra_args = list( positive = "TRUE", coordinate_names = coordinate_names,
    coords_as_features = FALSE, crs = crs))
    
    pre = po("encode") %>>% po("imputemode") %>>% po("removeconstants")
    g = pre %>>% 
     gunion(
        list(
          po("select") %>>% po("learner_cv", id = "kknn", lrn("classif.kknn")),
          po("pca") %>>% po("learner_cv", id = "featureless", lrn("classif.featureless")),
          po("subsample") %>>% po("learner_cv", id = "rpart", lrn("classif.rpart"))
          )
        ) %>>%
      po("featureunion") %>>%
      po("learner", lrn("classif.ranger",importance = "permutation")) 
      resampling_sp = rsmp("repeated_spcv_coords", folds = folds, repeats = 4)
      rr_sp = rsmp(
      task = tsk_regr, learner = g,
      resampling = resampling_sp
      )
      g$keep_results = "TRUE"
      # plt = g$plot()
      message(resample_method[2], immediate. = TRUE)
      g$
        train(tsk_clf)$
        predict(tsk_clf)$
        score(msr('classif.acc'))
      conf.mat = g$pipeops$classif.ranger$learner_model$model$confusion.matrix
      var.imp = g$pipeops$classif.ranger$learner_model$model$variable.importance
      vlp = names(var.imp[1:(round(length(var.imp)*0.1) + 1)])
      summary = g$pipeops$classif.ranger$learner_model$model
      tr.model = g$pipeops$classif.ranger$learner$train(tsk_clf)
      train_model = tr.model$predict_newdata
      response = tr.model$model$predictions
  }
  ## regr spcv ----
  else if (is.numeric(df.tr[,target.variable]) & crs == crs) {
    
    message( 
    paste0(task_type[2],"...", immediate. = TRUE)
    ) 
    if (is.null(method.list) & is.null(super.learner)) {
      method.list <- c("regr.kknn", "regr.featureless", "regr.rpart")
      super.learner <- "regr.ranger"
    }
    
    df.trf = as_data_backend(df.tr)
    
    tsk_regr = TaskRegrST$new(
    id = id, backend = df.trf, target = target.variable,
    extra_args = list(
      positive = "TRUE", coordinate_names = coordinate_names,
      coords_as_features = FALSE, crs = crs
      )
    )
    
    # pre =  po("encode") %>>% po("imputemode") %>>% po("removeconstants")
    g = #pre %>>% 
       gunion(
        list(
          po("select") %>>% po("learner_cv", id = "knn", lrn("regr.kknn")),
          po("pca") %>>% po("learner_cv", id = "featureless", lrn("regr.featureless")),
          po("scale") %>>% po("learner_cv", id = "rpart", lrn("regr.rpart"))
          
          )
        ) %>>%
      po("featureunion") %>>%
      po("learner", lrn("regr.ranger",importance = "permutation"))
    
    resampling_sp = rsmp("repeated_spcv_coords", folds = folds, repeats = 3)
    rr_sp = rsmp(
      task = tsk_regr, learner = g,
      resampling = resampling_sp
      )
    g$keep_results = "TRUE"
    plt = g$plot()
    message(resample_method[2], immediate. = TRUE)
    g$
      train(tsk_regr)$
      predict(tsk_regr)$
      score(msr('regr.rmse'))
    summary = g$pipeops$regr.ranger$learner_model$model
    tr.model = g$pipeops$regr.ranger$learner$train(tsk_regr)
    var.imp = tr.model$importance()
    vlp = names(var.imp[1:(round(length(var.imp)*0.1) + 1)])
    response = tr.model$model$predictions
    train_model = tr.model$predict_newdata
    response = tr.model$model$predictions
    }
  return(list(train_model, var.imp, summary, response, vlp, target))
  }
