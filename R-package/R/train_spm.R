#' Train Spatial Learner
#' @description
#' This is the abstract base class for training objects like [TaskClassif] and [TaskRegr].
#' For example, for a classification task columns must be marked as ID, df, target column.
#' train.spm  it multiple models/learners depending on the
#'class() of the target.variable and for now only returns a
#'trained model function so later on we could use it to train a new dataset.
#' @param df.tr observation data
#' @param target.variable response variable 
#' @param parallel parallel processing mode
#' @param predict_type e.g., response and prob
#' @param folds sub-item for cv
#' @param method.list learning methods
#' @param n_evals number of evaluation process
#' @param plot.workflow graph work flow
#' @param var.imp variable importance 
#' @param meta.learner super learner (mostly RF)
#' @param crs coordinate reference system for spcv
#' @return train.model
#' @return summary
#' @return var.imp
#' @author  \href{https://opengeohub.org/people/mohammadreza-sheykhmousa}{Mohammadreza Sheykhmousa} and  \href{https://opengeohub.org/people/tom-hengl}{Tom Hengl}
#' @export 
#' @examples
#' ## Meuse Demo
#' library(sp)
#' library(mlr3verse)
#' demo(meuse, echo=FALSE)
#' pr.vars = c("x","y","dist","elev","soil","lead")
#' df <- as.data.frame(meuse)
#' # df <- df[complete.cases(df[,pr.vars]),pr.vars]
#' df = na.omit(df[,])
#' summary(is.na(df))
#' crs = "+init=epsg:3035"
#' #target.variable = "landuse"
#' target.variable = "lead"
#' # define generic var ----
#' smp_size <- floor(0.5 * nrow(df))
#' set.seed(123)
#' train_ind <- sample(seq_len(nrow(df)), size = smp_size)
#' df.tr <- df[train_ind, c("x","y","dist","elev","soil","lead")]
#' df.ts <- df[-train_ind, c("x","y","dist","elev","soil")]
#' folds = 2
#' n_evals = 3
#' newdata = df.ts
#' tr = train_spm(df.tr, target.variable = target.variable, folds = folds ,n_evals = n_evals, plot.workflow = TRUE, coords, crs, var.imp)
#' train.model= tr[[1]]
#' var.imp = tr[[2]]
#' var.imp
#' summary = tr[[3]]
#' summary
#' 
train_spm = function(df.tr, target.variable, 
parallel = TRUE, predict_type = NULL, folds = folds, method.list = NULL,  n_evals = n_evals, plot.workflow = FALSE, var.imp = TRUE, meta.learner = NULL, crs = NULL,  coordinate_names = c("x","y")){
  id = deparse(substitute(df.tr))
  cv3 = rsmp("repeated_cv", folds = folds)
   if(is.factor(df.tr[,target.variable]) & is.null(crs)){
    message(paste("classification Task  ","resampling method: non-spatialCV ", "ncores: ",availableCores(), "..."), immediate. = TRUE)
        message(paste0("Using learners: ", paste("method.list", collapse = ", "), "..."), immediate. = TRUE)
        tsk_clf <- mlr3::TaskClassif$new(id = id, backend = df.tr, target = target.variable)
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
        at$train(tsk_clf)
        at$learner$train(tsk_clf)
        best.model = at$archive$best()
        var.imp = at$learner$importance()
        summary = at$learner$state$model
        tr.model = at$learner
        train.model = tr.model$predict_newdata
        response = tr.model$model$predictions
      } else if (is.numeric(df.tr[,target.variable]) & is.null(crs)) {
        if( missing(predict_type)){
          predict_type <- "response" }
      message(paste("Regr Task  ","resampling method: non-spatialCV ", "ncores: ",availableCores(), "..."), immediate. = TRUE)
      message(paste0("Using learners: ", paste("method.list", collapse = ", "), "..."), immediate. = TRUE)
      tsk_regr <- mlr3::TaskRegr$new(id = id, backend = df.tr, target = target.variable)
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
      at$train(tsk_regr)
      at$learner$train(tsk_regr)
      tr.model = at$learner
      summary = tr.model$model
      var.imp = tr.model$importance()
      train.model = tr.model$predict_newdata
      response = tr.model$model$predictions
    } else if (is.factor(df.tr[,target.variable]) & crs == crs){ 
      if(is.null(method.list) & is.null(meta.learner)){
        method.list <- c("classif.kknn", "classif.featureless")
        meta.learner = "classif.ranger"}
        df.trf = mlr3::as_data_backend(df.tr)
        tsk_clf = TaskClassifST$new(id = id, backend = df.trf, target = target.variable, extra_args = list( positive = "TRUE", coordinate_names = coordinate_names, coords_as_features = FALSE, crs = crs))
        tsk_clf$missings()
        pre =  po("encode") %>>%  po("imputemode") %>>% po("removeconstants")
        g = pre %>>% gunion(list(
          po("select") %>>% po("learner_cv", id = "cv1", lrn("classif.kknn")),
          po("pca") %>>% po("learner_cv", id = "cv2", lrn("classif.featureless")),
          po("nop")
        )) %>>%
          po("featureunion") %>>%
          po("learner", lrn("classif.ranger",importance ="permutation")) 
        g$param_set$values$cv1.resampling.method = "spcv_coords"
        g$param_set$values$cv2.resampling.method = "spcv_coords"
        g$keep_results = "TRUE"
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
        response = tr.model$model$predictions
  } else if(is.numeric(df.tr[,target.variable]) & crs == crs){
        if(is.null(method.list) & is.null(meta.learner)){
                   method.list <- c("regr.kknn", "regr.featureless")
                   meta.learner <- "regr.ranger"}
    df.trf = mlr3::as_data_backend(df.tr)
    tsk_regr = TaskRegrST$new(id = id, backend = df.trf, target = target.variable,
                              extra_args = list( positive = "TRUE", coordinate_names = c("x","y"), coords_as_features = FALSE,
                                                 crs = crs))
    pre =  po("encode") %>>%  po("imputemode") %>>% po("removeconstants")
    g = pre %>>% gunion(list(
      po("select") %>>% po("learner_cv", id = "cv1", lrn("regr.kknn")),
      po("pca") %>>% po("learner_cv", id = "cv2", lrn("regr.featureless")),
      po("nop")
    )) %>>%
      po("featureunion") %>>%
      po("learner", lrn("regr.ranger",importance ="permutation")) 
    resampling_sp = rsmp("repeated_spcv_coords", folds = folds, repeats = 4)
    rr_sp = resample(
      task = tsk_regr, learner = g,
      resampling = resampling_sp)
    g$keep_results = "TRUE"
    if(plot.workflow == "TRUE"){
      plt = g$plot()
    }
    message(paste( "         fit the regression model  (rsmp = SPCV by cooridinates) ..."), immediate. = TRUE)
    lgr::get_logger("bbotk")$set_threshold("warn")
    lgr::get_logger("mlr3")$set_threshold("warn")
    g$train(tsk_regr)
    g$predict(tsk_regr)
    summary = g$pipeops$regr.ranger$learner_model$model
    tr.model = g$pipeops$regr.ranger$learner$train(tsk_regr)
    var.imp = tr.model$importance()
    response = tr.model$model$predictions
    train.model = tr.model$predict_newdata
    tr.model$predict_newdata
    tr.model$predict_newdata(newdata )
        }
  return(list(train.model, var.imp, summary,response))

}



