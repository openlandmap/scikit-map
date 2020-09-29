
#' train.spm
#'@description
#' This is the abstract base class for training objects like [TaskClassif] and [TaskRegr].
#' For example, for a classification task columns must be marked as ID, df, target column.
#' train.spm  it multiple models/learners depending on the
#'class() of the target.variable and for now only returns a
#'trained model function so later on we could use it to train a new dataset.
#' 
#' 
#' @param df.tr 
#' @param target.variable 
#' @param parallel 
#' @param predict_type 
#' @param folds 
#' @param method.list 
#' @param n_evals 
#' @param plot.workflow 
#' @param var.imp 
#' @param meta.learner 
#' @param crs 
#'
#' @return train.model
#' 
#' @author  \href{https://opengeohub.org/people/mohammadreza-sheykhmousa}{Mohammadreza Sheykhmousa} and  \href{https://opengeohub.org/people/tom-hengl}{Tom Hengl}
#' 
#' @export 
#'
#' @examples
#' ## Splitting training (tr) and test (ts) sets and defining generic variables
#' ## Meuse Demo
#' library(sp)
#' library(mlr3verse)
#' data(meuse)
#' df <- meuse
#' df <- na.omit(df[,])
#' crs = "+init=epsg:3035"
#' target.variable = "lead"
#' ## define generic var
#' smp_size <- floor(0.5 * nrow(df))
#' set.seed(123)
#' train_ind <- sample(seq_len(nrow(df)), size = smp_size)
#' df.tr <- df[train_ind, ]
#' df.ts <- df[-train_ind, ]
#' folds = 2
#' xbins. = 50
#' tr = train.spm(df.tr, target.variable = target.variable, folds = folds ,n_evals = n_evals, plot.workflow = TRUE)
#' train.model= tr[[1]]
#' 
#' var.imp = tr[[2]]
#' var.imp
#' 
#' summary = tr[[3]]
#' summary
#' 
#' predict.variable = predict.spm(df.ts, task = NULL, train.model)
#' ## plot var
#' colorcut. = c(0,0.01,0.03,0.07,0.15,0.25,0.5,0.75,1)
#' colramp. = colorRampPalette(c("wheat2","red3"))
#' accuracy.plot.spm(x = df.ts[,target.variable], y = predict.variable)
#'
train.spm = function(df.tr, target.variable, 
parallel = TRUE, predict_type = NULL, folds = folds, method.list = NULL,  n_evals = n_evals, plot.workflow = FALSE, var.imp = TRUE, meta.learner = NULL, crs){
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
        best.model = at$archive$best()
        var.imp = at$learner$importance()
        summary = at$learner$state$model
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
      summary = tr.model$model
      var.imp = tr.model$importance()
      train.model = tr.mdl$predict_newdata

    } else if (is.factor(df.tr[,target.variable]) & crs == crs){ 
        method.list <- c("classif.ranger", "classif.rpart")
        meta.learner = "classif.ranger"
        df.trf = mlr3::as_data_backend(df.tr)
        tsk_clf1 = TaskClassifST$new(id = id, backend = df.trf, target = target.variable, extra_args = list( positive = "TRUE", coordinate_names = c("x", "y"), coords_as_features = FALSE,crs = crs))
        lrn = lrn("classif.ranger")
        gr = pipeline_robustify(tsk_clf1, lrn) %>>% po("learner", lrn)
        ede = resample(tsk_clf1, GraphLearner$new(gr), rsmp("holdout"))
        tsk_clf = ede$task$clone()
        tsk_clf$missings()
        
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
        tsk_regr1 = TaskRegrST$new(id = id, backend = df.trf, target = target.variable,
        extra_args = list( positive = "TRUE", coordinate_names = c("x", "y"), coords_as_features = FALSE,
        crs = crs))
        lrn = lrn("regr.rpart")
        gr = pipeline_robustify(tsk_regr1, lrn) %>>% po("learner", lrn)
        ede = resample(tsk_regr1, GraphLearner$new(gr), rsmp("holdout"))
        tsk_regr = ede$task$clone()
        tsk_regr$missings()
       
        g = gunion(list(
        po("learner_cv", id = "cv1", lrn("regr.ranger")),
        po("pca") %>>% po("learner_cv", id = "cv2", lrn("regr.rpart")),
        po("nop") %>>% po("encode") %>>%  po("imputemode") %>>% po("removeconstants")
        )) %>>%
        po("featureunion") %>>%
        po("learner", lrn("regr.ranger",importance ="permutation")) 
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
        var.imp = tr.model$importance()
        train.model = tr.model$predict_newdata
        
  }
  return(list(train.model, var.imp, summary))
}

  