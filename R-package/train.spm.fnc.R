## Wrapper of the mlr3 for (non)spatial data
## Project: GeoHarmonizer_INEA
## Mohammadreza.Sheykhmousa@opengeohub.org



### mlr3 function ----
train.spm = function(df.tr, target.variable, 
parallel = TRUE, predict_type = NULL, folds = folds, method.list = NULL,  n_evals = n_evals, plot.workflow = FALSE, var.ens = TRUE, meta.learner = NULL, crs){
  id = deparse(substitute(df.tr))
  #spcv = rsmp("spcv_coords", folds = folds)
  cv3 = rsmp("repeated_cv", folds = folds)
   if(is.factor(df.tr[,target.variable]) & missing(crs)){
    message(paste("classification Task  ","resampling method: non-spatialCV ", "ncores: ",availableCores(), "..."), immediate. = TRUE)
        ## CV classif ----
        message(paste0("Using learners: ", paste("method.list", collapse = ", "), "..."), immediate. = TRUE)
        tsk_clf <- mlr3::TaskClassif$new(id = id, backend = df.tr, target = target.variable)
        lrn = lrn("classif.rpart")
        gr = pipeline_robustify(tsk_clf, lrn) %>>% po("learner", lrn)
        ede = resample(tsk_clf, GraphLearner$new(gr), rsmp("holdout"))
        tsk_clasif1 = ede$task$clone()
        
        ranger_lrn = lrn("classif.ranger", predict_type = "response",importance ="permutation")
        # model = level0 %>>% ranger_lrn
        ps_ranger = ParamSet$new(
           list(
             ParamInt$new("mtry", lower = 1L, upper = 5L),
             ParamDbl$new("sample.fraction", lower = 0.5, upper = 1),
             ParamInt$new("num.trees", lower = 50L, upper = 500L),
             # ParamInt$new("classif.ranger.num.threads",  31, 32),
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
        ## CV regression model ----      
      message(paste0("Using learners: ", paste("method.list", collapse = ", "), "..."), immediate. = TRUE)
      # task_regr <-mlr3::TaskRegr$new(id = id, backend = df.tr, target = target.variable)
      tsk_rgr <- mlr3::TaskRegr$new(id = id, backend = df.tr, target = target.variable)
      lrn = lrn("regr.rpart")
      gr = pipeline_robustify(tsk_rgr, lrn) %>>% po("learner", lrn)
      ede = resample(tsk_rgr, GraphLearner$new(gr), rsmp("holdout"))
      tsk_regr1 = ede$task$clone()
      
      ranger_lrn = lrn("regr.ranger", predict_type = "response",importance ="permutation")
      # model = level0 %>>% ranger_lrn
      ps_ranger = ParamSet$new(
        list(
          ParamInt$new("mtry", lower = 1L, upper = 5L),
          ParamDbl$new("sample.fraction", lower = 0.5, upper = 1),
          ParamInt$new("num.trees", lower = 50L, upper = 500L),
          # ParamInt$new("regr.ranger.num.threads",  31, 32),
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
        # ----
        summary = g$pipeops$classif.ranger$learner_model$model
        tr.model = g$pipeops$classif.ranger$learner$train(tsk_clf)
        # tr.model$predict_newdata(df.ts ,tsk_cts)
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
        # conf.mat = g$pipeops$regr.ranger$learner_model$model$confusion.matrix
        # var.imp = g$pipeops$regr.ranger$learner_model$model$variable.importance
        summary = g$pipeops$regr.ranger$learner_model$model
        # ----
        tr.model = g$pipeops$regr.ranger$learner$train(tsk_regr)
        # tr.model$predict_newdata(df.ts ,tsk_ts)
        train.model = tr.model$predict_newdata
  }
  return(train.model)
}

  