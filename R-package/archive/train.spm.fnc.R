## Wrapper of the mlr3 for spatial data
## Project: GeoHarmonizer_INEA
## Mohammadreza.Sheykhmousa@opengeohub.org
setwd("/home/msheykhmousa/Documents/gitrepo/internal-planning/mlr3/Ensemble/")


# mlr3 function ----
train.spm = function(df.tr, target.variable, methodList=NULL, meta.learner = "regr.ranger",
parallel = "multicore", predict_type = NULL, folds = folds, feature.selection = c("information_gain", "Variance", "find_correlation"),  n_evals = n_evals, plot.workflow = FALSE, var.ens = TRUE, agg = TRUE, crs){
  id = deparse(substitute(df.tr))
  #spcv = rsmp("spcv_coords", folds = folds)
  cv3 = rsmp("repeated_cv", folds = folds)
  
  if(missing(crs)){
    message(paste("resampling method: non-spatialCV", availableCores(), "..."), immediate. = TRUE)
    if(is.factor(df.tr[,target.variable])){
      if(is.null(methodList)){
        methodList <- c("classif.kknn", "classif.svm", "classif.xgboost")
        if(is.null(meta.learner)){
          meta.learner <- "classif.ranger"
          if(missing(predict_type)){ 
            predict_type <- "response" 
          }
        }
      }
    } else {
      if(is.null(methodList)){
        methodList <- c("regr.featureless", "regr.glmnet", "regr.kknn", "regr.lm", "regr.svm", "regr.rpart")
        if(is.null(meta.learner)){
          meta.learner <- "regr.ranger"
          if(missing(predict_type)){ 
            predict_type <- "response" 
          }
        }
      }
    }
    if(parallel=="multicore"){
      nc = availableCores()
      message(paste("Starting parallelisation using", availableCores(), "cores..."), immediate. = TRUE)
      future::plan(multisession, globals = TRUE,
                   workers = nc)
    }
    
    ## CV classif ----
    if(is.factor(df.tr[,target.variable])){
      message(paste0("Using learners: ", paste(methodList, collapse = ", "), "..."), immediate. = TRUE)
      if(is.null(predict_type)){ predict_type = "response" }
        tsk_clf <- mlr3::TaskClassif$new(id = id, backend = df.tr, target = target.variable)
      
      mlr3filters::Filter -> FilterVariance
      mlr3filters::Filter -> FilterFindCorrelation
      knn_lrn  = lrn("classif.kknn", predict_type = "response")
      svm_lrn =  lrn("classif.svm", type = "C-classification", kernel= "radial",predict_type = "response")
      xgb_lrn = lrn("classif.xgboost", predict_type = "response")
      
      
      knn_cv1 = po("learner_cv", knn_lrn, id = "knn_1")
      svm_cv1 = po("learner_cv", svm_lrn, id = "svm_1")
      xgb_cv1 = po("learner_cv", xgb_lrn, id = "xgb_1")
      
      igain = po("filter", flt("information_gain"), id = "igain1")
      imptnc = po("filter", flt("importance"), id = "imptnc1")
      find_cor = po("filter", flt("find_correlation"), id = "cor1")
      
      ## level0 ----
      level0 = po("encode") %>>%  po("imputemode") %>>% po("removeconstants")  #po("scale") %>>% 
      #po("imputemedian") 
      ## level1 ----
      level1 = level0 %>>% gunion(list(
        igain %>>% svm_cv1,
        imptnc %>>% xgb_cv1,
        find_cor %>>% knn_cv1,
        po("nop", id = "nop1")))  %>>%
        po("featureunion", id = "union1") %>>%
        po("encode", id = "encode1") %>>%
        po("removeconstants", id = "removeconstants1")
      
      knn_cv2 = po("learner_cv", knn_lrn , id = "knn_2")
      svm_cv2 = po("learner_cv", svm_lrn, id = "svm_2")
      xgb_cv2 = po("learner_cv", xgb_lrn, id = "xgb_2")
      ## level2 ----
      level2 = level1 %>>%
        po("copy", 4) %>>%
        gunion(list(
          po("pca", id = "pca2_1", param_vals = list(scale. = TRUE)) %>>% svm_cv2,
          po("pca", id = "pca2_2", param_vals = list(scale. = TRUE)) %>>% knn_cv2,
          po("pca", id = "pca2_3", param_vals = list(scale. = TRUE)) %>>% xgb_cv2,
          po("nop", id = "nop2"))
        )  %>>%
        po("featureunion", id = "union2")
      
      ranger_lrn = lrn("classif.ranger", predict_type = "response",importance ="permutation")
      ensemble = level2 %>>% ranger_lrn
      if(plot.workflow==TRUE){
        message(paste0("plotting the ensemble: ", "..."), immediate. = TRUE)
        ensemble$plot(html = FALSE)
      }
      
      ps_ens = ParamSet$new(
        list(      
          ParamFct$new("igain1.type", levels = c("infogain","gainratio","symuncert"), default = "infogain"),
          ParamLgl$new("igain1.equal", default = FALSE),
          ParamLgl$new("igain1.discIntegers", default = TRUE), 
          ParamInt$new("igain1.threads", 20,  30),
          ParamInt$new("igain1.filter.nfeat", 1 , 10),
          ParamInt$new("imptnc1.minsplit", 1 , 5 ),
          ParamInt$new("imptnc1.filter.nfeat", 1 , 10),
          ParamDbl$new("imptnc1.cp", 0.1, 0.5 ),
          ParamInt$new("imptnc1.maxdepth", 1, 29 ),
          ParamFct$new("cor1.use","everything"),
          ParamInt$new("cor1.filter.nfeat",1, 10),
          ParamInt$new("pca2_1.rank.", 1, 10),
          ParamInt$new("pca2_2.rank.", 1, 10),
          ParamInt$new("pca2_3.rank.", 1, 10),
          ParamInt$new("knn_1.k", 1, 10),
          ParamDbl$new("knn_1.distance", 1, 2),
          ParamDbl$new("svm_1.cost", lower = 2^(-12), upper = 2^(0)),
          ParamDbl$new("svm_1.gamma", lower = 2^(-8), upper = 2^(-1)),
          #ParamFct$new("xgb_1.feature_selector",levels = c("cyclic","shuffle","random","greedy","thrifty"),default = "cyclic"),
          ParamInt$new("xgb_1.nthread", 10,30),
          ParamFct$new("xgb_1.booster", levels = c("gbtree","gblinear","dart"),default = "gblinear"),
          ParamInt$new("knn_2.k", 1, 10),
          ParamInt$new("knn_2.distance", 1, 4),
          ParamDbl$new("svm_2.cost", lower = 2^(-12), upper = 2^(0)),
          ParamDbl$new("svm_2.gamma", lower = 2^(-8), upper = 2^(-1)),
          #ParamFct$new("xgb_2.feature_selector",levels = c("cyclic","shuffle","random","greedy","thrifty"), default = "cyclic"),
          ParamInt$new("xgb_2.nthread", 10, 30),
          ParamFct$new("xgb_2.booster", levels = c("gbtree","gblinear","dart"),default = "gblinear"),
          ParamInt$new("classif.ranger.mtry", lower = 1, upper = 2),
          ParamDbl$new("classif.ranger.sample.fraction", lower = 0.5, upper = 0.8),
          ParamInt$new("classif.ranger.num.trees", lower = 50L, upper = 500L),
          ParamInt$new("classif.ranger.num.threads",  15, 32),
          ParamFct$new("classif.ranger.importance", "permutation")
        ))
      
      #as.data.table(ps_ens)
      ens_lrn = GraphLearner$new(ensemble)
      ens_lrn$predict_type = "response"
      ps_ranger = ParamSet$new(
        list(
          ParamInt$new("mtry", lower = 1L, upper = 2),
          ParamDbl$new("sample.fraction", lower = 0.5, upper = 1),
          ParamInt$new("num.trees", lower = 50L, upper = 200L)
        ))

      ## AutoTuner for the ensemble learner ----
      auto1 = AutoTuner$new(
        learner = ens_lrn,
        resampling = cv3,
        measure = msr("classif.acc"),
        search_space = ps_ens,
        terminator = trm("evals", n_evals = n_evals), 
        tuner = tnr("random_search")
      )
      auto1$store_tuning_instance = TRUE
      ## AutoTuner for the simple ranger learner ----
      auto2 = AutoTuner$new(
        learner = ranger_lrn,
        resampling = cv3,
        measure = msr("classif.acc"),
        search_space = ps_ranger,
        terminator = trm("evals", n_evals = n_evals), 
        tuner = tnr("random_search")
      )
      auto2$store_tuning_instance = TRUE

      
      set.seed(321)
      outer_hold = rsmp("holdout", ratio=.8)
      outer_hold$instantiate(tsk_clf)
      
      design = benchmark_grid(
        tasks = tsk_clf,
        learners = auto2,
        resamplings = outer_hold
      )
      
      requireNamespace("lgr")
      logger = lgr::get_logger("mlr3")
      logger$set_threshold("trace")
      
      lgr::get_logger("mlr3")$set_threshold("warn")
      lgr::get_logger("mlr3")$set_threshold("debug")
      message("           Fitting a ensemble ML using 'mlr3::TaskClassif'...", immediate. = TRUE)
      bmr = benchmark(design, store_models = TRUE)
      agg = bmr$aggregate(msr("classif.acc"))
      message("           Var.Imp '...", immediate. = TRUE)
      ensemble$pipeops$classif.ranger$train(list(tsk_clf))
      var.ens = ensemble$pipeops$classif.ranger$learner_model$importance()
      acc = round(bmr$aggregate(msr("classif.acc"))[1]$classif.acc, digits = 3)
      tr.model = bmr$learners$learner[[2]]
      train.model = tr.model$predict_newdata
      ## classifession result ----  
      return(train.model)
      
    } 
    
    ## CV regression model ----
    if(is.numeric(df.tr[,target.variable])){
      message(paste0("Using learners: ", paste(methodList, collapse = ", "), "..."), immediate. = TRUE)
      if(is.null(predict_type)){ predict_type = "response" }
      task_regr <-mlr3::TaskRegr$new(id = id, backend = df.tr, target = target.variable)
     
      
      mlr3filters::Filter -> FilterVariance
      mlr3filters::Filter -> FilterFindCorrelation
      knn_lrn  = lrn("regr.kknn", predict_type = "response")
      svm_lrn =  lrn("regr.svm", type = "eps-regression", kernel= "radial",predict_type = "response")
      xgb_lrn = lrn("regr.xgboost", predict_type = "response")
      
      knn_cv1 = po("learner_cv", knn_lrn, id = "knn_1")
      svm_cv1 = po("learner_cv", svm_lrn, id = "svm_1")
      xgb_cv1 = po("learner_cv", xgb_lrn, id = "xgb_1")
      
      igain = po("filter", flt("information_gain"), id = "igain1")
      variance = po("filter", flt("variance"), id = "var1")
      find_cor = po("filter", flt("find_correlation"), id = "cor1")
      # level0 ----
      level0 = po("scale") %>>% po("encode") %>>%
        po("imputemedian") %>>% po("removeconstants")
      # level1 ----
      level1 = level0 %>>% gunion(list(
        igain %>>% knn_cv1,
        variance %>>% svm_cv1,
        find_cor %>>% xgb_cv1,
        po("nop", id = "nop1")))  %>>%
        po("featureunion", id = "union1")
      
      knn_cv2 = po("learner_cv", knn_lrn , id = "knn_2")
      svm_cv2 = po("learner_cv", svm_lrn, id = "svm_2")
      xgb_cv2 = po("learner_cv", xgb_lrn, id = "xgb_2")
      # level2 ----
      level2 = level1 %>>%
        po("copy", 4) %>>%
        gunion(list(
          po("pca", id = "pca2_1", param_vals = list(scale. = TRUE)) %>>% knn_cv2,
          po("pca", id = "pca2_2", param_vals = list(scale. = TRUE)) %>>% svm_cv2,
          po("pca", id = "pca2_3", param_vals = list(scale. = TRUE)) %>>% xgb_cv2,
          po("nop", id = "nop2"))
        )  %>>%
        po("featureunion", id = "union2")
      
      ranger_lrn = lrn("regr.ranger", predict_type = "response",importance ="permutation")
      ensemble = level2 %>>% ranger_lrn
      if(plot.workflow==TRUE){
        message(paste0("plot the ensemble: ", "..."), immediate. = TRUE)
        ensemble$plot(html = FALSE)
      }
      
      ps_ens = ParamSet$new(
        list(      
          ParamInt$new("igain1.filter.nfeat", 1, 30),
          ParamInt$new("var1.filter.nfeat", 1, 30),
          ParamInt$new("cor1.filter.nfeat", 1, 30),
          ParamInt$new("pca2_1.rank.", 3, 10),
          ParamInt$new("pca2_2.rank.", 3, 10),
          ParamInt$new("pca2_3.rank.", 3, 10),
          ParamInt$new("knn_1.k", 1, 10),
          ParamDbl$new("knn_1.distance", 1, 5),
          ParamDbl$new("svm_1.cost", lower = 2^(-12), upper = 2^(4)),
          ParamDbl$new("svm_1.gamma", lower = 2^(-12), upper = 2^(-1)),
          ParamInt$new("knn_2.k", 1, 10),
          ParamInt$new("knn_2.distance", 1, 4),
          ParamDbl$new("svm_2.cost", lower = 2^(-12), upper = 2^(4)),
          ParamDbl$new("svm_2.gamma", lower = 2^(-12), upper = 2^(-1)),
          ParamInt$new("regr.ranger.mtry", lower = 1, upper = 2),
          ParamDbl$new("regr.ranger.sample.fraction", lower = 0.5, upper = 1),
          ParamInt$new("regr.ranger.num.trees", lower = 50L, upper = 500L),
          ParamFct$new("regr.ranger.importance", "permutation")
        ))
      
      ens_lrn = GraphLearner$new(ensemble)
      ens_lrn$predict_type = "response"
      ps_ranger = ParamSet$new(
        list(
          ParamInt$new("mtry", lower = 1L, upper = 2),
          ParamDbl$new("sample.fraction", lower = 0.5, upper = 1),
          ParamInt$new("num.trees", lower = 50L, upper = 200L)
        ))
      # AutoTuner for the ensemble learner ----
      auto1 = AutoTuner$new(
        learner = ens_lrn,
        resampling = cv3,
        measure = msr("regr.rmse"),
        search_space = ps_ens,
        terminator = trm("evals", n_evals = n_evals), 
        tuner = tnr("random_search")
      )
      auto1$store_tuning_instance = TRUE
      # AutoTuner for the simple ranger learner ----
      auto2 = AutoTuner$new(
        learner = ranger_lrn,
        resampling = cv3,
        measure = msr("regr.rmse"),
        search_space = ps_ranger,
        terminator = trm("evals", n_evals = n_evals), 
        tuner = tnr("random_search")
      )
      auto2$store_tuning_instance = TRUE
      learns = list(auto1, auto2)
      set.seed(321)
      outer_hold = rsmp("holdout", ratio=.8)
      outer_hold$instantiate(task_regr)
      
      design = benchmark_grid(
        tasks = task_regr,
        learners = learns,
        resamplings = outer_hold
      )
      
      requireNamespace("lgr")
      logger = lgr::get_logger("mlr3")
      logger$set_threshold("trace")
      
      lgr::get_logger("mlr3")$set_threshold("warn")
      lgr::get_logger("mlr3")$set_threshold("debug")
      message("Fitting a ensemble ML using 'mlr3::TaskRegr'...", immediate. = TRUE)
      bmr = benchmark(design, store_models = TRUE)
      
      #### ------
      tr.model = bmr$learners$learner[[2]]
      tr.model$train(task_regr)
      train.model = tr.model$predict_newdata
      agg = bmr$aggregate(msr("regr.rsq"))
      message("Var.Imp '...", immediate. = TRUE)
      ensemble$pipeops$regr.ranger$train(list(task_regr))
      var.ens = ensemble$pipeops$regr.ranger$learner_model$importance()
      RSQ = round(bmr$aggregate(msr("regr.rsq"))[1]$regr.rsq, digits = 3)
      RMSE = round(bmr$aggregate(msr("regr.rmse"))[1]$regr.rmse, digits = 3)
      p = ensemble$pipeops$regr.ranger$predict(list(task_regr))
      
      ## Regression result ----  

    }else {
      message("Stopping. Neither 'factor' nor 'numeric' variable selected.")
    }
    return(train.model)
  } else if(is.factor(df.tr[,target.variable]) & crs == crs){ 
        methodList <- c("classif.ranger", "classif.rpart")
        meta.learner = "classif.ranger"
        df.trf = mlr3::as_data_backend(df.tr)
        tsk_clf = TaskClassifST$new(id = id, backend = df.trf, target = target.variable, extra_args = list(
        positive = "TRUE", coordinate_names = c("x", "y"), coords_as_features = FALSE,crs = crs))
        
        df.tsf = mlr3::as_data_backend(df.ts)
        tsk_cts = TaskClassifST$new(id = id, backend = df.tsf, target = target.variable, extra_args = list(
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
        return(train.model)
  } else if(is.numeric(df.tr[,target.variable]) & crs == crs){
        if(is.null(methodList)){
                   methodList <- c("regr.ranger", "regr.rpart")
                   meta.learner <- "regr.ranger"}
        df.trf = mlr3::as_data_backend(df.tr)
        tsk_regr = TaskRegrST$new(id = id, backend = df.trf, target = target.variable,
        extra_args = list( positive = "TRUE", coordinate_names = c("x", "y"), coords_as_features = FALSE,
        crs = crs))
                    
        df.tsf = mlr3::as_data_backend(df.ts)
        tsk_ts = TaskRegrST$new(id = id, backend = df.tsf, target = target.variable,
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
        return(train.model)
        }
      }

  