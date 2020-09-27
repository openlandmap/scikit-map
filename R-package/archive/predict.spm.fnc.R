## Wrapper of the mlr3 for spatial data
## Project: GeoHarmonizer_INEA
## Mohammadreza.Sheykhmousa@opengeohub.org

setwd("/home/msheykhmousa/Documents/gitrepo/internal-planning/mlr3/Ensemble/")
source('train.spm.fnc.R')
# mlr3 function ----
predict.spm = function (df.ts , target.variable, train.model , crs){
   id = deparse(substitute(df.ts))
   if(missing(crs) & is.factor(df.ts[,target.variable])){
       tsk_clf <- mlr3::TaskClassif$new(id = id, backend = df.ts, target = target.variable)
       predict.variable = train.model(df.ts, tsk_clf) 
       y = df.ts[,target.variable]
       # x = predict.variable$truth
       return(y)
    } else if (missing(crs) & is.numeric(df.ts[,target.variable])) {
        task_regr <-mlr3::TaskRegr$new(id = id, backend = df.ts, target = target.variable)
        predict.variable = train.model(df.ts, task_regr)   
        y = df.ts[,target.variable]
        # x = predict.variable$truth
        return(y)

    } else if (is.factor(df.ts[,target.variable]) & crs == crs){ 
        df.tsf = mlr3::as_data_backend(df.ts)
        tsk_clf = TaskClassifST$new(id = id, backend = df.tsf, target = target.variable, extra_args = list(
        positive = "TRUE", coordinate_names = c("x", "y"), coords_as_features = FALSE,crs = crs))
        predict.variable = train.model(df.ts, tsk_clf)   
        y = df.ts[,target.variable]
        # x = predict.variable$truth
        return(y)
    
           
    } else if (is.numeric(df.ts[,target.variable]) & crs == crs){
      df.tsf = mlr3::as_data_backend(df.ts)
      tsk_regr = TaskRegrST$new( id = id, backend = df.tsf, target = target.variable, extra_args = list( positive = "TRUE", coordinate_names = c("x", "y"), coords_as_features = FALSE, crs = crs))
      predict.variable = train.model(df.ts, tsk_regr)   
      y = df.ts[,target.variable]
      return(y)
    }
}

