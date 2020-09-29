
#' predict.spm
#' @description
#' prediction on new dataset
#' @author  \href{https://opengeohub.org/people/mohammadreza-sheykhmousa}{Mohammadreza Sheykhmousa}
#' @param df.ts 
#' @param task 
#'
#' @return y
#' @export
predict.spm = function (df.ts , task = NULL, train.model){
  print('test')
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

