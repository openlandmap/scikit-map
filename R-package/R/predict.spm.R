
#' predict.spm
#' @description
#' prediction on new dataset
#' @author  \href{https://opengeohub.org/people/mohammadreza-sheykhmousa}{Mohammadreza Sheykhmousa}
#' @param newdata 
#' @param train.model
#' 
#' @return y
#' @export
predict.spm = function (train.model, newdata, task = NULL){
 # if (is.factor(df.tr[,target.variable])){ 
 #        # newdataf = mlr3::as_data_backend(newdata)
 #        # tsk_clf = TaskClassifST$new(id = id, backend = newdataf, target = target.variable, extra_args = list(
 #        # positive = "TRUE", coordinate_names = c("x", "y"), coords_as_features = FALSE,crs = crs))
 #        predict.variable = train.model(newdata)   
 #        y = newdata[,target.variable]
 # 
 #    } else if(is.numeric(df.tr[,target.variable])){
 #      # newdataf = mlr3::as_data_backend(newdata)
 #      # tsk_regr = TaskRegrST$new( id = id, backend = newdataf, target = target.variable, extra_args = list( positive = "TRUE", coordinate_names = c("x", "y"), coords_as_features = FALSE, crs = crs))
      predict.variable = train.model(newdata)   
      y = predict.variable$response
   return(y)
}

