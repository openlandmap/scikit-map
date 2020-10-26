#' predict_spm
#' @description
#' prediction on new dataset
#' @author  \href{https://opengeohub.org/people/mohammadreza-sheykhmousa}{Mohammadreza Sheykhmousa}
#' @param newdata data at new location
#' @param train.model trained model
#' @return y response values
#' @export
#' @example 
#' \dontrun{
#' predict.variable = predict_spm(train.model, newdata)
#' predict.variable
#' prd.all = predict_spm(train.model, df)
#' str(prd.all)
#' df$leadp = prd.all
#' }
#' 
#' 
#' 
predict_spm = function (train.model, newdata, task = NULL){
 if (is.factor(df.tr[,target.variable])){
        newdataf = mlr3::as_data_backend(newdata)
        tsk_clf = TaskClassifST$new(id = id, backend = newdataf, target = target.variable, extra_args = list(
        positive = "TRUE", coordinate_names = c("x", "y"), coords_as_features = FALSE,crs = crs))
        predict.variable = train.model(newdata)
        yy = newdata[,target.variable]
        y = predict.variable$response
 }
   return(y)
}

