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
predict_spm = function (train.model, newdata, task = NULL){
  if(is.factor(df.ts[,target.variable])){
        predict.variable = train.model(newdata)
        y = predict.variable$response
  } else if (is.numeric(df.ts[,target.variable])){
      predict.variable = train.model(newdata)
      y = predict.variable$response
   }
  return(y)
}
