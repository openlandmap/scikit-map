
#' @title  predict.spm
#' @description
#' prediction on new dataset
#' @author  \href{https://opengeohub.org/people/mohammadreza-sheykhmousa}{Mohammadreza Sheykhmousa}
#' @param newdata 
#' @param train.model
#' @
#' @return y
#' @export
predict.spm = function (train.model, newdata, task = NULL){
      predict.variable = train.model(newdata)   
      y = predict.variable$response
   return(y)
}

