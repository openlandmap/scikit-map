#' Title
#'
#' @param train.model 
#' @param newdata 
#' @param task 
#'
#' @return
#' @export
#'
#' @examples
predict_spm = function (train.model, newdata, task = NULL){
        predict.variable = train.model(newdata)
        if(!is.data.frame(newdata)){
          newdata = as.data.frame(newdata)
        }
        vrimp = newdata[1:nrow(newdata),vlp]
        y = predict.variable$response
        measure_test = predict.variable$score()
        return(list(y, vrimp,measure_test))
}
