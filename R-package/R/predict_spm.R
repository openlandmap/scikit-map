#' Predict_spm
#' @description
#' Prediction on a new dataset
#' @param object Trained model using `train_spm`,
#' @param newdata Data at new location,
#' @return Predicted variables, variable importance 
#' @export 
#' @author  \href{https://opengeohub.org/people/mohammadreza-sheykhmousa}{Mohammadreza Sheykhmousa}
#' @examples 
#' predict.variable = eumap::predict_spm(object, newdata)

predict_spm = function(object, newdata){
        predict.variable = object(newdata)
        if (!is.data.frame(newdata)) {
          newdata = as.data.frame(newdata)
        }
        vrimp = newdata[1:nrow(newdata), vlp]
        y = predict.variable$response
        # measure_test = predict.variable$score()
        return(list(y, vrimp))
}
