#' predict_spm
#' @description
#' prediction on new dataset
#' @param object trained model using `train_spm`,
#' @param newdata data at new location,
#' @return predicted variables, variable importance 
#' @export 
#' @author  \href{https://opengeohub.org/people/mohammadreza-sheykhmousa}{Mohammadreza Sheykhmousa}
#' @examples 
#' \dontrun{
#' predict.variable = eumap::predict_spm(object, newdata)
#' }

predict_spm = function (object, newdata){
        predict.variable = object(newdata)
        if(!is.data.frame(newdata)){
          newdata = as.data.frame(newdata)
        }
        vrimp = newdata[1:nrow(newdata),vlp]
        y = predict.variable$response
        # measure_test = predict.variable$score()
        return(list(y, vrimp))
}
