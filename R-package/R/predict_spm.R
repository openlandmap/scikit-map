#' predict_spm
#' @description
#' prediction on new dataset
#' @param newdata data at new location,
#' @param train_model trained model using `train_spm`,
#' @return predicted variables, variable importance 
#' @export 
#' @author  \href{https://opengeohub.org/people/mohammadreza-sheykhmousa}{Mohammadreza Sheykhmousa}
#' @examples 
#' \dontrun{
#' predict.variable = eumap::predict_spm(train_model, newdata)
#' }

predict_spm = function (train_model, newdata){
        predict.variable = train_model(newdata)
        if(!is.data.frame(newdata)){
          newdata = as.data.frame(newdata)
        }
        vrimp = newdata[1:nrow(newdata),vlp]
        y = predict.variable$response
        # measure_test = predict.variable$score()
        return(list(y, vrimp))
}
