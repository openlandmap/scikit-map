#' predict_spm
#' @description
#' prediction on new dataset
#' @param newdata data at new location,
#' @param train.model trained model from `train_spm`,
#' @return 
#' @export 
#' @author  \href{https://opengeohub.org/people/mohammadreza-sheykhmousa}{Mohammadreza Sheykhmousa}
#' @example 
#' \dontrun{
#' predict.variable = eumap::predict_spm(train.model, newdata)
#' }

predict_spm = function (train_model, newdata){
        predict.variable = train_model(newdata)
        if(!is.data.frame(newdata)){
          newdata = as.data.frame(newdata)
        }
        vrimp = newdata[1:nrow(newdata),vlp]
        y = predict.variable$response
<<<<<<< HEAD
        # measure_test = predict.variable$score()
        return(list(y, vrimp,measure_test))
=======
        #measure_test = predict.variable$score()
        return(list(y, vrimp))
>>>>>>> master
}
