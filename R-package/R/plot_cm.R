#' Function for plotting confusion matrices
#' @description The code is a modified version of [THIS LINK](https://stackoverflow.com/questions/23891140/r-how-to-visualize-confusion-matrix-using-the-caret-package/60150826#60150826)
#' @param cm A data frame that consist of prediction and truth; normally comes as an output of classification models.
#'
#' @return Annotated heat map for CM
#' @export
#'
#' @examples
#' \dontrun{
#' library(caret)
# simulated data
#' set.seed(23)
#' pred <- factor(sample(1:7,100,replace=T))
#' ref<- factor(sample(1:7,100,replace=T))
#' cm <- caret::confusionMatrix(pred,ref)
#' plot_cm(cm)
#' }
#' 
plot_cm <- function(cm){
  # extract the confusion matrix values as data.frame
  cm_d <- as.data.frame(cm$table)
  cm_d$diag <- cm_d$Prediction == cm_d$Reference # Get the Diagonal
  cm_d$ndiag <- cm_d$Prediction != cm_d$Reference # Off Diagonal     
  cm_d[cm_d == 0] <- NA # Replace 0 with NA for white tiles
  cm_d$Reference <-  reverse.levels(cm_d$Reference) # diagonal starts at top left
  cm_d$ref_freq <- cm_d$Freq * ifelse(is.na(cm_d$diag),-1,1)
  
  # plotting the matrix
  plt <-  ggplot(data = cm_d, aes(x = Prediction , y =  Reference, fill = Freq))+
    scale_x_discrete(position = "top") +
    geom_tile( data = cm_d,aes(fill = ref_freq)) +
    scale_fill_gradient2(guide = FALSE,low="red3",high="orchid4", midpoint = 0,na.value = 'white') +
    geom_text(aes(label = Freq), color = 'black', size = 3)+
    theme_light() +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          legend.position = "none",
          panel.border = element_blank(),
          plot.background = element_blank(),
          axis.line = element_blank())
  
  return(plt)
}




