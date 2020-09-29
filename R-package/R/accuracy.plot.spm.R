
pfun <- function(x,y, ...){
  panel.xyplot(x, y, ...)
  panel.hexbinplot(x,y, ...)  
  panel.abline(coef = c(0,1), col="black", size = 0.25, lwd = 2)
  return(pfun)
}

#' Accuracy plot
#' 
#' 
#' @author  \href{https://opengeohub.org/people/mohammadreza-sheykhmousa}{Mohammadreza Sheykhmousa}
#' 
#' @param x 
#' @param y 
#' @param main 
#' @param colramp 
#' @param xbins 
#' @param rng 
#'
#' @return plt
#' @return
#' @export

accuracy.plot.spm <- function(x, y, main, colramp, xbins = xbins. , rng ="nat"){
if(rng == "norm"){
    x.= normalize(x, method = "range", range = c(0, 1))
    y. = normalize(y, method = "range", range = c(0, 1))
    CCC <- signif(ccc(data.frame(x,y), x, y)$.estimate, digits=3)
    plt <- hexbinplot(x. ~ y., xbins = xbins., mincnt = 1, xlab=expression(italic("0~1 measured")), ylab=expression(italic("0~1 predicted")), inner=0.2, cex.labels=1, colramp = colramp., aspect = 1, main= paste0('CCC: ', CCC), colorcut= colorcut., type="g",panel = pfun) 
  
  }
   if(rng == "nat"){
      CCC <- signif(ccc(data.frame(x,y), x, y)$.estimate, digits=3)
      plt <- hexbinplot(x ~ y, mincnt = 1, xbins=35, xlab="measured", ylab="predicted (ensemble)", inner=0.2, cex.labels=1, colramp= colramp., aspect = 1, main=paste0('CCC: ', CCC), colorcut=colorcut., type="g",panel = pfun) 
    plt  
   }
  print(plt)
  return(plt)
}
