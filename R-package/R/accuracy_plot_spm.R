pfun <- function(x,y, ...){
  panel.xyplot(x, y, ...)
  panel.hexbinplot(x,y, ...)  
  panel.abline(coef = c(0,1), col="black", size = 0.25, lwd = 2)
  return(pfun)
}

#' Accuracy plot
#' @author  \href{https://opengeohub.org/people/mohammadreza-sheykhmousa}{Mohammadreza Sheykhmousa}
#' 
#' @param x observation vector
#' @param y estimated vector
#' @param main title
#' @param colramp color
#' @param xbins number of bins
#' @param rng methods for data representaion e.g., `norm`; normalized , `nat`;real values
#' @return plt accuracy plot
#' @export
#' @example 
#' \dontrun{
#' colorcut. = c(0,0.01,0.03,0.07,0.15,0.25,0.5,0.75,1)
#' colramp. = colorRampPalette(c("wheat2","red3"))
#' xbins. = 50
#' library("stats")  
#' library("dplyr")
#' library("EnvStats")
#' library("grid")
#' library("hexbin")
#' library("BBmisc")
#' library("lattice")
#' library("MASS")
#' library("gridExtra")
#' library("MLmetrics")
#' library("yardstick")
#' library("eumap")
#' plt = eumap::plot_spm(x = runif(1e3,1,1e9) , y = runif(1e3,0,1), rng = "norm")
#' }
#' 
plot_spm <- function(x, y, main, colramp, xbins = xbins. , rng ="nat"){
if(rng == "norm"){
  #summary(unique(df.tr$x & df.tr$y %in% df.ts$x & df.ts$y))
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
