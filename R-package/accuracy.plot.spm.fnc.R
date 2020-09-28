## Accuracy plot function
## Project: GeoHarmonizer_INEA
## Mohammadreza.Sheykhmousa@opengeohub.org

source('predict.spm.fnc.R')

pfun <- function(x,y, ...){
  panel.xyplot(x, y, ...)
  panel.hexbinplot(x,y, ...)  
  panel.abline(coef = c(0,1), col="black", size = 0.25, lwd = 2)
}

accuracy.plot.spm <- function(x, y, main, colramp, xbins = xbins. , rng ="nat"){
if(rng == "norm"){
    x.= normalize(x, method = "range", range = c(0, 1))
    y. = normalize(y, method = "range", range = c(0, 1))
    plt <- hexbinplot(x. ~ y., xbins = xbins., mincnt = 1, xlab=expression(italic("0~1 measured")), ylab=expression(italic("0~1 predicted")), inner=0.2, cex.labels=1, colramp = colramp., aspect = 1, main= paste0('RMSE: ', '    RSQ: '), colorcut= colorcut., type="g",panel = pfun) 
  
  }
   if(rng == "nat"){
      plt <- hexbinplot(x ~ y, mincnt = 1, xbins=35, xlab="measured", ylab="predicted (ensemble)", inner=0.2, cex.labels=1, colramp= colramp., aspect = 1, main=paste0('RMSE: ', '    RSQ: '), colorcut=colorcut., type="g",panel = pfun) 
    plt  
   }
  print(plt)
  return(plt)
  }



