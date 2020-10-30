
pfun <- function(x,y, ...){
  panel.xyplot(x, y, ...)
  panel.hexbinplot(x,y, ...)  
  panel.abline(coef = c(0,1), col="black", size = 0.25, lwd = 2)
  return(pfun)
}

prepfun <- function(x, y, ...){
  prepanel.hexbinplot(x,y, ...)
  return(prepfun)
  } 

#' Accuracy plot
#' @author  \href{https://opengeohub.org/people/mohammadreza-sheykhmousa}{Mohammadreza Sheykhmousa}
#' 
#' @param x observation vector
#' @param y estimated vector
#' @param main title
#' @param colramp color
#' @param xbins number of bins
#' @param mode methods for data representaion e.g., `norm`; normalized , `nat`;real values
#' @return plt accuracy plot
#' @export
#' @example 
#' \dontrun{
#' colorcut = c(0,0.01,0.03,0.07,0.15,0.25,0.5,0.75,1)
#' colramp = colorRampPalette(c("wheat2","red3"))
#' xbins = 50
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
#' plt = eumap::plot_spm(x = runif(1e3,1,1e9) , y = runif(1e3,0,1), gmode = "norm")
#' }


plot_spm <- function(x, y, z = NULL, main = NULL, palet  = NULL, colorcut = c(0,0.01,0.03,0.07,0.15,0.25,0.5,0.75,1),
xbins = 60 , gvar_imp = TRUE, gmode  = c("root","log10","norm","log2","nat"), gtype = c("accuracy", "correlation"), aggregate = NULL, aspect = 1){
  
  if(gvar_imp == TRUE){
    var_imp = barplot(var.imp, horiz = TRUE, las = 1, col = gray.colors(10))
    title(main = "variable importance", font.main = 4)
    print(var_imp)
  }
  if(is.null(palet)){
      palet=colorRampPalette(c("wheat2","yellow" ,"red","red3","orchid","orchid4") )
  }
  if(length(x) <= 500 ) {
    plt <- xyplot(x ~ y, asp=1, 
                   par.settings = list(plot.symbol = list(col=scales::alpha("black", 0.6), fill=scales::alpha("red", 0.6), pch=21, cex=0.6)), 
                   scales = list(x=list(log=TRUE, equispaced.log=FALSE), y=list(log=TRUE, equispaced.log=FALSE)),
                   ylab="measured", xlab="predicted")
    print(plt)
  #From Tom's book with a slight modification
    } else {
      if(gtype == "accuracy"){
        #get everything from accuracy.plot  here and enjoy!
        
        if(is.null(main)){
          CCC <- signif(ccc(data.frame(x,y), x, y)$.estimate, digits=3)
          RMSE <- signif(rmse(data.frame(x,y), x, y)$.estimate, digits=3)
          main = paste0(expression(CCC) ,": ",  CCC, "  RMSE: ", RMSE)
        }
        
        if(gmode == "norm"){
          df.x = normalize(x, method = "range", range = c(0, 1))
          df.y = normalize(y, method = "range", range = c(0, 1))
          xlab = expression(italic("0~1 measured"))
          ylab = expression(italic("0~1 predicted"))
          xscale.components = xscale.components.logpower
          yscale.components = yscale.components.logpower
        } else if(gmode == "nat"){
          df.x = x
          df.y = y
          xlab = expression(italic("measured"))
          ylab = expression(italic("predicted (ensemble)"))
          xscale.components = xscale.components.logpower
          yscale.components = yscale.components.logpower
        } else if(gmode=="log10"){
          df.x = exp(x)
          df.y = exp(y)
          xlab = expression(italic("log10 (measured)"))
          ylab = expression(italic("log10 (predicted)"))
          xscale.components = xscale.components.logpower
          yscale.components = yscale.components.log10ticks
        } else if(gmode=="log2"){
          df.x = 2^x
          df.y = 2^y
          xlab = expression(italic("log2 (measured)"))
          ylab = expression(italic("log2 (predicted)"))
          xscale.components = xscale.components.log
          yscale.components = yscale.components.log
        } else if(gmode=="root"){
          df.x = sqrt(x)
          df.y = sqrt(y)
          xlab = expression(sqrt(italic(measured)))
          ylab =expression(sqrt(italic(predicted)))
          xscale.components = xscale.components.subticks
          yscale.components = yscale.components.subticks
        }
      
        plt <- hexbinplot(
          df.x ~ df.y, xbins=xbins, scales = list(x = df.x, y = df.y), 
          xscale.components = xscale.components, yscale.components = yscale.components, 
          mincnt = 1, xlab = xlab , ylab = ylab, inner=0.2, cex.labels=1, colramp = palet, 
          aspect = aspect, main=main[1], colorcut = colorcut, type="g", panel = pfun
        ) 
        print(plt)
      } else if(gtype == "correlation") {
        #first we need to find var.imp and related vals <- ok
        #id = strsplit(deparse(target.variable),"\"")[[1]][2] #this was stupid of me
        df.pcor = data.frame(target.variable = df.tr[,target.variable],z = valu.imp)
        pcorr = pcor(df.pcor)
        main = "Partial correlation"
        #fekri behale namayeshe multiple figure bekon ba fit line mesle naghale
        #now we have 2 varimp see how to generate the maps automatically
        if(length(vlp) == 1){
          plt <- hexbinplot(df.pcor[,"target.variable"] ~ df.pcor[,2],
          mincnt = 1,xbins=xbins,xlab=colnames(df.pcor)[2],ylab="target.variable",
          colramp =  palet, main=main, colorcut = colorcut, type="g",panel = pfun) 
        }
        if(length(vlp) > 1){
          df.pcorr = df.pcor
          df.pcorr[,1] <- NULL
          x = df.pcor[,"target.variable"]
          plt <- lapply(1:length(vlp), function(i) {
          # y =  df.pcorr[,i]  
          hexbinplot(x ~  df.pcorr[,i],
          xlab=colnames(df.pcorr)[i],ylab="target.variable", colramp =  palet,
          main=main, colorcut = colorcut, type="g",panel = pfun) 
          })
          print(do.call(grid.arrange, c(plt, ncol=2)))
        }
    }
  }
  return(plt)
  }

