
pfun <- function(x,y, ...){
  panel.xyplot(x, y, ...)
  panel.hexbinplot(x,y, ...)  
  panel.abline(coef = c(0,1), col="black", size = 0.25, lwd = 2)
  return(pfun)
}

#' Plot Spatial Matrix,
#' @description Dedicated function to plot diagnostic graphs resulted from `train_spm ` and `predict_spm`,
#' @param x Object that contains observational data, if x <= 500 observation, then `plot_spm` generates a density plot, regardless of the other arguments that is passed,
#' @param y Object that contains predicted data,
#' @param z Object that contains variable importance data to be used in partial dependency plot,
#' @param Vim Object that contains named numeric variable importance,
#' @param main Title of the plot,
#' @param palet Default values is palet=colorRampPalette(c("wheat2","yellow" ,"red","red3","orchid","orchid4),
#' @param colorcut Default value is colorcut = c(0,0.01,0.03,0.07,0.15,0.25,0.5,0.75,1) ,
#' @param xbins Number of bins, default value is 50,
#' @param gtype Graphical type; user can choose among: gtype = c("accuracy", "correlation","var.imp")."accuracy" provides accuracy plot for a regression matrix. "correlation" provides partial correlation plot for the regression matrix. "var.imp" provides a graph of top 10 percent of the most informative features,
#' @param gmode Graphical mode; in case `gtype=accuracy` gives user five options for representation of the accuracy plot as following: c("root","log10","norm","log2","nat"). "root" shows root square representation of the data,"norm" represent normalized representation of the data, "nat" represent natural  representation of the data,
#' @param aspect Default values is aspect = 1,
#' @param ... Other arguments that can be passed on to \code{hexbin}.
#' 
#' @export
#' @author  \href{https://opengeohub.org/people/mohammadreza-sheykhmousa}{Mohammadreza Sheykhmousa}
#' @examples
#' \dontrun{
#' plt = eumap::plot_spm(df, gmode  = "norm" , gtype = "var.imp")
#' }
plot_spm <- function( x = NULL, y = NULL , z = NULL, Vim = NULL, main = NULL, palet  = NULL, colorcut = NULL, xbins = 60 , gtype = c("accuracy", "correlation","var.imp"), gmode = c("root","log10","norm","log2","nat"), aspect = 1, ...){
  if(gtype == "var.imp" && !is.null(Vim)){
    plt = raster:: barplot(sort((Vim),decreasing = TRUE), horiz = TRUE, las = 1)
    title(main = "variable importance", font.main = 4)
  }
   else if(gtype == "var.imp" && is.null(Vim)){
      stop('varibale importance is missing!')
    }
  if(is.null(x) || is.null(y)) {
    stop("x and or y are missing")  #For version eumap 0.0.4 different scenarios for df.tr, x, y will be added.
    # x = df.tr[1]
    # y = df.tr[2]
    # message('x and y are missing, first and second columns of data frame is used as x and y')
  } else{
    df = data.frame(x,y)
    colnames(df)[2] <- 'y'
    ccc <- signif( ccc(df, x, y)$.estimate, digits=3)
    RMSE <- signif( rmse(df, x, y)$.estimate, digits=3)
    # x = as.numeric(df.tr[,target.variable])
    # y = pred.v[[1]]
    # z = valu.imp
  }

  if(is.null(colorcut)){
    colorcut = c(0,0.01,0.03,0.07,0.15,0.25,0.5,0.75,1)  
  }
  if(is.null(palet)){
    palet=grDevices::colorRampPalette(c("wheat2","yellow" ,"red","red3","orchid","orchid4") )
  }
  if(length(x) <= 500 ) {
    plt <-  lattice::xyplot(x ~ y, asp = 1, 
                            par.settings = list(plot.symbol = list(alpha(col="black",0.6), fill=alpha("red", 0.6), pch=21, cex=0.6)), 
                            scales = list(x=list(log=TRUE, equispaced.log=FALSE), y=list(log=TRUE, equispaced.log=FALSE)), main = paste0(expression(ccc) ,": ",  ccc, "  RMSE: ", RMSE)
                            , ylab="measured", xlab="predicted")
    plt
    print(plt)
    message('Because of the LOW number of observations a density plot is displayed.')
  } else {
    if(gtype == "accuracy"){
      if(is.null(main)){
        main = paste0(expression(ccc) ,": ",  ccc, "  RMSE: ", RMSE)
      }
      if(gmode == "norm" | missing(gmode) | is.null(gmode) ){
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
        aspect = aspect, main=main[1], colorcut = colorcut, type="g", panel =  function(x,y){
          panel.xyplot(x, y)
          panel.hexbinplot(x,y)
          panel.abline(coef = c(0,1), col="black", size = 0.25, lwd = 2)
        }
      ) 
      print(plt)
    }
    if(gtype == "correlation" & is.null(z)) {
      stop('z has to be provided')
    }
    if(gtype == "correlation" & !is.null(z)) {
      df.pcor = data.frame(x, z)
      pcorr =pcor(df.pcor)
      main = "Partial correlation"
      #now we have 2 varimp see how to generate the maps automatically
      if(length(vlp) == 1){
        plt <- hexbinplot(df.pcor[,"target.variable"] ~ df.pcor[,2],
                          mincnt = 1,xbins=xbins,xlab=colnames(df.pcor)[2],ylab="target.variable",
                          colramp =  palet, main=main, colorcut = colorcut, type="g",panel =  function(x,y){
                            panel.xyplot(x, y)
                            panel.hexbinplot(x,y)
                            panel.abline(coef = c(0,1), col="black", size = 0.25, lwd = 2)
                          }
        ) 
      }
      if(length(vlp) > 1){
        df.pcorr = df.pcor
        df.pcorr[,1] <- NULL
        x = df.pcor[,"target.variable"]
        plt <- lapply(1:length(vlp), function(i) {
          hexbinplot(x ~  df.pcorr[,i],
                     xlab=colnames(df.pcorr)[i],ylab="target.variable", colramp =  palet,
                     main=main, colorcut = colorcut, type="g",panel =  function(x,y){
                       panel.xyplot(x, y)
                       panel.hexbinplot(x,y)
                       panel.abline(coef = c(0,1), col="black", size = 0.25, lwd = 2)
                     }
          ) 
        }
        )
        print(do.call(grid.arrange, c(plt, ncol=2)))
      }
    }
  }
  return(plt)
}


