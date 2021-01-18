
pfun <- function(x,y, ...){
  panel.xyplot(x, y, ...)
  panel.hexbinplot(x,y, ...)  
  panel.abline(coef = c(0,1), col = "black", size = 0.25, lwd = 2)
  return(pfun)
}

#' Plot Spatial Matrix,
#' @description Dedicated function to plot diagnostic graphs resulted from `train_spm ` and `predict_spm`,
#'
#' @param x Object that contains observational data, if x <= 500 observation, then `plot_spm` generates a density plot, regardless of the other arguments that is passed,
#' @param y Object that contains predicted data,
#' @param z Object that contains top variable importance values to be used in partial dependency plot,
#' @param Vim Object that contains named numeric variable importance,
#' @param main Title of the plot,
#' @param palet Default values is palet=colorRampPalette(c("wheat2","yellow" ,"red","red3","orchid","orchid4),
#' @param colorcut Default value is colorcut = c(0,0.01,0.03,0.07,0.15,0.25,0.5,0.75,1) ,
#' @param xbins Number of bins, default value is 50,
#' @param gtype Graphical type; user can choose among: gtype = c("accuracy", "correlation","var.imp")."accuracy" provides accuracy plot for a regression matrix. "correlation" provides partial correlation plot for the regression matrix. "var.imp" provides a graph of top 10 percent of the most informative features,
#' @param gmode Graphical mode; in case `gtype = accuracy` gives user five options for representation of the accuracy plot as following: c("root","log10","norm","log2","nat"). "root" shows root square representation of the data,"norm" represent normalized representation of the data, "nat" represent natural  representation of the data. If `gtype = cm`; function for plotting confusion matrix. `plot_spm` returns an annotated heat map inputs are `pred` and `truth`; normally comes as an output of classification models,
#' @param aspect Default values is aspect = 1,
#' @param ... Other arguments that can be passed on to \code{hexbin}.
#' 
#' @export
#' @author  \href{https://opengeohub.org/people/mohammadreza-sheykhmousa}{Mohammadreza Sheykhmousa}
#' @examples
#' \dontrun{
#' plt = eumap::plot_spm(df, gmode  = "norm" , gtype = "var.imp")
#' }
plot_spm <- function(x = NULL, y = NULL , z = NULL, Vim = NULL, main = NULL, palet  = NULL, colorcut = NULL, xbins = 60, gtype = c("accuracy", "correlation", "var.imp", "cm"), gmode = c("root","log10","norm","log2","nat"), aspect = 1, ...){
  
  if (gtype == "cm") {
    # extract the confusion matrix values as data.frame: The code is a modified version of [THIS LINK](https://stackoverflow.com/questions/23891140/r-how-to-visualize-confusion-matrix-using-the-caret-package/60150826#60150826)
    cm = confusionMatrix(x, y) #caret::
    cm_d <- as.data.frame(cm$table)
    cm_d$diag <- cm_d$Prediction == cm_d$Reference # Get the Diagonal
    cm_d$ndiag <- cm_d$Prediction != cm_d$Reference # Off Diagonal     
    cm_d[cm_d == 0] <- NA # Replace 0 with NA for white tiles
    cm_d$Reference <-  reverse.levels(cm_d$Reference) # diagonal starts at top left likert::
    cm_d$ref_freq <- cm_d$Freq * ifelse(is.na(cm_d$diag),-1,1)
    
    # plotting the matrix
    plt <-  ggplot(data = cm_d, aes(x = Prediction , y =  Reference, fill = Freq)) + #ggplot2::
      scale_x_discrete(position = "top") +
      geom_tile( data = cm_d,aes(fill = ref_freq)) +
      scale_fill_gradient2(guide = FALSE,low = "red3",high = "orchid4", midpoint = 0,na.value = 'white') +
      geom_text(aes(label = Freq), color = 'black', size = 3) +
      theme_light() +
      theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
            legend.position = "none",
            panel.border = element_blank(),
            plot.background = element_blank(),
            axis.line = element_blank())
  } else if (gtype != "cm") {
    if (is.null(colorcut)) {
      colorcut = c(0,0.01,0.03,0.07,0.15,0.25,0.5,0.75,1)  
    }
    
    if (is.null(palet)) {
      palet = grDevices::colorRampPalette(c("wheat2","yellow" ,"red","red3","orchid","orchid4") )
    }
    
    if (gtype == "var.imp" ) {
      
      plt = raster::barplot(Vim , horiz = TRUE, las = 1)
      title(main = "variable importance", font.main = 4)
    }
    
    if (gtype == "var.imp" & is.null(Vim)) {
      message('Vim is missing!')
    }
    
    if (gtype != "var.imp" & is.null(x) & is.null(y)) {
      stop('x and or y are missing!')
      #For version eumap 0.0.4 different scenarios for df.tr, x, y will be added.
      # x = df.tr[1]
      # y = df.tr[2]
      # message('x and y are missing, first and second columns of data frame is used as x and y')
    } else if (gtype != "var.imp" & !is.null(x) & !is.null(y)) {
      dff = data.frame(x,y)
      #colnames(dff)[1:2] <- c('x','y')
      CCC <- signif(ccc(dff, x, y)$.estimate, digits = 3)
      RMSE <- signif( rmse(dff, x, y)$.estimate, digits = 3)
      main = paste0(expression(CCC) ,": ",  CCC, "  RMSE: ", RMSE)
    }
    if (gtype != "var.imp" & length(x) <= 500 ) {
      plt <-  lattice::xyplot(x ~ y, asp = 1, 
                              par.settings = list(plot.symbol = list(alpha(col = "black", 0.6), fill = alpha("red", 0.6), pch = 21, cex = 0.6)), main = main
                              , ylab = "measured", xlab = "predicted")# scales = list(x=list(log=TRUE, equispaced.log=FALSE), y=list(log=TRUE, equispaced.log=FALSE)),
      # print(plt)
      # print(plt)
      message('Because of the LOW number of observations a density plot is displayed.')
    } else {
      if (gtype == "accuracy") {
        if (gmode == "norm" | missing(gmode) | is.null(gmode) ) {
          df.x = normalize(x, method = "range", range = c(0, 1))
          df.y = normalize(y, method = "range", range = c(0, 1))
          xlab = expression(italic("0~1 measured"))
          ylab = expression(italic("0~1 predicted"))
          xscale.components = xscale.components.logpower
          yscale.components = yscale.components.logpower
        } else if (gmode == "nat") {
          df.x = x
          df.y = y
          xlab = expression(italic("measured"))
          ylab = expression(italic("predicted (ensemble)"))
          xscale.components = xscale.components.logpower
          yscale.components = yscale.components.logpower
        } else if (gmode == "log10") {
          df.x = exp(x)
          df.y = exp(y)
          xlab = expression(italic("log10 (measured)"))
          ylab = expression(italic("log10 (predicted)"))
          xscale.components = xscale.components.logpower
          yscale.components = yscale.components.log10ticks
        } else if (gmode == "log2") {
          df.x = 2^x
          df.y = 2^y
          xlab = expression(italic("log2 (measured)"))
          ylab = expression(italic("log2 (predicted)"))
          xscale.components = xscale.components.log
          yscale.components = yscale.components.log
        } else if (gmode == "root") {
          df.x = sqrt(x)
          df.y = sqrt(y)
          xlab = expression(sqrt(italic(measured)))
          ylab = expression(sqrt(italic(predicted)))
          xscale.components = xscale.components.subticks
          yscale.components = yscale.components.subticks
        }
        
        plt <- hexbinplot(
          df.x ~ df.y, xbins = xbins, scales = list(x = df.x, y = df.y), 
          xscale.components = xscale.components, yscale.components = yscale.components, 
          mincnt = 1, xlab = xlab , ylab = ylab, inner = 0.2, cex.labels = 1, colramp = palet, 
          aspect = aspect, main = main[1], colorcut = colorcut, type = "g", panel =  function(x,y){
            panel.xyplot(x, y)
            panel.hexbinplot(x,y)
            panel.abline(coef = c(0,1), col = "black", size = 0.25, lwd = 2)
          }
        ) 
        print(plt)
      }
      if (gtype == "correlation" & is.null(z)) {
        stop('z has to be provided')
      }
      if (gtype == "correlation" & !is.null(z)) {
        df.pcor = data.frame(x, z)
        pcorr = pcor(df.pcor)
        main = "Partial correlation"
        #now we have 2 varimp see how to generate the maps automatically
        if (length(vlp) == 1) {
          plt <- hexbinplot(df.pcor[,"target.variable"] ~ df.pcor[,2],
                            mincnt = 1,xbins = xbins,xlab = colnames(df.pcor)[2],ylab = "target.variable",
                            colramp =  palet, main = main, colorcut = colorcut, type = "g",panel =  function(x,y){
                              panel.xyplot(x, y)
                              panel.hexbinplot(x,y)
                              panel.abline(coef = c(0,1), col = "black", size = 0.25, lwd = 2)
                            }
          ) 
        }
        if (length(vlp) > 1) {
          df.pcorr = df.pcor
          df.pcorr[,1] <- NULL
          x = df.pcor[,"target.variable"]
          plt <- lapply(1:length(vlp), function(i){
            hexbinplot(x ~  df.pcorr[,i],
                       xlab = colnames(df.pcorr)[i],ylab = "target.variable", colramp =  palet,
                       main = main, colorcut = colorcut, type = "g",panel =  function(x,y){
                         panel.xyplot(x, y)
                         panel.hexbinplot(x,y)
                         panel.abline(coef = c(0,1), col = "black", size = 0.25, lwd = 2)
                       }
            ) 
          }
          )
          print(do.call(grid.arrange, c(plt, ncol = 2)))
        }
      }
    }
    return(plt)
   }
  }


