pkgname <- "eumap"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
base::assign(".ExTimings", "eumap-Ex.timings", pos = 'CheckExEnv')
base::cat("name\tuser\tsystem\telapsed\n", file=base::get(".ExTimings", pos = 'CheckExEnv'))
base::assign(".format_ptime",
function(x) {
  if(!is.na(x[4L])) x[1L] <- x[1L] + x[4L]
  if(!is.na(x[5L])) x[2L] <- x[2L] + x[5L]
  options(OutDec = '.')
  format(x[1L:3L], digits = 7L)
},
pos = 'CheckExEnv')

### * </HEADER>
library('eumap')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("extract_tif")
### * extract_tif

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: extract_tif
### Title: Extract points from a GeoTIFF using spatiotemporal overlay
### Aliases: extract_tif

### ** Examples

## Not run: 
##D library(terra)
##D tif.name = "R-sample-tiles/9529/2000/landsat_ard_fall_blue_p50.tif"
##D strip_dates(tif.name, type="begin")
##D tif1.lst = list.files("R-sample-tiles/9529", pattern=".tif", full.names=TRUE, recursive=TRUE) 
##D year.span = c(2000:2020)
##D df = readOGR("/data/eumap/sample-data/R-sample-tiles/9529_croatia_landcover_samples.gpkg")
##D df <- as.data.frame(df)
##D df$Date = format.Date(as.Date(paste(df$survey_date), format="%Y/%m/%d"), "%Y-%m-%d")
##D df$row.id = 1:nrow(df)
##D begin.tif1.lst = sapply(tif1.lst, function(i){strip_years(i, type="begin")})
##D end.tif1.lst = sapply(tif1.lst, function(i){strip_years(i, type="end")})
##D x = extract_tif(tif=tif1.lst, df, date="Date", begin=begin, end=end, coords=c("x","y"))
## End(Not run)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("extract_tif", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("plot_cm")
### * plot_cm

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: plot_cm
### Title: Function for plotting confusion matrices
### Aliases: plot_cm

### ** Examples

## Not run: 
##D library(caret)
##D set.seed(23)
##D pred <- factor(sample(1:7,100,replace=T))
##D ref<- factor(sample(1:7,100,replace=T))
##D cm <- caret::confusionMatrix(pred,ref)
##D plot_cm(cm)
## End(Not run)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("plot_cm", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("plot_spm")
### * plot_spm

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: plot_spm
### Title: Plot Spatial Matrix,
### Aliases: plot_spm

### ** Examples

## Not run: 
##D plt = eumap::plot_spm(df, gmode  = "norm" , gtype = "var.imp")
## End(Not run)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("plot_spm", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("predict_spm")
### * predict_spm

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: predict_spm
### Title: Predict_spm
### Aliases: predict_spm

### ** Examples

## Not run: 
##D predict.variable = eumap::predict_spm(object, newdata)
## End(Not run)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("predict_spm", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("strip_dates")
### * strip_dates

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: strip_dates
### Title: Strip dates from tifs
### Aliases: strip_dates

### ** Examples

## Not run: 
##D tif.name = "R-sample-tiles/9529/2000/landsat_ard_fall_blue_p50.tif"
##D strip_dates(tif.name, type="begin")
##D strip_dates(tif.name, type="end")
## End(Not run)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("strip_dates", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("strip_years")
### * strip_years

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: strip_years
### Title: Strip dates from tifs
### Aliases: strip_years

### ** Examples

## Not run: 
##D tif.name = "R-sample-tiles/9529/2000/landsat_ard_fall_blue_p50.tif"
##D strip_years(tif.name, type="begin")
##D strip_years(tif.name, type="end")
## End(Not run)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("strip_years", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("train_spm")
### * train_spm

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: train_spm
### Title: Train a spatial prediction model, from a (spatial) matrix, using
###   ensemble machine learning,
### Aliases: train_spm

### ** Examples

## Not run: 
##D  
##D ## Meuse Demo
##D library(sp)
##D library(mlr3verse)
##D library(mlr3spatiotempcv)
##D library(checkmate)
##D library(future)
##D library(progress)
##D library(scales)
##D library(eumap)
##D demo(meuse, echo=FALSE)
##D df <- as.data.frame(meuse)
##D df.grid <- as.data.frame(meuse.grid)
##D df = na.omit(df[,])
##D df.grid = na.omit(df.grid[,])
##D smp_size <- floor(0.8 * nrow(df))
##D set.seed(123)
##D train_ind <- sample(seq_len(nrow(df)), size = smp_size)
##D df.tr <- df[, c("x","y","dist","ffreq","soil","lead")]
##D df.ts <- df.grid[, c("x","y","dist","ffreq","soil")]
##D newdata <-df.ts
##D tr = eumap::train_spm(df.tr, target.variable = "lead",crs )
##D train_model= tr[[1]]
##D #var.imp = tr[[2]]
##D summary = tr[[3]]
##D response = tr[[4]]
##D vlp = tr[[5]]
##D target = tr[[6]]
##D predict.variable = eumap::predict_spm(train_model, newdata)
##D pred.v = predict.variable[[1]]
##D valu.imp= predict.variable[[2]]
##D plt = eumap::plot_spm(df, gmode  = "norm" , gtype = "var.imp")
##D df.ts$leadp = predict.variable
##D coordinates(df.ts) <- ~x+y
##D proj4string(df.ts) <- CRS("+init=epsg:28992")
##D gridded(df.ts) = TRUE
##D ## regression grid 
##D #make a spatial prediction map 
##D plot(df.ts[,"leadp"])
##D points(meuse, pch="+")
## End(Not run)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("train_spm", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
