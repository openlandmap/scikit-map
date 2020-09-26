
rm(list=ls())
q()

load(".RData")
setwd("/home/msheykhmousa/Documents/gitrepo/internal-planning/mlr3/Ensemble/")
source('train.spm.fnc.R')
source('predict.spm.fnc.R')
source('accuracy.plot.spm.fnc.R')


library("mlr3verse")
library("bbotk")
library("ggplot2")
library("mltools")
library("data.table")
library("mlr3fselect")
library("FSelectorRcpp")
library("future")
library("future.apply")
library("magrittr")
library("progress")
library("mlr3spatiotempcv")
library("sp")
library("landmap")  
library("GSIF")
library("dplyr")
library("EnvStats")
library("grid")
library("hexbin")
library("BBmisc")
library("lattice")
library("MASS")
library("gridExtra")
library("MLmetrics")
library("yardstick")
library("plotKML")
library("latticeExtra")
library("devtools")
# pskgs <- c("mlr3verse","EnvStats","grid","hexbin","BBmisc","lattice","MASS","gridExtra","MLmetrics","yardstick","plotKML","latticeExtra","magick","devtools","dplyr", "GSIF" ,"landmap", "sp", "mlr3spatiotempcv", "magrittr" ,"future" , "future.apply",  "FSelectorRcpp", "mlr3fselect", "data.table", "mltools", "ggplot2", "bbotk")
# install.packages(pskgs, dependencies = TRUE)

# Edgeroi Demo ----
data(edgeroi)
edgeroi <- na.omit(edgeroi)
edgeroi <- left_join(edgeroi$horizons,edgeroi$sites,"SOURCEID")
edgeroi <- na.omit(edgeroi)
edgeroi = one_hot(as.data.table(edgeroi), cols = c("TAXGAUC","HZDUSD"))
edgeroi$NOTEOBS = as.numeric(as.factor(edgeroi$NOTEOBS))
edgeroi$SOURCEID <- NULL
edgeroi$NOTEOBS <- NULL
df <- edgeroi
target.variable <- "ORCDRC"

# Meuse Demo ----
data(meuse)
df <- meuse
df <- na.omit(meuse[,])
crs = "+proj=lcc +lat_1=40.66666666666666 +lat_2=41.03333333333333 +lat_0=40.16666666666666 +lon_0=-74 +x_0=300000 +y_0=0 +datum=NAD83 +units=us-ft +no_defs"

#classif
target.variable = "lime"

#regression
df <- na.omit(meuse[,c("lead","soil","dist","elev")])

target.variable = "lead"

# define generic var ----
smp_size <- floor(0.5 * nrow(df))
set.seed(123)
train_ind <- sample(seq_len(nrow(df)), size = smp_size)
df.tr <- df[train_ind, ]
df.ts <- df[-train_ind, ]
folds = 2
n_evals = 1

# plot var ----
colramp. = colorRampPalette(c("wheat2","red3"))
xbins. = 30

# fnc for data preparation ----
df.meuse = getmeuse()
df.edgeroi = getedgeroi()
  

# MODELS ----
train.model = train.spm(df.tr, target.variable = target.variable, parallel = TRUE, var.ens = TRUE, folds = folds ,n_evals = n_evals, plot.workflow = TRUE,agg = TRUE ,crs = crs )

predict.variable = predict.spm(df.ts,target.variable = target.variable, train.model,crs = crs)

accuracy.plot.spm(x = df.ts[,target.variable], y = predict.variable)



  