## Testing R code

library(mlr)
library(matrixStats)
library(mlbench)
data(BostonHousing, package = "mlbench")
tsk = makeRegrTask(data = BostonHousing, target = "medv")
BostonHousing$chas = as.numeric(BostonHousing$chas)
base = c("regr.rpart", "regr.svm", "regr.ranger")
lrns = lapply(base, makeLearner)
m = makeStackedLearner(base.learners = lrns, predict.type = "response", method = "stack.cv", super.learner = "regr.lm")
tmp = train(m, tsk)
summary(tmp$learner.model$super.model$learner.model)

## Testing 

library(resemble)
library(tidyr)
library(prospectr)
data(NIRsoil)
# Filter the data using the Savitzky and Golay smoothing filter
sg <- savitzkyGolay(NIRsoil$spc, p = 3, w = 11, m = 0) 
# Replace the original spectra with the filtered ones
NIRsoil$spc <- sg
Xu <- NIRsoil$spc[!as.logical(NIRsoil$train),]
Yu <- NIRsoil$CEC[!as.logical(NIRsoil$train)]
Yr <- NIRsoil$CEC[as.logical(NIRsoil$train)]
Xr <- NIRsoil$spc[as.logical(NIRsoil$train),]
Xu <- Xu[!is.na(Yu),]
Xr <- Xr[!is.na(Yr),]
Yu <- Yu[!is.na(Yu)]
Yr <- Yr[!is.na(Yr)]

# Example 1
# A mbl implemented in Ramirez-Lopez et al. (2013, the spectrum-based learner)
ctrl_1.3 <- mblControl(sm = "pls", pcSelection = list("opc", 40), 
                       valMethod = "NNv", progress = FALSE,
                       scaled = FALSE, center = TRUE)
sbl_1.3 <- mbl(Yr = Yr, Xr = Xr, Yu = Yu, Xu = Xu,
               mblCtrl = ctrl_1.3,
               dissUsage = "predictors",
               k = seq(40, 120, by = 10), 
               method = "gpr")
sbl_1.3

ctrl <- mblControl(sm = "pc", pcSelection = list("opc", 40), 
                   valMethod = "NNv", progress = FALSE,
                   scaled = FALSE, center = TRUE)
mbl.p <- mbl(Yr = Yr, Xr = Xr, Yu = Yu, Xu = Xu,
             mblCtrl = ctrl, 
             dissUsage = "none",
             k = seq(40, 120, by = 10), 
             method = "gpr")
mbl.p

## Spatial-cross validation
library(sp)
demo("meuse", echo=FALSE)
str(meuse)
library(mlr)
spatial.task = makeRegrTask(data = meuse@data[,c("zinc","dist","ffreq")], target = "zinc", 
                            coordinates = data.frame(meuse@coords))
learner.rf = makeLearner("regr.ranger")
r1 = makeResampleDesc("SpRepCV", fold = 5, reps = 2)
r2 = makeResampleDesc("RepCV", fold = 5, reps = 2)
set.seed(123)
out1 = resample(learner = learner.rf, task = spatial.task, resampling = r1)
out2 = resample(learner = learner.rf, task = spatial.task, resampling = r2)

