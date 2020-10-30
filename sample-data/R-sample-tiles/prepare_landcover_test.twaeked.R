

library(rgdal)
library(terra)
library(eumap)
library(R.utils)
library(lubridate)
library(dplyr)
library(ggplot2)
library(tidyr)
# reading  Croatia data ----
tif1.lst = list.files("/data/eumap/sample-data/R-sample-tiles/9529", pattern=".tif", full.names=TRUE, recursive=TRUE) 
df = readOGR("/data/eumap/sample-data/R-sample-tiles/9529_croatia_landcover_samples.gpkg")
df <- as.data.frame(df)
df$Date = format.Date(as.Date(paste(df$survey_date), format="%Y/%m/%d"), "%Y-%m-%d")
df$row.id = 1:nrow(df)
df$year = year(df$Date)

## stripe ----
begin.tif1.lst = sapply(tif1.lst, function(i){strip_dates.yr(i, type="begin")})
end.tif1.lst = sapply(tif1.lst, function(i){strip_dates.yr(i, type="end")})

## run ----
cores = ifelse(parallel::detectCores()<length(tif1.lst), parallel::detectCores(), length(tif1.lst))
ov.pnts <- parallel::mclapply(1:length(tif1.lst), function(i){ eumap::extract_tif(tif=tif1.lst[i], df, date="Date", date.tif.begin=begin.tif1.lst[i], date.tif.end=end.tif1.lst[i], coords=c("coords.x1","coords.x2")) }, mc.cores=cores)
gc()
ov.pnts = ov.pnts[!sapply(ov.pnts, is.null)]

############################
library(data.table)
commcols <- Reduce(intersect, lapply(ov.pnts, names))
L.dt <- lapply(ov.pnts, function(x) setkeyv(data.table(x), commcols))
cmt <- do.call(cbind, L.dt) 
uq.lst <- unique(colnames(cmt))
cm.tif <- cmt[, .SD, .SDcols = unique(names(cmt))]
df <- as.data.table(df)
cm <- Reduce(merge,list(df,cm.tif))
tt = cbind(cm,df$year)
write.csv(tt, '/home/msheykhmousa/Documents/gitrepo/internal-planning/mlr3/Ensemble/Archive/9529_tif_croatia.csv', row.names = FALSE)
saveRDS(tt, '/home/msheykhmousa/Documents/gitrepo/internal-planning/mlr3/Ensemble/Archive/9529_tif_croatia.rds')


###########################################


## Sweden ----
x = download.file("https://zenodo.org/record/4058447/files/22497_sweden_rasters_gapfilled.zip?download=1", "/data/eumap/sample-data/R-sample-tiles/22497_sweden_rasters_gapfilled.zip")
unzip("/data/eumap/sample-data/R-sample-tiles/22497_sweden_rasters_gapfilled.zip", exdir = "/data/eumap/sample-data/R-sample-tiles/22497/")
x = download.file("https://zenodo.org/record/4058447/files/22497_sweden_landcover_samples.gpkg?download=1", "/data/eumap/sample-data/R-sample-tiles/22497_sweden_landcover_samples.gpkg")
tif2.lst = list.files("/data/eumap/sample-data/R-sample-tiles/22497", pattern=".tif", full.names=TRUE, recursive=TRUE) 
df2 = readOGR("/data/eumap/sample-data/R-sample-tiles/22497_sweden_landcover_samples.gpkg")
df2 <- as.data.frame(df2)
df2$Date = format.Date(as.Date(paste(df2$survey_date), format="%Y/%m/%d"), "%Y-%m-%d")
hist(as.Date(df2$Date), "years")
df2$row.id = 1:nrow(df2)
begin.tif2.lst = sapply(tif2.lst, function(i){strip_dates(i, type="begin")})
end.tif2.lst = sapply(tif2.lst, function(i){strip_dates(i, type="end")})
## overlay
ov2.pnts <- parallel::mclapply(1:length(tif2.lst), function(i){ eumap::extract_tif(tif=tif2.lst[i], df2, date="Date", date.tif.begin=begin.tif2.lst[i], date.tif.end=end.tif2.lst[i], coords=c("coords.x1","coords.x2")) }, mc.cores=cores)
ov2.pnts = ov2.pnts[!sapply(ov2.pnts, is.null)]
ov2.tifs = plyr::join_all(ov2.pnts, by="row.id", type="full")
cm.sweden = plyr::join(df2, ov2.tifs)
saveRDS(cm.sweden, "./sample-data/R-sample-tiles/cm_22497_sweden_landcover_samples.rds")


