## Download and prepare sample tiles for spacetime ML
## tom.hengl@opengeohub.org

library(rgdal)
library(terra)
library(eumap)
library(R.utils)

## Croatia ----
x = download.file("https://zenodo.org/record/4058447/files/9529_croatia_rasters_gapfilled.zip?download=1", "/data/eumap/sample-data/R-sample-tiles/9529_croatia_rasters_gapfilled.zip")
unzip("/data/eumap/sample-data/R-sample-tiles/9529_croatia_rasters_gapfilled.zip", exdir = "/data/eumap/sample-data/R-sample-tiles/9529/")

tif1.lst = list.files("/data/eumap/sample-data/R-sample-tiles/9529", pattern=".tif", full.names=TRUE, recursive=TRUE) 
## 1745
year.span = c(2000:2020)

x = download.file("https://zenodo.org/record/4058447/files/9529_croatia_landcover_samples.gpkg?download=1", "/data/eumap/sample-data/R-sample-tiles/9529_croatia_landcover_samples.gpkg")

df = readOGR("/data/eumap/sample-data/R-sample-tiles/9529_croatia_landcover_samples.gpkg")
df <- as.data.frame(df)
df$Date = format.Date(as.Date(paste(df$survey_date), format="%Y/%m/%d"), "%Y-%m-%d")
head(df)
hist(as.Date(df$Date), "years")
summary(as.factor(df$Date))
df$row.id = 1:nrow(df)
## get dates based on the file name
begin.tif1.lst = sapply(tif1.lst, function(i){strip_dates(i, type="begin")})
#begin.tif1.lstM = as.Date(paste0(substr(begin.tif1.lst-1, 1, 4), "-12-12"))
end.tif1.lst = sapply(tif1.lst, function(i){strip_dates(i, type="end")})
write.csv(data.frame(tif=basename(tif1.lst), dirname(tif1.lst), begin.tif1.lst, end.tif1.lst), "/data/eumap/sample-data/R-sample-tiles/9529_tif_samples.csv", row.names = FALSE)

## test
x = extract_tif(tif=tif1.lst[1], df, date="Date", date.tif.begin=begin.tif1.lst[1], date.tif.end=end.tif1.lst[1], coords=c("coords.x1","coords.x2"))
x = extract_tif(tif=tif1.lst[43], df, date="Date", date.tif.begin=begin.tif1.lst[43], date.tif.end=end.tif1.lst[43], coords=c("coords.x1","coords.x2"))
## run in parallel
cores = ifelse(parallel::detectCores()<length(tif1.lst), parallel::detectCores(), length(tif1.lst))
#library(snowfall)
#sfInit(parallel=TRUE, cpus=cores)
#sfExport("df", "tif1.lst", "begin.tif1.lst", "end.tif1.lst")
#sfLibrary(eumap)
#sfLibrary(terra)
#sfLibrary(tools)
#ov.pnts <- sfClusterApplyLB(1:length(tif1.lst), function(i){ eumap::extract_tif(tif=tif1.lst[i], df, date="Date", date.tif.begin=begin.tif1.lst[i], date.tif.end=end.tif1.lst[i], coords=c("coords.x1","coords.x2")) })
#sfStop()
ov.pnts <- parallel::mclapply(1:length(tif1.lst), function(i){ eumap::extract_tif(tif=tif1.lst[i], df, date="Date", date.tif.begin=begin.tif1.lst[i], date.tif.end=end.tif1.lst[i], coords=c("coords.x1","coords.x2")) }, mc.cores=cores)
gc()
ov.pnts = ov.pnts[!sapply(ov.pnts, is.null)]
ov.tifs = plyr::join_all(ov.pnts, by="row.id", type="full")
## subset to complete cases:
str(ov.tifs)
cm.croatia = plyr::join(df, ov.tifs)
head(cm.croatia)
hist(cm.croatia$landsat_ard_summer_green_p50)
hist(cm.croatia$dtm_elevation)
saveRDS(cm.croatia, "./sample-data/R-sample-tiles/cm_9529_croatia_landcover_samples.rds")

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
