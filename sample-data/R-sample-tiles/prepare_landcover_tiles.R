## Download and prepare sample tiles for spacetime ML
## tom.hengl@opengeohub.org

library(rgdal)
library(terra)
library(eumap)
library(R.utils)

x = download.file("https://zenodo.org/record/4058447/files/9529_croatia_rasters_gapfilled.zip?download=1", paste0(getwd(), "/sample-data/R-sample-tiles/9529_croatia_rasters_gapfilled.zip"))
unzip("./sample-data/R-sample-tiles/9529_croatia_rasters_gapfilled.zip", exdir = "./sample-data/R-sample-tiles/9529/")

tif.lst = list.files(paste0(getwd(), "/sample-data/R-sample-tiles/9529"), pattern=".tif", full.names=TRUE, recursive=TRUE) 
date.tif.lst = c(2000:2020)
cores = ifelse(parallel::detectCores()<length(tif.lst), parallel::detectCores(), length(tif.lst))
df = readOGR("./sample-data/R-sample-tiles/")
df <- as.data.frame(df)
ov.pnts = parallel::mclapply(1:length(tif.lst), function(i){ 
  extract.tif(df, date="Year", tif=tif.lst[i], date.tif.begin=date.tif.lst[i]) }, 
  mc.cores = cores )
ov.pnts = do.call(rbind, ov.pnts)
str(ov.pnts)
}