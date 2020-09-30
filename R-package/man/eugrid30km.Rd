\docType{data}
\name{eugrid30km}
\alias{eugrid30km}
\title{eugrid30km: Tiling system continental Europe at 30km}
\description{
Tiling system with the cartographic and image coordinates. Can be used to speed up processing of the data.
The \code{eugrid30km} is an object of type \code{SpatialPolygonsDataFrame} with the following columns:
\describe{
  \item{\code{xl}}{numeric; lower left x-map coordinates}
  \item{\code{yl}}{numeric; lower left y-map coordinates}
  \item{\code{xu}}{numeric; upper right x-map coordinates}
  \item{\code{yu}}{numeric; upper right y-map coordinates}
  \item{\code{offst_y}}{numeric; offsets in the image coordinates lines (starting from top left)}
  \item{\code{offst_x}}{numeric; offsets in the image coordinates rows (starting from top left)}
  \item{\code{rgn_dm_y}}{numeric; tile dimensions in image coordinates}
  \item{\code{rgn_dm_x}}{numeric; tile dimensions in image coordinates}
}
}
\examples{
library(sp)
data(eugrid30km)
## Get tile ID:
eugrid30km$ID = row.names(eugrid30km@data)
\dontrun{
library(landmap)
objF = GDALinfo("lcv_landcover.12_pflugmacher2019_c_1m_s0..0m_2014..2016_eumap_epsg3035_v0.1.tif")
tileF.tbl <- landmap::getSpatialTiles(objF, block.x=3e4, return.SpatialPolygons=FALSE)
tileF.pol <- landmap::getSpatialTiles(objF, block.x=3e4, return.SpatialPolygons=TRUE)
tileF.pol <- sp::SpatialPolygonsDataFrame(tileF.pol, tileF.tbl)
rgdal::writeOGR(tileF.pol, "adm_tiling.system_30km_c_1m_s0..0m_2020_eumap_epsg3035_v0.1.gpkg", "tiling.system", "GPKG")
}
}