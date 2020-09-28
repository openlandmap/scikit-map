#' Extract points from a GeoTIFF using spatiotemporal overlay
#'
#' @param df data frame with coordinates and dates,
#' @param date date column in \code{df},
#' @param tif tif filename,
#' @param date.tif.begin reference begin date for the tif filename,
#' @param date.tif.end reference end date for the tif filename,
#' @param coords coordinate columns e.g. \code{"x", "y"},
#' @param crs projection system for the coordinates,
#'
#' @return data frame with point IDs and results of overlay
#' 
#' @details Extends the \code{terra::extract} functionality. 
#' Works extremely fast if the GeoTIFF are Cloud-Optimized and located on SSD or similar.
#' 
#' @export
#'
#' @examples
#' \dontrun{
#' library(terra)
#' library(dplyr)
#' library(parallel)
#' ## run overlay in parallel:
#' tif.lst = list.files(pattern=".tif", full.names=TRUE) 
#' date.tif.lst = c(2000:2020)
#' cores = ifelse(parallel::detectCores()<length(tif.lst), parallel::detectCores(), length(tif.lst))
#' ov.pnts = parallel::mclapply(1:length(tif.lst), function(i){ 
#'   extract.tif(df, date="Year", tif=tif.lst[i], date.tif.begin=date.tif.lst[i]) }, 
#'   mc.cores = cores )
#' ov.pnts = do.call(rbind, ov.pnts)
#' str(ov.pnts)
#' }
extract.tif <- function(df, date, tif, date.tif.begin, date.tif.end, coords=c("x","y"), crs="+init=epsg:3035"){
  if(missing(date.tif.end)){ 
    date.tif.end = date.tif.begin
  }
  if(is.character(date)){
    date.v = df[,"date"]
  } else {
    date.v <- rep(nrow(df), date)
  }
  sel <- date.v <= date.tif.end & date.v >= date.tif.begin
  if(sum(sel)>0){
    df.v = terra::vect(as.matrix(df[sel,coords]), crs)
    if(file.exists(tif)){
      ov = terra::extract(terra::rast(tif), df.v)
    } else {
      ov = matrix(nrow=length(df.v), ncol=2)
    }
  }
  names(ov)[2] = basename(tif)
  ov$row.id = which(sel)
  ov$date = date
  return(ov)
}
