#' Extract points from a GeoTIFF using spatiotemporal overlay
#'
#' @param tif filename,
#' @param df data frame with coordinates and dates,
#' @param date date column in \code{df},
#' @param date.tif.begin reference begin date for the tif filename,
#' @param date.tif.end reference end date for the tif filename,
#' @param coords coordinate columns e.g. \code{"x", "y"},
#' @param crs projection system for the coordinates,
#' @param format.date default formatting date,
#'
#' @return list of data frames with row.ids and results of overlay
#' 
#' @details Extends the \code{terra::extract} functionality. 
#' Works best if the GeoTIFF are Cloud-Optimized and located on SSD or similar.
#' 
#' @export 
#' 
#' @examples
#' \dontrun{
#' library(terra)
#' tif.name = "R-sample-tiles/9529/2000/landsat_ard_fall_blue_p50.tif"
#' strip_dates(tif.name, type="begin")
#' tif1.lst = list.files("R-sample-tiles/9529", pattern=".tif", full.names=TRUE, recursive=TRUE) 
#' year.span = c(2000:2020)
#' df = readOGR("/data/eumap/sample-data/R-sample-tiles/9529_croatia_landcover_samples.gpkg")
#' df <- as.data.frame(df)
#' df$Date = format.Date(as.Date(paste(df$survey_date), format="%Y/%m/%d"), "%Y-%m-%d")
#' df$row.id = 1:nrow(df)
#' begin.tif1.lst = sapply(tif1.lst, function(i){strip_years(i, type="begin")})
#' end.tif1.lst = sapply(tif1.lst, function(i){strip_years(i, type="end")})
#' x = extract_tif(tif=tif1.lst, df, date="Date", begin=begin, end=end, coords=c("x","y"))
#' }
extract_tif <- function(tif, df, date, date.tif.begin, date.tif.end, coords=c("x","y"), crs="+proj=epsg:3035", format.date="%Y-%m-%d"){
  if(any(!coords %in% colnames(df))){
    stop(paste("Coordinate columns", coords, "could not be found"))
  }
  if(is.character(date) & length(date)==1 & date %in% colnames(df)){
    date = as.Date(df[,date], format=format.date, origin="1970-01-01")
  } else {
    stop(paste("Column name", date, "could not be found in the dataframe"))
  }
  if(missing(date.tif.end)){ 
    date.tif.end = date.tif.begin
  }
  sel <- date <= as.Date(date.tif.end, format=format.date, origin="1970-01-01") & date >= as.Date(date.tif.begin, format=format.date, origin="1970-01-01")
  if(sum(sel)>0){
    pnts = as.matrix(df[sel, coords])
    attr(pnts, "dimnames")[[2]] = c("x","y") 
    df.v = terra::vect(pnts, crs=crs)
    if(file.exists(tif)){
      ov = terra::extract(terra::rast(tif), df.v)
    } else {
      ov = matrix(nrow=length(df.v), ncol=2)
    }
    ov = as.data.frame(ov)
    names(ov) = c("ID", tools::file_path_sans_ext(basename(tif)))
    ov$row.id = which(sel)
    ov$ID = NULL
    return(ov)
  }
}