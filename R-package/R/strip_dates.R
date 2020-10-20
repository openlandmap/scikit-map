#' Strip dates from tifs
#'
#' @param tif.name input tif file
#' @param year.span year span begin end
#' @param type begin or end format
#' @param begin.seasons date (\code{\%m-\%d}) of the start of season
#' @param season.name season names as used in the file names
#' @param timeless if timeless then consider the whole time span
#'
#' @return character or vector of dates formatted as \code{\%Y-\%m-\%d}.
#' 
#' @export
#'
#' @examples
#' tif.name = "R-sample-tiles/9529/2000/landsat_ard_fall_blue_p50.tif"
#' strip_dates(tif.name, type="begin")
#' strip_dates(tif.name, type="end")
strip_dates <- function(tif.name, year.span=c(2000:2020), type=c("begin", "end")[1], begin.seasons=c("12-02", "03-21", "06-25", "09-13", "12-02"), season.name=c("winter", "spring", "summer", "fall"), timeless="timeless"){
  if(length(grep(timeless, tif.name))>0){
    if(type=="begin"){
      x = paste0(year.span[1]-1, "-", begin.seasons[1])
    }
    if(type=="end"){
      x = paste0(year.span[length(year.span)], "-", begin.seasons[5])
    }
  } else {
    s = which(sapply(sapply(season.name[1:(length(begin.seasons)-1)], function(i){ grep(i, tif.name) }), function(i){length(i)>0}))
    if(length(s)==0 & type=="begin") { s <- 1 }
    if(length(s)==0 & type=="end") { s <- length(season.name) }
    y = which(sapply(sapply(year.span, function(i){ grep(i, tif.name, fixed = TRUE) }), function(i){length(i)>0}))
    if(length(y)==0 & type=="begin") { y <- year.span[1] }
    if(length(y)==0 & type=="begin") { y <- year.span[length(year.span)] }
    if(type=="begin"){
      if(s == 1){
        x = paste0(year.span[y]-1, "-", begin.seasons[s])  
      } else {
        x = paste0(year.span[y], "-", begin.seasons[s])
      }
    }
    if(type=="end"){
      x = paste0(year.span[y], "-", begin.seasons[s+1])
    }
  }
  return(x)
}