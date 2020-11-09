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
##D tif1.lst = list.files("/data/eumap/sample-data/R-sample-tiles/9529", pattern=".tif", full.names=TRUE, recursive=TRUE) 
##D year.span = c(2000:2020)
##D df = readOGR("/data/eumap/sample-data/R-sample-tiles/9529_croatia_landcover_samples.gpkg")
##D df <- as.data.frame(df)
##D df$Date = format.Date(as.Date(paste(df$survey_date), format="%Y/%m/%d"), "%Y-%m-%d")
##D df$row.id = 1:nrow(df)
##D begin.tif1.lst = sapply(tif1.lst, function(i){strip_years(i, type="begin")})
##D end.tif1.lst = sapply(tif1.lst, function(i){strip_years(i, type="end")})
##D x = extract_tif(tif=tif1.lst[43], df, date="Date", date.tif.begin=begin.tif1.lst[43], date.tif.end=end.tif1.lst[43], coords=c("coords.x1","coords.x2"))
## End(Not run)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("extract_tif", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("strip_dates")
### * strip_dates

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: strip_dates
### Title: Strip dates from tifs
### Aliases: strip_dates

### ** Examples

tif.name = "R-sample-tiles/9529/2000/landsat_ard_fall_blue_p50.tif"
strip_dates(tif.name, type="begin")
strip_dates(tif.name, type="end")




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

tif.name = "R-sample-tiles/9529/2000/landsat_ard_fall_blue_p50.tif"
strip_years(tif.name, type="begin")
strip_years(tif.name, type="end")





base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("strip_years", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
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
