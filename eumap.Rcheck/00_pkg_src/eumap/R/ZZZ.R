.onAttach <- function(lib, pkg)  {
  packageStartupMessage("version: ", utils::packageDescription("eumap", field="Version"), appendLF = TRUE)
}
