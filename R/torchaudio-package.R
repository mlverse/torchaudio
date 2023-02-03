utils::globalVariables(c("..", "N"))

.onLoad <- function(libname, pkgname) {
  torchaudio.loader <- getOption("torchaudio.loader", default = get("av_loader"))
}
