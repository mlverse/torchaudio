utils::globalVariables(c("..", "N"))

.onLoad <- function(libname, pkgname) {
  op <- options(
    torchaudio.loader = av_loader,
    torchaudio.loader.name = "av_loader"
  )
}
