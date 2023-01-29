utils::globalVariables(c("..", "N"))

.onLoad <- function(libname, pkgname) {
  op <- options(
    torchaudio.loader = tuneR_loader,
    torchaudio.loader.name = "tuneR_loader"
  )
}
