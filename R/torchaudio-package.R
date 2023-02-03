utils::globalVariables(c("..", "N"))

.onLoad <- function(libname, pkgname) {
  set_audio_backend(getOption("torchaudio.loader", default ="av_loader"))
}
