## usethis namespace: start
#' @importFrom Rcpp sourceCpp
#' @useDynLib torchaudio, .registration = TRUE
## usethis namespace: end
NULL

utils::globalVariables(c("..", "N"))


.onLoad <- function(libname, pkgname) {
  op <- options(
    torchaudio.loader = av_loader
  )
}
