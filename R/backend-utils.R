#' List Available Audio Backends
#'
#' @return character vector with the list of available backends.
#'
#' @export
backend_utils_list_audio_backends <- function() {
  backends = c()
  if(is_module_available('soundfile')) backends = c(backends, 'soundfile')
  if(is_module_available('sox')) backends = c(backends, 'sox')
  if(is_module_available('tuneR')) backends = c(backends, 'tuneR')

  return(backends)
}
