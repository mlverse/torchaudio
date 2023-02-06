
#' tuneR_loader
#'
#' @keywords internal
tuneR_loader <- function(
    filepath,
    offset = 0L,
    duration = Inf,
    unit = "samples") {
  rlang::check_installed("tuneR")

  file_ext <- tools::file_ext(filepath)
  valid <- c("mp3", "wav")
  validate_audio_extension(file_ext, valid, "tuneR")

  from <- offset
  to <- offset + duration

  unit <- unit[1]
  if (file_ext == "mp3") {
    wave_obj <- tuneR::readMP3(filepath)
    if (from > 0 | is.finite(to)) {
      wave_obj <- tuneR::extractWave(wave_obj, from = unit == "samples", to = to - from, xunit = unit)
    }
  } else if (file_ext == "wav") {
    if (unit == "time") unit <- "seconds"
    if (unit %in% c("samples", "sample")) {
      to <- to - 1
      from <- max(1, from)
    }
    wave_obj <- tuneR::readWave(filepath, from = from, to = to, unit = unit)
  }
  return(wave_obj)
}
