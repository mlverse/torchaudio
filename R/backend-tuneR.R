
#' tuneR_loader
#'
#' @keywords internal
tuneR_loader <- function(
    filepath,
    offset = 0L,
    duration = Inf,
    unit = "samples") {
  package_required("tuneR")

  from <- offset
  to <- offset + duration

  file_ext <- tools::file_ext(filepath)
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
  } else {
    runtime_error(glue::glue("Only .mp3 and .wav formats are supported. Got {file_ext}."))
  }
  return(wave_obj)
}
