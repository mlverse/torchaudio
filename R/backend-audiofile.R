#' @keywords internal
audiofile_read_wav <- function(filepath, from = 0, to = Inf, unit = "samples") {
  file_ext <- tools::file_ext(filepath)
  if(!file_ext %in% "wav")
    runtime_error(glue::glue("audiofile_loader supports .wav formats only. Got {file_ext}."))
  unit <- unit[1]
  if(is.finite(to) | from > 0)
    not_implemented_error("interval load with audiofile_loader not implemented yet.")
  to <- 99999
  wave_obj <- audiofile_read_wav_cpp(filepath, from = from, to = to, unit = unit)
  class(wave_obj) <- c("audiofile", class(wave_obj))
  wave_obj
}
