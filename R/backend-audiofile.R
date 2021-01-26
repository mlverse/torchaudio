audiofile_read_wav <- function(filepath, from = 0, to = Inf, unit = "samples") {
  file_ext <- tools::file_ext(filepath)
  if(!file_ext %in% "wav") runtime_error(glue::glue("audiofile_loader supports .wav formats only. Got {file_ext}."))
  unit <- unit[1]
  to <- 99999
  wave_obj <- audiofile_read_wav_cpp(filepath, from = from, to = to, unit = unit)
  class(wave_obj) <- c("audiofile", class(wave_obj))
  wave_obj
}

#' @export
audiofile_loader <- function(
  filepath,
  offset = 0L,
  duration = 0L,
  unit = c("samples", "time"),
  normalization = TRUE,
  signalinfo = NULL,
  encodinginfo = NULL,
  filetype = NULL
){
  if(is.null(normalization)) value_error('Argument "normalization" is missing. Should it be set to `TRUE`?')
  if(!is.null(signalinfo)) value_warning('Argument "signalinfo" is meaningful for sox backend only and will be ignored.')
  if(!is.null(encodinginfo)) value_error('Argument "encodinginfo" is meaningful for sox backend only and will be ignored.')

  filepath = as.character(filepath)

  # check if valid file
  if(!fs::is_file(filepath))
    runtime_error(glue::glue("{filepath} not found or is a directory"))

  if(duration < -1)
    value_error("Expected value for num_samples -1 (entire file) or >=0")
  if(duration %in% c(-1, 0))
    duration = Inf
  if(offset < 0)
    value_error("Expected positive offset value")

  # load audio file
  audiofile_read_wav(filepath, from = offset, to = offset + duration, unit = unit)
}
