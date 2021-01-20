tuneR_read_mp3_or_wav <- function(filepath, from = 0, to = Inf, unit = "samples") {
  file_ext <- tools::file_ext(filepath)
  if(file_ext == "mp3") {
    info <- suppressWarnings(audio_info(filepath))
    to_ <- to
    if(unit == "samples") {
      from <-max(1, from)/info$sample_rate
      to_ <- to_/info$sample_rate
    }
    from <-  max(0.01, from)
    to_ <- min(to_, info$duration)
    to_ <- max(to_, from + 0.015)
    to_ <- to_*1.1
    wave_obj <- monitoR::readMP3(filepath, from = from, to = to_)
    if(unit[1] == "seconds") unit <- "time"
    wave_obj <- tuneR::extractWave(wave_obj, from = from, to = to, xunit = unit[1])
  } else if(file_ext == "wav") {
    if(unit[1] == "time") unit <- "seconds"
    wave_obj <- tuneR::readWave(filepath, from = from, to = to, unit = unit[1])
  } else {
    runtime_error(glue::glue("Only .mp3 and .wav formats are supported. Got {file_ext}."))
  }
  return(wave_obj)
}



#' @export
tuneR_loader <- function(
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
  tuneR_read_mp3_or_wav(filepath, from = offset, to = offset + duration, unit = unit)
}

tuneR_info <- function() {}

tuneR_save <- function() {}
