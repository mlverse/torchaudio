#' @keywords internal
tuneR_read_mp3_or_wav <- function(filepath, from = 0, to = Inf, unit = "samples") {
  full_lenght = from == 0 & is.infinite(to)

  file_ext <- tools::file_ext(filepath)
  unit <- unit[1]
  if(file_ext == "mp3") {
    if(full_lenght) {
      wave_obj <- monitoR::readMP3(filepath)
    } else {
      info <- info(filepath)
      to_ <- to
      from_ <- from
      duration <- info$num_frames
      if(unit == "samples") {
        from_ <-max(1, from_)/info$sample_rate
        to_ <- to_/info$sample_rate
        duration <- duration/info$sample_rate
      }
      to_ <- min(to_, duration)
      to_ <- max(to_, from_ + 0.015)
      to_ <- 0.05 + to_*1.01
      wave_obj <- monitoR::readMP3(filepath, from = from_, to = to_)
    }

    if(from > 0 | is.finite(to))
      wave_obj <- tuneR::extractWave(wave_obj, from = unit=="samples", to = to - from, xunit = unit)
  } else if(file_ext == "wav") {
    if(unit == "time") unit <- "seconds"
    if(unit %in% c("samples", "sample")) {
      to <- to - 1
      from <- max(1, from)
    }
    wave_obj <- tuneR::readWave(filepath, from = from, to = to, unit = unit)
  } else {
    runtime_error(glue::glue("Only .mp3 and .wav formats are supported. Got {file_ext}."))
  }
  return(wave_obj)
}



#' tuneR_loader
#'
#' Load an audio located at 'filepath' using tuneR package.
#'
#' @param filepath (str) path to the audio file.
#' @param offset (num) the sample (or the second if unit = 'time') where the audio should start.
#' @param duration (num) how many samples (or how many seconds if unit = 'time') should be extracted.
#' @param unit (str) 'samples' or 'time'
#'
#' @export
tuneR_loader <- function(
  filepath,
  offset = 0L,
  duration = 0L,
  unit = c("samples", "time")
){
  package_required("tuneR")
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


