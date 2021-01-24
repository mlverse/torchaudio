av_read_mp3_or_wav <- function(filepath, from = 0, to = Inf, unit = "samples") {
  file_ext <- tools::file_ext(filepath)
  unit <- unit[1]
  info <- suppressWarnings(audio_info(filepath))
  to_ <- to
  from_ <- from
  if(unit == "samples") {
    from_ <- from_/info$sample_rate
    to_ <- to_/info$sample_rate
  }

  to_ <- max(to_, from_ + 0.015) + 0.05
  av_obj <- av::read_audio_bin(audio = filepath, start_time = from_, end_time = to_)
  channels <- attr(av_obj, "channels")
  samples <- length(av_obj)
  dim(av_obj) <- c(channels, samples/channels)
  attrs <- attributes(av_obj)
  if(unit == "samples") {
    len <- min(to - from, samples)
    av_obj <- av_obj[channels, 1:(len), drop = FALSE]
    attrs$dim <- c(channels, len)
    attributes(av_obj) <- attrs
  }

  return(av_obj)
}



#' @export
av_loader <- function(
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
  av_read_mp3_or_wav(filepath, from = offset, to = offset + duration, unit = unit)
}

av_info <- function() {}

av_save <- function() {}
