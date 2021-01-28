#' @keywords internal
av_read_mp3_or_wav <- function(filepath, from = 0, to = Inf, unit = "samples") {
  file_ext <- tools::file_ext(filepath)
  unit <- unit[1]
  info <- info(filepath)
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
  class(av_obj) <- c("av", class(av_obj))
  return(av_obj)
}



#' @export
av_loader <- function(
  filepath,
  offset = 0L,
  duration = 0L,
  unit = c("samples", "time")
){
  package_required("av")
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

