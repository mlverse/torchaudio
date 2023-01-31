
#' av_loader
#'
#' @keywords internal

av_loader <- function(
    filepath,
    offset = 0L,
    duration = Inf,
    unit = "samples") {

  from <- offset
  to <- offset + duration
  info <- torchaudio_info(filepath)

  from_secs <- if(unit == "samples") from / info$sample_rate else from
  to_secs <- if(unit == "samples") to / info$sample_rate else to

  to_secs <- max(to_secs, from_secs + 0.015) + 0.05
  av_obj <- av::read_audio_bin(audio = filepath, start_time = from_secs, end_time = to_secs)

  channels <- attr(av_obj, "channels")
  samples <- length(av_obj)
  dim(av_obj) <- c(channels, samples / channels)
  attrs <- attributes(av_obj)

  if (unit == "samples") {
    len <- min(to - from, samples)
    av_obj <- av_obj[channels, 1:(len), drop = FALSE]
    attrs$dim <- c(channels, len)
    attributes(av_obj) <- attrs
  }
  class(av_obj) <- c("av", class(av_obj))
  return(av_obj)
}
