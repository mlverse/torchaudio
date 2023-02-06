
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
  sample_rate <- av::av_media_info(filepath)$audio$sample_rate

  from_secs <- if(unit == "samples") from / sample_rate else from
  to_secs <- if(unit == "samples") to / sample_rate else to

  to_secs <- max(to_secs, from_secs + 0.015) + 0.05
  av_obj <- av::read_audio_bin(audio = filepath, start_time = from_secs, end_time = to_secs)

  channels <- attr(av_obj, "channels")
  samples <- length(av_obj)
  samples_per_channel <- samples / channels
  dim(av_obj) <- c(channels, samples_per_channel)
  attrs <- attributes(av_obj)

  if (unit == "samples") {
    len <- min(to - from, samples_per_channel)
    av_obj <- av_obj[ , 1:len, drop = FALSE]
    attrs$dim <- c(channels, len)
    attributes(av_obj) <- attrs
  }

  class(av_obj) <- c("av", class(av_obj))
  return(av_obj)
}
