#' @keywords internal
av_read_mp3_or_wav <- function(filepath, from = 0, to = Inf, unit = "samples") {
  file_ext <- tools::file_ext(filepath)
  unit <- unit[1]
  info <- torchaudio_info(filepath)
  to_ <- to
  from_ <- from
  if (unit == "samples") {
    from_ <- from_ / info$sample_rate
    to_ <- to_ / info$sample_rate
  }

  to_ <- max(to_, from_ + 0.015) + 0.05
  av_obj <- av::read_audio_bin(audio = filepath, start_time = from_, end_time = to_)
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


#' av_loader
#'
#' Load an audio located at 'filepath' using av package.
#'
#' @param filepath (str) path to the audio file.
#' @param offset (num) the sample (or the second if unit = 'time') where the audio should start.
#' @param duration (num) how many samples (or how many seconds if unit = 'time') should be extracted.
#' @param unit (str) 'samples' or 'time'
#'
#' @export
av_loader <- function(
    filepath,
    offset = 0L,
    duration = 0L,
    unit = c("samples", "time")) {
  # load audio file
  av_read_mp3_or_wav(filepath, from = offset, to = offset + duration, unit = unit)
}
