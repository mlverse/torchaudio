#' @keywords internal
validate_audio_extension <- function(actual, valid, backend) {
  if (!actual %in% valid) {
    value_error(glue::glue("{actual} is not a valid audio extension using the {backend} backend."))
  }
}

#' @keywords internal
AudioMetaData <- R6::R6Class(
  "AudioMetaData",
  public = list(
    sample_rate = NULL,
    num_frames = NULL,
    num_channels = NULL,
    initialize = function(sample_rate,
                          num_frames,
                          num_channels) {
      self$sample_rate <- sample_rate
      self$num_frames <- num_frames
      self$num_channels <- num_channels
    }
  )
)

#' Convert an audio object into a tensor
#'
#' Converts a numeric vector, as delivered by the backend, into a `torch_tensor` of shape (channels x samples).
#' If provided by the backend, attributes "channels" and "sample_rate" will be used.
#'
#' @param audio (numeric): A numeric vector, as delivered by the backend.
#' @param out (Tensor): An optional output tensor to use instead of creating one. (Default: ``NULL``)
#' @param normalization (bool, float or function): Optional normalization.
#'         If boolean `TRUE`, then output is divided by `2^(bits-1)`.
#'         If `bits` info is not available it assumes the input is signed 32-bit audio.
#'         If `numeric`, then output is divided by that number.
#'         If `function`, then the output is passed as a parameter to the given function,
#'         then the output is divided by the result. (Default: ``TRUE``)
#' @param channels_first (bool): Set channels first or length first in result. (Default: ``TRUE``)
#'
#' @return
#'     list(Tensor, int), containing
#'     - the audio content, encoded as `[C x L]` or `[L x C]` where L is the number of audio frames and
#'         C is the number of channels
#'     - the sample rate of the audio (as listed in the metadata of the file)
#'
#' @export
transform_to_tensor <- function(
    audio,
    out = NULL,
    normalization = TRUE,
    channels_first = TRUE) {
  UseMethod("transform_to_tensor")
}

#' @export
transform_to_tensor.Wave <- function(
    audio,
    out = NULL,
    normalization = TRUE,
    channels_first = TRUE) {
  l_wave_obj <- length(audio)

  channels <- if (audio@stereo) 2 else 1
  out_tensor <- torch::torch_zeros(channels, l_wave_obj)
  if (length(audio@left) > 0) out_tensor[1] <- audio@left
  if (length(audio@right) > 0 & channels == 2) out_tensor[2] <- audio@right

  if (!channels_first) {
    out_tensor <- out_tensor$t()
  }

  # normalize if needed
  if (is.null(normalization)) normalization <- TRUE
  if (is.logical(normalization) && isTRUE(normalization)) {
    bits <- audio@bit %||% 32
    normalization <- 2^(bits - 1)
  }
  internal__normalize_audio(out_tensor, normalization)

  sample_rate <- audio@samp.rate

  return(list(out_tensor, sample_rate))
}

#' @importFrom methods as
#' @export
transform_to_tensor.WaveMC <- function(
    audio,
    out = NULL,
    normalization = TRUE,
    channels_first = TRUE) {
  audio <- as(audio, "Wave")
  transform_to_tensor(audio, out, normalization, channels_first)
}

#' @export
transform_to_tensor.av <- function(
    audio,
    out = NULL,
    normalization = TRUE,
    channels_first = TRUE) {
  sample_rate <- attr(audio, "sample_rate")
  out_tensor <- torch::torch_tensor(audio, dtype = torch::torch_float())

  if (!channels_first) {
    out_tensor <- out_tensor$t()
  }

  # normalize if needed
  if (is.null(normalization)) normalization <- TRUE
  internal__normalize_audio(out_tensor, normalization)

  return(list(out_tensor, sample_rate))
}

#' Audio Information
#'
#' Retrieve audio metadata.
#'
#' @param filepath (str) path to the audio file.
#' @return AudioMetaData: an R6 class with fields sample_rate, channels, samples.
#'
#' @examples
#' path <- system.file("waves_yesno/1_1_0_1_1_0_1_1.wav", package = "torchaudio")
#' torchaudio_info(path)
#'
#' @export
torchaudio_info <- function(filepath) {
  audio <- av::read_audio_bin(filepath)
  num_samples <- length(audio) / attr(audio, "channels")
  AudioMetaData$new(
    sample_rate = attr(audio, "sample_rate"),
    num_frames = num_samples,
    num_channels = attr(audio, "channels")
  )
}

#' Set the backend for I/O operation
#'
#' @param backend (str): one of `'av_loader'`,
#' `'audiofile_loader'` or `'tuneR_loader'`.
#'
#' @return invisible(NULL).
#'
#' It will set `torchaudio.loader` options:``
#' options(
#'   torchaudio.loader = rlang::as_function(backend),
#' )
#'
#' @export
set_audio_backend <- function(backend) {
  options(
    torchaudio.loader = rlang::as_function(backend)
  )
}

#' Load Audio File
#'
#' Loads an audio file from disk using the default loader (getOption("torchaudio.loader")).
#'
#' @param filepath (str): Path to audio file
#' @param offset (int): Number of frames (or seconds) from the start of the file to begin data loading. (Default: `0`)
#' @param duration (int): Number of frames (or seconds) to load.  `-1` to load everything after the offset. (Default: `-1`)
#' @param unit (str): "sample" or "time". If "sample" duration and offset will be interpreted as frames, and as seconds otherwise.
#'
#'
#' @export
torchaudio_load <- function(
    filepath,
    offset = 0L,
    duration = -1L,
    unit = c("samples", "time")) {
  loader <- getOption("torchaudio.loader", default = tuneR_loader)

  filepath <- as.character(filepath)
  if (!fs::is_file(filepath)) {
    runtime_error(glue::glue("{filepath} not found or is a directory"))
  }
  if ((duration < -1) || (duration == 0)) {
    value_error("Expected value for num_samples -1 (entire file) or > 0")
  }
  if (duration == -1) {
    duration <- Inf
  }
  if (offset < 0) {
    value_error("Expected positive offset value")
  }

  loader(
    filepath,
    offset = offset,
    duration = duration,
    unit = unit[1]
  )
}
