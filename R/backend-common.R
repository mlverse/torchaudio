#' @keywords internal
validate_audio_extension <- function(file_extension) {
  valid_extensions <- c("mp3", "wav")
  if(!file_extension %in% valid_extensions)
    value_error(glue::glue("{file_extension} is not a valid audio extension."))
}

#' @keywords internal
AudioMetaData <- R6::R6Class(
  "AudioMetaData",
  public = list(
    sample_rate = NULL,
    num_frames = NULL,
    num_channels = NULL,
    initialize = function(
      sample_rate,
      num_frames,
      num_channels
    ) {
      self$sample_rate = sample_rate
      self$num_frames = num_frames
      self$num_channels = num_channels
    }
  )
)

#' Convert an audio object into a tensor
#'
#' Converts a tuneR Wave object or numeric vector into a `torch_tensor` of shape (Channels x Samples).
#' Convert Audio Object to Tensor.
#'
#' If audio is a numeric vector, attributes "channels" and "sample_rate" will be used if exists.
#' Numeric vectors returned from [av::read_audio_bin] have both attributes by default.
#'
#' @param audio (numeric or Wave): A numeric vector or Wave object, usually from [tuneR::readMP3] or [tuneR::readWave].
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
#'     list(Tensor, int): An output tensor of size `[C x L]` or `[L x C]` where
#'         L is the number of audio frames and
#'         C is the number of channels.
#'         An integer which is the sample rate of the audio (as listed in the metadata of the file)
#'
#' @export
transform_to_tensor <- function(
  audio,
  out = NULL,
  normalization = TRUE,
  channels_first = TRUE
) {
  UseMethod("transform_to_tensor")
}

#' @export
transform_to_tensor.Wave <- function(
  audio,
  out = NULL,
  normalization = TRUE,
  channels_first = TRUE
) {
  l_wave_obj <- length(audio)

  channels <- if(audio@stereo) 2 else 1
  out_tensor <- torch::torch_zeros(channels, l_wave_obj)
  if(length(audio@left) > 0) out_tensor[1] = audio@left
  if(length(audio@right) > 0 & channels == 2) out_tensor[2] = audio@right

  if(!channels_first)
    out_tensor = out_tensor$t()

  # normalize if needed
  if(is.null(normalization)) normalization <- TRUE
  if(is.logical(normalization) && isTRUE(normalization)) {
    bits <- audio@bit %||% 32
    normalization <- 2^(bits-1)
  }
  internal__normalize_audio(out_tensor, normalization)

  sample_rate = audio@samp.rate

  return(list(out_tensor, sample_rate))
}

#' @importFrom methods as
#' @export
transform_to_tensor.WaveMC <- function(
    audio,
    out = NULL,
    normalization = TRUE,
    channels_first = TRUE
) {
  audio <- as(audio, 'Wave')
  transform_to_tensor(audio, out, normalization, channels_first)
}

#' @export
transform_to_tensor.av <- function(
  audio,
  out = NULL,
  normalization = TRUE,
  channels_first = TRUE
) {

  sample_rate <- attr(audio, "sample_rate")
  out_tensor <- torch::torch_tensor(audio, dtype = torch::torch_float())

  if(!channels_first)
    out_tensor = out_tensor$t()

  # normalize if needed
  if(is.null(normalization)) normalization <- TRUE
  internal__normalize_audio(out_tensor, normalization)

  return(list(out_tensor, sample_rate))
}

to_tensor_float <- function(x) torch::torch_tensor(x, dtype = torch::torch_float())

#' @export
transform_to_tensor.audiofile <- function(
  audio,
  out = NULL,
  normalization = FALSE,
  channels_first = TRUE
) {

  sample_rate <- audio$sample_rate

  if(audio$channels > 1) {
    out_tensor <- Map(to_tensor_float, audio$waveform)
    out_tensor <- torch::torch_stack(out_tensor)
  } else {
    out_tensor <- to_tensor_float(audio$waveform[[1]])$unsqueeze(1)
  }

  if(!channels_first)
    out_tensor = out_tensor$t()

  # normalize if needed
  if(is.null(normalization)) normalization <- FALSE
  if(is.logical(normalization) && isTRUE(normalization)) {
    bits <- audio$bit %||% 32
    normalization <- 2^(bits-1)
  }
  internal__normalize_audio(out_tensor, normalization)

  return(list(out_tensor, sample_rate))
}

#' MP3 Information
#'
#' Retrive metadata from mp3 without load the audio samples in memory.
#'
#' @param filepath (chr) path to mp3 file
#'
#' @return AudioMetaData: sample_rate, channels, samples
#'
#' @examples
#' mp3_path <- system.file("sample_audio_1.mp3", package = "torchaudio")
#' mp3_info(mp3_path)
#'
#' @export
mp3_info <- function(filepath) {
  info <- get_info_mp3(filepath)
  AudioMetaData$new(
    sample_rate = info$hz,
    num_frames = info$samples,
    num_channels = info$channels
  )
}

#' Wave Information
#'
#' Retrive metadata from wav without load the audio samples in memory.
#'
#' @param filepath (chr) path to wav file
#'
#' @return AudioMetaData: sample_rate, channels, samples
#'
#' @examples
#' wav_path <- system.file("waves_yesno/1_1_0_1_1_0_1_1.wav", package = "torchaudio")
#' wav_info(wav_path)
#'
#' @export
wav_info <- function(filepath) {
  info <- tuneR::readWave(filepath, header = TRUE)
  AudioMetaData$new(
    sample_rate = info$sample.rate,
    num_frames = info$samples,
    num_channels = info$channels
  )
}

#' Audio Information
#'
#' Retrive metadata from mp3 or wave file without load the audio samples in memory.
#'
#' @param filepath (str) path to the mp3/wav file.
#'
#' @return list(sample_rate, channels, samples)
#'
#' @export
info <- function(filepath) {
  file_ext <- tools::file_ext(filepath)
  validate_audio_extension(file_ext)

  if(file_ext == "mp3")
    info <- mp3_info(filepath)
  if(file_ext == "wav")
    info <- wav_info(filepath)

  info
}

#' Set the backend for I/O operation
#'
#' @param backend (str): one of `'av_loader'`,
#' `'audiofile_loader'` or `'tuneR_loader'`.
#'
#' @return invisible(NULL).
#'
#' It will set `torchaudio.loader` and `torchaudio.loader.name` options:``
#' options(
#'   torchaudio.loader = rlang::as_function(backend),
#'   torchaudio.loader.name = backend
#' )
#'
#' @export
set_audio_backend <- function(backend) {
  options(
    torchaudio.loader = rlang::as_function(backend),
    torchaudio.loader.name = backend
  )
}

#' Load Audio File
#'
#' Loads an audio file from disk using the default loader (getOption("torchaudio.loader")).
#'
#' @param filepath (str): Path to audio file
#' @param offset (int): Number of frames (or seconds) from the start of the file to begin data loading. (Default: ``0``)
#' @param duration (int): Number of frames (or seconds) to load.  0 to load everything after the offset. (Default: ``0``)
#' @param unit (str): "sample" or "time". If "sample" duration and offset will be interpreted as frames, and as seconds otherwise.
#'
#'
#' @export
torchaudio_load <- function(
  filepath,
  offset = 0L,
  duration = 0L,
  unit = c("samples", "time")
) {
  loader <- getOption("torchaudio.loader", default = tuneR_loader)
  loader(
    filepath,
    offset = offset,
    duration = duration,
    unit = unit[1]
  )
}

