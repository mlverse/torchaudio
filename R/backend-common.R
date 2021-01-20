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
#' @param audio (numeric or Wave): A numeric vector or Wave object, usually from [tuneR::readMP3], [tuneR::readWave] or [monitoR::readMP3].
#' @param out (Tensor): An optional output tensor to use instead of creating one. (Default: ``NULL``)
#' @param normalization (bool, float or function): Optional normalization.
#'         If boolean `TRUE`, then output is divided by `2^31`.
#'         Assuming the input is signed 32-bit audio, this normalizes to `[-1, 1]`.
#'         If `numeric`, then output is divided by that number.
#'         If `function`, then the output is passed as a paramete to the given function,
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
transform_to_tensor <- function(audio) {
  UseMethod("transform_to_tensor", audio)
}

#' @export
transform_to_tensor.Wave <- function(
  wave_obj,
  out = NULL,
  normalization = NULL,
  channels_first = TRUE
) {
  l_wave_obj <- length(wave_obj)
  bits <- wave_obj@bit

  out_tensor <- torch::torch_zeros(2, l_wave_obj)
  if(length(wave_obj@left) > 0) out_tensor[1] = wave_obj@left
  if(length(wave_obj@right) > 0) out_tensor[2] = wave_obj@right

  if(!channels_first)
    out_tensor = out_tensor$t()

  # normalize if needed
  internal__normalize_audio(out_tensor, 2^(bits-1))

  sample_rate = wave_obj@samp.rate

  return(list(out_tensor, sample_rate))
}

#' Audio Information
#'
#' @param filepath (str) path to the mp3/wav file.
#'
#' @return data.frame.
#'
#' @export
audio_info <- function(filepath) {
  package_required("av")
  info <- av::av_media_info(filepath)
  info$audio$duration <- info$duration
  info$audio
}

