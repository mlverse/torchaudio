# https://github.com/pytorch/audio/blob/e5eb485726088fb3a3f61750b66b2660ba91643d/torchaudio/_internal/misc_ops.py

#' Audio Normalization
#'
#' Audio normalization of a tensor in-place.  The normalization can be a bool,
#' a number, or a function that takes the audio tensor as an input. SoX uses
#' 32-bit signed integers internally, thus bool normalizes based on that assumption.
#'
#' @param signal (Tensor): waveform
#' @param normalization (bool, int or function): Optional normalization.
#'         If boolean `TRUE`, then output is divided by `2^31`.
#'         Assuming the input is signed 32-bit audio, this normalizes to `[-1, 1]`.
#'         If `numeric`, then output is divided by that number.
#'         If `function`, then the output is passed as a paramete to the given function,
#'         then the output is divided by the result. (Default: ``TRUE``)
#'
#' @keywords internal
internal__normalize_audio <- function(signal, normalization = TRUE) {
  if(is.function(normalization)) {
    invisible(signal$div_(normalization(signal)))
  } else {

    # do not normalize
    if(!normalization)
      return(invisible(NULL))

    # assumes it has 32-bit rate (sox default)
    if(is.logical(normalization))
      return(invisible(signal$div_(2^31)))

    # normalize with custom value
    if(is.numeric(normalization))
      return(invisible(signal$div_(normalization)))

  }
}
