#' Load Audio File
#'
#' Loads an audio file from disk into a tensor
#'
#' @param filepath (str): Path to audio file
#' @param out (Tensor): An optional output tensor to use instead of creating one. (Default: ``NULL``)
#' @param normalization (bool, float or function): Optional normalization.
#'         If boolean `TRUE`, then output is divided by `2^31`.
#'         Assuming the input is signed 32-bit audio, this normalizes to `[-1, 1]`.
#'         If `numeric`, then output is divided by that number.
#'         If `function`, then the output is passed as a paramete to the given function,
#'         then the output is divided by the result. (Default: ``TRUE``)
#'
#' @param channels_first (bool): Set channels first or length first in result. (Default: ``TRUE``)
#' @param num_frames (int): Number of frames to load.  0 to load everything after the offset. (Default: ``0``)
#' @param offset (int): Number of frames from the start of the file to begin data loading. (Default: ``0``)
#' @param signalinfo (str): A sox_signalinfo_t type, which could be helpful if the
#'         audio type cannot be automatically determined. (Default: ``NULL``)
#' @param encodinginfo (str): A sox_encodinginfo_t type, which could be set if the
#'         audio type cannot be automatically determined. (Default: ``NULL``)
#' @param filetype (str): A filetype or extension to be set if sox cannot determine it
#'         automatically. (Default: ``NULL``)
#'
#' @return
#'     list(Tensor, int): An output tensor of size `[C x L]` or `[L x C]` where
#'         L is the number of audio frames and
#'         C is the number of channels.
#'         An integer which is the sample rate of the audio (as listed in the metadata of the file)
#'
#' @examples
#' data = torchaudio::torchaudio_load('foo.mp3')
#' print(data[[1]]$size())
#' data_vol_normalized = torchaudio::torchaudio_load('foo.mp3', normalization= function(x) torch::torch_abs(x)$max())
#' print(data_vol_normalized[[1]]$abs()$max())
#'
#' @export
torchaudio_load <- function(
  filepath,
  out = NULL,
  normalization = TRUE,
  channels_first = TRUE,
  num_frames = 0L,
  offset = 0L,
  signalinfo = NULL,
  encodinginfo = NULL,
  filetype = NULL
) {

  # if tuneR is installed
  if(requireNamespace("tuneR", quietly = TRUE)) {
    backend_tuneR_backend_load(
      filepath = filepath,
      out = out,
      normalization = normalization,
      channels_first = channels_first,
      num_frames = num_frames,
      offset = offset,
      signalinfo = signalinfo,
      encodinginfo = encodinginfo,
      filetype = filetype
    )
  } else {
    package_required_error("tuneR")
  }
}


torchaudio_info <- function() {}
