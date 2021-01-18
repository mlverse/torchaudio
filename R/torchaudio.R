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
#' @param duration (int): Number of frames (or seconds) to load.  0 to load everything after the offset. (Default: ``0``)
#' @param offset (int): Number of frames (or seconds) from the start of the file to begin data loading. (Default: ``0``)
#' @param unit: (str): "sample" or "time". If "sample" duration and offset will be interpreted as frames, and as seconds otherwise.
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
#' \dontrun{
#' if(torch::torch_is_installed()) {
#' mp3_filename <- system.file("sample_audio_2.mp3", package = "torchaudio")
#' data = torchaudio_load(mp3_filename)
#' print(data[[1]]$size())
#' norm_fun <- function(x) torch::torch_abs(x)$max()
#' data_vol_normalized = torchaudio_load(mp3_filename, normalization= norm_fun)
#' print(data_vol_normalized[[1]]$abs()$max())
#' }
#'
#' }
#'
#' @export
torchaudio_load <- function(
  filepath,
  out = NULL,
  normalization = TRUE,
  channels_first = TRUE,
  duration = 0L,
  offset = 0L,
  unit = c("sample", "time"),
  signalinfo = NULL,
  encodinginfo = NULL,
  filetype = NULL
) {
  package_required("tuneR")

  # if tuneR is installed
  backend_tuneR_backend_load(
    filepath = filepath,
    out = out,
    normalization = normalization,
    channels_first = channels_first,
    duration = duration,
    offset = offset,
    unit = unit[1],
    signalinfo = signalinfo,
    encodinginfo = encodinginfo,
    filetype = filetype
  )
}



torchaudio_info <- function() {}
