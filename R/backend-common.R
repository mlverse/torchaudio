#' @export
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

.LOAD_DOCSTRING = "Loads an audio file from disk into a tensor

Args:
    filepath: Path to audio file

    out: An optional output tensor to use instead of creating one. (Default: ``NULL``)

    normalization: Optional normalization.
        If boolean `TRUE`, then output is divided by `1 << 31`.
        Assuming the input is signed 32-bit audio, this normalizes to `[-1, 1]`.
        If `float`, then output is divided by that number.
        If `function`, then the output is passed as a paramete to the given function,
        then the output is divided by the result. (Default: ``TRUE``)

    channels_first: Set channels first or length first in result. (Default: ``TRUE``)

    num_frames: Number of frames to load.  0 to load everything after the offset.
        (Default: ``0``)

    offset: Number of frames from the start of the file to begin data loading.
        (Default: ``0``)

    signalinfo: A sox_signalinfo_t type, which could be helpful if the
        audio type cannot be automatically determined. (Default: ``NULL``)

    encodinginfo: A sox_encodinginfo_t type, which could be set if the
        audio type cannot be automatically determined. (Default: ``NULL``)

    filetype: A filetype or extension to be set if sox cannot determine it
        automatically. (Default: ``NULL``)

Returns:
    (Tensor, int): An output tensor of size `[C x L]` or `[L x C]` where
        L is the number of audio frames and
        C is the number of channels.
        An integer which is the sample rate of the audio (as listed in the metadata of the file)

Example
    > data = torchaudio::backend_load('foo.mp3')
    > print(data[[1]]$size())
    torch.Size([2, 278756])
    > print(sample_rate)
    44100
    > data_vol_normalized = torchaudio::backend_load('foo.mp3', normalization= function(x) torch::torch_abs(x)$max())
    > print(data_vol_normalized[[1]]$abs()$max())
    1.
"

.LOAD_WAV_DOCSTRING = " Loads a wave file.

It assumes that the wav file uses 16 bit per sample that needs normalization by
shifting the input right by 16 bits.

Args:
    filepath: Path to audio file

Returns:
    (Tensor, int): An output tensor of size `[C x L]` or `[L x C]` where L is the number
        of audio frames and C is the number of channels. An integer which is the sample rate of the
        audio (as listed in the metadata of the file)
"

.SAVE_DOCSTRING = "Saves a Tensor on file as an audio file

Args:
    filepath: Path to audio file
    src: An input 2D tensor of shape `[C x L]` or `[L x C]` where L is
        the number of audio frames, C is the number of channels
    sample_rate: An integer which is the sample rate of the
        audio (as listed in the metadata of the file)
    precision Bit precision (Default: ``16``)
    channels_first (bool, optional): Set channels first or length first in result. (
        Default: ``TRUE``)
"

.INFO_DOCSTRING = "Gets metadata from an audio file without loading the signal.

Args:
    filepath: Path to audio file

Returns:
    (sox_signalinfo_t, sox_encodinginfo_t): A si (sox_signalinfo_t) signal
        info as a python object. An ei (sox_encodinginfo_t) encoding info

Example
    > si_ei = torchaudio::backend_info('foo.wav')
    > rate = si_ei[[1]]$rate
    > channels = si_ei[[1]]$channels
    > encoding = si_ei[[2]]$encoding
"

backend__impl_load <- function(func){
  attr(func, 'doc') <- .LOAD_DOCSTRING
  return(func)
}

backend__impl_load_wav <- function(func){
  attr(func, 'doc') <- .LOAD_WAV_DOCSTRING
  return(func)
}

backend__impl_save <- function(func){
  attr(func, 'doc') <- .SAVE_DOCSTRING
  return(func)
}

backend__impl_info <- function(func){
  attr(func, 'doc') <- .INFO_DOCSTRING
  return(func)
}
