#' Mel Scale
#'
#' Turn a normal STFT into a mel frequency STFT, using a conversion
#' matrix. This uses triangular filter banks.
#'
#' @param specgram (Tensor): A spectrogram STFT of dimension (..., freq, time).
#' @param n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
#' @param sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
#' @param f_min (float, optional): Minimum frequency. (Default: ``0.``)
#' @param f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
#' @param n_stft (int, optional): Number of bins in STFT. Calculated from first input
#' if None is given.  See ``n_fft`` in :class:`Spectrogram`. (Default: ``None``)
#'
#' @return `tensor`: Mel frequency spectrogram of size (..., ``n_mels``, time).
#'
#' @export
mel_scale <- function(
  specgram,
  n_mels= 128,
  sample_rate = 16000,
  f_min = 0.0,
  f_max = NULL,
  n_stft = NULL
) {
  if(is.null(f_max)) f_max = as.numeric(sample_rate %/% 2)
  if(f_min > f_max) value_error(glue::glue("Require f_min: {f_min} < f_max: {f_max}"))

  # pack batch
  shape = specgram$size()
  ls = length(shape)
  specgram = specgram$reshape(list(-1, shape[ls-1], shape[ls]))

  if(is.null(n_stft)) n_stft = specgram$size(2)

  fb = create_fb_matrix(
    n_freqs = n_stft,
    f_min = f_min,
    f_max = f_max,
    n_mels = n_mels,
    sample_rate = sample_rate
  )

  mel_specgram = torch_matmul(specgram$transpose(2L, 3L), fb)$transpose(2L, 3L)

  # unpack batch
  lspec = length(mel_specgram$shape)
  mel_specgram = mel_specgram$reshape(c(shape[-((ls-1):ls)], mel_specgram$shape[(lspec-2):lspec]))

  return(mel_specgram)
}

#' DB to Amplitude
#'
#' Turn a tensor from the decibel scale to the power/amplitude scale.
#'
#'
#' @param x (Tensor): Input tensor before being converted to power/amplitude scale.
#' @param ref (float): Reference which the output will be scaled by. (Default: ``1.0``)
#' @param power (float): If power equals 1, will compute DB to power. If 0.5, will compute
#'  DB to amplitude. (Default: ``1.0``)
#'
#' @return `Tensor`: Output tensor in power/amplitude scale.
#'
#' @export
DB_to_amplitude <- function(x, ref = 1.0, power = 1.0) {
  ref * torch::torch_pow(torch::torch_pow(10.0, 0.1 * x), power)
}
