#' Spectrogram
#'
#' Create a spectrogram or a batch of spectrograms from a raw audio signal.
#' The spectrogram can be either magnitude-only or complex.
#'
#' @param waveform (tensor): Tensor of audio of dimension (..., time)
#' @param pad (integer): Two sided padding of signal
#' @param window (tensor or function): Window tensor that is applied/multiplied to each
#' frame/window or a function that generates the window tensor.
#' @param n_fft (integer): Size of FFT
#' @param hop_length (integer): Length of hop between STFT windows
#' @param win_length (integer): Window size
#' @param power (numeric): Exponent for the magnitude spectrogram, (must be > 0) e.g.,
#'  1 for energy, 2 for power, etc. If None, then the complex spectrum is returned instead.
#' @param normalized (logical): Whether to normalize by magnitude after stft
#' @param Arguments for window function.
#'
#' @return tensor: Dimension (..., freq, time), freq is n_fft %/% 2 + 1 and n_fft is the
#' number of Fourier bins, and time is the number of window hops (n_frame).
#' @export
spectrogram <- function(
  waveform,
  pad = 0,
  n_fft = 400,
  hop_length = NULL,
  win_length = NULL,
  window = torch::torch_hann_window,
  power = 2,
  normalized = FALSE,
  ...
) {
  if(is.null(win_length)) win_length <- n_fft
  if(is.null(hop_length)) hop_length <- win_length %/% 2
  if(is.function(window)) window <- window(window_length = win_length, dtype = torch::torch_float(), ...)
  if(pad > 0) waveform <- torch::nnf_pad(waveform, c(pad, pad))
  if(!is_torch_tensor(waveform)) waveform <- torch::torch_tensor(as.vector(as.array(waveform)), dtype = torch::torch_float())

  # default values are consistent with librosa.core.spectrum._spectrogram
  spec_f <- torch::torch_stft(
    input = waveform, n_fft = n_fft,
    hop_length = hop_length, win_length = win_length,
    window = window, center = FALSE,
    pad_mode = "reflect", normalized = FALSE,
    onesided = TRUE
  )

  if(normalized) spec_f <- spec_f/sqrt(sum(window^2))
  if(!is.null(power)) spec_f <- complex_norm(spec_f, power = power)

  return(spec_f)
}


#' Frequency Bin Conversion Matrix
#'
#' Create a frequency bin conversion matrix.
#'
#' @param n_freqs (int): Number of frequencies to highlight/apply
#' @param f_min (float): Minimum frequency (Hz)
#' @param f_max (float): Maximum frequency (Hz)
#' @param n_mels (int): Number of mel filterbanks
#' @param sample_rate (int): Sample rate of the audio waveform
#' @param norm (chr) (Optional): If 'slaney', divide the triangular
#'  mel weights by the width of the mel band (area normalization). (Default: `None`)
#'
#' @return `tensor`: Triangular filter banks (fb matrix) of size (`n_freqs`, `n_mels`)
#'         meaning number of frequencies to highlight/apply to x the number of filterbanks.
#'         Each column is a filterbank so that assuming there is a matrix A of
#'         size (..., `n_freqs`), the applied result would be
#'         ``A * create_fb_matrix(A.size(-1), ...)``.
#' @export
create_fb_matrix <- function() {

}

#' Complex Norm
#'
#' Compute the norm of complex tensor input.
#'
#' @param complex_tensor (tensor): Tensor shape of `(..., complex=2)`
#' @param power (numeric): Power of the norm. (Default: `1.0`).
#'
#' @return tensor: Power of the normed input tensor. Shape of `(..., )`
#'
#' @export
complex_norm <- function(complex_tensor, power = 1) {
  complex_tensor$pow(2.)$sum(-1)$pow(0.5 * power)
}
