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
functional_spectrogram <- function(
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


  # pack batch
  shape = waveform$size()
  ls = length(shape)
  waveform = waveform$reshape(list(-1, shape[ls]))

  # default values are consistent with librosa.core.spectrum._spectrogram
  spec_f <- torch::torch_stft(
    input = waveform, n_fft = n_fft,
    hop_length = hop_length, win_length = win_length,
    window = window, center = FALSE,
    pad_mode = "reflect", normalized = FALSE,
    onesided = TRUE
  )

  # unpack batch
  lspec = length(spec_f$shape)
  spec_f = spec_f$reshape(c(shape[-ls], spec_f$shape[(lspec-2):lspec]))

  if(normalized) spec_f <- spec_f/sqrt(sum(window^2))
  if(!is.null(power)) spec_f <- functional_complex_norm(spec_f, power = power)

  return(spec_f)
}

#' Frequency Bin Conversion Matrix
#'
#' Create a frequency bin conversion matrix.
#'
#' @param n_freqs (int): Number of frequencies to highlight/apply
#' @param n_mels (int): Number of mel filterbanks
#' @param sample_rate (int): Sample rate of the audio waveform
#' @param f_min (float): Minimum frequency (Hz)
#' @param f_max (float): Maximum frequency (Hz)
#' @param norm (chr) (Optional): If 'slaney', divide the triangular
#'  mel weights by the width of the mel band (area normalization). (Default: `None`)
#'
#' @return `tensor`: Triangular filter banks (fb matrix) of size (`n_freqs`, `n_mels`)
#'         meaning number of frequencies to highlight/apply to x the number of filterbanks.
#'         Each column is a filterbank so that assuming there is a matrix A of
#'         size (..., `n_freqs`), the applied result would be
#'         ``A * create_fb_matrix(A.size(-1), ...)``.
#' @export
functional_create_fb_matrix <- function(
  n_freqs,
  n_mels,
  sample_rate = 16000,
  f_min = 0,
  f_max = sample_rate %/% 2,
  norm = NULL
) {
  if(!is.null(norm) && norm != "slaney")
    type_error("norm must be one of NULL or 'slaney'")

  # freq bins
  all_freqs <- torch::torch_linspace(0, sample_rate %/% 2, n_freqs)

  # calculate mel freq bins
  # hertz to mel(f) is 2595. * math.log10(1. + (f / 700.))
  m_min = linear_to_mel_frequency(f_min)
  m_max = linear_to_mel_frequency(f_max)
  m_pts = torch::torch_linspace(m_min, m_max, n_mels + 2)
  # mel to hertz(mel) is 700. * (10**(mel / 2595.) - 1.)
  f_pts = 700.0 * (10 ^ (m_pts / 2595.0) - 1.0)
  # calculate the difference between each mel point and each stft freq point in hertz
  len_f_pts <- length(f_pts)
  f_diff = f_pts[2:len_f_pts] - f_pts[1:(len_f_pts - 1)]  # (n_mels + 1)
  slopes = f_pts$unsqueeze(1L) - all_freqs$unsqueeze(2L)  # (n_freqs, n_mels + 2)
  # create overlapping triangles
  zero = torch::torch_zeros(1L)
  down_slopes = (-1.0 * slopes[ , 1:(ncol(slopes) - 2)]) / f_diff[1:(length(f_diff) - 1)]  # (n_freqs, n_mels)
  up_slopes = slopes[ , 3:ncol(slopes)] / f_diff[2:length(f_diff)]  # (n_freqs, n_mels)
  fb = torch::torch_max(zero, other = torch::torch_min(down_slopes, other = up_slopes))

  if(!is.null(norm) && norm == "slaney") {
    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (f_pts[3:(n_mels + 2)] - f_pts[1:n_mels])
    fb = fb + enorm$unsqueeze(1)
  }

  return(fb)
}

#' DCT transformation matrix
#'
#' Create a DCT transformation matrix with shape (``n_mels``, ``n_mfcc``),
#' normalized depending on norm.
#' [http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II]()
#'
#' @param n_mfcc (int): Number of mfc coefficients to retain
#' @param n_mels (int): Number of mel filterbanks
#' @param norm (chr or NULL): Norm to use (either 'ortho' or None)
#'
#' @return `Tensor`: The transformation matrix, to be right-multiplied to
#'     row-wise data of size (``n_mels``, ``n_mfcc``).
#'
#' @export
functional_create_dct <- function(
  n_mfcc,
  n_mels,
  norm = NULL
) {
  n = torch::torch_arange(0, n_mels)
  k = torch::torch_arange(0, n_mfcc)$unsqueeze(2)
  dct = torch::torch_cos(pi / n_mels * ((n + 0.5) * k))  # size (n_mfcc, n_mels)
  if(is.null(norm)) {
    dct = dct * 2.0
  } else {
    if(norm != 'ortho') value_error("Argument norm must be 'ortho' or NULL.")
    dct[1] = dct[1] * 1.0 / torch::torch_sqrt(2.0)
    dct = dct * torch::torch_sqrt(2.0 / n_mels)
  }

  return(dct$t())
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
functional_complex_norm <- function(complex_tensor, power = 1) {
  complex_tensor$pow(2.)$sum(-1)$pow(0.5 * power)
}

#' Amplitude to DB
#'
#' Turn a tensor from the power/amplitude scale to the decibel scale.
#'
#' This output depends on the maximum value in the input tensor, and so
#' may return different values for an audio clip split into snippets vs. a
#' a full clip.
#'
#' @param x (Tensor): Input tensor before being converted to decibel scale
#' @param multiplier (float): Use 10.0 for power and 20.0 for amplitude (Default: ``10.0``)
#' @param amin (float): Number to clamp ``x`` (Default: ``1e-10``)
#' @param ref_value (float): Reference which the output will be scaled by.
#' @param db_multiplier (float): Log10(max(ref_value and amin))
#' @param top_db (float or NULL, optional): Minimum negative cut-off in decibels. A reasonable number
#'     is 80. (Default: ``NULL``)
#'
#' @return `Tensor`: Output tensor in decibel scale
#'
#' @export
functional_amplitude_to_db <- function(
  x,
  multiplier = 10.0,
  amin = 1e-10,
  ref_value = 1.0,
  db_multiplier = log10(max(amin, ref_value)),
  top_db = NULL
) {
  x_db = multiplier * torch::torch_log10(torch::torch_clamp(x, min=amin))
  x_db = x_db - multiplier * db_multiplier

  if(!is.null(top_db)){
    x_db = x_db$clamp(min=x_db$max()$item() - top_db)
  }

  return(x_db)
}

#' DB to Amplitude (functional)
#'
#' Turn a tensor from the decibel scale to the power/amplitude scale.
#'
#' @param x (Tensor): Input tensor before being converted to power/amplitude scale.
#' @param ref (float): Reference which the output will be scaled by. (Default: ``1.0``)
#' @param power (float): If power equals 1, will compute DB to power. If 0.5, will compute
#'  DB to amplitude. (Default: ``1.0``)
#'
#' @return `Tensor`: Output tensor in power/amplitude scale.
#'
#' @export
functional_db_to_amplitude <- function(x, ref = 1.0, power = 1.0) {
  ref * torch::torch_pow(torch::torch_pow(10.0, 0.1 * x), power)
}


