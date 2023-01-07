#' Spectrogram (functional)
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
#'  1 for energy, 2 for power, etc. If NULL, then the complex spectrum is returned instead.
#' @param normalized (logical): Whether to normalize by magnitude after stft
#'
#' @return `tensor`: Dimension (..., freq, time), freq is n_fft %/% 2 + 1 and n_fft is the
#' number of Fourier bins, and time is the number of window hops (n_frame).
#' @export
functional_spectrogram <- function(
  waveform,
  pad,
  window,
  n_fft,
  hop_length,
  win_length,
  power,
  normalized
) {
  if(pad > 0) waveform <- torch::nnf_pad(waveform, c(pad, pad))

  # pack batch
  shape = waveform$size()
  ls = length(shape)
  waveform = waveform$reshape(list(-1, shape[ls]))
  lws = length(waveform$size())

  # default values are consistent with librosa.core.spectrum._spectrogram
  spec_f <- torch::torch_stft(
    input = waveform, n_fft = n_fft,
    hop_length = hop_length, win_length = win_length,
    window = window, center = TRUE,
    pad_mode = "reflect", normalized = FALSE,
    onesided = TRUE, return_complex = TRUE
  )

  # unpack batch
  lspec = length(spec_f$shape)
  spec_f = spec_f$reshape(c(shape[-ls], spec_f$shape[(lspec-1):lspec]))

  if(normalized) spec_f <- spec_f/sqrt(sum(window^2))
  if(!is.null(power)) {
    if(power == 1)
      return(spec_f$abs())
    return(spec_f$abs()$pow(power))
  }
  return(torch::torch_view_as_real(spec_f))
}

#' Frequency Bin Conversion Matrix (functional)
#'
#' Create a frequency bin conversion matrix.
#'
#' @param n_freqs (int): Number of frequencies to highlight/apply
#' @param n_mels (int): Number of mel filterbanks
#' @param sample_rate (int): Sample rate of the audio waveform
#' @param f_min (float): Minimum frequency (Hz)
#' @param f_max (float or NULL): Maximum frequency (Hz). If NULL defaults to sample_rate %/% 2
#' @param norm (chr) (Optional): If 'slaney', divide the triangular
#'  mel weights by the width of the mel band (area normalization). (Default: `NULL`)
#'
#' @return `tensor`: Triangular filter banks (fb matrix) of size (`n_freqs`, `n_mels`)
#'         meaning number of frequencies to highlight/apply to x the number of filterbanks.
#'         Each column is a filterbank so that assuming there is a matrix A of
#'         size (..., `n_freqs`), the applied result would be
#'         ``A * functional_create_fb_matrix(A.size(-1), ...)``.
#'
#' @export
functional_create_fb_matrix <- function(
  n_freqs,
  f_min,
  f_max,
  n_mels,
  sample_rate,
  norm = NULL
) {
  if(!is.null(norm) && norm != "slaney")
    type_error("norm must be one of NULL or 'slaney'")

  # freq bins
  all_freqs <- torch::torch_linspace(0, sample_rate %/% 2, n_freqs)

  # calculate mel freq bins
  # hertz to mel(f) is 2595. * math.log10(1. + (f / 700.))
  f_max = if(is.null(f_max)) sample_rate %/% 2 else f_max
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

#' DCT transformation matrix (functional)
#'
#' Create a DCT transformation matrix with shape (``n_mels``, ``n_mfcc``),
#' normalized depending on norm.
#' [https://en.wikipedia.org/wiki/Discrete_cosine_transform]()
#'
#' @param n_mfcc (int): Number of mfc coefficients to retain
#' @param n_mels (int): Number of mel filterbanks
#' @param norm (chr or NULL): Norm to use (either 'ortho' or NULL)
#'
#' @return `tensor`: The transformation matrix, to be right-multiplied to
#'     row-wise data of size (``n_mels``, ``n_mfcc``).
#'
#' @export
functional_create_dct <- function(
  n_mfcc,
  n_mels,
  norm = NULL
) {
  n = torch::torch_arange(0, n_mels-1)
  k = torch::torch_arange(0, n_mfcc-1)$unsqueeze(2)
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

#' Complex Norm (functional)
#'
#' Compute the norm of complex tensor input.
#'
#' @param complex_tensor (tensor): Tensor shape of `(..., complex=2)`
#' @param power (numeric): Power of the norm. (Default: `1.0`).
#'
#' @return `tensor`: Power of the normed input tensor. Shape of `(..., )`
#'
#' @export
functional_complex_norm <- function(complex_tensor, power = 1) {
  complex_tensor$pow(2.)$sum(-1)$pow(0.5 * power)
}

#' Amplitude to DB (functional)
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
#' @param db_multiplier (float): Log10(max(ref_value and amin))
#' @param top_db (float or NULL, optional): Minimum negative cut-off in decibels. A reasonable number
#'     is 80. (Default: ``NULL``)
#'
#' @return `tensor`: Output tensor in decibel scale
#'
#' @export
functional_amplitude_to_db <- function(
  x,
  multiplier,
  amin,
  db_multiplier,
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
#' @return `tensor`: Output tensor in power/amplitude scale.
#'
#' @export
functional_db_to_amplitude <- function(x, ref, power) {
  ref * torch::torch_pow(torch::torch_pow(10.0, 0.1 * x), power)
}

#' Mel Scale (functional)
#'
#' Turn a normal STFT into a mel frequency STFT, using a conversion
#' matrix. This uses triangular filter banks.
#'
#' @param specgram (Tensor): A spectrogram STFT of dimension (..., freq, time).
#' @param n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
#' @param sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
#' @param f_min (float, optional): Minimum frequency. (Default: ``0.``)
#' @param f_max (float or NULL, optional): Maximum frequency. (Default: ``sample_rate %/% 2``)
#' @param n_stft (int, optional): Number of bins in STFT. Calculated from first input
#' if NULL is given.  See ``n_fft`` in :class:`Spectrogram`. (Default: ``NULL``)
#'
#' @return `tensor`: Mel frequency spectrogram of size (..., ``n_mels``, time).
#'
#' @export
functional_mel_scale <- function(
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
  ls = length(shape)

  if(is.null(n_stft)) n_stft = specgram$size(2)

  fb = functional_create_fb_matrix(
    n_freqs = n_stft,
    f_min = f_min,
    f_max = f_max,
    n_mels = n_mels,
    sample_rate = sample_rate
  )

  mel_specgram = torch::torch_matmul(specgram$transpose(2L, 3L), fb)$transpose(2L, 3L)

  # unpack batch
  lspec = length(mel_specgram$shape)
  mel_specgram = mel_specgram$reshape(c(shape[-((ls-1):ls)], mel_specgram$shape[(lspec-2):lspec]))

  return(mel_specgram)
}


#' Mu Law Encoding (functional)
#'
#' Encode signal based on mu-law companding.  For more info see
#' the [Wikipedia Entry](https://en.wikipedia.org/wiki/M-law_algorithm)
#'
#' @param x (Tensor): Input tensor
#' @param quantization_channels (int): Number of channels
#'
#' @details
#' This algorithm assumes the signal has been scaled to between -1 and 1 and
#' returns a signal encoded with values from 0 to quantization_channels - 1.
#'
#' @return `tensor`: Input after mu-law encoding
#'
#' @export
functional_mu_law_encoding <- function(
  x,
  quantization_channels
) {
  mu = quantization_channels - 1.0
  if(!torch::torch_is_floating_point(x)) {
    x = x$to(torch::torch_float())
  }
  mu = torch::torch_tensor(mu, dtype=x$dtype)
  x_mu = torch::torch_sign(x) * torch::torch_log1p(mu * torch::torch_abs(x)) / torch::torch_log1p(mu)
  x_mu = ((x_mu + 1) / 2 * mu + 0.5)$to(torch::torch_int64())
  return(x_mu)
}

#' Mu Law Decoding (functional)
#'
#' Decode mu-law encoded signal.  For more info see the
#'  [Wikipedia Entry](https://en.wikipedia.org/wiki/M-law_algorithm)
#'
#' @param x_mu (Tensor): Input tensor
#' @param quantization_channels (int): Number of channels
#'
#' @details
#' This expects an input with values between 0 and quantization_channels - 1
#' and returns a signal scaled between -1 and 1.
#'
#' @return `tensor`: Input after mu-law decoding
#'
#' @export
functional_mu_law_decoding <- function(
  x_mu,
  quantization_channels
) {
  mu = quantization_channels - 1.0
  if(!torch::torch_is_floating_point(x_mu)) {
    x_mu = x_mu$to(torch::torch_float())
  }
  mu = torch::torch_tensor(mu, dtype=x_mu$dtype)
  x = ((x_mu)/mu) * 2 - 1.0
  x = torch::torch_sign(x) * (torch::torch_exp(torch::torch_abs(x) * torch::torch_log1p(mu)) - 1.0)/mu
  return(x)
}

#' Angle (functional)
#'
#' Compute the angle of complex tensor input.
#'
#' @param complex_tensor (Tensor): Tensor shape of `(..., complex=2)`
#'
#' @return `tensor`: Angle of a complex tensor. Shape of `(..., )`
#'
#' @export
functional_angle <- function(complex_tensor) {
  complex_tensor = complex_tensor$to(torch::torch_float())
  torch::torch_atan2(complex_tensor[.., 2], complex_tensor[.., 1])
}

#' Magnitude and Phase (functional)
#'
#' Separate a complex-valued spectrogram with shape `(.., 2)` into its magnitude and phase.
#'
#' @param complex_tensor (Tensor): Tensor shape of `(.., complex=2)`
#' @param power (float): Power of the norm. (Default: `1.0`)
#'
#' @return list(`tensor`, `tensor`): The magnitude and phase of the complex tensor
#'
#' @export
functional_magphase <- function(
  complex_tensor,
  power = 1.0
) {
  mag = functional_complex_norm(complex_tensor, power)
  phase = functional_angle(complex_tensor)
  return(list(mag, phase))
}

#' Phase Vocoder
#'
#' Given a STFT tensor, speed up in time without modifying pitch by a factor of ``rate``.
#'
#' @param complex_specgrams  (Tensor): Dimension of `(..., freq, time, complex=2)`
#' @param rate  (float): Speed-up factor
#' @param phase_advance  (Tensor): Expected phase advance in each bin. Dimension of (freq, 1)
#'
#' @return `tensor`: Complex Specgrams Stretch with dimension of `(..., freq, ceiling(time/rate), complex=2)`
#'
#' @examples
#' if(torch::torch_is_installed()) {
#' library(torch)
#' library(torchaudio)
#'
#' freq = 1025
#' hop_length = 512
#'
#' #  (channel, freq, time, complex=2)
#' complex_specgrams = torch_randn(2, freq, 300, 2)
#' rate = 1.3 # Speed up by 30%
#' phase_advance = torch_linspace(0, pi * hop_length, freq)[.., NULL]
#' x = functional_phase_vocoder(complex_specgrams, rate, phase_advance)
#' x$shape # with 231 == ceil (300 / 1.3)
#' # torch.Size ([2, 1025, 231, 2])
#' }
#'
#' @export
functional_phase_vocoder <- function(
  complex_specgrams,
  rate,
  phase_advance
) {
  # pack batch
  shape = complex_specgrams$size()
  ls = length(shape)
  complex_specgrams = complex_specgrams$reshape(c(-1, shape[(ls-2):ls]))
  shape = complex_specgrams$size()
  ls = length(shape)

  time_steps = torch::torch_arange(
    0,
    complex_specgrams$size(ls-1)-1,
    rate,
    device = complex_specgrams$device,
    dtype = complex_specgrams$dtype
  )

  alphas = time_steps %% 1.0
  phase_0 = functional_angle(complex_specgrams[.., 1:1, 1:N])

  # Time Padding
  complex_specgrams = torch::nnf_pad(complex_specgrams, c(0, 0, 0, 2))

  # (new_bins, freq, 2)
  complex_specgrams_0 = complex_specgrams$index_select(ls-1, (time_steps + 1)$to(dtype = torch::torch_long()))
  complex_specgrams_1 = complex_specgrams$index_select(ls-1, (time_steps + 2)$to(dtype = torch::torch_long()))

  angle_0 = functional_angle(complex_specgrams_0)
  angle_1 = functional_angle(complex_specgrams_1)

  norm_0 = torch::torch_norm(complex_specgrams_0, p=2, dim=-1)
  norm_1 = torch::torch_norm(complex_specgrams_1, p=2, dim=-1)

  phase = angle_1 - angle_0 - phase_advance
  phase = phase - 2 * pi * torch::torch_round(phase / (2 * pi))

  # Compute Phase Accum
  phase = phase + phase_advance
  phase = torch::torch_cat(list(phase_0, phase[.., 1:-2]), dim=-1)
  phase_acc = torch::torch_cumsum(phase, -1)

  mag = alphas * norm_1 + (1 - alphas) * norm_0

  real_stretch = mag * torch::torch_cos(phase_acc)
  imag_stretch = mag * torch::torch_sin(phase_acc)

  complex_specgrams_stretch = torch::torch_stack(list(real_stretch, imag_stretch), dim=-1)

  # unpack batch
  lcss <- length(complex_specgrams_stretch$shape)
  complex_specgrams_stretch = complex_specgrams_stretch$reshape(c(shape[1:(ls-3)], complex_specgrams_stretch$shape[2:lcss]))

  return(complex_specgrams_stretch)
}

#' Griffin-Lim Transformation (functional)
#'
#' Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation.
#'  Implementation ported from `librosa`.
#'
#' @param specgram (Tensor): A magnitude-only STFT spectrogram of dimension (..., freq, frames)
#'      where freq is ``n_fft %/% 2 + 1``.
#' @param window (Tensor): Window tensor that is applied/multiplied to each frame/window
#' @param n_fft (int): Size of FFT, creates ``n_fft %/% 2 + 1`` bins
#' @param hop_length (int): Length of hop between STFT windows.
#' @param win_length (int): Window size.
#' @param power (float): Exponent for the magnitude spectrogram,
#'      (must be > 0) e.g., 1 for energy, 2 for power, etc.
#' @param normalized (bool): Whether to normalize by magnitude after stft.
#' @param n_iter (int): Number of iteration for phase recovery process.
#' @param momentum (float): The momentum parameter for fast Griffin-Lim.
#'      Setting this to 0 recovers the original Griffin-Lim method.
#'      Values near 1 can lead to faster convergence, but above 1 may not converge.
#' @param length (int or NULL): Array length of the expected output.
#' @param rand_init (bool): Initializes phase randomly if TRUE, to zero otherwise.
#'
#' @return `tensor`: waveform of (..., time), where time equals the ``length`` parameter if given.
#'
#' @export
functional_griffinlim <- function(
  specgram,
  window,
  n_fft,
  hop_length,
  win_length,
  power,
  normalized,
  n_iter,
  momentum,
  length,
  rand_init
) {

  not_implemented_error("functional_griffinlim is not implemented yet.")

  if(momentum > 1) value_warning('momentum > 1 can be unstable')
  if(momentum < 0) value_error('momentum < 0')

  # pack batch
  shape = specgram$size()
  ls = length(shape)
  specgram = specgram$reshape(c(-1, shape[(ls-1):ls]))
  shape = specgram$size()
  ls = length(shape)

  specgram = specgram$pow(1 / power)

  # randomly initialize the phase
  ss = specgram$size()
  batch = ss[1]
  freq = ss[2]
  frames = ss[3]
  if(rand_init) {
    angles = 2 * pi * torch::torch_rand(batch, freq, frames)
  } else {
    angles = torch::torch_zeros(batch, freq, frames)
  }

  angles = torch::torch_stack(list(angles$cos(), angles$sin()), dim=-1)$to(dtype = specgram$dtype, device = specgram$device)
  specgram = specgram$unsqueeze(-1)$expand_as(angles)

  # And initialize the previous iterate to 0
  rebuilt = torch::torch_tensor(0.)

  for(i in seq.int(n_iter)) {
    # Store the previous iterate
    tprev = rebuilt
    # Invert with our current estimate of the phases
    inverse = torch::torch_istft(specgram * angles,
                                 n_fft=n_fft,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 window=window,
                                 length=length)$float()

    # Rebuild the spectrogram
    rebuilt = torch::torch_stft(inverse, n_fft, hop_length, win_length, window,
                                TRUE, 'reflect', FALSE, TRUE)

    # Update our phase estimates
    angles = rebuilt
    if(momentum) {
      angles = angles - tprev$mul_(momentum / (1 + momentum))
    }
    angles = angles$div(functional_complex_norm(angles)$add(1e-16)$unsqueeze(-1)$expand_as(angles))
  }

  # Return the final phase estimates
  waveform = torch::torch_istft(specgram * angles,
                                n_fft=n_fft,
                                hop_length=hop_length,
                                win_length=win_length,
                                window=window,
                                length=length)

  # unpack batch
  lws = length(waveform$shape)
  waveform = waveform$reshape(c(shape[1:(ls-2)], waveform$shape[lws]))

  return(waveform)
}

#' An IIR Filter (functional)
#'
#' Perform an IIR filter by evaluating difference equation.
#'
#' @param waveform (Tensor): audio waveform of dimension of ``(..., time)``.  Must be normalized to -1 to 1.
#' @param a_coeffs (Tensor): denominator coefficients of difference equation of dimension of ``(n_order + 1)``.
#'  Lower delays coefficients are first, e.g. ``[a0, a1, a2, ...]``.
#'  Must be same size as b_coeffs (pad with 0's as necessary).
#' @param b_coeffs (Tensor): numerator coefficients of difference equation of dimension of ``(n_order + 1)``.
#'  Lower delays coefficients are first, e.g. ``[b0, b1, b2, ...]``.
#'  Must be same size as a_coeffs (pad with 0's as necessary).
#' @param clamp (bool, optional): If ``TRUE``, clamp the output signal to be in the range \[-1, 1\] (Default: ``TRUE``)
#'
#' @return `tensor`: Waveform with dimension of ``(..., time)``.
#'
#' @export
functional_lfilter <- function(
  waveform,
  a_coeffs,
  b_coeffs,
  clamp = TRUE
) {
  # pack batch
  shape = waveform$size()
  ls = length(shape)
  waveform = waveform$reshape(c(-1, shape[ls]))
  ls = length(shape)

  if(a_coeffs$size(1) != b_coeffs$size(1)) value_error(glue::glue("Size of a_coeffs: {a_coeffs$size(1)} differs from size of b_coeffs: {b_coeffs$size(1)}"))
  if(length(waveform$size()) != 2) value_error(glue::glue("waveform size should be 1, got {length(waveform$size()) - 1}."))
  if(waveform$device$type != a_coeffs$device$type) runtime_error(glue::glue("waveform is in {waveform$device$type} device while a_coeffs is in {a_coeffs$device$type} device. They should share the same device."))
  if(b_coeffs$device$type != a_coeffs$device$type) runtime_error(glue::glue("b_coeffs is in {b_coeffs$device$type} device while a_coeffs is in {a_coeffs$device$type} device. They should share the same device."))

  device = waveform$device
  dtype = waveform$dtype
  n_channel = waveform$size(1)
  n_sample = waveform$size(2)
  n_order = a_coeffs$size(1)
  n_sample_padded = n_sample + n_order - 1
  if(n_order <= 0) value_error(glue::glue("a_coeffs$size(1) should be greater than zero, got {n_order}."))

  # Pad the input and create output
  padded_waveform = torch::torch_zeros(n_channel, n_sample_padded, dtype=dtype, device=device)
  padded_waveform[, (n_order):N] = waveform
  padded_output_waveform = torch::torch_zeros(n_channel, n_sample_padded, dtype=dtype, device=device)

  # Set up the coefficients matrix
  # Flip coefficients' order
  a_coeffs_flipped = a_coeffs$flip(1)
  b_coeffs_flipped = b_coeffs$flip(1)

  # calculate windowed_input_signal in parallel
  # create indices of original with shape (n_channel, n_order, n_sample)
  window_idxs = torch::torch_arange(0, n_sample-1, device=device)$unsqueeze(1) + torch::torch_arange(0, n_order-1, device=device)$unsqueeze(2)
  window_idxs = window_idxs$`repeat`(c(n_channel, 1, 1))
  window_idxs = window_idxs + (torch::torch_arange(0, n_channel-1, device=device)$unsqueeze(-1)$unsqueeze(-1) * n_sample_padded)
  # Indices/Index start at 1 in R.
  window_idxs = (window_idxs + 1)$to(torch::torch_long())
  # (n_order, ) matmul (n_channel, n_order, n_sample) -> (n_channel, n_sample)
  input_signal_windows = torch::torch_matmul(b_coeffs_flipped, torch::torch_take(padded_waveform, window_idxs))

  input_signal_windows$div_(a_coeffs[1])
  a_coeffs_flipped$div_(a_coeffs[1])
  for(i_sample in 1:ncol(input_signal_windows)) {
    o0 = input_signal_windows[,i_sample]
    windowed_output_signal = padded_output_waveform[ , i_sample:(i_sample + n_order-1)]
    o0$addmv_(windowed_output_signal, a_coeffs_flipped, alpha=-1)
    padded_output_waveform[ , i_sample + n_order - 1] = o0
  }

  output = padded_output_waveform[, (n_order):N]

  if(clamp) output = torch::torch_clamp(output, min=-1., max=1.)

  # unpack batch
  output = output$reshape(c(shape[-ls], output$shape[length(output$shape)]))

  return(output)
}

#' Biquad Filter (functional)
#'
#' Perform a biquad filter of input tensor.  Initial conditions set to 0.
#'     [https://en.wikipedia.org/wiki/Digital_biquad_filter]()
#'
#' @param waveform (Tensor): audio waveform of dimension of `(..., time)`
#' @param b0 (float): numerator coefficient of current input, x\[n\]
#' @param b1 (float): numerator coefficient of input one time step ago x\[n-1\]
#' @param b2 (float): numerator coefficient of input two time steps ago x\[n-2\]
#' @param a0 (float): denominator coefficient of current output y\[n\], typically 1
#' @param a1 (float): denominator coefficient of current output y\[n-1\]
#' @param a2 (float): denominator coefficient of current output y\[n-2\]
#'
#' @return `tensor`: Waveform with dimension of `(..., time)`
#'
#' @export
functional_biquad <- function(
  waveform,
  b0,
  b1,
  b2,
  a0,
  a1,
  a2
) {
  device = waveform$device
  dtype = waveform$dtype

  output_waveform = functional_lfilter(
    waveform,
    torch::torch_tensor(c(a0, a1, a2), dtype=dtype, device=device),
    torch::torch_tensor(c(b0, b1, b2), dtype=dtype, device=device)
  )
  return(output_waveform)
}

db_to_linear <- function(x) {
  exp(x * log(10) / 20.0)
}

#' High-pass Biquad Filter (functional)
#'
#' Design biquad highpass filter and perform filtering.  Similar to SoX implementation.
#'
#' @param waveform (Tensor): audio waveform of dimension of `(..., time)`
#' @param sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
#' @param cutoff_freq (float): filter cutoff frequency
#' @param Q (float, optional): [https://en.wikipedia.org/wiki/Q_factor]() (Default: ``0.707``)
#'
#' @return `tensor`: Waveform dimension of `(..., time)`
#'
#' @export
functional_highpass_biquad <- function(
  waveform,
  sample_rate,
  cutoff_freq,
  Q = 0.707
) {

  w0 = 2 * pi * cutoff_freq / sample_rate
  alpha = sin(w0) / 2. / Q

  b0 = (1 + cos(w0)) / 2
  b1 = -1 - cos(w0)
  b2 = b0
  a0 = 1 + alpha
  a1 = -2 * cos(w0)
  a2 = 1 - alpha
  return(functional_biquad(waveform, b0, b1, b2, a0, a1, a2))
}

#' Low-pass Biquad Filter (functional)
#'
#' Design biquad lowpass filter and perform filtering.  Similar to SoX implementation.
#'
#' @param waveform  (torch.Tensor): audio waveform of dimension of `(..., time)`
#' @param sample_rate  (int): sampling rate of the waveform, e.g. 44100 (Hz)
#' @param cutoff_freq  (float): filter cutoff frequency
#' @param Q  (float, optional): [https://en.wikipedia.org/wiki/Q_factor]() (Default: ``0.707``)
#'
#' @return `tensor`: Waveform of dimension of `(..., time)`
#'
#' @export
functional_lowpass_biquad <- function(
  waveform,
  sample_rate,
  cutoff_freq,
  Q = 0.707
) {

  w0 = 2 * pi * cutoff_freq / sample_rate
  alpha = sin(w0) / 2 / Q

  b0 = (1 - cos(w0)) / 2
  b1 = 1 - cos(w0)
  b2 = b0
  a0 = 1 + alpha
  a1 = -2 * cos(w0)
  a2 = 1 - alpha
  return(functional_biquad(waveform, b0, b1, b2, a0, a1, a2))
}

#' All-pass Biquad Filter (functional)
#'
#' Design two-pole all-pass filter. Similar to SoX implementation.
#'
#' @param waveform (Tensor): audio waveform of dimension of `(..., time)`
#' @param sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
#' @param central_freq (float): central frequency (in Hz)
#' @param Q (float, optional): [https://en.wikipedia.org/wiki/Q_factor]() (Default: ``0.707``)
#'
#' @return `tensor`: Waveform of dimension of `(..., time)`
#'
#' @references
#' - [https://sox.sourceforge.net/sox.html]()
#' - [https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html]()
#'
#' @export
functional_allpass_biquad <- function(
  waveform,
  sample_rate,
  central_freq,
  Q = 0.707
) {

  w0 = 2 * pi * central_freq / sample_rate
  alpha = sin(w0) / 2 / Q

  b0 = 1 - alpha
  b1 = -2 * cos(w0)
  b2 = 1 + alpha
  a0 = 1 + alpha
  a1 = -2 * cos(w0)
  a2 = 1 - alpha
  return(functional_biquad(waveform, b0, b1, b2, a0, a1, a2))
}

#' Band-pass Biquad Filter (functional)
#'
#' Design two-pole band-pass filter.  Similar to SoX implementation.
#'
#' @param waveform  (Tensor): audio waveform of dimension of `(..., time)`
#' @param sample_rate  (int): sampling rate of the waveform, e.g. 44100 (Hz)
#' @param central_freq  (float): central frequency (in Hz)
#' @param Q  (float, optional): [https://en.wikipedia.org/wiki/Q_factor]() (Default: ``0.707``)
#' @param const_skirt_gain  (bool, optional) : If ``TRUE``, uses a constant skirt gain (peak gain = Q).
#'   If ``FALSE``, uses a constant 0dB peak gain.  (Default: ``FALSE``)
#'
#' @return Tensor: Waveform of dimension of `(..., time)`
#'
#' @references
#' - [https://sox.sourceforge.net/sox.html]()
#' - [https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html]()
#'
#' @export
functional_bandpass_biquad <- function(
  waveform,
  sample_rate,
  central_freq,
  Q = 0.707,
  const_skirt_gain = FALSE
) {

  w0 = 2 * pi * central_freq / sample_rate
  alpha = sin(w0) / 2 / Q

  temp =if(const_skirt_gain) sin(w0) / 2 else alpha
  b0 = temp
  b1 = 0.
  b2 = -temp
  a0 = 1 + alpha
  a1 = -2 * cos(w0)
  a2 = 1 - alpha
  return(functional_biquad(waveform, b0, b1, b2, a0, a1, a2))
}

#' Band-reject Biquad Filter (functional)
#'
#' Design two-pole band-reject filter.  Similar to SoX implementation.
#'
#' @param waveform  (Tensor): audio waveform of dimension of `(..., time)`
#' @param sample_rate  (int): sampling rate of the waveform, e.g. 44100 (Hz)
#' @param central_freq  (float): central frequency (in Hz)
#' @param Q  (float, optional): [https://en.wikipedia.org/wiki/Q_factor]() (Default: ``0.707``)
#'
#' @return `tensor`: Waveform of dimension of `(..., time)`
#'
#' @references
#' - [https://sox.sourceforge.net/sox.html]()
#' - [https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html]()
#'
#' @export
functional_bandreject_biquad <- function(
  waveform,
  sample_rate,
  central_freq,
  Q = 0.707
) {

  w0 = 2 * pi * central_freq / sample_rate
  alpha = sin(w0) / 2 / Q

  b0 = 1.
  b1 = -2 * cos(w0)
  b2 = 1.
  a0 = 1 + alpha
  a1 = -2 * cos(w0)
  a2 = 1 - alpha
  return(functional_biquad(waveform, b0, b1, b2, a0, a1, a2))
}

#' Biquad Peaking Equalizer Filter (functional)
#'
#' Design biquad peaking equalizer filter and perform filtering.  Similar to SoX implementation.
#'
#' @param waveform  (Tensor): audio waveform of dimension of `(..., time)`
#' @param sample_rate  (int): sampling rate of the waveform, e.g. 44100 (Hz)
#' @param center_freq  (float): filter's central frequency
#' @param gain (float): desired gain at the boost (or attenuation) in dB
#' @param Q (float, optional): [https://en.wikipedia.org/wiki/Q_factor]() (Default: ``0.707``)
#'
#' @return Tensor: Waveform of dimension of `(..., time)`
#'
#' @export
functional_equalizer_biquad <- function(
  waveform,
  sample_rate,
  center_freq,
  gain,
  Q = 0.707
) {

  w0 = 2 * pi * center_freq / sample_rate
  A = exp(gain / 40.0 * log(10))
  alpha = sin(w0) / 2 / Q

  b0 = 1 + alpha * A
  b1 = -2 * cos(w0)
  b2 = 1 - alpha * A
  a0 = 1 + alpha / A
  a1 = -2 * cos(w0)
  a2 = 1 - alpha / A
  return(functional_biquad(waveform, b0, b1, b2, a0, a1, a2))
}

#' Two-pole Band Filter (functional)
#'
#' Design two-pole band filter.  Similar to SoX implementation.
#'
#' @param waveform  (Tensor): audio waveform of dimension of `(..., time)`
#' @param sample_rate  (int): sampling rate of the waveform, e.g. 44100 (Hz)
#' @param central_freq  (float): central frequency (in Hz)
#' @param Q  (float, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``).
#' @param noise  (bool, optional) : If ``TRUE``, uses the alternate mode for un-pitched audio
#' (e.g. percussion). If ``FALSE``, uses mode oriented to pitched audio, i.e. voice, singing,
#' or instrumental music  (Default: ``FALSE``).
#'
#' @return `tensor`: Waveform of dimension of `(..., time)`
#'
#' @references
#' - [https://sox.sourceforge.net/sox.html]()
#' - [https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html]()
#'
#' @export
functional_band_biquad <- function(
  waveform,
  sample_rate,
  central_freq,
  Q = 0.707,
  noise = FALSE
) {

  w0 = 2 * pi * central_freq / sample_rate
  bw_Hz = central_freq / Q

  a0 = 1.
  a2 = exp(-2 * pi * bw_Hz / sample_rate)
  a1 = -4 * a2 / (1 + a2) * cos(w0)

  b0 = sqrt(1 - a1 * a1 / (4 * a2)) * (1 - a2)

  if(noise) {
    mult = sqrt(((1 + a2) * (1 + a2) - a1 * a1) * (1 - a2) / (1 + a2)) / b0
    b0 = b0 * mult
  }

  b1 = 0.
  b2 = 0.

  return(functional_biquad(waveform, b0, b1, b2, a0, a1, a2))
}

#' Treble Tone-control Effect (functional)
#'
#' Design a treble tone-control effect.  Similar to SoX implementation.
#'
#' @param waveform  (Tensor): audio waveform of dimension of `(..., time)`
#' @param sample_rate  (int): sampling rate of the waveform, e.g. 44100 (Hz)
#' @param gain  (float): desired gain at the boost (or attenuation) in dB.
#' @param central_freq  (float, optional): central frequency (in Hz). (Default: ``3000``)
#' @param Q  (float, optional): [https://en.wikipedia.org/wiki/Q_factor]() (Default: ``0.707``).
#'
#' @return `tensor`: Waveform of dimension of `(..., time)`
#'
#' @references
#' - [https://sox.sourceforge.net/sox.html]()
#' - [https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html]()
#'
#' @export
functional_treble_biquad <- function(
  waveform,
  sample_rate,
  gain,
  central_freq = 3000,
  Q = 0.707
) {

  w0 = 2 * pi * central_freq / sample_rate
  alpha = sin(w0) / 2 / Q
  A = exp(gain / 40 * log(10))

  temp1 = 2 * sqrt(A) * alpha
  temp2 = (A - 1) * cos(w0)
  temp3 = (A + 1) * cos(w0)

  b0 = A * ((A + 1) + temp2 + temp1)
  b1 = -2 * A * ((A - 1) + temp3)
  b2 = A * ((A + 1) + temp2 - temp1)
  a0 = (A + 1) - temp2 + temp1
  a1 = 2 * ((A - 1) - temp3)
  a2 = (A + 1) - temp2 - temp1

  return(functional_biquad(waveform, b0, b1, b2, a0, a1, a2))
}

#' Bass Tone-control Effect (functional)
#'
#' Design a bass tone-control effect.  Similar to SoX implementation.
#'
#' @param waveform  (Tensor): audio waveform of dimension of `(..., time)`
#' @param sample_rate  (int): sampling rate of the waveform, e.g. 44100 (Hz)
#' @param gain  (float): desired gain at the boost (or attenuation) in dB.
#' @param central_freq  (float, optional): central frequency (in Hz). (Default: ``100``)
#' @param Q  (float, optional): [https://en.wikipedia.org/wiki/Q_factor]() (Default: ``0.707``).
#'
#' @return `tensor`: Waveform of dimension of `(..., time)`
#'
#' @references
#' - [https://sox.sourceforge.net/sox.html]()
#' - [https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html]()
#'
#' @export
functional_bass_biquad <- function(
  waveform,
  sample_rate,
  gain,
  central_freq = 100,
  Q = 0.707
) {

  w0 = 2 * pi * central_freq / sample_rate
  alpha = sin(w0) / 2 / Q
  A = exp(gain / 40 * log(10))

  temp1 = 2 * sqrt(A) * alpha
  temp2 = (A - 1) * cos(w0)
  temp3 = (A + 1) * cos(w0)

  b0 = A * ((A + 1) - temp2 + temp1)
  b1 = 2 * A * ((A - 1) - temp3)
  b2 = A * ((A + 1) - temp2 - temp1)
  a0 = (A + 1) + temp2 + temp1
  a1 = -2 * ((A - 1) + temp3)
  a2 = (A + 1) + temp2 - temp1

  return(functional_biquad(waveform, b0 / a0, b1 / a0, b2 / a0, a0 / a0, a1 / a0, a2 / a0))
}


#' ISO 908 CD De-emphasis IIR Filter (functional)
#'
#' Apply ISO 908 CD de-emphasis (shelving) IIR filter.  Similar to SoX implementation.
#'
#' @param waveform  (Tensor): audio waveform of dimension of `(..., time)`
#' @param sample_rate  (int): sampling rate of the waveform, Allowed sample rate ``44100`` or ``48000``
#'
#' @return Tensor: Waveform of dimension of `(..., time)`
#'
#' @references
#' - [https://sox.sourceforge.net/sox.html]()
#' - [https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html]()
#'
#' @export
functional_deemph_biquad <- function(
  waveform,
  sample_rate
) {
  if(sample_rate == 44100) {
    central_freq = 5283
    width_slope = 0.4845
    gain = -9.477
  } else if(sample_rate == 48000) {
    central_freq = 5356
    width_slope = 0.479
    gain = -9.62
  } else {
    value_error("Sample rate must be 44100 (audio-CD) or 48000 (DAT)")
  }

  w0 = 2 * pi * central_freq / sample_rate
  A = exp(gain / 40.0 * log(10))
  alpha = sin(w0) / 2 * sqrt((A + 1 / A) * (1 / width_slope - 1) + 2)

  temp1 = 2 * sqrt(A) * alpha
  temp2 = (A - 1) * cos(w0)
  temp3 = (A + 1) * cos(w0)

  b0 = A * ((A + 1) + temp2 + temp1)
  b1 = -2 * A * ((A - 1) + temp3)
  b2 = A * ((A + 1) + temp2 - temp1)
  a0 = (A + 1) - temp2 + temp1
  a1 = 2 * ((A - 1) - temp3)
  a2 = (A + 1) - temp2 - temp1

  return(functional_biquad(waveform, b0, b1, b2, a0, a1, a2))
}

#' RIAA Vinyl Playback Equalisation (functional)
#'
#' Apply RIAA vinyl playback equalisation.  Similar to SoX implementation.
#'
#' @param waveform  (Tensor): audio waveform of dimension of `(..., time)`
#' @param sample_rate  (int): sampling rate of the waveform, e.g. 44100 (Hz).
#'  Allowed sample rates in Hz : ``44100``,``48000``,``88200``,``96000``
#'
#' @return `tensor`: Waveform of dimension of `(..., time)`
#'
#' @references
#' - [https://sox.sourceforge.net/sox.html]()
#' - [https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html]()
#'
#' @export
functional_riaa_biquad <- function(
  waveform,
  sample_rate
) {


  if(sample_rate == 44100) {
    zeros = c(-0.2014898, 0.9233820)
    poles = c(0.7083149, 0.9924091)

  } else if(sample_rate == 48000) {
    zeros = c(-0.1766069, 0.9321590)
    poles = c(0.7396325, 0.9931330)

  } else if(sample_rate == 88200) {
    zeros = c(-0.1168735, 0.9648312)
    poles = c(0.8590646, 0.9964002)

  } else if(sample_rate == 96000) {
    zeros = c(-0.1141486, 0.9676817)
    poles = c(0.8699137, 0.9966946)

  } else {
    value_error("Sample rate must be 44.1k, 48k, 88.2k, or 96k")
  }

  # polynomial coefficients with roots zeros[0] and zeros[1]
  b0 = 1.
  b1 = -(zeros[1] + zeros[2])
  b2 = (zeros[1] * zeros[2])

  # polynomial coefficients with roots poles[0] and poles[1]
  a0 = 1.
  a1 = -(poles[1] + poles[2])
  a2 = (poles[1] * poles[2])

  # Normalise to 0dB at 1kHz
  y = 2 * pi * 1000 / sample_rate
  b_re = b0 + b1 * cos(-y) + b2 * cos(-2 * y)
  a_re = a0 + a1 * cos(-y) + a2 * cos(-2 * y)
  b_im = b1 * sin(-y) + b2 * sin(-2 * y)
  a_im = a1 * sin(-y) + a2 * sin(-2 * y)
  g = 1 / sqrt((b_re ** 2 + b_im ** 2) / (a_re ** 2 + a_im ** 2))

  b0 = b0 * g
  b1 = b1 * g
  b2 = b2 * g

  return(functional_biquad(waveform, b0, b1, b2, a0, a1, a2))
}


#' Contrast Effect (functional)
#'
#' Apply contrast effect.  Similar to SoX implementation.
#' Comparable with compression, this effect modifies an audio signal to
#' make it sound louder
#'
#' @param waveform  (Tensor): audio waveform of dimension of `(..., time)`
#' @param enhancement_amount  (float): controls the amount of the enhancement
#' Allowed range of values for enhancement_amount : 0-100
#' Note that enhancement_amount = 0 still gives a significant contrast enhancement
#'
#' @return `tensor`: Waveform of dimension of `(..., time)`
#'
#' @references
#' - [https://sox.sourceforge.net/sox.html]()
#'
#' @export
functional_contrast <- function(
  waveform,
  enhancement_amount = 75.0
) {

  if(enhancement_amount < 0 | enhancement_amount > 100) {
    value_error("Allowed range of values for enhancement_amount : 0-100")
  }
  contrast = enhancement_amount / 750.

  temp1 = waveform * (pi / 2)
  temp2 = contrast * torch::torch_sin(temp1 * 4)
  output_waveform = torch::torch_sin(temp1 + temp2)

  return(output_waveform)
}

#' DC Shift (functional)
#'
#' Apply a DC shift to the audio. Similar to SoX implementation.
#'    This can be useful to remove a DC offset (caused perhaps by a
#'    hardware problem in the recording chain) from the audio
#'
#' @param waveform  (Tensor): audio waveform of dimension of `(..., time)`
#' @param shift  (float): indicates the amount to shift the audio
#'  Allowed range of values for shift : -2.0 to +2.0
#' @param limiter_gain  (float): It is used only on peaks to prevent clipping
#'  It should have a value much less than 1  (e.g. 0.05 or 0.02)
#'
#' @return `tensor`: Waveform of dimension of `(..., time)`
#'
#' @references
#' - [https://sox.sourceforge.net/sox.html]()
#'
#' @export
functional_dcshift <- function(
  waveform,
  shift,
  limiter_gain = NULL
) {
  output_waveform = waveform
  limiter_threshold = 0.0

  if(!is.null(limiter_gain)) {
    limiter_threshold = 1.0 - (abs(shift) - limiter_gain)
  }

  if(!is.null(limiter_gain) & shift > 0) {
    mask = (waveform > limiter_threshold)
    temp = (waveform[mask] - limiter_threshold) * limiter_gain / (1 - limiter_threshold)
    output_waveform[mask] = (temp + limiter_threshold + shift)$clamp(max = limiter_threshold)
    output_waveform[!mask] = (waveform[!mask] + shift)$clamp(min=-1.0, max=1.0)
  } else if(!is.null(limiter_gain) & shift < 0) {
    mask = waveform < -limiter_threshold
    temp = (waveform[mask] + limiter_threshold) * limiter_gain / (1 - limiter_threshold)
    output_waveform[mask] = (temp - limiter_threshold + shift)$clamp(min=-limiter_threshold)
    output_waveform[!mask] = (waveform[!mask] + shift)$clamp(min=-1.0, max=1.0)
  } else {
    output_waveform = (waveform + shift)$clamp(min=-1.0, max=1.0)
  }

  return(output_waveform)
}

#' Overdrive Effect (functional)
#'
#' Apply a overdrive effect to the audio. Similar to SoX implementation.
#'    This effect applies a non linear distortion to the audio signal.
#'
#' @param waveform  (Tensor): audio waveform of dimension of `(..., time)`
#' @param gain  (float): desired gain at the boost (or attenuation) in dB
#'  Allowed range of values are 0 to 100
#' @param colour  (float):  controls the amount of even harmonic content in
#' the over-driven output. Allowed range of values are 0 to 100
#'
#' @return Tensor: Waveform of dimension of `(..., time)`
#'
#' @references
#' - [https://sox.sourceforge.net/sox.html]()
#'
#' @export
functional_overdrive <- function(
  waveform,
  gain = 20,
  colour = 20
) {

  actual_shape = waveform$shape
  device = waveform$device
  dtype = waveform$dtype

  # convert to 2D (..,time)
  waveform = waveform$view(c(-1, actual_shape[length(actual_shape)]))
  lws = length(waveform$shape)
  gain = db_to_linear(gain)
  colour = colour / 200
  last_in = torch::torch_zeros(waveform$shape[-lws], dtype=dtype, device=device)
  last_out = torch::torch_zeros(waveform$shape[-lws], dtype=dtype, device=device)

  temp = waveform * gain + colour

  mask1 = temp < -1
  temp[mask1] = torch::torch_tensor(-2.0 / 3.0, dtype=dtype, device=device)
  # Wrapping the constant with Tensor is required for Torchscript

  mask2 = temp > 1
  temp[mask2] = torch::torch_tensor(2.0 / 3.0, dtype=dtype, device=device)

  mask3 = (!mask1 & !mask2)
  temp[mask3] = temp[mask3] - (temp[mask3]**3) * (1. / 3)

  output_waveform = torch::torch_zeros_like(waveform, dtype=dtype, device=device)

  # TODO: Implement a torch CPP extension
  for(i in seq(waveform$shape[lws])) {
    last_out = temp[ , i] - last_in + 0.995 * last_out
    last_in = temp[ , i]
    output_waveform[ , i] = waveform[ , i] * 0.5 + last_out * 0.75
  }

  return(output_waveform$clamp(min=-1.0, max=1.0)$view(actual_shape))
}

#' Phasing Effect (functional)
#'
#' Apply a phasing effect to the audio. Similar to SoX implementation.
#'
#' @param waveform  (Tensor): audio waveform of dimension of `(..., time)`
#' @param sample_rate  (int): sampling rate of the waveform, e.g. 44100 (Hz)
#' @param gain_in  (float): desired input gain at the boost (or attenuation) in dB.
#'  Allowed range of values are 0 to 1
#' @param gain_out  (float): desired output gain at the boost (or attenuation) in dB.
#'  Allowed range of values are 0 to 1e9
#' @param delay_ms  (float): desired delay in milli seconds.
#'  Allowed range of values are 0 to 5.0
#' @param decay  (float):  desired decay relative to gain-in. Allowed range of values are 0 to 0.99
#' @param mod_speed  (float):  modulation speed in Hz.
#'  Allowed range of values are 0.1 to 2
#' @param sinusoidal  (bool):  If ``TRUE``, uses sinusoidal modulation (preferable for multiple instruments).
#'  If ``FALSE``, uses triangular modulation  (gives single instruments a sharper phasing effect)
#' (Default: ``TRUE``)
#'
#' @return `tensor`: Waveform of dimension of `(..., time)`
#'
#' @references
#' - [https://sox.sourceforge.net/sox.html]()
#'
#' @export
functional_phaser <- function(
  waveform,
  sample_rate,
  gain_in = 0.4,
  gain_out = 0.74,
  delay_ms = 3.0,
  decay = 0.4,
  mod_speed = 0.5,
  sinusoidal = TRUE
) {
  actual_shape = waveform$shape
  device = waveform$device
  dtype = waveform$dtype

  # convert to 2D (channels,time)
  las = length(actual_shape)
  waveform = waveform$view(c(-1, actual_shape[las]))

  delay_buf_len = as.integer((delay_ms * .001 * sample_rate) + .5)
  delay_buf = torch::torch_zeros(waveform$shape[1], delay_buf_len, dtype=dtype, device=device)

  mod_buf_len = as.integer(sample_rate / mod_speed + .5)

  if(sinusoidal) {
    wave_type = 'SINE'
  } else {
    wave_type = 'TRIANGLE'
  }

  mod_buf = functional__generate_wave_table(
    wave_type = wave_type,
    data_type = 'INT',
    table_size = mod_buf_len,
    min = 1.0,
    max = as.numeric(delay_buf_len),
    phase = pi / 2,
    device = device
  )

  delay_pos = 1
  mod_pos = 1

  output_waveform_pre_gain_list = list()
  waveform = waveform * gain_in
  delay_buf = delay_buf * decay
  waveform_list = if(ncol(waveform) > 0) Map(function(i) waveform[,i], seq(ncol(waveform))) else list()
  delay_buf_list = if(ncol(delay_buf) > 0) Map(function(i) delay_buf[,i], seq(ncol(delay_buf))) else list()
  mod_buf_list = if(nrow(mod_buf) > 0) Map(function(i) mod_buf[i], seq(nrow(mod_buf))) else list()

  lws = length(waveform$shape)
  for(i in seq.int(waveform$shape[lws])) {
    idx = as.integer((delay_pos + mod_buf_list[[mod_pos]]) %% delay_buf_len)
    mod_pos = (mod_pos + 1) %% mod_buf_len
    delay_pos = (delay_pos + 1) %% delay_buf_len
    temp = (waveform_list[[i]]) + (delay_buf_list[[idx+1]])
    delay_buf_list[[delay_pos+1]] = temp * decay
    output_waveform_pre_gain_list[[length(output_waveform_pre_gain_list) + 1]] <- temp
  }

  output_waveform = torch::torch_stack(output_waveform_pre_gain_list, dim=2)$to(dtype=dtype, device=device)
  output_waveform$mul_(gain_out)

  return(output_waveform$clamp(min=-1.0, max=1.0)$view(actual_shape))
}

#' Wave Table Generator (functional)
#'
#' A helper function for phaser. Generates a table with given parameters
#'
#' @param wave_type  (str): 'SINE' or 'TRIANGULAR'
#' @param data_type  (str): desired data_type ( `INT` or `FLOAT` )
#' @param table_size  (int): desired table size
#' @param min  (float): desired min value
#' @param max  (float): desired max value
#' @param phase  (float): desired phase
#' @param device  (torch_device): Torch device on which table must be generated
#'
#' @return `tensor`: A 1D tensor with wave table values
#'
#' @export
functional__generate_wave_table <- function(
  wave_type,
  data_type,
  table_size,
  min,
  max,
  phase,
  device
) {

  phase_offset = as.integer(phase / pi / 2 * table_size + 0.5)
  t = torch::torch_arange(0, table_size-1, device=device, dtype=torch::torch_int32())
  point = (t + phase_offset) %% table_size
  d = torch::torch_zeros_like(point, device=device, dtype=torch::torch_float64())

  if(wave_type == 'SINE') {
    d = (torch::torch_sin(point$to(torch::torch_float64()) / table_size * 2 * pi) + 1) / 2
  } else if(wave_type == 'TRIANGLE') {
    d = (point$to(torch::torch_float64()) * 2) / table_size
    value = (4 * point) %/% table_size
    d[value == 0] = d[value == 0] + 0.5
    d[value == 1] = 1.5 - d[value == 1]
    d[value == 2] = 1.5 - d[value == 2]
    d[value == 3] = d[value == 3] - 1.5
  }

  d = d * (max - min) + min

  if(data_type == 'INT') {
    mask = d < 0
    d[mask] = d[mask] - 0.5
    d[!mask] = d[!mask] + 0.5
    d = d$to(torch::torch_int32())
  } else if(data_type == 'FLOAT') {
    d = d$to(torch::torch_float32())
  }

  return(d)
}

#' Flanger Effect (functional)
#'
#' Apply a flanger effect to the audio. Similar to SoX implementation.
#'
#' @param waveform  (Tensor): audio waveform of dimension of `(..., channel, time)` .
#'            Max 4 channels allowed
#' @param sample_rate  (int): sampling rate of the waveform, e.g. 44100 (Hz)
#' @param delay  (float): desired delay in milliseconds(ms).
#'            Allowed range of values are 0 to 30
#' @param depth  (float): desired delay depth in milliseconds(ms).
#'            Allowed range of values are 0 to 10
#' @param regen  (float): desired regen(feeback gain) in dB.
#'            Allowed range of values are -95 to 95
#' @param width  (float):  desired width(delay gain) in dB.
#'            Allowed range of values are 0 to 100
#' @param speed  (float):  modulation speed in Hz.
#'            Allowed range of values are 0.1 to 10
#' @param phase  (float):  percentage phase-shift for multi-channel.
#'            Allowed range of values are 0 to 100
#' @param modulation  (str):  Use either "sinusoidal" or "triangular" modulation. (Default: ``sinusoidal``)
#' @param interpolation  (str): Use either "linear" or "quadratic" for delay-line interpolation. (Default: ``linear``)
#'
#' @return `tensor`: Waveform of dimension of `(..., channel, time)`
#'
#' @references
#' - [https://sox.sourceforge.net/sox.html]()
#'- Scott Lehman, Effects Explained, [https://web.archive.org/web/20051125072557/http://www.harmony-central.com/Effects/effects-explained.html]()
#'
#' @export
functional_flanger <- function(
  waveform,
  sample_rate,
  delay = 0.0,
  depth = 2.0,
  regen = 0.0,
  width = 71.0,
  speed = 0.5,
  phase = 25.0,
  modulation = "sinusoidal",
  interpolation = "linear"
) {

  if(!modulation %in% c("sinusoidal", "triangular")) {
    value_error("Only 'sinusoidal' or 'triangular' modulation allowed")
  }

  if(!interpolation %in% c("linear", "quadratic")) {
    value_error("Only 'linear' or 'quadratic' interpolation allowed")
  }

  actual_shape = waveform$shape
  device = waveform$device
  dtype = waveform$dtype

  las = length(actual_shape)
  if(actual_shape[las-1] > 4) {
    value_error("Max 4 channels allowed")
  }

  # convert to 3D (batch, channels, time)
  waveform = waveform$view(c(-1, actual_shape[las-1], actual_shape[las]))

  # Scaling
  feedback_gain = regen / 100
  delay_gain = width / 100
  channel_phase = phase / 100
  delay_min = delay / 1000
  delay_depth = depth / 1000

  lws = length(waveform$shape)
  n_channels = waveform$shape[lws-1]

  if(modulation == "sinusoidal") {
    wave_type = "SINE"
  } else {
    wave_type = "TRIANGLE"
  }

  # Balance output:
  in_gain = 1. / (1 + delay_gain)
  delay_gain = delay_gain / (1 + delay_gain)

  # Balance feedback loop:
  delay_gain = delay_gain * (1 - abs(feedback_gain))

  delay_buf_length = as.integer((delay_min + delay_depth) * sample_rate + 0.5)
  delay_buf_length = as.integer(delay_buf_length + 2)

  delay_bufs = torch::torch_zeros(waveform$shape[1], n_channels, delay_buf_length, dtype=dtype, device=device)
  delay_last = torch::torch_zeros(waveform$shape[1], n_channels, dtype=dtype, device=device)

  lfo_length = as.integer(sample_rate / speed)

  table_min = floor(delay_min * sample_rate + 0.5)
  table_max = delay_buf_length - 2.

  lfo = functional__generate_wave_table(
    wave_type = wave_type,
    data_type = "FLOAT",
    table_size = lfo_length,
    min = as.numeric(table_min),
    max = as.numeric(table_max),
    phase = 3 * pi / 2,
    device = device
  )

  output_waveform = torch::torch_zeros_like(waveform, dtype=dtype, device=device)

  delay_buf_pos = 0L
  lfo_pos = 0L
  channel_idxs = torch::torch_arange(0, n_channels-1, device=device)$to(torch::torch_long())

  for(i in seq.int(waveform$shape[lws])) {
    delay_buf_pos = (delay_buf_pos + delay_buf_length - 1L) %% delay_buf_length

    cur_channel_phase = (channel_idxs * lfo_length * channel_phase + .5)$to(torch::torch_long())
    delay_tensor = lfo[((lfo_pos + cur_channel_phase) %% lfo_length)$to(torch::torch_long()) + 1L]
    frac_delay = torch::torch_frac(delay_tensor)
    delay_tensor = torch::torch_floor(delay_tensor)

    int_delay = delay_tensor$to(torch::torch_long())

    temp = waveform[ ,  , i]

    delay_bufs[ ,  , delay_buf_pos+1] = temp + delay_last * feedback_gain

    delayed_0 = torch_index(
      delay_bufs,
      list(torch_arange(1, dim(delay_bufs)[1])$to(dtype = torch_long()),
           channel_idxs + 1L,
           (delay_buf_pos + int_delay) %% delay_buf_length + 1L
           )
    )

    int_delay = int_delay + 1L

    delayed_1 = torch_index(
      delay_bufs,
      list(torch_arange(1, dim(delay_bufs)[1])$to(dtype = torch_long()),
           channel_idxs + 1L,
           (delay_buf_pos + int_delay) %% delay_buf_length + 1L
      )
    )

    int_delay = int_delay + 1L

    if(interpolation == "linear") {
      delayed = delayed_0 + (delayed_1 - delayed_0) * frac_delay
    } else {
      delayed_2 =
        torch_index(
          delay_bufs,
          list(torch_arange(1, dim(delay_bufs)[1])$to(dtype = torch_long()),
               channel_idxs + 1L,
               (delay_buf_pos + int_delay) %% delay_buf_length + 1L
          )
        )
      int_delay = int_delay + 1L

      delayed_2 = delayed_2 - delayed_0
      delayed_1 = delayed_1 - delayed_0
      a = delayed_2 * .5 - delayed_1
      b = delayed_1 * 2 - delayed_2 * .5

      delayed = delayed_0 + (a * frac_delay + b) * frac_delay
    }

    delay_last = delayed
    output_waveform[ ,  , i] = waveform[ ,  , i] * in_gain + delayed * delay_gain

    lfo_pos = (lfo_pos + 1) %% lfo_length
  }

  return(output_waveform$clamp(min=-1.0, max=1.0)$view(actual_shape))
}

#' Mask Along Axis IID (functional)
#'
#' Apply a mask along ``axis``. Mask will be applied from indices ``[v_0, v_0 + v)``, where
#' ``v`` is sampled from ``uniform (0, mask_param)``, and ``v_0`` from ``uniform(0, max_v - v)``.
#'
#' @param specgrams  (Tensor): Real spectrograms (batch, channel, freq, time)
#' @param mask_param  (int): Number of columns to be masked will be uniformly sampled from ``[0, mask_param]``
#' @param mask_value  (float): Value to assign to the masked columns
#' @param axis  (int): Axis to apply masking on (3 -> frequency, 4 -> time)
#'
#' @return `tensor`: Masked spectrograms of dimensions (batch, channel, freq, time)
#'
#' @export
functional_mask_along_axis_iid <- function(
  specgrams,
  mask_param,
  mask_value,
  axis
) {

  if(axis != 3 & axis != 4)
    value_error("Only Frequency (axis 3) and Time (axis 4) masking are supported")

  device = specgrams$device
  dtype = specgrams$dtype

  value = torch::torch_rand(specgrams$shape[1:2], device=device, dtype=dtype) * mask_param
  min_value = torch::torch_rand(specgrams$shape[1:2], device=device, dtype=dtype) * (specgrams$size(axis) - value)

  # Create broadcastable mask
  mask_start = min_value[.., NULL, NULL]
  mask_end = (min_value + value)[.., NULL, NULL]
  mask = torch::torch_arange(0, specgrams$size(axis)-1, device=device, dtype=dtype)

  # Per batch example masking
  specgrams = specgrams$transpose(axis, -1)
  specgrams$masked_fill_((mask >= mask_start) & (mask < mask_end), mask_value)
  specgrams = specgrams$transpose(axis, -1)

  return(specgrams)
}

#' Mask Along Axis (functional)
#'
#' Apply a mask along ``axis``. Mask will be applied from indices ``[v_0, v_0 + v)``, where
#' ``v`` is sampled from ``uniform (0, mask_param)``, and ``v_0`` from ``uniform(0, max_v - v)``.
#' All examples will have the same mask interval.
#'
#' @param specgram  (Tensor): Real spectrogram (channel, freq, time)
#' @param mask_param  (int): Number of columns to be masked will be uniformly sampled from ``[0, mask_param]``
#' @param mask_value  (float): Value to assign to the masked columns
#' @param axis  (int): Axis to apply masking on (2 -> frequency, 3 -> time)
#'
#' @return Tensor: Masked spectrogram of dimensions (channel, freq, time)
#'
#' @export
functional_mask_along_axis <- function(
  specgram,
  mask_param,
  mask_value,
  axis
) {
  # pack batch
  shape = specgram$size()
  ls = length(shape)
  specgram = specgram$reshape(c(-1, shape[(ls-1):ls]))
  shape = specgram$size()
  ls = length(shape)

  value = torch::torch_rand(1) * mask_param
  min_value = torch::torch_rand(1) * (specgram$size(axis) - value)

  mask_start = as.integer((min_value$to(torch::torch_long()))$squeeze())
  mask_end = as.integer((min_value$to(torch::torch_long()) + value$to(torch::torch_long()))$squeeze())

  if(as.logical((mask_end - mask_start) >= mask_param)) {
    value_error("mask_end - mask_start >= mask_param")
  }

  if(axis == 2) {
    specgram[ , mask_start:mask_end] = mask_value
  } else if(axis == 3) {
    specgram[ ,  , mask_start:mask_end] = mask_value
  } else {
    value_error("Only Frequency and Time masking are supported")
  }

  # unpack batch
  lss = length(specgram$shape)
  specgram = specgram$reshape(c(shape[1:(ls-2)], specgram$shape[(lss-1):lss]))

  return(specgram)
}

#' Delta Coefficients (functional)
#'
#' Compute delta coefficients of a tensor, usually a spectrogram.
#'
#' math:
#'  \deqn{d_t = \frac{\sum_{n=1}^{N} n  (c_{t+n} - c_{t-n})}{2 \sum_{n=1}^{N} n^2}}
#'
#'  where `d_t` is the deltas at time `t`, `c_t` is the spectrogram coeffcients at time `t`,
#'  `N` is `` (win_length-1) %/% 2``.
#'
#' @param specgram  (Tensor): Tensor of audio of dimension (..., freq, time)
#' @param win_length  (int, optional): The window length used for computing delta (Default: ``5``)
#' @param mode  (str, optional): Mode parameter passed to padding (Default: ``"replicate"``)
#'
#' @return `tensor`: Tensor of deltas of dimension (..., freq, time)
#'
#' @examples
#' if(torch::torch_is_installed()) {
#' library(torch)
#' library(torchaudio)
#' specgram = torch_randn(1, 40, 1000)
#' delta = functional_compute_deltas(specgram)
#' delta2 = functional_compute_deltas(delta)
#' }
#'
#' @export
functional_compute_deltas <- function(
  specgram,
  win_length = 5,
  mode = "replicate"
) {
  specgram = torch::torch_randn(1, 4, 10)
  device = specgram$device
  dtype = specgram$dtype

  # pack batch
  shape = specgram$size()
  ls = length(shape)
  specgram = specgram$reshape(c(1, -1, shape[ls]))
  ls = length(shape)
  if(win_length < 3) value_error("win_length must be >= 3.")

  n = (win_length - 1) %/% 2

  # twice sum of integer squared
  denom = n * (n + 1) * (2 * n + 1) / 3

  specgram = torch::nnf_pad(specgram, c(n, n), mode=mode)
  kernel = torch::torch_arange(-n, n, 1, device=device, dtype=dtype)$`repeat`(c(specgram$shape[2], 1, 1))
  output = torch::nnf_conv1d(specgram, kernel, groups=specgram$shape[2]) / denom

  # unpack batch
  output = output$reshape(shape)

  return(output)
}

#' Gain (functional)
#'
#' Apply amplification or attenuation to the whole waveform.
#'
#' @param waveform  (Tensor): Tensor of audio of dimension (..., time).
#' @param gain_db  (float, optional) Gain adjustment in decibels (dB) (Default: ``1.0``).
#'
#' @return `tensor`: the whole waveform amplified by gain_db.
#'
#' @export
functional_gain <- function(
  waveform,
  gain_db = 1.0
) {

  if((gain_db == 0)) {
    return(waveform)
  }

  ratio = 10 ** (gain_db / 20)

  return(waveform * ratio)
}

#' Noise Shaping (functional)
#'
#' Noise shaping is calculated by error:
#'    error\[n\] = dithered\[n\] - original\[n\]
#'    noise_shaped_waveform\[n\] = dithered\[n\] + error\[n-1\]
#'
#' @param dithered_waveform (Tensor) dithered
#' @param waveform (Tensor) original
#'
#' @return `tensor` of the noise shaped waveform
#'
#' @export
functional_add_noise_shaping <- function(
  dithered_waveform,
  waveform
) {
  wf_shape = waveform$size()
  lws = length(wf_shape)
  waveform = waveform$reshape(c(-1, wf_shape[lws]))

  dithered_shape = dithered_waveform$size()
  lds = length(dithered_shape)
  dithered_waveform = dithered_waveform$reshape(c(-1, dithered_shape[lds]))

  error = dithered_waveform - waveform

  # add error[n-1] to dithered_waveform[n], so offset the error by 1 index
  zeros = torch::torch_zeros(1, dtype=error$dtype, device=error$device)
  for(index in seq.int(error$size()[1])) {
    err = error[index]
    error_offset = torch::torch_cat(list(zeros, err))
    error[index] = error_offset[1:waveform$size()[2]]
  }

  noise_shaped = dithered_waveform + error
  return(noise_shaped$reshape(c(dithered_shape[-lds], noise_shaped$shape[length(noise_shaped$shape)])))
}

#' Probability Distribution Apply (functional)
#'
#' Apply a probability distribution function on a waveform.
#'
#' @param waveform  (Tensor): Tensor of audio of dimension (..., time)
#' @param density_function  (str, optional): The density function of a
#' continuous random variable  (Default: ``"TPDF"``)
#'           Options: Triangular Probability Density Function - `TPDF`
#'                    Rectangular Probability Density Function - `RPDF`
#'                    Gaussian Probability Density Function - `GPDF`
#'
#' @details
#'
#' - **Triangular** probability density function  (TPDF) dither noise has a
#'    triangular distribution; values in the center of the range have a higher
#'    probability of occurring.
#'
#' - **Rectangular** probability density function  (RPDF) dither noise has a
#'    uniform distribution; any value in the specified range has the same
#'    probability of occurring.
#'
#' - **Gaussian** probability density function  (GPDF) has a normal distribution.
#'    The relationship of probabilities of results follows a bell-shaped,
#'    or Gaussian curve, typical of dither generated by analog sources.
#'
#' @return `tensor`: waveform dithered with TPDF
#'
#' @export
functional_apply_probability_distribution <- function(
  waveform,
  density_function = "TPDF"
) {
  # pack batch
  shape = waveform$size()
  ls = length(shape)
  waveform = waveform$reshape(c(-1, shape[ls]))
  shape = waveform$size()
  ls = length(shape)

  channel_size = waveform$size()[1]
  time_size = waveform$size()[ls]

  random_channel = if(channel_size > 0) as.integer(torch::torch_randint(1, channel_size, list(1))$item()) else 0L

  random_time = if(time_size > 0) as.integer(torch::torch_randint(1, time_size, list(1))$item()) else 0L

  number_of_bits = 16
  up_scaling = 2 ** (number_of_bits - 1) - 2
  signal_scaled = waveform * up_scaling
  down_scaling = 2 ** (number_of_bits - 1)

  signal_scaled_dis = waveform
  if (density_function == "RPDF") {
    RPDF = waveform[random_channel][random_time] - 0.5
    signal_scaled_dis = signal_scaled + RPDF
  } else if((density_function == "GPDF")) {
    # TODO Replace by distribution code once
    # https://github.com/pytorch/pytorch/issues/29843 is resolved
    # gaussian = torch::torch_distributions.normal.Normal(torch::torch_mean(waveform, -1), 1).sample()

    num_rand_variables = 6
    gaussian = waveform[random_channel][random_time]
    for(ws in  rep(time_size, num_rand_variables)) {
      rand_chan = as.integer(torch::torch_randint(1, channel_size, list(1))$item())
      gaussian = gaussian + waveform[rand_chan][as.integer(torch::torch_randint(1, ws,  list(1))$item())]
    }

    signal_scaled_dis = signal_scaled + gaussian
  } else {
    # dtype needed for https://github.com/pytorch/pytorch/issues/32358
    TPDF = torch::torch_bartlett_window(time_size, dtype=signal_scaled$dtype, device=signal_scaled$device)
    TPDF = TPDF$`repeat`(c((channel_size), 1))
    signal_scaled_dis = signal_scaled + TPDF
  }

  quantised_signal_scaled = torch::torch_round(signal_scaled_dis)
  quantised_signal = quantised_signal_scaled / down_scaling

  # unpack batch
  return(quantised_signal$reshape(c(shape[-ls], quantised_signal$shape[length(quantised_signal$shape)])))
}

#' Dither (functional)
#'
#' Dither increases the perceived dynamic range of audio stored at a
#'    particular bit-depth by eliminating nonlinear truncation distortion
#'    (i.e. adding minimally perceived noise to mask distortion caused by quantization).
#' @param waveform  (Tensor): Tensor of audio of dimension (..., time)
#' @param density_function  (str, optional): The density function of a continuous random variable (Default: ``"TPDF"``)
#'           Options: Triangular Probability Density Function - `TPDF`
#'                    Rectangular Probability Density Function - `RPDF`
#'                    Gaussian Probability Density Function - `GPDF`
#' @param noise_shaping  (bool, optional): a filtering process that shapes the spectral
#'  energy of quantisation error  (Default: ``FALSE``)
#'
#' @return `tensor`: waveform dithered
#'
#' @export
functional_dither <- function(
  waveform,
  density_function = "TPDF",
  noise_shaping = FALSE
) {

  dithered = functional_apply_probability_distribution(waveform, density_function=density_function)

  if(noise_shaping) {
    return(functional_add_noise_shaping(dithered, waveform))
  } else {
    return(dithered)
  }
}

#' Normalized Cross-Correlation Function (functional)
#'
#' Compute Normalized Cross-Correlation Function  (NCCF).
#'
#' @param waveform  (Tensor): Tensor of audio of dimension (..., time)
#' @param sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
#' @param frame_time (float)
#' @param freq_low (float)
#'
#' @return `tensor` of nccf``
#'
#' @export
functional__compute_nccf <- function(
  waveform,
  sample_rate,
  frame_time,
  freq_low
) {

  EPSILON = 10 ** (-9)

  # Number of lags to check
  lags = as.integer(ceiling(sample_rate / freq_low))

  frame_size = as.integer(ceiling(sample_rate * frame_time))

  lws = length(waveform$size())
  waveform_length = waveform$size()[lws]
  num_of_frames = as.integer(ceiling(waveform_length / frame_size))

  p = lags + num_of_frames * frame_size - waveform_length
  waveform = torch::nnf_pad(waveform, c(0, p))

  # Compute lags
  output_lag = list()
  for(lag in 1:(lags)) {
    s1 = waveform[.., 1:(waveform$size()[lws] - lag)]$unfold(-1, frame_size, frame_size)[.., 1:num_of_frames, ]
    s2 = waveform[.., (1 + lag):(waveform$size()[lws])]$unfold(-1, frame_size, frame_size)[.., 1:num_of_frames, ]

    output_frames = (
      (s1 * s2)$sum(-1)
      / (EPSILON + torch::torch_norm(s1, p=2L, dim=-1))$pow(2)
      / (EPSILON + torch::torch_norm(s2, p=2L, dim=-1))$pow(2)
    )

    output_lag[[length(output_lag) + 1]] <- output_frames$unsqueeze(-1)
  }

  nccf = torch::torch_cat(output_lag, -1)

  return(nccf)
}

#' Combine Max (functional)
#'
#' Take value from first if bigger than a multiplicative factor of the second, elementwise.
#'
#' @param a (list(tensor, tensor))
#' @param b (list(tensor, tensor))
#' @param thresh (float) Default: 0.99
#'
#' @return `list(tensor, tensor)`: a list with values tensor and indices tensor.
#'
#' @export
functional__combine_max <- function(
  a,
  b,
  thresh = 0.99
) {
  mask = (a[[1]] > thresh * b[[1]])
  values = mask * a[[1]] + (!mask) * b[[1]]
  indices = mask * a[[2]] + (!mask) * b[[2]]

  return(list(values, indices))
}

#' Find Max Per Frame (functional)
#'
#'  For each frame, take the highest value of NCCF,
#'  apply centered median smoothing, and convert to frequency.
#'
#' @param nccf (tensor): Usually a tensor returned by [torchaudio::functional__compute_nccf]
#' @param sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
#' @param freq_high  (int): Highest frequency that can be detected (Hz)
#'
#'  Note: If the max among all the lags is very close
#'  to the first half of lags, then the latter is taken.
#'
#' @return `tensor` with indices
#'
#' @export
functional__find_max_per_frame <- function(
  nccf,
  sample_rate,
  freq_high
) {

  lag_min = as.integer(ceiling(sample_rate / freq_high))

  # Find near enough max that is smallest

  lns = length(nccf$shape)
  best = torch::torch_max(nccf[.., (lag_min+1):nccf$shape[lns]], -1)

  half_size = nccf$shape[lns] %/% 2
  half = torch::torch_max(nccf[.., (lag_min+1):half_size], -1)

  best = functional__combine_max(half, best)
  indices = best[[2]]

  # Add back minimal lag
  indices = indices + lag_min
  # Add 1 empirical calibration offset
  indices = indices + 1

  return(indices)
}

#' Median Smoothing (functional)
#'
#' Apply median smoothing to the 1D tensor over the given window.
#'
#' @param indices (Tensor)
#' @param win_length (int)
#'
#' @return `tensor`
#'
#' @export
functional__median_smoothing <- function(
  indices,
  win_length
) {
  # Centered windowed
  pad_length = (win_length - 1) %/% 2

  # "replicate" padding in any dimension
  indices = torch::nnf_pad(
    indices, c(pad_length, 0), mode="constant", value=0.
  )

  indices[.., 1:pad_length] = torch::torch_cat(replicate(pad_length, indices[.., pad_length + 1]$unsqueeze(-1)), dim=-1)
  roll = indices$unfold(-1, win_length, 1)

  values = torch::torch_median(roll, -1)[[1]]
  return(values)
}

#' Detect Pitch Frequency (functional)
#'
#' It is implemented using normalized cross-correlation function and median smoothing.
#'
#' @param waveform  (Tensor): Tensor of audio of dimension (..., freq, time)
#' @param sample_rate  (int): The sample rate of the waveform (Hz)
#' @param frame_time  (float, optional): Duration of a frame (Default: ``10 ** (-2)``).
#' @param win_length  (int, optional): The window length for median smoothing (in number of frames) (Default: ``30``).
#' @param freq_low  (int, optional): Lowest frequency that can be detected (Hz) (Default: ``85``).
#' @param freq_high  (int, optional): Highest frequency that can be detected (Hz) (Default: ``3400``).
#'
#' @return Tensor: Tensor of freq of dimension (..., frame)
#'
#' @export
functional_detect_pitch_frequency <- function(
  waveform,
  sample_rate,
  frame_time = 10 ** (-2),
  win_length = 30,
  freq_low = 85,
  freq_high = 3400
) {

  # pack batch
  shape = waveform$size()
  ls = length(shape)
  waveform = waveform$reshape(c(-1, shape[ls]))

  nccf = functional__compute_nccf(waveform, sample_rate, frame_time, freq_low)
  indices = functional__find_max_per_frame(nccf, sample_rate, freq_high)
  indices = functional__median_smoothing(indices, win_length)

  # Convert indices to frequency
  EPSILON = 10 ** (-9)
  freq = sample_rate / (EPSILON + indices$to(torch::torch_float()))

  # unpack batch
  lfs = length(freq$shape)
  freq = freq$reshape(c(shape[-ls], freq$shape[lfs]))

  return(freq)
}

#' sliding-window Cepstral Mean Normalization (functional)
#'
#' Apply sliding-window cepstral mean  (and optionally variance) normalization per utterance.
#'
#' @param waveform  (Tensor): Tensor of audio of dimension (..., freq, time)
#' @param cmn_window  (int, optional): Window in frames for running average CMN computation (int, default = 600)
#' @param min_cmn_window  (int, optional):  Minimum CMN window used at start of decoding (adds latency only at start).
#'  Only applicable if center == ``FALSE``, ignored if center==``TRUE``  (int, default = 100)
#' @param center  (bool, optional): If ``TRUE``, use a window centered on the current frame
#'  (to the extent possible, modulo end effects). If ``FALSE``, window is to the left. (bool, default = ``FALSE``)
#' @param norm_vars  (bool, optional): If ``TRUE``, normalize variance to one. (bool, default = ``FALSE``)
#'
#' @return `tensor`: Tensor of freq of dimension (..., frame)
#'
#' @export
functional_sliding_window_cmn <- function(
  waveform,
  cmn_window = 600,
  min_cmn_window = 100,
  center = FALSE,
  norm_vars = FALSE
) {

  input_shape = waveform$shape
  lis = length(input_shape)
  if(lis < 2) value_error("waveform should have at least 2 dimensions. Expected Tensor(..., freq, time).")

  num_frames = input_shape[lis-1]
  num_feats = input_shape[lis]
  waveform = waveform$view(c(-1, num_frames, num_feats))
  num_channels = waveform$shape[1]

  dtype = waveform$dtype
  device = waveform$device
  last_window_start = last_window_end = -1
  cur_sum = torch::torch_zeros(num_channels, num_feats, dtype=dtype, device=device)
  cur_sumsq = torch::torch_zeros(num_channels, num_feats, dtype=dtype, device=device)
  cmn_waveform = torch::torch_zeros(num_channels, num_frames, num_feats, dtype=dtype, device=device)
  for(t in seq.int(num_frames)) {
    window_start = 0
    window_end = 0

    if(center) {
      window_start = (t - cmn_window) %/% 2
      window_end = window_start + cmn_window
    } else {
      window_start = t - cmn_window
      window_end = t + 1
    }

    if(window_start < 0) {
      window_end = window_end - window_start
      window_start = 0
    }

    if(!center) {
      if(window_end > t) {
        window_end = max(t + 1, min_cmn_window)
      }
    }

    if(window_end > num_frames) {
      window_start = window_start - (window_end - num_frames)
      window_end = num_frames
      if(window_start < 0) {
        window_start = 0
      }
    }
    if(last_window_start == -1) {
      input_part = waveform[ , (window_start + 1): (window_end - window_start), ]
      cur_sum = cur_sum + torch::torch_sum(input_part, 2)
      if(norm_vars) {
        cur_sumsq = cur_sumsq + torch::torch_cumsum(input_part ** 2, 2)[, -1, ]
      }
    } else {
      if(window_start > last_window_start) {
        frame_to_remove = waveform[ , last_window_start, ]
        cur_sum = cur_sum - frame_to_remove
        if(norm_vars) {
          cur_sumsq = cur_sumsq - (frame_to_remove ** 2)
        }
      }

      if(window_end > last_window_end) {
        frame_to_add = waveform[ , last_window_end, ]
        cur_sum = cur_sum + frame_to_add
        if(norm_vars) {
          cur_sumsq = cur_sumsq + (frame_to_add ** 2)
        }
      }
    }
    window_frames = window_end - window_start
    last_window_start = window_start
    last_window_end = window_end
    cmn_waveform[ , t, ] = waveform[ , t, ] - cur_sum / window_frames
    if(norm_vars) {
      if(window_frames == 1) {
        cmn_waveform[ , t, ] = torch::torch_zeros(num_channels, num_feats, dtype=dtype, device=device)
      } else {
        variance = cur_sumsq
        variance = variance / window_frames
        variance = variance - ((cur_sum ** 2) / (window_frames ** 2))
        variance = torch::torch_pow(variance, -0.5)
        cmn_waveform[ , t, ] = cmn_waveform[ , t, ] * variance
      }
    }
  }
  cmn_waveform = cmn_waveform$view(c(input_shape[-((lis-1):lis)], num_frames, num_feats))
  if(length(input_shape) == 2) {
    cmn_waveform = cmn_waveform$squeeze(1)
  }
  return(cmn_waveform)
}



functional_measure <- function(
  measure_len_ws,
  samples,
  spectrum,
  noise_spectrum,
  spectrum_window,
  spectrum_start,
  spectrum_end,
  cepstrum_window,
  cepstrum_start,
  cepstrum_end,
  noise_reduction_amount,
  measure_smooth_time_mult,
  noise_up_time_mult,
  noise_down_time_mult,
  index_ns,
  boot_count
) {

  lss = length(spectrum$size())
  lns = length(noise_spectrum$size())
  if(spectrum$size()[lss] != noise_spectrum$size()[lns]) value_error("spectrum$size()[-1] != noise_spectrum$size()[-1]")

  lsas = length(samples$size())
  samplesLen_ns = samples$size()[lsas]
  dft_len_ws = spectrum$size()[lss]

  dftBuf = torch::torch_zeros(dft_len_ws)

  .index_ns = torch::torch_tensor(c(index_ns, (index_ns + seq(1, measure_len_ws-1)) %% samplesLen_ns ))$to(torch::torch_long())
  dftBuf[1:measure_len_ws] = samples[.index_ns + 1L] * spectrum_window[1:measure_len_ws]

  # memset(c->dftBuf + i, 0, (p->dft_len_ws - i) * sizeof(*c->dftBuf));
  dftBuf[measure_len_ws:(dft_len_ws-1)]$zero_()

  # lsx_safe_rdft((int)p->dft_len_ws, 1, c->dftBuf);
  .dftBuf = torch::torch_fft_rfft(dftBuf)

  # memset(c->dftBuf, 0, p->spectrum_start * sizeof(*c->dftBuf));
  .dftBuf[1:spectrum_start]$zero_()

  mult = if(boot_count >= 0) boot_count / (1. + boot_count) else measure_smooth_time_mult

  spectrum_end_minus_1 = spectrum_end - 1
  .d = .dftBuf[spectrum_start:spectrum_end_minus_1]$abs()
  spectrum[spectrum_start:spectrum_end_minus_1]$mul_(mult)$add_(.d * (1 - mult))
  .d = spectrum[spectrum_start:spectrum_end_minus_1] ** 2

  .zeros = torch::torch_zeros(spectrum_end - spectrum_start )

  .mult = if(boot_count >= 0) {
    .zeros
  } else {
    torch::torch_where(
      .d > noise_spectrum[spectrum_start:spectrum_end_minus_1],
      torch::torch_tensor(noise_up_time_mult), # if
      torch::torch_tensor(noise_down_time_mult) # else
    )
  }

  noise_spectrum[spectrum_start:spectrum_end_minus_1]$mul_(.mult)$add_(.d * (1 - .mult))
  .d = torch::torch_sqrt(
    torch::torch_max(
      .zeros,
      other = .d - noise_reduction_amount * noise_spectrum[spectrum_start:spectrum_end_minus_1]
    )
  )

  .cepstrum_Buf = torch::torch_zeros(dft_len_ws %/% 2)
  .cepstrum_Buf[spectrum_start:spectrum_end_minus_1] = .d * cepstrum_window
  .cepstrum_Buf[spectrum_end:(dft_len_ws %/% 2)]$zero_()


  # lsx_safe_rdft((int)p->dft_len_ws >> 1, 1, c->dftBuf);
  .cepstrum_Buf = torch::torch_fft_rfft(.cepstrum_Buf, 1)

  result = as.numeric(
    torch::torch_sum(
      .cepstrum_Buf[cepstrum_start:cepstrum_end]$abs()$pow(2)
    )
  )

  result = if(result > 0) log(result / (cepstrum_end - cepstrum_start)) else -Inf

  return(max(0, 21 + result))
}


#' Voice Activity Detector (functional)
#'
#' Voice Activity Detector. Similar to SoX implementation.
#'    Attempts to trim silence and quiet background sounds from the ends of recordings of speech.
#'    The algorithm currently uses a simple cepstral power measurement to detect voice,
#'    so may be fooled by other things, especially music.
#'
#'    The effect can trim only from the front of the audio,
#'    so in order to trim from the back, the reverse effect must also be used.
#'
#' @param waveform  (Tensor): Tensor of audio of dimension `(..., time)`
#' @param sample_rate  (int): Sample rate of audio signal.
#' @param trigger_level  (float, optional): The measurement level used to trigger activity detection.
#'            This may need to be cahnged depending on the noise level, signal level,
#'     and other characteristics of the input audio.  (Default: 7.0)
#' @param trigger_time  (float, optional): The time constant (in seconds)
#'   used to help ignore short bursts of sound.  (Default: 0.25)
#' @param search_time  (float, optional): The amount of audio (in seconds)
#'  to search for quieter/shorter bursts of audio to include prior
#'  to the detected trigger point.  (Default: 1.0)
#' @param allowed_gap  (float, optional): The allowed gap (in seconds) between
#'   quiteter/shorter bursts of audio to include prior
#'  to the detected trigger point.  (Default: 0.25)
#' @param pre_trigger_time  (float, optional): The amount of audio (in seconds) to preserve
#'  before the trigger point and any found quieter/shorter bursts.  (Default: 0.0)
#' @param boot_time  (float, optional) The algorithm (internally) uses adaptive noise
#'            estimation/reduction in order to detect the start of the wanted audio.
#'  This option sets the time for the initial noise estimate.  (Default: 0.35)
#' @param noise_up_time  (float, optional) Time constant used by the adaptive noise estimator
#'   for when the noise level is increasing.  (Default: 0.1)
#' @param noise_down_time  (float, optional) Time constant used by the adaptive noise estimator
#' for when the noise level is decreasing.  (Default: 0.01)
#' @param noise_reduction_amount  (float, optional) Amount of noise reduction to use in
#'  the detection algorithm  (e.g. 0, 0.5, ...). (Default: 1.35)
#' @param measure_freq  (float, optional) Frequency of the algorithm’s
#'  processing/measurements.  (Default: 20.0)
#' @param measure_duration  (float, optional) Measurement duration.
#'  (Default: Twice the measurement period; i.e. with overlap.)
#' @param measure_smooth_time  (float, optional) Time constant used to smooth
#'   spectral measurements.  (Default: 0.4)
#' @param hp_filter_freq  (float, optional) "Brick-wall" frequency of high-pass filter applied
#'  at the input to the detector algorithm.  (Default: 50.0)
#' @param lp_filter_freq  (float, optional) "Brick-wall" frequency of low-pass filter applied
#'  at the input to the detector algorithm.  (Default: 6000.0)
#' @param hp_lifter_freq  (float, optional) "Brick-wall" frequency of high-pass lifter used
#' in the detector algorithm.  (Default: 150.0)
#' @param lp_lifter_freq  (float, optional) "Brick-wall" frequency of low-pass lifter used
#'   in the detector algorithm.  (Default: 2000.0)
#'
#' @return Tensor: Tensor of audio of dimension (..., time).
#'
#' @references
#' - [https://sox.sourceforge.net/sox.html]()
#'
#' @export
functional_vad <- function(
  waveform,
  sample_rate,
  trigger_level = 7.0,
  trigger_time = 0.25,
  search_time = 1.0,
  allowed_gap = 0.25,
  pre_trigger_time = 0.0,
  # Fine-tuning parameters
  boot_time = .35,
  noise_up_time = .1,
  noise_down_time = .01,
  noise_reduction_amount = 1.35,
  measure_freq = 20.0,
  measure_duration = NULL,
  measure_smooth_time = .4,
  hp_filter_freq = 50.,
  lp_filter_freq = 6000.,
  hp_lifter_freq = 150.,
  lp_lifter_freq = 2000.
) {

  measure_duration = if(is.null(measure_duration)) 2.0 / measure_freq else measure_duration

  measure_len_ws = as.integer(sample_rate * measure_duration + .5)
  measure_len_ns = measure_len_ws
  # for (dft_len_ws = 16; dft_len_ws < measure_len_ws; dft_len_ws <<= 1);
  dft_len_ws = 16
  while (dft_len_ws < measure_len_ws) {
    dft_len_ws = dft_len_ws * 2
  }

  measure_period_ns = as.integer(sample_rate / measure_freq + .5)
  measures_len = ceiling(search_time * measure_freq)
  search_pre_trigger_len_ns = measures_len * measure_period_ns
  gap_len = as.integer(allowed_gap * measure_freq + .5)

  fixed_pre_trigger_len_ns = as.integer(pre_trigger_time * sample_rate + .5)
  samplesLen_ns = fixed_pre_trigger_len_ns + search_pre_trigger_len_ns + measure_len_ns

  # lsx_apply_hann(spectrum_window, (int)measure_len_ws);
  spectrum_window =  (2.0 / sqrt(measure_len_ws)) * torch::torch_hann_window(measure_len_ws, dtype=torch::torch_float())

  spectrum_start = as.integer(hp_filter_freq / sample_rate * dft_len_ws + .5)
  spectrum_start = max(spectrum_start, 1L)
  spectrum_end = as.integer(lp_filter_freq / sample_rate * dft_len_ws + .5)
  spectrum_end = min(spectrum_end, dft_len_ws %/% 2)

  cepstrum_window =  (2.0 / sqrt(spectrum_end - spectrum_start)) * torch::torch_hann_window(spectrum_end - spectrum_start, dtype=torch::torch_float())

  cepstrum_start = ceiling(sample_rate * .5 / lp_lifter_freq)
  cepstrum_end = floor(sample_rate * .5 / hp_lifter_freq)
  cepstrum_end = min(cepstrum_end, dft_len_ws %/% 4)

  if(cepstrum_end <= cepstrum_start) value_error("cepstrum_end <= cepstrum_start")

  noise_up_time_mult = exp(-1. / (noise_up_time * measure_freq))
  noise_down_time_mult = exp(-1. / (noise_down_time * measure_freq))
  measure_smooth_time_mult = exp(-1. / (measure_smooth_time * measure_freq))
  trigger_meas_time_mult = exp(-1. / (trigger_time * measure_freq))

  boot_count_max = as.integer(boot_time * measure_freq - .5)
  measure_timer_ns = measure_len_ns
  boot_count = measures_index = flushedLen_ns = samplesIndex_ns = 0

  # pack batch
  shape = waveform$size()
  ls = length(shape)
  waveform = waveform$view(c(-1, shape[ls]))
  shape = waveform$size()
  ls = length(shape)

  n_channels = waveform$size(1)
  ilen = waveform$size(2)

  mean_meas = torch::torch_zeros(n_channels)
  samples = torch::torch_zeros(n_channels, samplesLen_ns)
  spectrum = torch::torch_zeros(n_channels, dft_len_ws)
  noise_spectrum = torch::torch_zeros(n_channels, dft_len_ws)
  measures = torch::torch_zeros(n_channels, measures_len)

  has_triggered = FALSE
  num_measures_to_flush = 0
  pos = 0


  while(pos < ilen & !has_triggered) {
    measure_timer_ns = measure_timer_ns - 1
    for(i in seq.int(n_channels)) {
      samples[i, samplesIndex_ns+1] = waveform[i, pos+1]
      # if((!p->measure_timer_ns)
      if (measure_timer_ns == 0) {
        index_ns = (samplesIndex_ns + samplesLen_ns - measure_len_ns) %% samplesLen_ns
        meas = functional_measure(
          measure_len_ws = measure_len_ws,
          samples=samples[i],
          spectrum=spectrum[i],
          noise_spectrum=noise_spectrum[i],
          spectrum_window=spectrum_window,
          spectrum_start=spectrum_start,
          spectrum_end=spectrum_end,
          cepstrum_window=cepstrum_window,
          cepstrum_start=cepstrum_start,
          cepstrum_end=cepstrum_end,
          noise_reduction_amount=noise_reduction_amount,
          measure_smooth_time_mult=measure_smooth_time_mult,
          noise_up_time_mult=noise_up_time_mult,
          noise_down_time_mult=noise_down_time_mult,
          index_ns=index_ns,
          boot_count=boot_count)
        measures[i, measures_index+1] = meas
        mean_meas[i] = mean_meas[i] * trigger_meas_time_mult + meas * (1. - trigger_meas_time_mult)

        has_triggered = has_triggered | as.logical(mean_meas[i] >= trigger_level)
        if(has_triggered) {
          n = measures_len
          k = measures_index
          jTrigger = n
          jZero = n
          j = 0

          for(j in 0:(n-1)) {
            if(as.logical(measures[i, k+1] >= trigger_level) & as.logical(j <= jTrigger + gap_len)) {
              jZero = jTrigger = j
            } else if(as.logical(measures[i, k+1] == 0) & as.logical(jTrigger >= jZero)) {
              jZero = j
            }
            k = (k + n - 1) %% n
          }

          j = min(j, jZero)
          # num_measures_to_flush = range_limit(j, num_measures_to_flush, n);
          num_measures_to_flush = (min(max(num_measures_to_flush, j), n))
        } # end if(has_triggered)
      } # end if (measure_timer_ns == 0))
    } # end for

    samplesIndex_ns = samplesIndex_ns + 1
    pos = pos + 1
    if(samplesIndex_ns == samplesLen_ns) {
      samplesIndex_ns = 0
    }

    if(measure_timer_ns == 0) {
      measure_timer_ns = measure_period_ns
      measures_index = measures_index + 1
      measures_index = measures_index %% measures_len
      if(boot_count >= 0) {
        boot_count = if(boot_count == boot_count_max) -1 else boot_count + 1
      }
    }

    if(has_triggered) {
      flushedLen_ns = (measures_len - num_measures_to_flush) * measure_period_ns
      samplesIndex_ns = (samplesIndex_ns + flushedLen_ns) %% samplesLen_ns
    }

  } # end while

  res = waveform[ , (pos + 1 - samplesLen_ns + flushedLen_ns):waveform$size(2)]
  # unpack batch
  lrs = length(res$shape)
  return(res$view(c(shape[-ls], res$shape[lrs])))
}
