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
#' @param Arguments for window function.
#'
#' @return `tensor`: Dimension (..., freq, time), freq is n_fft %/% 2 + 1 and n_fft is the
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
#' @param f_max (float or NULL): Maximum frequency (Hz). If NULL defaults to sample_rate %/% 2
#' @param norm (chr) (Optional): If 'slaney', divide the triangular
#'  mel weights by the width of the mel band (area normalization). (Default: `NULL`)
#'
#' @return `tensor`: Triangular filter banks (fb matrix) of size (`n_freqs`, `n_mels`)
#'         meaning number of frequencies to highlight/apply to x the number of filterbanks.
#'         Each column is a filterbank so that assuming there is a matrix A of
#'         size (..., `n_freqs`), the applied result would be
#'         ``A * create_fb_matrix(A.size(-1), ...)``.
#'
#' @export
functional_create_fb_matrix <- function(
  n_freqs,
  n_mels,
  sample_rate = 16000,
  f_min = 0,
  f_max = NULL,
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

#' DCT transformation matrix
#'
#' Create a DCT transformation matrix with shape (``n_mels``, ``n_mfcc``),
#' normalized depending on norm.
#' [http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II]()
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
#' @param ref_value (float): Reference which the output will be scaled by.
#' @param db_multiplier (float): Log10(max(ref_value and amin))
#' @param top_db (float or NULL, optional): Minimum negative cut-off in decibels. A reasonable number
#'     is 80. (Default: ``NULL``)
#'
#' @return `tensor`: Output tensor in decibel scale
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
#' @return `tensor`: Output tensor in power/amplitude scale.
#'
#' @export
functional_db_to_amplitude <- function(x, ref = 1.0, power = 1.0) {
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
#' @param f_max (float or NULL, optional): Maximum frequency. (Default: ``sample_rate // 2``)
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


#' Mu Law Encoding
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

#' Mu Law Decoding
#'
#' Decode mu-law encoded signal.  For more info see the
#'  [Wikipedia Entry](https://en.wikipedia.org/wiki/M-law_algorithm)
#'
#' @param x (Tensor): Input tensor
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

#' Angle
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

#' Magnitude and Phase
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



#' Griffin-Lim Transformation
#'
#' Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation.
#'  Implementation ported from `librosa`.
#'
#'
#' @param specgram (Tensor): A magnitude-only STFT spectrogram of dimension (..., freq, frames)
#'      where freq is ``n_fft %/% 2 + 1``.
#' @param window (Tensor): Window tensor that is applied/multiplied to each frame/window
#' @param n_fft (int): Size of FFT, creates ``n_fft %/% 2 + 1`` bins
#' @param hop_length (int): Length of hop between STFT windows. (Default: ``win_length %/% 2``)
#' @param win_length (int): Window size. (Default: ``n_fft``)
#' @param power (float): Exponent for the magnitude spectrogram,
#'      (must be > 0) e.g., 1 for energy, 2 for power, etc.
#' @param normalized (bool): Whether to normalize by magnitude after stft.
#' @param n_iter (int): Number of iteration for phase recovery process.
#' @param momentum (float): The momentum parameter for fast Griffin-Lim.
#'      Setting this to 0 recovers the original Griffin-Lim method.
#'      Values near 1 can lead to faster convergence, but above 1 may not converge.
#' @param length (int or NULL): Array length of the expected output.
#' @param rand_init (bool): Initializes phase randomly if True, to zero otherwise.
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
  # if(momentum > 1) value_warning('momentum > 1 can be unstable')
  # if(momentum < 0) value_error('momentum < 0')
  #
  # # pack batch
  # shape = specgram$size()
  # specgram = specgram$reshape([-1] + list(shape[-2:]))
  #
  # specgram = specgram$pow(1 / power)
  #
  # # randomly initialize the phase
  # ss = specgram$size()
  # batch = ss[1]
  # freq = ss[2]
  # frames = ss[3]
  # if(rand_init) {
  #   angles = 2 * pi * torch::torch_rand(batch, freq, frames)
  # } else {
  #   angles = torch::Torch_zeros(batch, freq, frames)
  # }
  #
  # angles = torch::torch_stack([angles.cos(), angles.sin()], dim=-1).to(dtype=specgram.dtype, device=specgram.device)
  # specgram = specgram.unsqueeze(-1).expand_as(angles)
  #
  # # And initialize the previous iterate to 0
  # rebuilt = torch::torch_tensor(0.)
  #
  # for _ in range(n_iter):
  #   # Store the previous iterate
  #   tprev = rebuilt
  #
  # # Invert with our current estimate of the phases
  # inverse = torch::torch_istft(specgram * angles,
  #                       n_fft=n_fft,
  #                       hop_length=hop_length,
  #                       win_length=win_length,
  #                       window=window,
  #                       length=length)$float()
  #
  # # Rebuild the spectrogram
  # rebuilt = torch.stft(inverse, n_fft, hop_length, win_length, window,
  #                      True, 'reflect', False, True)
  #
  # # Update our phase estimates
  # angles = rebuilt
  # if momentum:
  #   angles = angles - tprev.mul_(momentum / (1 + momentum))
  # angles = angles.div(complex_norm(angles).add(1e-16).unsqueeze(-1).expand_as(angles))
  #
  # # Return the final phase estimates
  # waveform = torch.istft(specgram * angles,
  #                        n_fft=n_fft,
  #                        hop_length=hop_length,
  #                        win_length=win_length,
  #                        window=window,
  #                        length=length)
  #
  # # unpack batch
  # waveform = waveform$reshape(shape[:-2] + waveform.shape[-1:])
  #
  # return(waveform)
  not_implemented_error("TO DO (waiting for torch_istft() implementation)")
}

#' An IIR Filter
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
#' @param clamp (bool, optional): If ``TRUE``, clamp the output signal to be in the range [-1, 1] (Default: ``TRUE``)
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
  window_idxs = torch::torch_arange(0, n_sample, device=device)$unsqueeze(1) + torch::torch_arange(0, n_order, device=device)$unsqueeze(2)
  window_idxs = window_idxs$`repeat`(c(n_channel, 1, 1))
  window_idxs = window_idxs + (torch::torch_arange(0, n_channel, device=device)$unsqueeze(-1)$unsqueeze(-1) * n_sample_padded)
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

#' Biquad Filter
#'
#' Perform a biquad filter of input tensor.  Initial conditions set to 0.
#'     [https://en.wikipedia.org/wiki/Digital_biquad_filter]()
#'
#' @param waveform (Tensor): audio waveform of dimension of `(..., time)`
#' @param b0 (float): numerator coefficient of current input, x[n]
#' @param b1 (float): numerator coefficient of input one time step ago x[n-1]
#' @param b2 (float): numerator coefficient of input two time steps ago x[n-2]
#' @param a0 (float): denominator coefficient of current output y[n], typically 1
#' @param a1 (float): denominator coefficient of current output y[n-1]
#' @param a2 (float): denominator coefficient of current output y[n-2]
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

#' High-pass Biquad Filter
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

#' Low-pass Biquad Filter
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

#' All-pass Biquad Filter
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
#' - [http://sox.sourceforge.net/sox.html]()
#' - [https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF]()
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

#' Band-pass Biquad Filter
#'
#' Design two-pole band-pass filter.  Similar to SoX implementation.
#'
#' @param waveform  (Tensor): audio waveform of dimension of `(..., time)`
#' @param sample_rate  (int): sampling rate of the waveform, e.g. 44100 (Hz)
#' @param central_freq  (float): central frequency (in Hz)
#' @param Q  (float, optional): [https://en.wikipedia.org/wiki/Q_factor]() (Default: ``0.707``)
#' @param const_skirt_gain  (bool, optional) : If ``FALSE``, uses a constant skirt gain (peak gain = Q).
#' @param If ``TRUE``, uses a constant 0dB peak gain.  (Default: ``TRUE``)
#'
#' @return Tensor: Waveform of dimension of `(..., time)`
#'
#' @references
#' - [http://sox.sourceforge.net/sox.html]()
#' - [https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF]()
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

#' Band-reject Biquad Filter
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
#' - [http://sox.sourceforge.net/sox.html]()
#' - [https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF]()
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

#' Biquad Peaking Equalizer Filter
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

#' Two-pole Band Filter
#'
#' Design two-pole band filter.  Similar to SoX implementation.
#'
#' @param waveform  (Tensor): audio waveform of dimension of `(..., time)`
#' @param sample_rate  (int): sampling rate of the waveform, e.g. 44100 (Hz)
#' @param central_freq  (float): central frequency (in Hz)
#' @param Q  (float, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``).
#' @param noise  (bool, optional) : If ``FALSE``, uses the alternate mode for un-pitched audio
#' (e.g. percussion). If ``TRUE``, uses mode oriented to pitched audio, i.e. voice, singing,
#' or instrumental music  (Default: ``TRUE``).
#'
#' @return `tensor`: Waveform of dimension of `(..., time)`
#'
#' @references
#' - [http://sox.sourceforge.net/sox.html]()
#' - [https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF]()
#'
#' @export
functional_band_biquad <- function(
  waveform,
  sample_rate,
  central_freq,
  Q = 0.707,
  noise = TRUE
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

#' Treble Tone-control Effect
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
#' - [http://sox.sourceforge.net/sox.html]()
#' - [https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF]()
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

#' Bass Tone-control Effect
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
#' - [http://sox.sourceforge.net/sox.html]()
#' - [https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF]()
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


#' ISO 908 CD De-emphasis IIR Filter
#'
#' Apply ISO 908 CD de-emphasis (shelving) IIR filter.  Similar to SoX implementation.
#'
#' @param waveform  (Tensor): audio waveform of dimension of `(..., time)`
#' @param sample_rate  (int): sampling rate of the waveform, Allowed sample rate ``44100`` or ``48000``
#'
#' @return Tensor: Waveform of dimension of `(..., time)`
#'
#' @references
#' - [http://sox.sourceforge.net/sox.html]()
#' - [https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF]()
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

#' RIAA Vinyl Playback Equalisation
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
#' - [http://sox.sourceforge.net/sox.html]()
#' - [https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF]()
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


#' Contrast Effect
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
#' - [http://sox.sourceforge.net/sox.html]()
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

#' DC Shift
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
#' - [http://sox.sourceforge.net/sox.html]()
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

#' Overdrive Effect
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
#' - [http://sox.sourceforge.net/sox.html]()
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

#' Phasing Effect
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
#' @param sinusoidal  (bool):  If ``FALSE``, uses sinusoidal modulation (preferable for multiple instruments).
#'  If ``TRUE``, uses triangular modulation  (gives single instruments a sharper phasing effect)
#' (Default: ``FALSE``)
#'
#' @return `tensor`: Waveform of dimension of `(..., time)`
#'
#' @references
#' - [http://sox.sourceforge.net/sox.html]()
#' - Scott Lehman, Effects Explained, [http://harmony-central.com/Effects/effects-explained.html]()
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
  sinusoidal = FALSE
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

  mod_buf = functional_generate_wave_table(
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

#' Wave Table Generator
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
functional_generate_wave_table <- function(
  wave_type,
  data_type,
  table_size,
  min,
  max,
  phase,
  device
) {

  phase_offset = as.integer(phase / pi / 2 * table_size + 0.5)
  t = torch::torch_arange(0, table_size, device=device, dtype=torch::torch_int32())
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

#' Flanger Effect
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
#' - [http://sox.sourceforge.net/sox.html]()
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

  lfo = functional_generate_wave_table(
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
  channel_idxs = torch::torch_arange(0, n_channels, device=device)$to(torch::torch_long())

  for(i in seq.int(waveform$shape[lws])) {
    delay_buf_pos = (delay_buf_pos + delay_buf_length - 1L) %% delay_buf_length

    cur_channel_phase = (channel_idxs * lfo_length * channel_phase + .5)$to(torch::torch_long())
    delay_tensor = lfo[((lfo_pos + cur_channel_phase) %% lfo_length)$to(torch::torch_long())]
    frac_delay = torch::torch_frac(delay_tensor)
    delay_tensor = torch::torch_floor(delay_tensor)

    int_delay = delay_tensor$to(torch::torch_long())

    temp = waveform[ ,  , i]

    delay_bufs[ ,  , delay_buf_pos+1] = temp + delay_last * feedback_gain

    delayed_0 = delay_bufs[ , channel_idxs, (delay_buf_pos + int_delay) %% delay_buf_length]

    int_delay = int_delay + 1L

    delayed_1 = delay_bufs[ , channel_idxs, (delay_buf_pos + int_delay) %% delay_buf_length]

    int_delay = int_delay + 1

    if(interpolation == "linear") {
      delayed = delayed_0 + (delayed_1 - delayed_0) * frac_delay
    } else {
      delayed_2 = delay_bufs[ , channel_idxs, (delay_buf_pos + int_delay) %% delay_buf_length]

      int_delay = int_delay + 1

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
#' Mask Along Axis IID
#'
#' Apply a mask along ``axis``. Mask will be applied from indices ``[v_0, v_0 + v)``, where
#' ``v`` is sampled from ``uniform (0, mask_param)``, and ``v_0`` from ``uniform(0, max_v - v)``.
#'
#' @param specgrams  (Tensor): Real spectrograms (batch, channel, freq, time)
#' @param mask_param  (int): Number of columns to be masked will be uniformly sampled from [0, mask_param]
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

  if(axis != 3 & axis != 4) {
    value_error("Only Frequency (axis 3) and Time (axis 4) masking are supported")
  }

  device = specgrams$device
  dtype = specgrams$dtype

  value = torch::torch_rand(specgrams$shape[1:2], device=device, dtype=dtype) * mask_param
  min_value = torch::torch_rand(specgrams$shape[1:2], device=device, dtype=dtype) * (specgrams$size(axis) - value)

  # Create broadcastable mask
  mask_start = min_value[.., NULL, NULL]
  mask_end = (min_value + value)[.., NULL, NULL]
  mask = torch::torch_arange(0, specgrams$size(axis), device=device, dtype=dtype)

  # Per batch example masking
  specgrams = specgrams$transpose(axis, -1)
  specgrams$masked_fill_((mask >= mask_start) & (mask < mask_end), mask_value)
  specgrams = specgrams$transpose(axis, -1)

  return(specgrams)
}
#' Mask Along Axis
#'
#' Apply a mask along ``axis``. Mask will be applied from indices ``[v_0, v_0 + v)``, where
#' ``v`` is sampled from ``uniform (0, mask_param)``, and ``v_0`` from ``uniform(0, max_v - v)``.
#' All examples will have the same mask interval.
#'
#' @param specgram  (Tensor): Real spectrogram (channel, freq, time)
#' @param mask_param  (int): Number of columns to be masked will be uniformly sampled from [0, mask_param]
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

#' Delta Coefficients
#'
#' Compute delta coefficients of a tensor, usually a spectrogram.
#'
#' math:
#'  \deqn{
#'  d_t = \frac{\sum_{n=1}^{N} n  (c_{t+n} - c_{t-n})}{2 \sum_{n=1}^{N} n^2}
#'  }
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
#' specgram = torch::torch_randn(1, 40, 1000)
#' delta = functional_compute_deltas(specgram)
#' delta2 = functional_compute_deltas(delta)
#'
#' @export
functional_compute_deltas <- function(
  specgram,
  win_length = 5,
  mode = "replicate"
) {
  device = specgram$device
  dtype = specgram$dtype

  # pack batch
  shape = specgram$size()
  ls = length(shape)
  specgram = specgram$reshape(c(1, -1, shape[ls]))
  if(win_length < 3) value_error("win_length must be >= 3.")

  n = (win_length - 1) %/% 2

  # twice sum of integer squared
  denom = n * (n + 1) * (2 * n + 1) / 3

  specgram = torch::nnf_pad(specgram, c(n, n), mode=mode)
  kernel = torch::torch_arange(-n, n + 1, 1, device=device, dtype=dtype)$`repeat`(c(specgram$shape[2], 1, 1))
  output = torch::nnf_conv1d(specgram, kernel, groups=specgram$shape[2]) / denom

  # unpack batch
  output = output$reshape(shape)

  return(output)
}
