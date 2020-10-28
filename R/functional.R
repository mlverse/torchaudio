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

