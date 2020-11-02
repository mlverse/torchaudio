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
#'  1 for energy, 2 for power, etc. If NULL, then the complex spectrum is returned instead.
#' @param normalized (logical): Whether to normalize by magnitude after stft
#' @param ... (optional) Arguments for window function.
#'
#' @return tensor: Dimension (..., freq, time), freq is n_fft %/% 2 + 1 and n_fft is the
#' number of Fourier bins, and time is the number of window hops (n_frame).
#'
#' @export
transform_spectrogram <- torch::nn_module(
  "Spectrogram",
  initialize = function(
    n_fft = 400,
    win_length = NULL,
    hop_length = NULL,
    pad = 0L,
    window_fn = torch::torch_hann_window,
    power = 2,
    normalized = FALSE,
    ...
  ) {
    self$n_fft = n_fft

    # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
    # number of frequecies due to onesided=True in torch.stft
    self$win_length = win_length %||% n_fft
    self$hop_length = hop_length %||% self$win_length %/% 2
    window = window_fn(window_length = self$win_length, dtype = torch::torch_float(), ...)
    self$register_buffer('window', window)
    self$pad = pad
    self$power = power
    self$normalized = normalized
  },

  forward = function(waveform){
    functional_spectrogram(
      waveform = waveform,
      pad = self$pad,
      n_fft = self$n_fft,
      window = self$window,
      hop_length = self$hop_length,
      win_length = self$win_length,
      power = self$power,
      normalized = self$normalized
    )
  }
)

#' Mel Scale
#'
#' Turn a normal STFT into a mel frequency STFT, using a conversion
#' matrix. This uses triangular filter banks.
#'
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
transform_mel_scale <- torch::nn_module(
  "MelScale",
  initialize = function(
    n_mels = 128,
    sample_rate = 16000,
    f_min = 0.0,
    f_max = NULL,
    n_stft = NULL
  ) {
    self$n_mels = n_mels
    self$sample_rate = sample_rate
    self$f_max = if(is.null(f_max)) as.numeric(sample_rate %/% 2) else f_max
    self$f_min = f_min

    if(self$f_min > self$f_max) value_error(glue::glue("Require f_min: {self$f_min} < f_max: {self$f_max}"))

    fb = if(is.null(n_stft)) {
      torch::torch_empty(0)
    } else {
      functional_create_fb_matrix(
        n_freqs = n_stft,
        f_min = f_min,
        f_max = f_max,
        n_mels = n_mels,
        sample_rate = sample_rate
      )
    }
    self$register_buffer('fb', fb)
  },
  forward = function(specgram) {
    # pack batch
    shape = specgram$size()
    ls = length(shape)
    specgram = specgram$reshape(list(-1, shape[ls-1], shape[ls]))

    if(self$fb$numel() == 0) {
      tmp_fb = functional_create_fb_matrix(
        n_freqs = specgram$size(2),
        f_min = self$f_min,
        f_max = self$f_max,
        n_mels = self$n_mels,
        sample_rate = self$sample_rate
      )
      # Attributes cannot be reassigned outside __init__ so workaround
      self$fb$resize_(tmp_fb$size())
      self$fb$copy_(tmp_fb)
    }

    # (channel, frequency, time).transpose(...) dot (frequency, n_mels)
    # -> (channel, time, n_mels).transpose(...)
    mel_specgram = torch::torch_matmul(specgram$transpose(2L, 3L), self$fb)$transpose(2L, 3L)

    # unpack batch
    lspec = length(mel_specgram$shape)
    mel_specgram = mel_specgram$reshape(c(shape[-((ls-1):ls)], mel_specgram$shape[(lspec-2):lspec]))

    return(mel_specgram)
  }
)

#' Amplitude to DB
#'
#' Turn a tensor from the power/amplitude scale to the decibel scale.
#'
#' This output depends on the maximum value in the input tensor, and so
#' may return different values for an audio clip split into snippets vs. a
#' a full clip.
#'
#' @param x (Tensor): Input tensor before being converted to decibel scale
#' @param stype (str, optional): scale of input tensor ('power' or 'magnitude'). The
#' power being the elementwise square of the magnitude. (Default: ``'power'``)
#' @param top_db (float or NULL, optional): Minimum negative cut-off in decibels. A reasonable number
#' is 80. (Default: ``NULL``)
#'
#' @return `tensor`: Output tensor in decibel scale
#'
#' @export
transform_amplitude_to_db <- torch::nn_module(
  "AmplitudeToDB",
  initialize = function(
    stype = 'power',
    top_db = NULL
  ) {
    self$stype = stype
    if(!is.null(top_db) && top_db < 0) value_error("top_db must be positive value")
    self$top_db = top_db
    self$multiplier = if(stype == 'power') 10.0 else 20.0
    self$amin = 1e-10
    self$ref_value = 1.0
    self$db_multiplier = log10(max(self$amin, self$ref_value))
  },

  forward = function(x) {
    functional_amplitude_to_db(
      x = x,
      multiplier = self$multiplier,
      amin = self$amin,
      db_multiplier = self$db_multiplier,
      top_db = self$top_db
    )
  }
)

#' Mel Spectrogram
#'
#' Create MelSpectrogram for a raw audio signal. This is a composition of Spectrogram
#'   and MelScale.
#'
#' @param waveform (Tensor): Tensor of audio of dimension (..., time).
#' @param sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
#' @param win_length (int or NULL, optional): Window size. (Default: ``n_fft``)
#' @param hop_length (int or NULL, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
#' @param n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
#' @param f_min (float, optional): Minimum frequency. (Default: ``0.``)
#' @param f_max (float or NULL, optional): Maximum frequency. (Default: ``NULL``)
#' @param pad (int, optional): Two sided padding of signal. (Default: ``0``)
#' @param n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
#' @param window_fn (function, optional): A function to create a window tensor
#'  that is applied/multiplied to each frame/window. (Default: ``torch_hann_window``)
#' @param ... (optional): Arguments for window function.
#'
#' @return `tensor`: Mel frequency spectrogram of size (..., ``n_mels``, time).
#'
#' @section  Sources:
#' - [https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe]()
#' - [https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html]()
#' - [http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html]()
#'
#' @examples #'   Example
#' sample_mp3 <- tuneR::readMP3(system.file("sample_audio_1.mp3", package = "torchaudio"))
#' mel_specgram <- transform_mel_spectrogram(sample_rate = sample_mp3@@samp.rate)(sample_mp3@@left)  # (channel, n_mels, time)
#'
#' @export
transform_mel_spectrogram <- torch::nn_module(
  "MelSpectrogram",
  initialize = function(
    sample_rate = 16000,
    n_fft = 400,
    win_length = NULL,
    hop_length = NULL,
    f_min = 0.0,
    f_max = NULL,
    pad = 0,
    n_mels = 128,
    window_fn = torch::torch_hann_window,
    power = 2.,
    normalized = FALSE,
    ...
  ) {
    self$sample_rate = sample_rate
    self$n_fft = n_fft
    self$win_length = if(!is.null(win_length)) win_length else n_fft
    self$hop_length = if(!is.null(hop_length)) hop_length else self$win_length %/% 2
    self$pad = pad
    self$power = power
    self$normalized = normalized
    self$n_mels = n_mels  # number of mel frequency bins
    self$f_max = f_max
    self$f_min = f_min
    self$spectrogram = transform_spectrogram(
      n_fft = self$n_fft,
      win_length = self$win_length,
      hop_length = self$hop_length,
      pad = self$pad,
      window_fn = window_fn,
      power = self$power,
      normalized = self$normalized,
      ...
    )

    self$mel_scale = transform_mel_scale(
      n_mels = self$n_mels,
      sample_rate = self$sample_rate,
      f_min = self$f_min,
      f_max = self$f_max,
      n_stft = self$n_fft %/% 2 + 1
    )
  },

  forward = function(waveform) {
    specgram = self$spectrogram(waveform)
    mel_specgram = self$mel_scale(specgram)
    return(mel_specgram)
  }
)

#' Mel-frequency Cepstrum Coefficients
#'
#' Create the Mel-frequency cepstrum coefficients from an audio signal.
#'
#' @param waveform (tensor): Tensor of audio of dimension (..., time)
#' @param sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
#' @param n_mfcc (int, optional): Number of mfc coefficients to retain. (Default: ``40``)
#' @param dct_type (int, optional): type of DCT (discrete cosine transform) to use. (Default: ``2``)
#' @param norm (str, optional): norm to use. (Default: ``'ortho'``)
#' @param log_mels (bool, optional): whether to use log-mel spectrograms instead of db-scaled. (Default: ``FALSE``)
#' @param ... (optional): arguments for [torchaudio::transform_mel_spectrogram].
#'
#' @details By default, this calculates the MFCC on the DB-scaled Mel spectrogram.
#' This output depends on the maximum value in the input spectrogram, and so
#' may return different values for an audio clip split into snippets vs. a
#' a full clip.
#'
#' @return `tensor`: specgram_mel_db of size (..., ``n_mfcc``, time).
#'
#' @export
transform_mfcc <- torch::nn_module(
  "MFCC",
  initialize = function(
    sample_rate = 16000,
    n_mfcc = 40,
    dct_type = 2,
    norm = 'ortho',
    log_mels = FALSE,
    ...
  ) {

    supported_dct_types = c(2)
    if(!dct_type %in% supported_dct_types) {
      value_error(paste0('DCT type not supported:', dct_type))
    }
    self$sample_rate = sample_rate
    self$n_mfcc = n_mfcc
    self$dct_type = dct_type
    self$norm = norm
    self$top_db = 80.0
    self$amplitude_to_db = transform_amplitude_to_db('power', self$top_db)

    self$mel_spectrogram = transform_mel_spectrogram(sample_rate = self$sample_rate, ...)

    if(self$n_mfcc > self$mel_spectrogram$n_mels) value_error('Cannot select more MFCC coefficients than # mel bins')

    dct_mat = functional_create_dct(
      n_mfcc = self$n_mfcc,
      n_mels = self$mel_spectrogram$n_mels,
      norm = self$norm
    )

    self$register_buffer('dct_mat', dct_mat)
    self$log_mels = log_mels
  },

  forward = function(waveform) {

    # pack batch
    shape = waveform$size()
    ls = length(shape)
    waveform = waveform$reshape(list(-1, shape[ls]))

    mel_specgram = self$mel_spectrogram(waveform)
    if(self$log_mels) {
      log_offset = 1e-6
      mel_specgram = torch::torch_log(mel_specgram + log_offset)
    } else {
      mel_specgram = self$amplitude_to_db(mel_specgram)
    }

    # (channel, n_mels, time).tranpose(...) dot (n_mels, n_mfcc)
    # -> (channel, time, n_mfcc).tranpose(...)

    mfcc = torch::torch_matmul(mel_specgram$transpose(3, 4), self$dct_mat)$transpose(3, 4)

    # unpack batch
    lspec = length(mfcc$shape)
    mfcc = mfcc$reshape(c(shape[-((ls-1):ls)], mfcc$shape[(lspec-2):lspec]))

    return(mfcc)
  }
)

#' Inverse Mel Scale
#'
#'  Solve for a normal STFT from a mel frequency STFT, using a conversion
#'  matrix.  This uses triangular filter banks.
#'
#' @param melspec  (Tensor): A Mel frequency spectrogram of dimension (..., ``n_mels``, time)
#' @param n_stft  (int): Number of bins in STFT. See ``n_fft`` in [torchaudio::transform_spectrogram].
#' @param n_mels  (int, optional): Number of mel filterbanks. (Default: ``128``)
#' @param sample_rate  (int, optional): Sample rate of audio signal. (Default: ``16000``)
#' @param f_min  (float, optional): Minimum frequency. (Default: ``0.``)
#' @param f_max  (float or NULL, optional): Maximum frequency. (Default: ``sample_rate %/% 2``)
#' @param max_iter  (int, optional): Maximum number of optimization iterations. (Default: ``100000``)
#' @param tolerance_loss  (float, optional): Value of loss to stop optimization at. (Default: ``1e-5``)
#' @param tolerance_change  (float, optional): Difference in losses to stop optimization at. (Default: ``1e-8``)
#' @param ...  (optional): Arguments passed to the SGD optimizer. Argument lr will default to 0.1 if not specied.(Default: ``NULL``)
#'
#' @details
#' It minimizes the euclidian norm between the input mel-spectrogram and the product between
#' the estimated spectrogram and the filter banks using SGD.
#'
#' @return Tensor: Linear scale spectrogram of size (..., freq, time)
#'
#' @export
transform_inverse_mel_scale <- torch::nn_module(
  "InverseMelScale",
  initialize = function(
    n_stft,
    n_mels = 128,
    sample_rate = 16000,
    f_min = 0.,
    f_max = NULL,
    max_iter = 100000,
    tolerance_loss = 1e-5,
    tolerance_change = 1e-8,
    ...
  ) {
    self$n_mels = n_mels
    self$sample_rate = sample_rate
    self$f_max = f_max %||% as.numeric(sample_rate %/% 2)
    self$f_min = f_min
    self$max_iter = max_iter
    self$tolerance_loss = tolerance_loss
    self$tolerance_change = tolerance_change
    self$sgdargs = list(...) %||% list('lr' = 0.1, 'momentum' = 0.9)
    self$sgdargs$lr = self$sgdargs$lr %||% 0.1 # lr is required for torch::optim_sgd()

    if(f_min > self$f_max)
      value_error(glue::glue('Require f_min: {f_min} < f_max: {self$f_max}'))

    fb = functional_create_fb_matrix(
      n_freqs = n_stft,
      f_min = self$f_min,
      f_max = self$f_max,
      n_mels = self$n_mels,
      sample_rate = self$sample_rate
    )
    self$register_buffer('fb', fb)
  },

  forward = function(melspec) {
    # pack batch
    shape = melspec$size()
    ls = length(shape)
    melspec = melspec$view(c(-1, shape[ls-1], shape[ls]))

    n_mels = shape[ls-1]
    time = shape[ls]

    freq = self$fb$size(1) # (freq, n_mels)
    melspec = melspec$transpose(-1, -2)
    if(self$n_mels != n_mels) runtime_error("self$n_mels != n_mels")

    specgram = torch::torch_rand(melspec$size()[1], time, freq, requires_grad=TRUE,
                                 dtype=melspec$dtype, device=melspec$device)
    self$sgdargs$params <- specgram
    optim = do.call(torch::optim_sgd, self$sgdargs)

    loss = Inf
    for(i in seq.int(self$max_iter)){
      optim$zero_grad()
      diff = melspec - specgram$matmul(self$fb)
      new_loss = diff$pow(2)$sum(dim=-1)$mean()
      # take sum over mel-frequency then average over other dimensions
      # so that loss threshold is applied par unit timeframe
      new_loss$backward()
      optim$step()
      specgram$set_data(specgram$data()$clamp(min=0))

      new_loss = new_loss$item()
      if(new_loss < self$tolerance_loss | abs(loss - new_loss) < self$tolerance_change)
        break

      loss = new_loss
    }

    specgram$requires_grad_(FALSE)
    specgram = specgram$clamp(min=0)$transpose(-1, -2)

    # unpack batch
    specgram = specgram$view(c(shape[1:(ls-2)], freq, time))
    return(specgram)
  }
)

#' Mu Law Encoding
#'
#' Encode signal based on mu-law companding.  For more info see
#' the [Wikipedia Entry](https://en.wikipedia.org/wiki/M-law_algorithm)
#'
#' @param x  (Tensor): A signal to be encoded.
#' @param quantization_channels  (int, optional): Number of channels. (Default: ``256``)
#'
#' @return x_mu (Tensor): An encoded signal.
#'
#' @details
#' This algorithm assumes the signal has been scaled to between -1 and 1 and
#' returns a signal encoded with values from 0 to quantization_channels - 1.
#'
#' @export
transform_mu_law_encoding <- torch::nn_module(
  "MuLawEncoding",
  initialize = function(quantization_channels = 256) {
    self$quantization_channels = quantization_channels
  },

  forward = function(x) {
    return(functional_mu_law_encoding(x, self$quantization_channels))
  }
)

#' Mu Law Decoding
#'
#' Decode mu-law encoded signal.  For more info see the
#'  [Wikipedia Entry](https://en.wikipedia.org/wiki/M-law_algorithm)
#'
#'    This expects an input with values between 0 and quantization_channels - 1
#'    and returns a signal scaled between -1 and 1.
#'
#' @param x_mu  (Tensor): A mu-law encoded signal which needs to be decoded.
#' @param quantization_channels  (int, optional): Number of channels. (Default: ``256``)
#'
#' @return Tensor: The signal decoded.
#'
#' @export
transform_mu_law_decoding <- torch::nn_module(
  "MuLawDecoding",
  initialize = function(quantization_channels = 256) {
    self$quantization_channels = quantization_channels
  },

  forward = function(x_mu) {
    return(functional_mu_law_decoding(x_mu, self$quantization_channels))
  }
)

#' Signal Resample
#'
#' Resample a signal from one frequency to another. A resampling method can be given.
#'
#' @param waveform  (Tensor): Tensor of audio of dimension (..., time).
#' @param orig_freq  (float, optional): The original frequency of the signal. (Default: ``16000``)
#' @param new_freq  (float, optional): The desired frequency. (Default: ``16000``)
#' @param resampling_method  (str, optional): The resampling method. (Default: ``'sinc_interpolation'``)
#'
#' @return Tensor: Output signal of dimension (..., time).
#'
#' @export
transform_resample <- torch::nn_module(
  "Resample",
  initialize = function(
    orig_freq = 16000,
    new_freq = 16000,
    resampling_method = 'sinc_interpolation'
  ) {
    self$orig_freq = orig_freq
    self$new_freq = new_freq
    self$resampling_method = resampling_method
  },

  forward = function(waveform) {
    if(self$resampling_method == 'sinc_interpolation') {

      # pack batch
      shape = waveform$size()
      ls = length(shape)
      waveform = waveform$view(c(-1, shape[ls]))

      waveform = kaldi_resample_waveform(waveform, self$orig_freq, self$new_freq)

      # unpack batch
      lws = length(waveform$shape)
      waveform = waveform$view(c(shape[-ls], waveform$shape[lws]))

      return(waveform)

    } else {
      value_error(glue::glue('Invalid resampling method: {self$resampling_method}'))
    }
  }
)

#' Complex Norm
#'
#' Compute the norm of complex tensor input.
#'
#' @param complex_tensor  (Tensor): Tensor shape of `(..., complex=2)`.
#' @param power  (float, optional): Power of the norm. (Default: to ``1.0``)
#'
#' @return Tensor: norm of the input tensor, shape of `(..., )`.
#'
#' @export
trasnform_complex_norm <- torch::nn_module(
  "ComplexNorm",
  initialize = function(power = 1.0) {
    self$power = power
  },

  forward = function(complex_tensor) {
    return(functional_complex_norm(complex_tensor, self$power))
  }
)

#' Delta Coefficients
#'
#' Compute delta coefficients of a tensor, usually a spectrogram.
#'
#' @param specgram  (Tensor): Tensor of audio of dimension (..., freq, time).
#' @param win_length  (int): The window length used for computing delta. (Default: ``5``)
#' @param mode  (str): Mode parameter passed to padding. (Default: ``'replicate'``)
#'
#' @details See [torchaudio::functional_compute_deltas] for more details.
#'
#' @return Tensor: Tensor of deltas of dimension (..., freq, time).
#'
#' @export
transform_compute_deltas <- torch::nn_module(
  "ComputeDeltas",
  initialize = function( win_length = 5, mode = "replicate") {
    self$win_length = win_length
    self$mode = mode
  },

  forward = function(specgram) {
    return(functional_compute_deltas(specgram, win_length=self$win_length, mode=self$mode))
  }
)

#' Time Stretch
#'
#' Stretch stft in time without modifying pitch for a given rate.
#'
#' @param complex_specgrams  (Tensor): complex spectrogram (..., freq, time, complex=2).
#' @param hop_length  (int or NULL, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
#' @param n_freq  (int, optional): number of filter banks from stft. (Default: ``201``)
#' @param fixed_rate  (float or NULL, optional): rate to speed up or slow down by.
#'        If NULL is provided, rate must be passed to the forward method.  (Default: ``NULL``)
#' @param overriding_rate  (float or NULL, optional): speed up to apply to this batch.
#' @param If no rate is passed, use ``self$fixed_rate``.  (Default: ``NULL``)
#'
#' @return Tensor: Stretched complex spectrogram of dimension (..., freq, ceil(time/rate), complex=2).
#'
#' @export
transform_time_stretch <- torch::nn_module(
  "TimeStretch",
  initialize = function(
    hop_length = NULL,
    n_freq = 201,
    fixed_rate = NULL
  ) {
    self$fixed_rate = fixed_rate
    n_fft = (n_freq - 1) * 2
    hop_length = if(!is.null(hop_length)) hop_length  else n_fft %/% 2
    self$register_buffer('phase_advance', torch::torch_linspace(0, pi * hop_length, n_freq)[.., NULL])
  },

  forward = function(complex_specgrams, overriding_rate = NULL) {
    lcs = length(complex_specgrams$size())
    if(complex_specgrams$size()[lcs] != 2)
      value_error("complex_specgrams should be a complex tensor, shape (..., complex=2)")

    if(is.null(overriding_rate)) {
      rate = self$fixed_rate
      if(is.null(rate)) {
        value_error("If no fixed_rate is specified, must pass a valid rate to the forward method.")
      }
    } else {
      rate = overriding_rate
    }

    if(rate == 1.0) {
      return(complex_specgrams)
    } else {
      return(functional_phase_vocoder(complex_specgrams, rate, self$phase_advance))
    }
  }
)

#' Fade In/Out
#'
#' Add a fade in and/or fade out to an waveform.
#'
#' @param waveform  (Tensor): Tensor of audio of dimension (..., time).
#' @param fade_in_len  (int, optional): Length of fade-in (time frames). (Default: ``0``)
#' @param fade_out_len  (int, optional): Length of fade-out (time frames). (Default: ``0``)
#' @param fade_shape  (str, optional): Shape of fade. Must be one of: "quarter_sine",
#'                     "half_sine", "linear", "logarithmic", "exponential".  (Default: ``"linear"``)
#'
#' @return Tensor: Tensor of audio of dimension (..., time).
#'
#' @export
transform_fade <- torch::nn_module(
  "Fade",
  initialize = function(
    fade_in_len = 0,
    fade_out_len = 0,
    fade_shape = "linear"
  ) {
    self$fade_in_len = fade_in_len
    self$fade_out_len = fade_out_len
    self$fade_shape = fade_shape
  },

  forward = function(waveform) {
    lws = length(waveform$size())
    waveform_length = waveform$size()[lws]
    device = waveform$device
    return(self$.fade_in(waveform_length)$to(device = device) *  self$.fade_out(waveform_length)$to(device = device) * waveform)
  },

  .fade_in = function(waveform_length) {
    fade = torch::torch_linspace(0, 1, self$fade_in_len)
    ones = torch::torch_ones(waveform_length - self$fade_in_len)

    if(self$fade_shape == "linear")
      fade = fade

    if(self$fade_shape == "exponential")
      fade = torch::torch_pow(2, (fade - 1)) * fade

    if(self$fade_shape == "logarithmic")
      fade = torch::torch_log10(.1 + fade) + 1

    if(self$fade_shape == "quarter_sine")
      fade = torch::torch_sin(fade * pi / 2)

    if(self$fade_shape == "half_sine")
      fade = torch::torch_sin(fade * pi - pi / 2) / 2 + 0.5

    return(torch::torch_cat(list(fade, ones))$clamp_(0, 1))
  },

  .fade_out = function( waveform_length) {
    fade = torch::torch_linspace(0, 1, self$fade_out_len)
    ones = torch::torch_ones(waveform_length - self$fade_out_len)

    if(self$fade_shape == "linear")
      fade = - fade + 1

    if(self$fade_shape == "exponential")
      fade = torch::torch_pow(2, - fade) * (1 - fade)

    if(self$fade_shape == "logarithmic")
      fade = torch::torch_log10(1.1 - fade) + 1

    if(self$fade_shape == "quarter_sine")
      fade = torch::torch_sin(fade * pi / 2 + pi / 2)

    if(self$fade_shape == "half_sine")
      fade = torch::torch_sin(fade * pi + pi / 2) / 2 + 0.5

    return(torch::torch_cat(list(ones, fade))$clamp_(0, 1))
  }
)

#' Axis Masking
#'
#' Apply masking to a spectrogram.
#'
#' @param mask_param  (int): Maximum possible length of the mask.
#' @param axis  (int): What dimension the mask is applied on.
#' @param iid_masks  (bool): Applies iid masks to each of the examples in the batch dimension.
#'                   This option is applicable only when the input tensor is 4D.
#' @param specgram  (Tensor): Tensor of dimension (..., freq, time).
#' @param mask_value  (float): Value to assign to the masked columns.
#'
#' @return Tensor: Masked spectrogram of dimensions (..., freq, time).
#'
#' @export
transform__axismasking <- torch::nn_module(
  "_AxisMasking",
  initialize = function(mask_param, axis, iid_masks) {
    self$mask_param = mask_param
    self$axis = axis
    self$iid_masks = iid_masks
  },

  forward = function(specgram, mask_value = 0.) {
    # if(iid_masks flag marked and specgram has a batch dimension
    if(self$iid_masks & specgram$dim() == 4) {
      return(functional_mask_along_axis_iid(specgram, self$mask_param, mask_value, self$axis + 1L))
    } else {
      return(functional_mask_along_axis(specgram, self$mask_param, mask_value, self$axis))
    }
  }
)

#' Frequency-domain Masking
#'
#' Apply masking to a spectrogram in the frequency domain.
#'
#' @param freq_mask_param  (int): maximum possible length of the mask.
#'            Indices uniformly sampled from [0, freq_mask_param).
#' @param iid_masks  (bool, optional): whether to apply different masks to each
#'            example/channel in the batch.  (Default: ``FALSE``)
#'            This option is applicable only when the input tensor is 4D.
#'
#' @export
transform_frequencymasking <- function() {
  not_implemented_error("Class _AxisMasking to be implemented yet.")
}
#   R6::R6Class(
#   "FrequencyMasking",
#   inherit = transform__axismasking,
#   initialize = function(freq_mask_param, iid_masks = FALSE) {
#     # super(FrequencyMasking, self).__init__(freq_mask_param, 1, iid_masks)
#     # https://pytorch.org/audio/_modules/torchaudio/transforms.html#FrequencyMasking
#   }
# )

#' Time-domain Masking
#'
#' Apply masking to a spectrogram in the time domain.
#'
#' @param time_mask_param  (int): maximum possible length of the mask.
#'            Indices uniformly sampled from [0, time_mask_param).
#' @param iid_masks  (bool, optional): whether to apply different masks to each
#'            example/channel in the batch.  (Default: ``FALSE``)
#'            This option is applicable only when the input tensor is 4D.
#'
#' @export
transform_timemasking <- function() {
  not_implemented_error("Class _AxisMasking to be implemented yet.")
}
# torchaudio::transform_axismasking(
#   "TimeMasking",
#   initialize = function(time_mask_param, iid_masks = FALSE) {
#     # super(TimeMasking, self).__init__(time_mask_param, 2, iid_masks)
#     # https://pytorch.org/audio/_modules/torchaudio/transforms.html#TimeMasking
#     not_implemented_error("Class _AxisMasking to be implemented yet.")
#   }
# )


#' Add a volume to an waveform.
#'
#' @param waveform  (Tensor): Tensor of audio of dimension (..., time).
#' @param gain  (float): Interpreted according to the given gain_type:
#'            If ``gain_type`` = ``amplitude``, ``gain`` is a positive amplitude ratio.
#'            If ``gain_type`` = ``power``, ``gain`` is a power  (voltage squared).
#'            If ``gain_type`` = ``db``, ``gain`` is in decibels.
#' @param gain_type  (str, optional): Type of gain. One of: ``amplitude``, ``power``, ``db`` (Default: ``amplitude``)
#'
#' @return Tensor: Tensor of audio of dimension (..., time).
#'
#' @export
transform_vol <- torch::nn_module(
  "Vol",
  initialize = function(
    gain,
    gain_type = 'amplitude'
  ) {
    self$gain = gain
    self$gain_type = gain_type

    if(gain_type %in% c('amplitude', 'power') & gain < 0)
      value_error("If gain_type = amplitude or power, gain must be positive.")
  },

  forward = function(waveform) {

    if(self$gain_type == "amplitude")
      waveform = waveform * self$gain

    if(self$gain_type == "db")
      waveform = functional_gain(waveform, self$gain)

    if(self$gain_type == "power")
      waveform = functional_gain(waveform, 10 * log10(self$gain))

    return(torch::torch_clamp(waveform, -1, 1))
  }
)

#' sliding-window Cepstral Mean Normalization
#'
#'  Apply sliding-window cepstral mean  (and optionally variance) normalization per utterance.
#'
#' @param waveform  (Tensor): Tensor of audio of dimension (..., time).
#' @param cmn_window  (int, optional): Window in frames for running average CMN computation (int, default = 600)
#' @param min_cmn_window  (int, optional):  Minimum CMN window used at start of decoding (adds latency only at start).
#' @param Only applicable if center == ``FALSE``, ignored if center==``TRUE``  (int, default = 100)
#' @param center  (bool, optional): If ``TRUE``, use a window centered on the current frame
#'   (to the extent possible, modulo end effects). If ``FALSE``, window is to the left. (bool, default = ``FALSE``)
#' @param norm_vars  (bool, optional): If ``TRUE``, normalize variance to one. (bool, default = ``FALSE``)
#'
#' @return Tensor: Tensor of audio of dimension (..., time).
#'
#' @export
transform_sliding_window_cmn <- torch::nn_module(
  "SlidingWindowCmn",
  initialize = function(
    cmn_window = 600,
    min_cmn_window = 100,
    center = FALSE,
    norm_vars = FALSE
  ) {
    self$cmn_window = cmn_window
    self$min_cmn_window = min_cmn_window
    self$center = center
    self$norm_vars = norm_vars
  },

  forward = function(waveform) {
    cmn_waveform = functional_sliding_window_cmn(
      waveform,
      self$cmn_window,
      self$min_cmn_window,
      self$center,
      self$norm_vars
    )
    return(cmn_waveform)
  }
)


#' Voice Activity Detector
#'
#' Voice Activity Detector. Similar to SoX implementation.
#'
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
#'            and other characteristics of the input audio.  (Default: 7.0)
#' @param trigger_time  (float, optional): The time constant (in seconds)
#'            used to help ignore short bursts of sound.  (Default: 0.25)
#' @param search_time  (float, optional): The amount of audio (in seconds)
#'            to search for quieter/shorter bursts of audio to include prior
#'            the detected trigger point.  (Default: 1.0)
#' @param allowed_gap  (float, optional): The allowed gap (in seconds) between
#'            quiteter/shorter bursts of audio to include prior
#'            to the detected trigger point.  (Default: 0.25)
#' @param pre_trigger_time  (float, optional): The amount of audio (in seconds) to preserve
#'            before the trigger point and any found quieter/shorter bursts.  (Default: 0.0)
#' @param boot_time  (float, optional) The algorithm (internally) uses adaptive noise
#'            estimation/reduction in order to detect the start of the wanted audio.
#'            This option sets the time for the initial noise estimate.  (Default: 0.35)
#' @param noise_up_time  (float, optional) Time constant used by the adaptive noise estimator
#'            for when the noise level is increasing.  (Default: 0.1)
#' @param noise_down_time  (float, optional) Time constant used by the adaptive noise estimator
#'            for when the noise level is decreasing.  (Default: 0.01)
#' @param noise_reduction_amount  (float, optional) Amount of noise reduction to use in
#'            the detection algorithm  (e.g. 0, 0.5, ...). (Default: 1.35)
#' @param measure_freq  (float, optional) Frequency of the algorithm’s
#'            processing/measurements.  (Default: 20.0)
#' @param measure_duration:  (float, optional) Measurement duration. (Default: Twice the measurement period; i.e. with overlap.)
#' @param measure_smooth_time  (float, optional) Time constant used to smooth spectral measurements.  (Default: 0.4)
#' @param hp_filter_freq  (float, optional) "Brick-wall" frequency of high-pass filter applied
#'            at the input to the detector algorithm.  (Default: 50.0)
#' @param lp_filter_freq  (float, optional) "Brick-wall" frequency of low-pass filter applied
#'            at the input to the detector algorithm.  (Default: 6000.0)
#' @param hp_lifter_freq  (float, optional) "Brick-wall" frequency of high-pass lifter used
#'            in the detector algorithm.  (Default: 150.0)
#' @param lp_lifter_freq  (float, optional) "Brick-wall" frequency of low-pass lifter used
#'            in the detector algorithm.  (Default: 2000.0)
#'
#' @references
#' - [http://sox.sourceforge.net/sox.html]()
#'
#' @export
transform_vad <- torch::nn_module(
  "Vad",
  initialize = function(
    sample_rate,
    trigger_level = 7.0,
    trigger_time = 0.25,
    search_time = 1.0,
    allowed_gap = 0.25,
    pre_trigger_time = 0.0,
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

    self$sample_rate = sample_rate
    self$trigger_level = trigger_level
    self$trigger_time = trigger_time
    self$search_time = search_time
    self$allowed_gap = allowed_gap
    self$pre_trigger_time = pre_trigger_time
    self$boot_time = boot_time
    self$noise_up_time = noise_up_time
    self$noise_down_time = noise_up_time
    self$noise_reduction_amount = noise_reduction_amount
    self$measure_freq = measure_freq
    self$measure_duration = measure_duration
    self$measure_smooth_time = measure_smooth_time
    self$hp_filter_freq = hp_filter_freq
    self$lp_filter_freq = lp_filter_freq
    self$hp_lifter_freq = hp_lifter_freq
    self$lp_lifter_freq = lp_lifter_freq
  },

  forward = function(waveform) {
    return(functional_vad(
      waveform=waveform,
      sample_rate=self$sample_rate,
      trigger_level=self$trigger_level,
      trigger_time=self$trigger_time,
      search_time=self$search_time,
      allowed_gap=self$allowed_gap,
      pre_trigger_time=self$pre_trigger_time,
      boot_time=self$boot_time,
      noise_up_time=self$noise_up_time,
      noise_down_time=self$noise_up_time,
      noise_reduction_amount=self$noise_reduction_amount,
      measure_freq=self$measure_freq,
      measure_duration=self$measure_duration,
      measure_smooth_time=self$measure_smooth_time,
      hp_filter_freq=self$hp_filter_freq,
      lp_filter_freq=self$lp_filter_freq,
      hp_lifter_freq=self$hp_lifter_freq,
      lp_lifter_freq=self$lp_lifter_freq
    ))
  }
)

