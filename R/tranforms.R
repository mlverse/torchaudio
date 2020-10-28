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
#' @param Arguments for window function.
#'
#' @return tensor: Dimension (..., freq, time), freq is n_fft %/% 2 + 1 and n_fft is the
#' number of Fourier bins, and time is the number of window hops (n_frame).
#' @export
transform_spectrogram <- torch::nn_module(
  "Spectrogram",
  initialize = function(
    n_fft = 400,
    win_length = NULL,
    hop_length = NULL,
    pad = 0,
    window_fn = torch::torch_hann_window,
    power = 2,
    normalized = FALSE,
    ...
  ) {
    self$n_fft = n_fft

    # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
    # number of frequecies due to onesided=True in torch.stft
    self$win_length = if(!is.null(win_length)) win_length else n_fft
    self$hop_length = if(!is.null(hop_length)) hop_length else self$win_length %/% 2
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
#' @param f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
#' @param n_stft (int, optional): Number of bins in STFT. Calculated from first input
#' if None is given.  See ``n_fft`` in :class:`Spectrogram`. (Default: ``None``)
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
      create_fb_matrix(
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
