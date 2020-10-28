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
    browser()
    mfcc = torch::torch_matmul(mel_specgram$transpose(3, 4), self$dct_mat)$transpose(3, 4)

    # unpack batch
    lspec = length(mfcc$shape)
    mfcc = mfcc$reshape(c(shape[-((ls-1):ls)], mfcc$shape[(lspec-2):lspec]))

    return(mfcc)
  }
)
