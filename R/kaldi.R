#' Linear Resample Indices And Weights
#'
#' Based on LinearResample::SetIndexesAndWeights where it retrieves the weights for
#' resampling as well as the indices in which they are valid. LinearResample  (LR) means
#' that the output signal is at linearly spaced intervals  (i.e the output signal has a
#' frequency of ``new_freq``).
#'
#' @param orig_freq  (float): The original frequency of the signal
#' @param new_freq  (float): The desired frequency
#' @param output_samples_in_unit  (int): The number of output samples in the
#'   smallest repeating unit: num_samp_out = new_freq / Gcd (orig_freq, new_freq)
#' @param window_width  (float): The width of the window which is nonzero
#' @param lowpass_cutoff  (float): The filter cutoff in Hz. The filter cutoff needs to be less
#'            than samp_rate_in_hz/2 and less than samp_rate_out_hz/2.
#' @param lowpass_filter_width  (int): Controls the sharpness of the filter, more == sharper but less
#'            efficient. We suggest around 4 to 10 for normal use.
#' @param device (torch_device): Torch device on which output must be generated.
#' @param dtype (torch::torch_\<dtype\>): Torch dtype such as [torch::torch_float]
#'
#' @return Tensor, Tensor): A tuple of ``min_input_index`` (which is the minimum indices
#'         where the window is valid, size  (``output_samples_in_unit``)) and ``weights`` (which is the weights
#'         which correspond with min_input_index, size  (``output_samples_in_unit``, ``max_weight_width``)).
#'
#' @details
#'    It uses sinc/bandlimited interpolation to upsample/downsample
#'    the signal.
#'
#'    The reason why the same filter is not used for multiple convolutions is because the
#'    sinc function could sampled at different points in time. For example, suppose
#'    a signal is sampled at the timestamps  (seconds)
#'    0         16        32
#'    and we want it to be sampled at the timestamps  (seconds)
#'    0 5 10 15   20 25 30  35
#'    at the timestamp of 16, the delta timestamps are
#'    16 11 6 1   4  9  14  19
#'    at the timestamp of 32, the delta timestamps are
#'    32 27 22 17 12 8 2    3
#'
#'    As we can see from deltas, the sinc function is sampled at different points of time
#'    assuming the center of the sinc function is at 0, 16, and 32  (the deltas \[..., 6, 1, 4, ....\]
#'    for 16 vs \[...., 2, 3, ....\] for 32)
#'
#'    Example, one case is when the ``orig_freq`` and ``new_freq`` are multiples of each other then
#'    there needs to be one filter.
#'
#'    A windowed filter function  (i.e. Hanning * sinc) because the ideal case of sinc function
#'    has infinite support  (non-zero for all values) so instead it is truncated and multiplied by
#'    a window function which gives it less-than-perfect rolloff \[1\].
#'
#'    \[1\] Chapter 16: Windowed-Sinc Filters, https://www.dspguide.com/ch16/1.htm
#'
#' @export
kaldi__get_lr_indices_and_weights <- function(
  orig_freq,
  new_freq,
  output_samples_in_unit,
  window_width,
  lowpass_cutoff,
  lowpass_filter_width,
  device,
  dtype
) {
  if(lowpass_cutoff >= min(orig_freq, new_freq) / 2) value_error("lowpass_cutoff >= min(orig_freq, new_freq) / 2")
  output_t = torch::torch_arange(0., output_samples_in_unit, device=device, dtype=dtype) / new_freq
  min_t = output_t - window_width
  max_t = output_t + window_width

  min_input_index = torch::torch_ceil(min_t * orig_freq) # size (output_samples_in_unit)
  max_input_index = torch::torch_floor(max_t * orig_freq) # size (output_samples_in_unit)
  num_indices = max_input_index - min_input_index + 1 # size (output_samples_in_unit)

  max_weight_width = num_indices$max()
  # create a group of weights of size (output_samples_in_unit, max_weight_width)
  j = torch::torch_arange(0, as.numeric(max_weight_width$to(device = "cpu")), device=device, dtype=dtype)$unsqueeze(1)
  input_index = min_input_index$unsqueeze(2) + j
  delta_t = (input_index / orig_freq) - output_t$unsqueeze(2)

  weights = torch::torch_zeros_like(delta_t)
  inside_window_indices = delta_t$abs()$lt(window_width)
  # raised-cosine (Hanning) window with width `window_width`
  weights[inside_window_indices] = 0.5 * (1 + torch::torch_cos(2 * pi * lowpass_cutoff /
                                                                 lowpass_filter_width * delta_t[inside_window_indices]))

  t_eq_zero_indices = delta_t$eq(0.0)
  t_not_eq_zero_indices = !t_eq_zero_indices
  # sinc filter function
  weights[t_not_eq_zero_indices] = weights[t_not_eq_zero_indices] * torch::torch_sin(
    2 * pi * lowpass_cutoff * delta_t[t_not_eq_zero_indices]) / (pi * delta_t[t_not_eq_zero_indices])
  # limit of the function at t = 0
  weights[t_eq_zero_indices] = weights[t_eq_zero_indices] * 2 * lowpass_cutoff

  weights = weights / orig_freq # size (output_samples_in_unit, max_weight_width)
  return(list(min_input_index, weights))
}

#' @keywords internal
kaldi__lcm <- function(a, b) {
  package_required("numbers")
  return(abs(a * b) %/% numbers::GCD(a, b))
}

#' Linear Resample Output Samples
#'
#'  Based on LinearResample::GetNumOutputSamples. LinearResample  (LR) means that
#'  the output signal is at linearly spaced intervals  (i.e the output signal has a
#'  frequency of ``new_freq``). It uses sinc/bandlimited interpolation to upsample/downsample
#'  the signal.
#'
#' @param input_num_samp  (int): The number of samples in the input
#' @param samp_rate_in  (float): The original frequency of the signal
#' @param samp_rate_out  (float): The desired frequency
#'
#' @return int: The number of output samples
#'
#' @export
kaldi__get_num_lr_output_samples <- function(
  input_num_samp,
  samp_rate_in,
  samp_rate_out
) {

  # For exact computation, we measure time in "ticks" of 1.0 / tick_freq,
  # where tick_freq is the least common multiple of samp_rate_in and
  # samp_rate_out.
  samp_rate_in = as.integer(samp_rate_in)
  samp_rate_out = as.integer(samp_rate_out)

  tick_freq = kaldi__lcm(samp_rate_in, samp_rate_out)
  ticks_per_input_period = tick_freq %/% samp_rate_in

  # work out the number of ticks in the time interval
  # [ 0, input_num_samp/samp_rate_in ).
  interval_length_in_ticks = input_num_samp * ticks_per_input_period
  if(interval_length_in_ticks <= 0)
    return(0)

  ticks_per_output_period = tick_freq %/% samp_rate_out
  # Get the last output-sample in the closed interval, i.e. replacing [ ) with
  # [ ]. Note: integer division rounds down. See
  # http://en.wikipedia.org/wiki/Interval_(mathematics) for an explanation of
  # the notation.
  last_output_samp = interval_length_in_ticks %/% ticks_per_output_period
  # We need the last output-sample in the open interval, so if(it takes us to
  # the end of the interval exactly, subtract one.
  if((last_output_samp * ticks_per_output_period) == interval_length_in_ticks) {
    last_output_samp = last_output_samp - 1
  }
  # First output-sample index is zero, so the number of output samples
  # is the last output-sample plus one.
  num_output_samp = last_output_samp + 1
  return(num_output_samp)
}


#' Kaldi's Resample Waveform
#'
#' Resamples the waveform at the new frequency.
#'
#' @param waveform  (Tensor): The input signal of size (c, n)
#' @param orig_freq  (float): The original frequency of the signal
#' @param new_freq  (float): The desired frequency
#' @param lowpass_filter_width  (int, optional): Controls the sharpness of the filter, more == sharper
#'  but less efficient. We suggest around 4 to 10 for normal use.  (Default: ``6``)
#'
#' @details This matches Kaldi's OfflineFeatureTpl ResampleWaveform
#' which uses a LinearResample (resample a signal at linearly spaced intervals to upsample/downsample
#' a signal). LinearResample (LR) means that the output signal is at linearly spaced intervals (i.e
#' the output signal has a frequency of ``new_freq``). It uses sinc/bandlimited interpolation to
#' upsample/downsample the signal.
#'
#' @references
#' - [https://ccrma.stanford.edu/~jos/resample/Theory_Ideal_Bandlimited_Interpolation.html]()
#' - [https://github.com/kaldi-asr/kaldi/blob/master/src/feat/resample.h#L56]()
#'
#' @return Tensor: The waveform at the new frequency
#'
#' @export
kaldi_resample_waveform <- function(
  waveform,
  orig_freq,
  new_freq,
  lowpass_filter_width = 6
) {
  package_required("numbers")

  device = waveform$device
  dtype = waveform$dtype

  if(waveform$dim() != 2) value_error("waveform$dim() != 2")
  if(!(orig_freq > 0.0 & new_freq > 0.0)) value_error("orig_freq <= 0.0 or new_freq <= 0.0)")

  min_freq = min(orig_freq, new_freq)
  lowpass_cutoff = 0.99 * 0.5 * min_freq

  if(lowpass_cutoff * 2 > min_freq) value_error("lowpass_cutoff * 2 > min_freq")

  base_freq = numbers::GCD(as.integer(orig_freq), as.integer(new_freq))
  input_samples_in_unit = as.integer(orig_freq) %/% base_freq
  output_samples_in_unit = as.integer(new_freq) %/% base_freq

  window_width = lowpass_filter_width / (2.0 * lowpass_cutoff)
  first_indices_and_weights = kaldi__get_lr_indices_and_weights(
    orig_freq, new_freq,
    output_samples_in_unit,
    window_width,
    lowpass_cutoff,
    lowpass_filter_width,
    device,
    dtype
  )
  first_indices = first_indices_and_weights[[1]]
  weights = first_indices_and_weights[[2]]

  if(first_indices$dim() != 1) value_error("first_indices$dim() != 1")
  # TODO figure a better way to do this. conv1d reaches every element i*stride + padding
  # all the weights have the same stride but have different padding.
  # Current implementation takes the input and applies the various padding before
  # doing a conv1d for that specific weight.
  conv_stride = input_samples_in_unit
  conv_transpose_stride = output_samples_in_unit
  waveform_size = waveform$size()
  num_channels = waveform_size[1]
  wave_len = waveform_size[2]
  window_size = weights$size(2)
  tot_output_samp = kaldi__get_num_lr_output_samples(
    wave_len,
    orig_freq,
    new_freq
  )
  output = torch::torch_zeros(num_channels, tot_output_samp,
                              device=device, dtype=dtype)
  # eye size: (num_channels, num_channels, 1)
  eye = torch::torch_eye(num_channels, device=device, dtype=dtype)$unsqueeze(3)
  for(i in seq.int(first_indices$size(1))) {
    wave_to_conv = waveform
    first_index = as.integer(first_indices[i]$item())
    if(first_index >= 0) {
      # trim the signal as the filter will not be applied before the first_index
      wave_to_conv = wave_to_conv[.., first_index:N]
    }
    # pad the right of the signal to allow partial convolutions meaning compute
    # values for partial windows (e.g. end of the window is outside the signal length)
    max_unit_index = (tot_output_samp - 1) %/% output_samples_in_unit
    end_index_of_last_window = max_unit_index * conv_stride + window_size
    current_wave_len = wave_len - first_index
    right_padding = max(0, end_index_of_last_window + 1 - current_wave_len)

    left_padding = max(0, -first_index)
    if(left_padding != 0 | right_padding != 0) {
      wave_to_conv = torch::nnf_pad(wave_to_conv, c(left_padding, right_padding))
    }
    conv_wave = torch::nnf_conv1d(
      wave_to_conv$unsqueeze(1),
      weights[i]$`repeat`(c(num_channels, 1, 1)),
      stride=conv_stride,
      groups=num_channels
    )

    # we want conv_wave[:, i] to be at output[:, i + n*conv_transpose_stride]
    dilated_conv_wave = torch::nnf_conv_transpose1d(
      conv_wave,
      eye,
      stride = conv_transpose_stride
    )$squeeze(1)

    # pad dilated_conv_wave so it reaches the output length if(needed.
    dialated_conv_wave_len = dilated_conv_wave$size(-1)
    left_padding = i-1
    right_padding = max(0, tot_output_samp - (left_padding + dialated_conv_wave_len))
    dilated_conv_wave = torch::nnf_pad(dilated_conv_wave, c(left_padding, right_padding))[.., 1:tot_output_samp]

    output = output + dilated_conv_wave
  }
  return(output)
}
