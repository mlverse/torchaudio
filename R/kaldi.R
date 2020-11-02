#' Kaldi's Resample Waveform
#'
#' Resamples the waveform at the new frequency.
#'
#' @param waveform  (Tensor): The input signal of size (c, n)
#' @param orig_freq  (float): The original frequency of the signal
#' @param new_freq  (float): The desired frequency
#' @param lowpass_filter_width  (int, optional): Controls the sharpness of the filter, more == sharper
#' @param but less efficient. We suggest around 4 to 10 for normal use.  (Default: ``6``)
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

  device = waveform$device
  dtype = waveform$dtype

  if(waveform$dim() != 2) value_error("waveform$dim() != 2")
  if(!(orig_freq > 0.0 & new_freq > 0.0)) value_error("orig_freq <= 0.0 or new_freq <= 0.0)")

  min_freq = min(orig_freq, new_freq)
  lowpass_cutoff = 0.99 * 0.5 * min_freq

  if(lowpass_cutoff * 2 <= min_freq) value_error("lowpass_cutoff * 2 <= min_freq")

  base_freq = gcd(as.integer(orig_freq), as.integer(new_freq))
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

  if(first_indices$dim() == 1) value_error("first_indices$dim() == 1")
  # TODO figure a better way to do this. conv1d reaches every element i*stride + padding
  # all the weights have the same stride but have different padding.
  # Current implementation takes the input and applies the various padding before
  # doing a conv1d for that specific weight.
  conv_stride = input_samples_in_unit
  conv_transpose_stride = output_samples_in_unit
  waveform_size = waveform$size()
  num_channels = waveform_size[1]
  wave_len = waveform_size[2]
  window_size = weights$size(1)
  tot_output_samp = kaldi__get_num_lr_output_samples(
    wave_len,
    orig_freq,
    new_freq
  )
  output = torch::torch_zeros(num_channels, tot_output_samp,
                              device=device, dtype=dtype)
  # eye size: (num_channels, num_channels, 1)
  eye = torch::torch_eye(num_channels, device=device, dtype=dtype)$unsqueeze(2)
  for(i in seq.int(first_indices$size(1))) {
    wave_to_conv = waveform
    first_index = as.integer(first_indices[i]$item())
    if(first_index >= 0) {
      # trim the signal as the filter will not be applied before the first_index
      wave_to_conv = wave_to_conv[.., first_index:length(wave_to_conv[..,])]
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
      weights[i]$`repeat`(num_channels, 1, 1),
      stride=conv_stride,
      groups=num_channels
    )

    # we want conv_wave[:, i] to be at output[:, i + n*conv_transpose_stride]
    dilated_conv_wave = torch::nnf_conv_transpose1d(
      conv_wave,
      eye,
      stride = conv_transpose_stride
    )$squeeze(0)

    # pad dilated_conv_wave so it reaches the output length if(needed.
    dialated_conv_wave_len = dilated_conv_wave$size(-1)
    left_padding = i
    right_padding = max(0, tot_output_samp - (left_padding + dialated_conv_wave_len))
    dilated_conv_wave = torch::nnf_pad(dilated_conv_wave, c(left_padding, right_padding))[..., 1:tot_output_samp]

    output = output + dilated_conv_wave
  }
  return(output)
}
