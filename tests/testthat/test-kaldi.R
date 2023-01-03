test_that("kaldi__get_lr_indices_and_weights", {
  if(requireNamespace("numbers", quietly = TRUE)) {
    expect_no_error(x <- kaldi__get_lr_indices_and_weights(
      orig_freq = 4,
      new_freq = 3,
      output_samples_in_unit = 6,
      window_width = 2,
      lowpass_cutoff = 1,
      lowpass_filter_width = 2,
      device = torch::torch_device("cpu"),
      dtype = torch::torch_float()
    ))
    expect_tensor(x[[1]])
    expect_tensor(x[[2]])
    expect_error(x[[3]])
  }
})

test_that("kaldi__get_num_lr_output_samples", {
  if(requireNamespace("numbers", quietly = TRUE)) {
    expect_no_error(x <- kaldi__get_num_lr_output_samples(100, 10, 5))
    expect_equal(x, 50)
  }
})

test_that("kaldi_resample_waveform", {
  if(requireNamespace("numbers", quietly = TRUE)) {
    expect_no_error(x <- kaldi_resample_waveform(
      waveform = torch::torch_rand(2, 100),
      orig_freq = 10,
      new_freq = 9,
      lowpass_filter_width = 6
    ))
    expect_tensor(x)
  }
})

