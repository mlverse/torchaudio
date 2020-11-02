test_that("kaldi__get_lr_indices_and_weights", {
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
})

test_that("kaldi_resample_waveform", {
})

