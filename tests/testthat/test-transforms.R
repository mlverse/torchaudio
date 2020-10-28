sample_mp3 <- tuneR::readMP3(system.file("sample_audio_1.mp3", package = "torchaudio"))

test_that("transform_mel_scale", {
  n_fft = 400
  samples = length(sample_mp3@left)
  expect_no_error(spec <- functional_spectrogram(sample_mp3@left, n_fft = n_fft))

  ms = transform_mel_scale()(spec)
  expect_tensor(ms)
  expect_equal(dim(ms), c(1, 128, 1709))
})
