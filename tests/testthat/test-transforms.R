sample_mp3 <- tuneR::readMP3(system.file("sample_audio_1.mp3", package = "torchaudio"))

test_that("transform_spectrogram", {
  samples = length(sample_mp3@left)
  expect_no_error(spec <- transform_spectrogram()(sample_mp3@left))
  expect_tensor(spec)
  expect_equal(dim(spec)[1], 400 %/% 2 + 1)
})

test_that("transform_mel_scale", {
  n_fft = 400
  samples = length(sample_mp3@left)
  expect_no_error(spec <- functional_spectrogram(sample_mp3@left, n_fft = n_fft))

  ms = transform_mel_scale()(spec)
  expect_tensor(ms)
  expect_equal(dim(ms), c(1, 128, 1709))
})

test_that("transform_amplitude_to_db", {
  x1 <- torch::torch_arange(0,3, dtype = torch::torch_float())

  # amplitude_to_db
  x2 <- transform_amplitude_to_db()(x1)
  expect_tensor(x2)
  expect_tensor_shape(x2, c(3))

  # top_db
  x2 <- transform_amplitude_to_db(top_db = 1.0)(x1)
  expect_tensor(x2)
  expect_tensor_shape(x2, c(3))

  # DB_to_amplitude
  expect_lt( as.numeric(sum(functional_db_to_amplitude(transform_amplitude_to_db()(x1)) - x1)), 1e-8)
})
