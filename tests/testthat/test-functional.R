sample_mp3 <- tuneR::readMP3(system.file("sample_audio_1.mp3", package = "torchaudio"))

test_that("spectrogram", {
  n_fft = 400
  samples = length(sample_mp3@left)
  expect_no_error(spec <- spectrogram(sample_mp3@left, n_fft = n_fft))
  expect_tensor(spec)
  expect_equal(dim(spec)[1], n_fft %/% 2 + 1)

})

test_that("create_fb_matrix", {
  x <- create_fb_matrix(
    n_freqs = 10,
    n_mels = 13,
    f_min = 0,
    f_max = 8000,
    sample_rate = 16000,
    norm = NULL
  )
  expect_tensor(x)
  expect_tensor_shape(x, c(10, 13))
})

test_that("complex_norm", {
  tensor_r <- c(1,2,3)
  tensor_t <- torch::torch_tensor(tensor_r)

  tensor_r_cn <- sum((tensor_r^2))^(0.5)
  tensor_t_cn <- complex_norm(tensor_t)
  expect_lt(abs(as.numeric(tensor_t_cn) - tensor_r_cn), 1e-6)
})

test_that("create_dct", {
  x <- create_dct(
    n_mfcc = 10,
    n_mels = 64,
    norm = NULL
  )
  expect_tensor(x)
  expect_tensor_shape(x, c(64, 10))
})
