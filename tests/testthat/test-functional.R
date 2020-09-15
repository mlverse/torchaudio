sample_mp3 <- tuneR::readMP3(system.file("sample_audio.mp3", package = "torchaudio"))

test_that("spectrogram", {
  n_fft = 400
  samples = length(sample_mp3@left)
  expect_no_error(spec <- spectrogram(sample_mp3@left, n_fft = n_fft))
  expect_tensor(spec)
  expect_equal(dim(spec)[1], n_fft %/% 2 + 1)

})

test_that("complex_norm", {
  tensor_r <- c(1,2,3)
  tensor_t <- torch::torch_tensor(tensor_r)

  tensor_r_cn <- sum((tensor_r^2))^(0.5)
  tensor_t_cn <- complex_norm(tensor_t)
  expect_identical()
  expect_lt(abs(as.numeric(tensor_t_cn) - tensor_r_cn), 1e-6)
})
