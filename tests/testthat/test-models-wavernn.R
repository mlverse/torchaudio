library(purrr)
cmuarctic_df <- torchaudio::cmuarctic_dataset(system.file("", package = "torchaudio"))
spectrogram_spec <- torchaudio::transform_spectrogram(n_fft = 255)

spectrograms = map(seq.int(cmuarctic_df), ~{
  s <-spectrogram_spec(cmuarctic_df[.x][[1]])
  s <- s[..,1:100]
  s <- torch::torch_mean(s, 1)
})
spectrograms = torch::torch_stack(spectrograms)

test_that("model_resblock", {
  resblock = model_resblock()
  expect_no_error(x <- resblock(spectrograms))
  expect_tensor(x)
  expect_true(x$requires_grad)
})

test_that("model_melresnet", {
   melresnet <- model_melresnet()
   expect_no_error(x <- melresnet(spectrograms))
   expect_tensor(x)
   expect_true(x$requires_grad)
})

test_that("model_stretch2d", {
  stretch2d <- model_stretch2d(freq_scale = 1L, time_scale = 1L)
  expect_no_error(x <- stretch2d(spectrograms))
  expect_tensor(x)
  expect_false(x$requires_grad)
})

test_that("model_upsample_network", {
   upsamplenetwork = model_upsample_network(upsample_scales=c(4, 4, 16))
   input = torch::torch_rand (3, 128, 10)  # a random spectrogram
   output = upsamplenetwork (input)  # shape: (10, 1536, 128), (10, 1536, 128)
   expect_equal(dim(output[[1]]), c(3, 128, (10 - 5 + 1)*(4*4*16)))
   expect_equal(dim(output[[2]]), c(3, 128, (10 - 5 + 1)*(4*4*16)))
})

test_that("model_wavernn", {
  wavernn = model_wavernn(upsample_scales=c(2,2,3), n_classes=5, hop_length=12)

  waveform = torch::torch_rand(3,1,(10 - 5 + 1)*12)
  spectrogram = torch::torch_rand(3,1,128,10)
  # waveform shape:  (n_batch, n_channel, (n_time - kernel_size + 1) * hop_length)
  expect_no_error(output <- wavernn (waveform, spectrogram))
  expect_tensor(output)
  expect_equal(dim(output), c(3, 1, (10 - 5 + 1) * 12, 5))
  # output shape:  (n_batch, n_channel, (n_time - kernel_size + 1) * hop_length, n_classes)
})



