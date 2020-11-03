cmuarctic_df <- torchaudio::cmuarctic_dataset(system.file("", package = "torchaudio"))
spectrogram_spec <- torchaudio::transform_spectrogram(n_fft = 255)

spectrograms = purrr::map(seq.int(cmuarctic_df), ~{
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
   input = torch::torch_rand (10, 128, 10)  # a random spectrogram
   output = upsamplenetwork (input)  # shape: (10, 1536, 128), (10, 1536, 128)
   expect_equal(dim(output[[1]]), c(10, 1536, 128))
   expect_equal(dim(output[[2]]), c(10, 1536, 128))
})

test_that("model_wavernn", {
  wavernn = model_wavernn(upsample_scales=c(5,5,8), n_classes=512, hop_length=200)

  waveform_and_sample_rate = torchaudio::torchaudio_load (system.file("sample_audio_1.mp3", package = "torchaudio"))
  waveform = waveform_and_sample_rate[[1]]
  sample_rate = waveform_and_sample_rate[[2]]
  # waveform shape:  (n_batch, n_channel, (n_time - kernel_size + 1) * hop_length)
  specgram = transform_mel_spectrogram(sample_rate)(waveform)  # shape: (n_batch, n_channel, n_freq, n_time)
  expect_no_error(output <- wavernn (waveform, specgram))
  expect_tensor(output)
  expect_equal(dim(output), c(1, 2, (length(waveform) - 200 + 1) * 128, 512))
  # output shape:  (n_batch, n_channel, (n_time - kernel_size + 1) * hop_length, n_classes)
})



