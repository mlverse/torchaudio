filepath_mp3 <- system.file("sample_audio_1.mp3", package = "torchaudio")
sample_mp3 <- tuneR::readMP3(filepath_mp3)
filepath_wav <- system.file("waves_yesno/1_1_0_1_1_0_1_1.wav", package = "torchaudio")
sample_wav <- tuneR::readWave(filepath_wav)

test_that("av_loader works", {

})

test_that("tuneR_loader works", {
  # MP3
  by_samples <- tuneR_loader(filepath_mp3, offset = 1, duration = 40000, unit = "samples")

  expect_gt(length(by_samples@left), 40000)
  expect_lt(length(by_samples@left), 50000)

  by_time <-tuneR_loader(filepath_mp3, offset = 1, duration = 1, unit = "time")

  expect_gt(length(by_time@left), 40000)
  expect_lt(length(by_time@left), 50000)

  # WAV
  by_samples <- tuneR_loader(filepath_wav, offset = 1, duration = 40000, unit = "samples")

  expect_gt(length(by_samples@left), 40000)
  expect_lt(length(by_samples@left), 50000)

  by_time <-tuneR_loader(filepath_wav, offset = 1, duration = 1, unit = "seconds")

  expect_gt(length(by_time@left), 7999)
  expect_lt(length(by_time@left), 8001)
})


test_that("transform_to_tensor works", {
  # Wave
  by_samples <- tuneR_loader(filepath_mp3, offset = 1, duration = 48000, unit = "samples")
  waveform_and_sample_rate <- transform_to_tensor(by_samples)
  expect_equal(waveform_and_sample_rate[[2]], sample_mp3@samp.rate)
  expect_tensor(waveform_and_sample_rate[[1]])

  # numeric vector
  #...
})
