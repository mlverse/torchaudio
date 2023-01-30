filepath_mp3 <- system.file("sample_audio_1.mp3", package = "torchaudio")
sample_mp3 <- tuneR::readMP3(filepath_mp3)
filepath_wav <- system.file("waves_yesno/1_1_0_1_1_0_1_1.wav", package = "torchaudio")
sample_wav <- tuneR::readWave(filepath_wav)

test_that("tuneR_loader works", {
  # MP3
  by_samples <- tuneR_loader(filepath_mp3, offset = 1, duration = 40000, unit = "samples")
  expect_equal(length(by_samples@left), 40000)
  expect_equal(class(by_samples)[1], c("Wave"))

  by_samples <- tuneR_loader(filepath_mp3, offset = 1, duration = 100, unit = "samples")
  expect_equal(length(by_samples@left), 100)
  expect_equal(class(by_samples)[1], c("Wave"))

  by_time <- tuneR_loader(filepath_mp3, offset = 1, duration = 1, unit = "time")
  expect_gt(length(by_time@left), 40000)
  expect_lt(length(by_time@left), 50000)
  expect_equal(class(by_time)[1], c("Wave"))

  tuneR_Wave <- tuneR_loader(system.file("sample_audio_1.mp3", package = "torchaudio"))
  expect_equal(class(tuneR_Wave)[1], "Wave")
  waveform_and_sample_rate <- transform_to_tensor(tuneR_Wave)
  expect_equal(waveform_and_sample_rate[[2]], sample_mp3@samp.rate)
  expect_tensor(waveform_and_sample_rate[[1]])
  expect_equal(length(waveform_and_sample_rate[[1]]), length(sample_mp3@left))

  tuneR_Wave <- tuneR_loader(
    system.file("sample_audio_1.mp3", package = "torchaudio"),
    offset = 2000,
    duration = 2
  )
  expect_equal(class(tuneR_Wave)[1], "Wave")
  waveform_and_sample_rate <- transform_to_tensor(tuneR_Wave)
  expect_equal(waveform_and_sample_rate[[2]], sample_mp3@samp.rate)
  expect_tensor(waveform_and_sample_rate[[1]])
  expect_equal(dim(waveform_and_sample_rate[[1]]), c(1, 2))

  # WAV
  by_samples <- tuneR_loader(filepath_wav, offset = 1, duration = 40000, unit = "samples")
  expect_equal(length(by_samples@left), 40000)
  expect_equal(class(by_samples)[1], c("Wave"))

  by_samples <- tuneR_loader(filepath_wav, offset = 1, duration = 100, unit = "samples")
  expect_equal(length(by_samples@left), 100)
  expect_equal(class(by_samples)[1], c("Wave"))

  by_time <- tuneR_loader(filepath_wav, offset = 1, duration = 1, unit = "seconds")
  expect_gte(length(by_time@left), 8000)
  expect_lte(length(by_time@left), 8000)
  expect_equal(class(by_time)[1], c("Wave"))
})
