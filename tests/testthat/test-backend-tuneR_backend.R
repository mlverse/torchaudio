sample_mp3 <- tuneR::readMP3(system.file("sample_audio_1.mp3", package = "torchaudio"))

test_that("tuneR_loader", {
  tuneR_Wave <- tuneR_loader(system.file("sample_audio_1.mp3", package = "torchaudio"))
  expect_equal(class(tuneR_Wave)[1], "Wave")

  waveform_and_sample_rate <- transform_to_tensor(tuneR_Wave)

  expect_equal(waveform_and_sample_rate[[2]], sample_mp3@samp.rate)
  expect_tensor(waveform_and_sample_rate[[1]])
  expect_equal(length(waveform_and_sample_rate[[1]]), length(sample_mp3@left))

  tuneR_Wave <- tuneR_loader(
    system.file("sample_audio_1.mp3", package = "torchaudio"),
    normalization = 10,
    offset = 2000,
    duration = 2
  )
  expect_equal(class(tuneR_Wave)[1], "Wave")

  waveform_and_sample_rate <- transform_to_tensor(tuneR_Wave)
  expect_equal(waveform_and_sample_rate[[2]], sample_mp3@samp.rate)
  expect_tensor(waveform_and_sample_rate[[1]])
  expect_equal(dim(waveform_and_sample_rate[[1]]), c(1,2))
})
