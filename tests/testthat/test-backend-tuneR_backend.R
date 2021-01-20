sample_mp3 <- tuneR::readMP3(system.file("sample_audio_1.mp3", package = "torchaudio"))

test_that("tuneR_loader", {
  waveform_and_sample_rate <- tuneR_loader(system.file("sample_audio_1.mp3", package = "torchaudio"))
  expect_equal(class(waveform_and_sample_rate)[1], "Wave")

  waveform_and_sample_rate <- wave_to_tensor(waveform_and_sample_rate)

  expect_equal(waveform_and_sample_rate[[2]], sample_mp3@samp.rate)
  expect_tensor(waveform_and_sample_rate[[1]])

  waveform_and_sample_rate <- tuneR_loader(
    system.file("sample_audio_1.mp3", package = "torchaudio"),
    normalization = 10,
    channels_first = FALSE,
    offset = 2000,
    duration = 2
  )
  expect_equal(class(waveform_and_sample_rate)[1], "Wave")

  waveform_and_sample_rate <- wave_to_tensor(waveform_and_sample_rate)
  expect_equal(waveform_and_sample_rate[[2]], sample_mp3@samp.rate)
  expect_tensor(waveform_and_sample_rate[[1]])
  expect_equal(dim(waveform_and_sample_rate[[1]]), c(2,2))
})
