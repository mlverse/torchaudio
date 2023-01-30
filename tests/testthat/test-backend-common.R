filepath_mp3 <- system.file("sample_audio_1.mp3", package = "torchaudio")
sample_mp3 <- tuneR::readMP3(filepath_mp3)
filepath_wav <- system.file("waves_yesno/1_1_0_1_1_0_1_1.wav", package = "torchaudio")
sample_wav <- tuneR::readWave(filepath_wav)

test_that("av_loader works", {

})


test_that("transform_to_tensor works", {
  # Wave
  by_samples <- tuneR_loader(filepath_mp3, offset = 1, duration = 48000, unit = "samples")
  waveform_and_sample_rate <- transform_to_tensor(by_samples)
  expect_equal(waveform_and_sample_rate[[2]], sample_mp3@samp.rate)
  expect_tensor(waveform_and_sample_rate[[1]])

  # av
  by_samples <- av_loader(filepath_mp3, offset = 1, duration = 48000, unit = "samples")
  waveform_and_sample_rate <- transform_to_tensor(by_samples)
  expect_equal(waveform_and_sample_rate[[2]], sample_mp3@samp.rate)
  expect_tensor(waveform_and_sample_rate[[1]])

})

test_that("set_audio_backend and torchaudio_load works", {
  loader <- getOption("torchaudio.loader")
  expect_equal(class(loader), "function")

  set_audio_backend(tuneR_loader)
  audio <- torchaudio_load(filepath_mp3, offset = 1, duration = 100, unit = "samples")
  expect_equal(class(audio)[1], "Wave")
  expect_equal(length(audio), 100)

  set_audio_backend(av_loader)
  audio <- torchaudio_load(filepath_mp3, offset = 1, duration = 100, unit = "samples")
  expect_equal(class(audio)[1], "av")
  expect_equal(length(audio), 100)
})

