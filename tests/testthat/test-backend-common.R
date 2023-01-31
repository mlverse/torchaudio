filepath_mp3 <- system.file("sample_audio_1.mp3", package = "torchaudio")

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

