sample_mp3 <- av::read_audio_bin(system.file("sample_audio_1.mp3", package = "torchaudio"))

test_that("av_loader", {
  av_obj <- av_loader(system.file("sample_audio_1.mp3", package = "torchaudio"))
  expect_equal(class(av_obj)[1], "matrix")

  waveform_and_sample_rate <- transform_to_tensor(av_obj)

  expect_equal(waveform_and_sample_rate[[2]], attr(sample_mp3, "sample_rate"))
  expect_tensor(waveform_and_sample_rate[[1]])

  av_obj <- av_loader(
    system.file("sample_audio_1.mp3", package = "torchaudio"),
    normalization = 10,
    offset = 5000,
    duration = 2
  )
  expect_equal(class(av_obj)[1], "matrix")

  waveform_and_sample_rate <- transform_to_tensor(av_obj)
  expect_equal(waveform_and_sample_rate[[2]], attr(sample_mp3, "sample_rate"))
  expect_tensor(waveform_and_sample_rate[[1]])
  expect_equal(dim(waveform_and_sample_rate[[1]]), c(1,2))
})
