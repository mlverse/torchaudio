filepath_mp3 <- system.file("sample_audio_1.mp3", package = "torchaudio")
mp3_info <- av::av_media_info(filepath_mp3)

filepath_wav <- system.file("waves_yesno/1_1_0_1_1_0_1_1.wav", package = "torchaudio")
filepath_wav_2chan <- tempfile(fileext = ".wav")
av::av_audio_convert(filepath_wav, channels = 2, output = filepath_wav_2chan)

test_that("av_loader works", {

  # mp3
  av_obj <- av_loader(filepath_mp3)
  expect_equal(class(av_obj)[1], "av")
  waveform_and_sample_rate <- transform_to_tensor(av_obj)
  expect_equal(waveform_and_sample_rate[[2]], mp3_info$audio$sample_rate)
  expect_tensor(waveform_and_sample_rate[[1]])
  expect_equal(length(waveform_and_sample_rate[[1]]), length(av_obj) / attr(av_obj, "channels"))

  av_obj <- av_loader(filepath_mp3, offset = 5000, duration = 2)
  expect_equal(class(av_obj)[1], "av")
  waveform_and_sample_rate <- transform_to_tensor(av_obj)
  expect_equal(waveform_and_sample_rate[[2]], mp3_info$audio$sample_rate)
  expect_tensor(waveform_and_sample_rate[[1]])
  expect_equal(dim(waveform_and_sample_rate[[1]]), c(1,2))

  # wav
  av_obj <- av_loader(filepath_wav)
  expect_equal(attr(av_obj, "channels"), 1)
  expect_equal(attr(av_obj, "dim")[2], 46960)
  av_obj <- av_loader(filepath_wav_2chan)
  expect_equal(attr(av_obj, "channels"), 2)
  expect_equal(attr(av_obj, "dim")[2], 46960)

})

test_that("transform_to_tensor works with av", {

  by_samples <- av_loader(filepath_mp3, offset = 1, duration = 48000, unit = "samples")
  waveform_and_sample_rate <- transform_to_tensor(by_samples)
  expect_equal(waveform_and_sample_rate[[2]], mp3_info$audio$sample_rate)
  expect_tensor(waveform_and_sample_rate[[1]])

})
