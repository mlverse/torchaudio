filepath <- system.file("sample_audio_1.mp3", package = "torchaudio")
sample_mp3 <- tuneR::readMP3(filepath)

test_that("av_loader works", {

})

test_that("tuneR_loader works", {
  by_samples <- tuneR_loader(filepath, offset = 1, duration = 48000, unit = "samples")

  expect_gt(length(by_samples@left), 40000)
  expect_lt(length(by_samples@left), 50000)

  by_time <-tuneR_loader(filepath, offset = 1, duration = 1, unit = "time")

  expect_gt(length(by_time@left), 40000)
  expect_lt(length(by_time@left), 50000)
})


test_that("transform_to_tensor works", {
  # Wave
  by_samples <- tuneR_loader(filepath, offset = 1, duration = 48000, unit = "samples")
  waveform_and_sample_rate <- transform_to_tensor(by_samples)
  expect_equal(waveform_and_sample_rate[[2]], sample_mp3@samp.rate)
  expect_tensor(waveform_and_sample_rate[[1]])

  # numeric vector
  #...
})
