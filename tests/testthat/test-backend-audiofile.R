sample_wav <- torchaudio:::audiofile_read_wav(system.file("ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav", package = "torchaudio"))

test_that("audiofile_loader", {
  audiofile_obj <- audiofile_loader(system.file("ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav", package = "torchaudio"))
  expect_equal(class(audiofile_obj)[1], "audiofile")

  waveform_and_sample_rate <- transform_to_tensor(audiofile_obj)

  expect_equal(waveform_and_sample_rate[[2]], sample_wav$sample_rate)
  expect_tensor(waveform_and_sample_rate[[1]])
  expect_equal(length(waveform_and_sample_rate[[1]]), length(sample_wav$waveform[[1]]))
  expect_equal(dim(waveform_and_sample_rate[[1]]), c(1, length(sample_wav$waveform[[1]])))
})
