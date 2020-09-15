sample_mp3 <- tuneR::readMP3(system.file("sample_audio.mp3", package = "torchaudio"))

test_that("spectrogram", {
  expect_no_error(spec <- spectrogram(sample_mp3))
  expect_tensor(spec)

  spec_r <- monitoR:::spectro(sample_mp3)
  expect_equal_to_r(spec, spec_r$amp)

})
