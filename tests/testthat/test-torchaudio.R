sample_mp3 <- tuneR::readMP3(system.file("sample_audio_1.mp3", package = "torchaudio"))

test_that("torchaudio_load", {
    waveform_and_sample_rate <- torchaudio::torchaudio_load(system.file("sample_audio_1.mp3", package = "torchaudio"))
    expect_equal(waveform_and_sample_rate[[2]], sample_mp3@samp.rate)
    expect_tensor(waveform_and_sample_rate[[1]])
})
