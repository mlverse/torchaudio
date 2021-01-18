sample_mp3 <- tuneR::readMP3(system.file("sample_audio_1.mp3", package = "torchaudio"))

test_that("torchaudio_load", {
    waveform_and_sample_rate <- torchaudio::torchaudio_load(system.file("sample_audio_1.mp3", package = "torchaudio"))
    expect_equal(waveform_and_sample_rate[[2]], sample_mp3@samp.rate)
    expect_tensor(waveform_and_sample_rate[[1]])

    waveform_and_sample_rate <- torchaudio::torchaudio_load(system.file("sample_audio_1.mp3", package = "torchaudio"), offset = 1, duration = 4)
    expect_equal(waveform_and_sample_rate[[2]], sample_mp3@samp.rate)
    expect_tensor(waveform_and_sample_rate[[1]])
    expect_equal(dim(waveform_and_sample_rate[[1]]), c(2, 4))

    waveform_and_sample_rate <- torchaudio::torchaudio_load(system.file("sample_audio_1.mp3", package = "torchaudio"), offset = 1, duration = 4, unit = "time")
    expect_equal(waveform_and_sample_rate[[2]], sample_mp3@samp.rate)
    expect_tensor(waveform_and_sample_rate[[1]])
    expect_equal(dim(waveform_and_sample_rate[[1]]), c(2, 144001))
})
