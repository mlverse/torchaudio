filepath_mp3 <- system.file("sample_audio_1.mp3", package = "torchaudio")
sample_mp3 <- tuneR::readMP3(filepath_mp3)
filepath_wav <- system.file("waves_yesno/1_1_0_1_1_0_1_1.wav", package = "torchaudio")
sample_wav <- tuneR::readWave(filepath_wav)

test_that("av_loader works", {

})

test_that("audiofile_loader works", {
  # WAV
  by_samples <- audiofile_loader(filepath_wav, offset = 1, duration = 40000, unit = "samples")
  expect_equal(length(by_samples$waveform[[1]]), 40000)
  expect_equal(class(by_samples), c("audiofile", "list"))

  # MP3
  expect_error(by_samples <- audiofile_loader(filepath_mp3), class = "runtime_error")
})

test_that("tuneR_loader works", {

  a <- torchaudio:::tuneR_read_mp3_or_wav(filepath_mp3, from = 1, to = 40001, unit = "samples")
  # MP3
  by_samples <- tuneR_loader(filepath_mp3, offset = 1, duration = 40000, unit = "samples")
  expect_equal(length(by_samples@left), 40000)
  expect_equal(class(by_samples)[1], c("Wave"))

  by_samples <- tuneR_loader(filepath_mp3, offset = 1, duration = 100, unit = "samples")
  expect_equal(length(by_samples@left), 100)
  expect_equal(class(by_samples)[1], c("Wave"))

  by_time <-tuneR_loader(filepath_mp3, offset = 1, duration = 1, unit = "time")
  expect_gt(length(by_time@left), 40000)
  expect_lt(length(by_time@left), 50000)
  expect_equal(class(by_time)[1], c("Wave"))

  # WAV
  by_samples <- tuneR_loader(filepath_wav, offset = 1, duration = 40000, unit = "samples")
  expect_equal(length(by_samples@left), 40000)
  expect_equal(class(by_samples)[1], c("Wave"))

  by_samples <- tuneR_loader(filepath_wav, offset = 1, duration = 100, unit = "samples")
  expect_equal(length(by_samples@left), 100)
  expect_equal(class(by_samples)[1], c("Wave"))

  by_time <-tuneR_loader(filepath_wav, offset = 1, duration = 1, unit = "seconds")
  expect_gte(length(by_time@left), 8000)
  expect_lte(length(by_time@left), 8000)
  expect_equal(class(by_time)[1], c("Wave"))
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

  # audiofile
  by_samples <- audiofile_loader(filepath_wav, offset = 1, duration = 48000, unit = "samples")
  waveform_and_sample_rate <- transform_to_tensor(by_samples)
  expect_equal(waveform_and_sample_rate[[2]], sample_mp3@samp.rate)
  expect_tensor(waveform_and_sample_rate[[1]])
})

test_that("set_audio_backend and torchaudio_loader works", {
  loader <- getOption("torchaudio.loader")
  expect_equal(class(loader), "function")

  set_audio_backend(tuneR_loader)
  audio <- torchaudio_loader(filepath_mp3, offset = 1, duration = 100, unit = "samples")
  expect_equal(class(audio)[1], "Wave")
  expect_equal(length(audio), 100)

  set_audio_backend(av_loader)
  audio <- torchaudio_loader(filepath_mp3, offset = 1, duration = 100, unit = "samples")
  expect_equal(class(audio)[1], "av")
  expect_equal(length(audio), 100)

  set_audio_backend(audiofile_loader)
  audio <- torchaudio_loader(filepath_wav, offset = 1, duration = 100, unit = "samples")
  expect_equal(class(audio)[1], "audiofile")
})

test_that("torchaudio_load works", {
  set_audio_backend(av_loader)
  waveform_and_sample_rate <- torchaudio_load(filepath_mp3, offset = 1, duration = 100, unit = "samples")
  expect_equal(class(waveform_and_sample_rate)[1], "list")
  expect_equal(waveform_and_sample_rate[[2]], sample_mp3@samp.rate)
  expect_tensor(waveform_and_sample_rate[[1]])

  set_audio_backend(tuneR_loader)
  waveform_and_sample_rate <- torchaudio_load(filepath_mp3, offset = 1, duration = 100, unit = "samples")
  expect_equal(class(waveform_and_sample_rate)[1], "list")
  expect_equal(waveform_and_sample_rate[[2]], sample_mp3@samp.rate)
  expect_tensor(waveform_and_sample_rate[[1]])

  set_audio_backend(av_loader)
  waveform_and_sample_rate <- torchaudio_load(filepath_wav, offset = 1, duration = 100, unit = "samples")
  expect_equal(class(waveform_and_sample_rate)[1], "list")
  expect_equal(waveform_and_sample_rate[[2]], sample_wav@samp.rate)
  expect_tensor(waveform_and_sample_rate[[1]])

  set_audio_backend(tuneR_loader)
  waveform_and_sample_rate <- torchaudio_load(filepath_wav, offset = 1, duration = 100, unit = "samples")
  expect_equal(class(waveform_and_sample_rate)[1], "list")
  expect_equal(waveform_and_sample_rate[[2]], sample_wav@samp.rate)
  expect_tensor(waveform_and_sample_rate[[1]])

  set_audio_backend(audiofile_loader)
  waveform_and_sample_rate <- torchaudio_load(filepath_wav, offset = 1, duration = 100, unit = "samples")
  expect_equal(class(waveform_and_sample_rate)[1], "list")
  expect_equal(waveform_and_sample_rate[[2]], sample_wav@samp.rate)
  expect_tensor(waveform_and_sample_rate[[1]])
})


filepath_mp3 <- system.file("sample_audio_1.mp3", package = "torchaudio")
sample_mp3 <- tuneR::readMP3(filepath_mp3)
filepath_wav <- system.file("SpeechCommands/speech_commands_v0.02/seven/0a2b400e_nohash_1.wav", package = "torchaudio")
sample_wav <- tuneR::readWave(filepath_wav)


test_that("loaders returns the same output", {

})

a1 <- data.frame(
  audiofile = torchaudio::audiofile_loader(filepath_wav)$waveform[[1]],
  tuner = torchaudio::tuneR_loader(filepath_wav)@left,
  av = as.vector(torchaudio::av_loader(filepath_wav))
)

GGally::ggpairs(a1)
purrr::map(a1, ~sum(is.na(.x)))

a2 <- data.frame(
  audiofile_wf = as.numeric(transform_to_tensor(torchaudio::audiofile_loader(filepath_wav))[[1]]),
  tuner_wf = as.numeric(transform_to_tensor(torchaudio::tuneR_loader(filepath_wav))[[1]]),
  av_wf = as.numeric(transform_to_tensor(torchaudio::av_loader(filepath_wav))[[1]])
)

GGally::ggpairs(a2)
purrr::map(a2, ~sum(is.na(.x)))



speechcommand_ds <- speechcommand_dataset(root = "/home/athos/R/x86_64-pc-linux-gnu-library/4.0/torchaudio/")

library(torchaudio)
aff <- function(loader) {
  set_audio_backend(loader)
  as.numeric(speechcommand_ds[1]$waveform)
}

a3 <- data.frame(
  audiofile_wf = aff(audiofile_loader),
  tuner_wf = aff(tuneR_loader),
  av_wf = aff(av_loader)
)

GGally::ggpairs(a3)
purrr::map(a3, ~sum(is.na(.x)))

