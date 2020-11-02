sample_mp3 <- tuneR::readMP3(system.file("sample_audio_1.mp3", package = "torchaudio"))
sample_mp3 <- tuneR::extractWave(sample_mp3, 20001, 30000)
sample_torch <- torch::torch_tensor(sample_mp3@left, dtype = torch::torch_float())
samples = length(sample_torch)

sample_torch2 <- torch::torch_stack(list(sample_torch, sample_torch))

test_that("transform_spectrogram", {
  expect_no_error(spec <- transform_spectrogram()(sample_torch))
  expect_tensor(spec)
  expect_equal(dim(spec), c(400 %/% 2 + 1, 49))

  expect_no_error(spec <- transform_spectrogram()(sample_torch2))
  expect_tensor(spec)
  expect_equal(dim(spec), c(2, 400 %/% 2 + 1, 49))
})

spec <- transform_spectrogram()(sample_torch)

test_that("transform_mel_scale", {
  ms = transform_mel_scale()(spec)
  expect_tensor(ms)
  expect_equal(dim(ms), c(1, 128, 49))
})

test_that("transform_amplitude_to_db", {
  x1 <- torch::torch_arange(0,3, dtype = torch::torch_float())

  # amplitude_to_db
  x2 <- transform_amplitude_to_db()(x1)
  expect_tensor(x2)
  expect_tensor_shape(x2, c(3))

  # top_db
  x2 <- transform_amplitude_to_db(top_db = 1.0)(x1)
  expect_tensor(x2)
  expect_tensor_shape(x2, c(3))

  # DB_to_amplitude
  expect_lt( as.numeric(sum(functional_db_to_amplitude(transform_amplitude_to_db()(x1)) - x1)), 1e-8)
})

test_that("transform_mfcc", {
  expect_no_error(m <- transform_mfcc()(sample_torch))
  expect_tensor(m)
})

test_that("transform_time_stretch", {
  expect_error(m <- transform_time_stretch()(sample_torch), class = "value_error")
  spec_complex = transform_spectrogram(power = NULL)(sample_torch)
  expect_no_error(m <- transform_time_stretch(fixed_rate = 2.0)(spec_complex), class = "value_error")
  expect_tensor(m)
})

test_that("transform_fade", {
  expect_no_error(m <- transform_fade(1000L, 1000L)(sample_torch))
  expect_tensor(m)
})

test_that("transform__axismasking", {
  expect_no_error(m <- transform__axismasking(10, 2, FALSE)(spec))
  expect_tensor(m)
  # transform_frequencymasking
  # transform_timemasking
})

test_that("transform_vol", {
  expect_error(transform_vol(-1, 'amplitude'), class = "value_error")
  expect_error(transform_vol(-1, 'power'), class = "value_error")

  vol = transform_vol(-1, "db")
  x1 <- torch::torch_arange(0,3, dtype = torch::torch_float())
  vol_x1 = vol(x1)
  expect_tensor(vol_x1)

})

test_that("transform_sliding_window_cmn", {
  sliding_window_cmn = transform_sliding_window_cmn()
  sliding_window_cmn = sliding_window_cmn(spec)
  expect_tensor(sliding_window_cmn)
  expect_equal(dim(sliding_window_cmn), dim(spec))
})

test_that("transform_vad", {
  vad = transform_vad(sample_mp3@samp.rate)
  expect_tensor(vad(sample_torch))
})

