sample_mp3 <- tuneR::readMP3(system.file("sample_audio_1.mp3", package = "torchaudio"))
sample_mp3 <- tuneR::extractWave(sample_mp3, 20001, 30000)
sample_torch <- torch::torch_tensor(sample_mp3@left, dtype = torch::torch_float())
samples = length(sample_torch)

sample_torch2 <- torch::torch_stack(list(sample_torch, sample_torch))

test_that("transform_spectrogram", {
  expect_no_error(spec <- transform_spectrogram()(sample_torch))
  expect_tensor(spec)
  expect_equal(dim(spec), c(400 %/% 2 + 1, 51))

  expect_no_error(spec <- transform_spectrogram()(sample_torch2))
  expect_tensor(spec)
  expect_equal(dim(spec), c(2, 400 %/% 2 + 1, 51))
})

spec = transform_spectrogram()(sample_torch)
spec_complex = transform_spectrogram(power = NULL)(sample_torch)

test_that("transform_mel_scale and functional_inverse_mel_scale", {
  expect_no_error(m <- transform_mel_scale(n_mels = 10)(spec[..,1:10]))
  expect_tensor(m)
  expect_equal(dim(m), c(10, 10))

  expect_no_error(m <- transform_inverse_mel_scale(n_stft = 10, n_mels = 10)(m))
  expect_tensor(m)
  expect_equal(dim(m), c(10, 10))
})

test_that("transform_mel_spectrogram", {
  expect_no_error(m <- transform_mel_spectrogram(hop_length = 200)(sample_torch))
  expect_tensor(m)
  expect_equal(dim(m), c(128, 51))
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
})

test_that("transform_mfcc", {
  sample_torch = torch::torch_tensor(sin(1:1000))
  expect_no_error(m <- transform_mfcc()(sample_torch))
  expect_tensor(m)
  expect_equal(dim(m), c(40, 6))
})

test_that("transform_mu_law_encoding and transform_mu_law_decoding", {
  expect_no_error(m <- transform_mu_law_encoding()(sample_torch))
  expect_tensor(m)
  expect_no_error(m2 <- transform_mu_law_decoding()(m))
  expect_tensor(m2)
  expect_gt(cor(as.numeric(m2), as.numeric(sample_torch)), 0.995)
})

test_that("transform_resample", {
  expect_no_error(m <- transform_resample()(sample_torch))
  # expect_tensor(m)
})

test_that("transform_complex_norm", {
  expect_no_error(m <- transform_complex_norm()(spec_complex))
  expect_tensor(m)
})


test_that("transform_compute_deltas", {
  expect_no_error(m <- transform_compute_deltas()(spec))
  expect_tensor(m)
})

test_that("transform_time_stretch", {
  expect_error(m <- transform_time_stretch()(sample_torch), class = "value_error")
  expect_no_error(m <- transform_time_stretch(fixed_rate = 2.0)(spec_complex))
  # expect_tensor(m)
})

test_that("transform_fade", {
  expect_no_error(m <- transform_fade(1000L, 1000L)(sample_torch))
  expect_tensor(m)
})

test_that("transform__axismasking", {
  expect_no_error(m <- transform__axismasking(10, 2, FALSE)(spec))
  # expect_tensor(m)
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

