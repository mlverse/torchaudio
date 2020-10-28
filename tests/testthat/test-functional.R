sample_mp3 <- tuneR::readMP3(system.file("sample_audio_1.mp3", package = "torchaudio"))
sample_rate = sample_mp3@samp.rate
samples = length(sample_mp3@left)

test_that("functional_spectrogram", {
  n_fft = 400
  expect_no_error(spec <- functional_spectrogram(sample_mp3@left, n_fft = n_fft))
  expect_tensor(spec)
  expect_equal(dim(spec)[1], n_fft %/% 2 + 1)
})

test_that("create_fb_matrix", {
  x <- functional_create_fb_matrix(
    n_freqs = 10,
    n_mels = 13,
    f_min = 0,
    f_max = 8000,
    sample_rate = 16000,
    norm = NULL
  )
  expect_tensor(x)
  expect_tensor_shape(x, c(10, 13))
})

test_that("complex_norm", {
  tensor_r <- c(1,2,3)
  tensor_t <- torch::torch_tensor(tensor_r)

  tensor_r_cn <- sum((tensor_r^2))^(0.5)
  tensor_t_cn <- functional_complex_norm(tensor_t)
  expect_lt(abs(as.numeric(tensor_t_cn) - tensor_r_cn), 1e-6)
})

test_that("functional_create_dct", {
  x <- functional_create_dct(
    n_mfcc = 10,
    n_mels = 64,
    norm = NULL
  )
  expect_tensor(x)
  expect_tensor_shape(x, c(64, 10))
  expect_no_error(functional_create_dct(2, 3, 'ortho'), class = "value_error")
  expect_error(functional_create_dct(2, 3, 'not ortho nor NULL norm'), class = "value_error")
})

test_that("functional_amplitude_to_db and functional_db_to_amplitude", {
  x1 <- torch::torch_arange(0,3, dtype = torch::torch_float())

  # amplitude_to_DB
  x2 <- functional_amplitude_to_db(x1)
  expect_tensor(x2)
  expect_tensor_shape(x2, c(3))

  # top_db
  x2 <- functional_amplitude_to_db(x1, top_db = 1.0)
  expect_tensor(x2)
  expect_tensor_shape(x2, c(3))

  # DB_to_amplitude
  expect_lt( as.numeric(sum(functional_db_to_amplitude(functional_amplitude_to_db(x1)) - x1)), 1e-8)
})


test_that("functional_mu_law_encoding and functional_mu_law_decoding", {
  # functional_mu_law_encoding
  # functional_mu_law_decoding
  stop("TO DO")
})

test_that("functional_angle", {
  # functional_angle
  stop("TO DO")
})

test_that("functional_magphase", {
  # functional_magphase
  stop("TO DO")
})

context("filters")
a_coeffs = torch::torch_tensor(c(1.0, 2.1, 3.3))
b_coeffs = torch::torch_tensor(c(3.1,3.1,10.0))
samp = torch::torch_tensor(c(0.5,0.5,0.5,0.5,0.5))

test_that("functional_lfilter and functional_biquad", {

  # functional_lfilter
  filtered_samp <- functional_lfilter(waveform = samp, a_coeffs = a_coeffs, b_coeffs = b_coeffs)
  expect_tensor(filtered_samp)
  expect_tensor_shape(filtered_samp, samp$shape)

  # functional_lfilter 2D
  samp = torch::torch_tensor(c(0.5,0.5,0.5,0.5,0.5))
  samp = torch::torch_stack(list(samp, samp, samp))
  filtered_samp <- functional_lfilter(waveform = samp, a_coeffs = a_coeffs, b_coeffs = b_coeffs)
  expect_tensor(filtered_samp)
  expect_tensor_shape(filtered_samp, samp$shape)

  # functional_biquad
  a_coeffs = as.numeric(a_coeffs)
  b_coeffs = as.numeric(b_coeffs)
  biquad_samp <- functional_biquad(
    waveform = samp,
    b_coeffs[1],
    b_coeffs[2],
    b_coeffs[3],
    a_coeffs[1],
    a_coeffs[2],
    a_coeffs[3]
  )
  expect_tensor(biquad_samp)
  expect_tensor_shape(biquad_samp, filtered_samp$shape)
  expect_equal(as.array(biquad_samp), as.array(filtered_samp))
})

test_that("allpass_biquad", {
  allpass_biquad <- functional_allpass_biquad(samp, sample_rate = 2, central_freq = 1)
  expect_tensor(allpass_biquad)
  expect_tensor_shape(allpass_biquad, samp$shape)
})

test_that("highpass_biquad", {
  highpass_biquad <- functional_highpass_biquad(samp, sample_rate = 2, cutoff_freq = 1)
  expect_tensor(highpass_biquad)
  expect_tensor_shape(highpass_biquad, samp$shape)
})

test_that("lowpass_biquad", {
  lowpass_biquad <- functional_lowpass_biquad(samp, sample_rate = 2, cutoff_freq = 1)
  expect_tensor(lowpass_biquad)
  expect_tensor_shape(lowpass_biquad, samp$shape)
})
