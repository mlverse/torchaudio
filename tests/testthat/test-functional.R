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
samp_1d = torch::torch_tensor(c(0.5,-0.5,0.9,-0.9,0.5))
samp = torch::torch_stack(list(samp_1d, samp_1d, samp_1d))

test_that("functional_lfilter and functional_biquad", {

  # functional_lfilter
  filtered_samp <- functional_lfilter(waveform = samp_1d, a_coeffs = a_coeffs, b_coeffs = b_coeffs)
  expect_tensor(filtered_samp)
  expect_tensor_shape(filtered_samp, samp_1d$shape)

  # functional_lfilter 2D
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

test_that("bandpass_biquad", {
  bandpass_biquad <- functional_bandpass_biquad(samp, sample_rate = 2, central_freq = 1)
  expect_tensor(bandpass_biquad)
  expect_tensor_shape(bandpass_biquad, samp$shape)
})

test_that("bandreject_biquad", {
  bandreject_biquad <- functional_bandreject_biquad(samp, sample_rate = 2, central_freq = 1)
  expect_tensor(bandreject_biquad)
  expect_tensor_shape(bandreject_biquad, samp$shape)
})

test_that("equalizer_biquad", {
  equalizer_biquad <- functional_equalizer_biquad(samp, sample_rate = 2, center_freq = 1, gain = 1)
  expect_tensor(equalizer_biquad)
  expect_tensor_shape(equalizer_biquad, samp$shape)
})

test_that("band_biquad", {
  band_biquad <- functional_band_biquad(samp, sample_rate = 2, central_freq = 1)
  expect_tensor(band_biquad)
  expect_tensor_shape(band_biquad, samp$shape)
})

test_that("treble_biquad", {
  treble_biquad <- functional_treble_biquad(samp, sample_rate = 2, gain = 1)
  expect_tensor(treble_biquad)
  expect_tensor_shape(treble_biquad, samp$shape)
})

test_that("bass_biquad", {
  bass_biquad <- functional_bass_biquad(samp, sample_rate = 2, gain = 1)
  expect_tensor(bass_biquad)
  expect_tensor_shape(bass_biquad, samp$shape)
})

test_that("deemph_biquad", {
  deemph_biquad <- functional_deemph_biquad(samp, sample_rate = 44100)
  expect_tensor(deemph_biquad)
  expect_tensor_shape(deemph_biquad, samp$shape)
  expect_error(functional_deemph_biquad(samp, sample_rate = 1), class = "value_error")
})

test_that("riaa_biquad", {
  riaa_biquad <- functional_riaa_biquad(samp, sample_rate = 44100)
  expect_tensor(riaa_biquad)
  expect_tensor_shape(riaa_biquad, samp$shape)
  expect_error(functional_riaa_biquad(samp, sample_rate = 1), class = "value_error")
})

context("other effects")

test_that("contrast", {
  contrast <- functional_contrast(samp)
  expect_tensor(contrast)
  expect_tensor_shape(contrast, samp$shape)
  expect_error(functional_contrast(samp, enhancement_amount = 101), class = "value_error")
})

test_that("dcshift", {
  dcshift <- functional_dcshift(samp, 5)
  expect_tensor(dcshift)
  expect_tensor_shape(dcshift, samp$shape)
})

test_that("overdrive", {
  overdrive <- functional_overdrive(samp)
  expect_tensor(overdrive)
  expect_tensor_shape(overdrive, samp$shape)
})

test_that("generate_wave_table", {
  wave_table <- functional_generate_wave_table(
    wave_type = 'TRIANGLE',
    data_type = 'INT',
    table_size = 800,
    min = 1.0,
    max = 1.0,
    phase = pi / 2,
    device = torch::torch_device("cpu")
  )
  expect_tensor(wave_table)
  expect_tensor_shape(wave_table, 800)
})

test_that("phaser", {
  phaser <- functional_phaser(samp, sample_rate = 400)
  expect_tensor(phaser)
  expect_tensor_shape(phaser, samp$shape)
})

test_that("flanger", {
  flanger <- functional_flanger(samp, sample_rate = 400)
  expect_tensor(flanger)
  expect_tensor_shape(flanger, samp$shape)
})

test_that("mask_along_axis_iid", {
  mask_along_axis_iid <- functional_mask_along_axis_iid(
    specgrams = torch::torch_rand(3, 2, 4, 5),
    mask_param = 3L,
    mask_value = 99,
    axis = 3
  )
  expect_tensor(mask_along_axis_iid)
  expect_tensor_shape(mask_along_axis_iid, c(3, 2, 4, 5))
})

test_that("mask_along_axis", {
  mask_along_axis <- functional_mask_along_axis(
    specgram = torch::torch_rand(2, 4, 5),
    mask_param = 3,
    mask_value = 99,
    axis = 2
  )
  expect_tensor(mask_along_axis)
  expect_tensor_shape(mask_along_axis, c(2, 4, 5))
})

test_that("compute_deltas", {
  compute_deltas <- functional_compute_deltas(
    specgram = torch::torch_randn(1, 4, 10)
  )
  expect_tensor(compute_deltas)
  expect_tensor_shape(compute_deltas, c(1, 4, 10))
})

test_that("gain", {
  gain <- functional_gain(samp)
  expect_tensor(gain)
  expect_tensor_shape(gain, samp$size())
})

test_that("add_noise_shaping", {
  add_noise_shaping <- functional_add_noise_shaping(torch::torch_arange(0,5), torch::torch_arange(0,5))
  expect_tensor(add_noise_shaping)
  expect_tensor_shape(add_noise_shaping, 5)
})

test_that("apply_probability_distribution", {
  TPDF <- functional_apply_probability_distribution(samp, "TPDF")
  RPDF <- functional_apply_probability_distribution(samp, "RPDF")
  GPDF <- functional_apply_probability_distribution(samp, "GPDF")
  expect_tensor(TPDF)
  expect_tensor(RPDF)
  expect_tensor(GPDF)
  expect_tensor_shape(TPDF, samp$shape)
  expect_tensor_shape(RPDF, samp$shape)
  expect_tensor_shape(GPDF, samp$shape)
})

test_that("dither", {
  TPDF <- functional_dither(samp, "TPDF")
  RPDF <- functional_dither(samp, "RPDF")
  GPDF <- functional_dither(samp, "GPDF")
  expect_tensor(TPDF)
  expect_tensor(RPDF)
  expect_tensor(GPDF)
  expect_tensor_shape(TPDF, samp$shape)
  expect_tensor_shape(RPDF, samp$shape)
  expect_tensor_shape(GPDF, samp$shape)
})

test_that("compute_nccf", {
  expect_no_error(
    compute_nccf <- functional_compute_nccf(
      waveform  = samp,
      sample_rate = 10,
      frame_time = 0.01,
      freq_low = 5
    ),
    class = "value_error"
  )

  # expect_tensor(compute_nccf)
  # expect_tensor_shape(compute_nccf, samp$shape)

})
