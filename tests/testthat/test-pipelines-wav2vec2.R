test_that("pipeline_wav2vec2_asr_base_960h", {
  bundle = pipeline_wav2vec2_asr_base_960h()

  model = bundle$get_model()

  SPEECH_FILE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
  c(waveform, sample_rate) %<-% transform_to_tensor(torchaudio_load(SPEECH_FILE))

  features = model$extract_features(waveform)[[1]]
})
