backend_tuneR_backend_load <- function(
  filepath,
  out = NULL,
  normalization = TRUE,
  channels_first = TRUE,
  num_frames = 0L,
  offset = 0L,
  signalinfo = NULL,
  encodinginfo = NULL,
  filetype = NULL
){

  if(!is.null(out)) not_implemented_error('Argument "out" not implemented yet. Please set it to NULL.')
  if(is.null(normalization)) value_error('Argument "normalization" is missing. Should it be set to `TRUE`?')
  if(!is.null(signalinfo)) value_warning('Argument "signalinfo" is meaningful for sox backend only and will be ignored.')
  if(!is.null(encodinginfo)) value_error('Argument "encodinginfo" is meaningful for sox backend only and will be ignored.')

  filepath = as.character(filepath)

  # check if valid file
  if(!fs::is_file(filepath))
    runtime_error(glue::glue("{filepath} not found or is a directory"))

  if(num_frames < -1)
    value_error("Expected value for num_samples -1 (entire file) or >=0")
  if(num_frames %in% c(-1, 0))
    num_frames = Inf
  if(offset < 0)
    value_error("Expected positive offset value")


  # load audio file
  file_ext <- tools::file_ext(filepath)
  if(file_ext == "mp3") {
    out <- tuneR::readMP3(filepath)
  } else if(file_ext == "wav") {
    out <- tuneR::readWave(filepath)
  } else {
    runtime_error(glue::glue("Only .mp3 and .wav formats are supported. Got {file_ext}."))
  }

  out <- tuneR::extractWave(out, from = offset+1, to = offset + num_frames, xunit = "samples", interact = FALSE)
  l_out <- length(out)

  out_tensor <- torch::torch_zeros(2, l_out)
  if(length(out@left) > 0) out_tensor[1] = out@left
  if(length(out@right) > 0) out_tensor[2] = out@right

  if(!channels_first)
    out_tensor = out_tensor$t()

  # normalize if needed
  internal__normalize_audio(out_tensor, normalization)

  sample_rate = out@samp.rate

  return(list(out_tensor, sample_rate))
}

backend_tuneR_backend_save <- function(){}
backend_tuneR_backend_info <- function(){}
