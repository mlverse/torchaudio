AudioMetaData <- R6::R6Class(
  "AudioMetaData",
  public = list(
    sample_rate = NULL,
    num_frames = NULL,
    num_channels = NULL,
    initialize = function(
      sample_rate,
      num_frames,
      num_channels
    ) {
      self$sample_rate = sample_rate
      self$num_frames = num_frames
      self$num_channels = num_channels
    }
  )
)
