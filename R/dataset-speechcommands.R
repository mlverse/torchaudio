#' @keywords internal
load_speechcommands_item <- function(filepath, path, hash_divider = "_nohash_", class_to_index, normalization = NULL) {
  if(length(filepath) != 1) value_error("length(filepath) should be 1.")
  relpath = fs::path_rel(filepath, path)
  relpath_split = unlist(fs::path_split(relpath))
  label = relpath_split[1]
  filename = relpath_split[2]
  speaker = tools::file_path_sans_ext(filename)
  speaker_id_and_utterance_number = unlist(strsplit(speaker, hash_divider))

  speaker_id = speaker_id_and_utterance_number[1]
  utterance_number = as.integer(speaker_id_and_utterance_number[2])

  # Load audio
  waveform_and_sample_rate = torchaudio_load(filepath, normalization = normalization)
  waveform = waveform_and_sample_rate[[1]][1]$unsqueeze(1)
  sample_rate = waveform_and_sample_rate[[2]]
  return(list(waveform = waveform,
              sample_rate = sample_rate,
              label_index = class_to_index(label),
              label = label, speaker_id,
              utterance_number = utterance_number))
}

#' Speech Commands Dataset
#'
#' @param root  (str): Path to the directory where the dataset is found or downloaded.
#' @param url  (str, optional): The URL to download the dataset from,
#'            or the type of the dataset to dowload.
#'            Allowed type values are ``"speech_commands_v0.01"`` and ``"speech_commands_v0.02"``
#'            (default: ``"speech_commands_v0.02"``)
#' @param folder_in_archive  (str, optional): The top-level directory of the dataset.  (default: ``"SpeechCommands"``)
#' @param download  (bool, optional): Whether to download the dataset if it is not found at root path.  (default: ``FALSE``).
#' @param normalization (NULL, bool, int or function): Optional normalization.
#'  If boolean TRUE, then output is divided by 2^31. Assuming the input is signed 32-bit audio,
#'  this normalizes to [-1, 1]. If numeric, then output is divided by that number.
#'  If function, then the output is passed as a paramete to the given function,
#'  then the output is divided by the result. (Default: NULL)
#'
#' @return a torch::dataset()
#'
#' @export
speechcommand_dataset <- torch::dataset(
  "SpeechCommands",

  HASH_DIVIDER = "_nohash_",
  EXCEPT_FOLDER = "_background_noise_",
  .CHECKSUMS = list(
    "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz" =
      "3cd23799cb2bbdec517f1cc028f8d43c",
    "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz" =
      "6b74f3901214cb2c2934e98196829835"
  ),

  initialize = function(
    root,
    url = "speech_commands_v0.02",
    folder_in_archive = "SpeechCommands",
    download = FALSE,
    normalization = NULL
  ) {

    self$URL <- url
    self$FOLDER_IN_ARCHIVE <- folder_in_archive
    self$normalization <- normalization

    if(url %in% "speech_commands_v0.01") {
      classes <- c("_background_noise_", "bed", "bird", "cat", "dog", "down", "eight", "five", "four",
                        "go", "happy", "house", "left", "marvin", "nine", "no", "off",
                        "on", "one", "right", "seven", "sheila", "six", "stop", "three",
                        "tree", "two", "up", "wow", "yes", "zero")
    } else {
      classes <- c("_background_noise_", "backward", "bed", "bird", "cat", "dog",
                        "down", "eight", "five", "follow", "forward", "four", "go", "happy",
                        "house", "learn", "left", "marvin", "nine", "no", "off", "on",
                        "one", "right", "seven", "sheila", "six", "stop", "three", "tree",
                        "two", "up", "visual", "wow", "yes", "zero")
    }
    self$classes <- classes[!(classes %in% self$EXCEPT_FOLDER)]
    if(url %in% c(
      "speech_commands_v0.01",
      "speech_commands_v0.02"
    )) {
      base_url = "https://storage.googleapis.com/download.tensorflow.org/data"
      ext_archive = ".tar.gz"
      url = file.path(base_url, paste0(url, ext_archive))
    }

    basename = basename(url)
    archive = file.path(root, basename)

    basename = sub(ext_archive, "", basename, fixed = TRUE)
    folder_in_archive = file.path(folder_in_archive, basename)

    self$.path = file.path(root, folder_in_archive)

    if(download) {
      if(!fs::is_dir(self$.path)) {
        if(!fs::is_file(archive)) {
          checksum = self$.CHECKSUMS[[url]]
          download_url(url = url, destfile = archive, checksum = checksum)
        }
        extract_archive(archive, self$.path)
      }
    }
    walker = walk_files(self$.path, suffix = "wav$", prefix = TRUE)
    walker = walker[!grepl(paste(self$EXCEPT_FOLDER, collapse = "|"), walker)]
    self$.walker = walker
  },

  .getitem = function(n) {
    force(n)
    if(length(n) != 1 || n <= 0) value_error("n should be a single positive integer.")
    fileid = self$.walker[n]
    output <- load_speechcommands_item(
      fileid, self$.path, self$HASH_DIVIDER,
      class_to_index = self$class_to_index,
      normalization = self$normalization
    )
    return(output)
  },

  .length = function() {
    length(self$.walker)
  },

  class_to_index = function(class) {
    torch::torch_scalar_tensor(which(self$classes == class))
  },

  index_to_class = function(index) {
    self$classes[as.numeric(index$item())]
  }
)
