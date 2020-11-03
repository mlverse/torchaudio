#' @keywords internal
load_speechcommands_item <- function(filepath, path, hash_divider = "_nohash_") {
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
  waveform_and_sample_rate = torchaudio::torchaudio_load(filepath)
  waveform = waveform_and_sample_rate[[1]]
  sample_rate = waveform_and_sample_rate[[2]]
  return(list(waveform = waveform,
              sample_rate = sample_rate,
              label = label, speaker_id,
              utterance_number = utterance_number))
}

#' Create a Dataset for Speech Commands.
#'
#' @param root  (str): Path to the directory where the dataset is found or downloaded.
#' @param url  (str, optional): The URL to download the dataset from,
#'            or the type of the dataset to dowload.
#'            Allowed type values are ``"speech_commands_v0.01"`` and ``"speech_commands_v0.02"``
#'            (default: ``"speech_commands_v0.02"``)
#' @param folder_in_archive  (str, optional): The top-level directory of the dataset.  (default: ``"SpeechCommands"``)
#' @param download  (bool, optional): Whether to download the dataset if it is not found at root path.  (default: ``FALSE``).
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
    download = FALSE
  ) {

    self$URL <- url
    self$FOLDER_IN_ARCHIVE <- folder_in_archive

    if(url %in% c(
      "speech_commands_v0.01",
      "speech_commands_v0.02"
    )
    ) {
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
        extract_archive(archive, fs::path_dir(archive))
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
    return(load_speechcommands_item(fileid, self$.path, self$HASH_DIVIDER))
  },

  .length = function() {
    length(self$.walker)
  }
)
