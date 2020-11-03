#' @keywords internal
load_yesno_item <- function(fileid, path, ext_audio) {
  if(length(fileid) != 1) value_error("length(fileid) should be 1.")

  # Read audio
  waveform_and_sample_rate = torchaudio::torchaudio_load(fileid)

  # Read label
  fileid <- basename(fileid)
  fileid <- tools::file_path_sans_ext(fileid)
  labels = as.integer(unlist(strsplit(fileid, "_", fixed = TRUE)))

  waveform_and_sample_rate[[3]] <- labels
  return(waveform_and_sample_rate)
}

#' YesNo Dataset
#'
#' Create a Dataset for YesNo
#'
#' @param root  (str): Path to the directory where the dataset is found or downloaded.
#' @param url  (str, optional): The URL to download the dataset from.
#'  (default: ``"[http://www.openslr.org/resources/1/waves_yesno.tar.gz]()"``)
#' @param folder_in_archive  (str, optional):
#' @param The top-level directory of the dataset.  (default: ``"waves_yesno"``)
#' @param download  (bool, optional):
#' @param Whether to download the dataset if it is not found at root path.  (default: ``FALSE``).
#' @param transform  (callable, optional): Optional transform applied on waveform. (default: ``NULL``)
#' @param target_transform  (callable, optional): Optional transform applied on utterance. (default: ``NULL``)
#'
#' @return tuple: ``(waveform, sample_rate, labels)``
#'
#' @export
yesno_dataset <- torch::dataset(
  "YESNO",

  .ext_audio = "wav",
  .CHECKSUMS = list(
    "http://www.openslr.org/resources/1/waves_yesno.tar.gz" = "962ff6e904d2df1126132ecec6978786"
  ),

  initialize = function(
    root,
    url = "http://www.openslr.org/resources/1/waves_yesno.tar.gz",
    folder_in_archive = "waves_yesno",
    download = FALSE,
    transform = NULL,
    target_transform = NULL
  ) {
    self$URL <- url
    self$FOLDER_IN_ARCHIVE <- "waves_yesno"

    if(!is.null(transform) | !is.null(target_transform)) {
      value_warning("In the next version, transforms will not be part of the dataset. Please remove the option `transform=TRUE` and `target_transform=TRUE` to suppress this warning.")
    }

    self$transform = transform
    self$target_transform = target_transform

    archive = basename(url)
    archive = file.path(root, archive)
    self$.path = file.path(root, folder_in_archive)

    if(download) {
      if(!fs::is_dir(self$.path)) {
        if(!fs::is_file(archive)){
          checksum = self$.CHECKSUMS[[url]]
          download_url(url = url, destfile = archive, checksum = checksum)
        }
        extract_archive(archive, fs::path_dir(archive))
      }
    }
    if(!fs::is_dir(self$.path))
      runtime_error("Dataset not found. Please use `download=TRUE` to download it.")

    self$.walker = walk_files(self$.path, suffix = paste0(self$.ext_audio, '$'), prefix = TRUE)
  },

  .getitem = function(n) {
    force(n)
    if(length(n) != 1 || n <= 0) value_error("n should be a single positive integer.")

    fileid = self$.walker[n]
    item = load_yesno_item(fileid, self$.path, self$.ext_audio)

    # TODO Upon deprecation, uncomment line below and remove following code
    # return(item)

    waveform = item[[1]]
    sample_rate = item[[2]]
    labels = item[[3]]
    if(!is.null(self$transform))
      waveform = self$transform(waveform)
    if(!is.null(self$target_transform))
      labels = self$target_transform(labels)

    return(list(waveform, sample_rate, labels))
  },

  .length = function() {
    length(self$.walker)
  }
)
