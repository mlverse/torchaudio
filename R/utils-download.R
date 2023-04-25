#' Download and store torchaudio assets
#'
#' Download and store torchaudio assets to local file system. If a file exists
#' at the download path, then that path is returned with or without hash
#' validation.
#'
#' @param key  (str): The asset identifier.
#' @param path  (path-like object, optional):
#'  By default, the downloaded asset is saved in a directory under
#'  `system.file(package = "torchaudio")` and intermediate directories based on
#'  the given `key` are created. This argument can be used to overwrite the
#'  target location. When this argument is provided, all the intermediate
#'  directories have to be created beforehand.
#'  @param storage_url url to service where the asset is stored.
#'  Default: "https://download.pytorch.org/torchaudio"
#' @param hash  (str, optional):
#'  The value of SHA256 hash of the asset. If provided, it is used to verify
#'  the downloaded / cached object. If not provided, then no hash validation
#'  is performed. This means if a file exists at the download path, then the path
#'  is returned as-is without verifying the identity of the file.
#'  @param progress (bool, optional) – whether or not to display a progress bar.
#'  Default: TRUE
#'
#' @note
#' Currently the valid key values are the route on ``download.pytorch.org/torchaudio``,
#' but this is an implementation detail.
#'
#' @return str: The path to the asset on the local file system.
#'
#' @export
download_asset <- function(
    key,
    path = .get_local_path(key),
    storage_url = "https://download.pytorch.org/torchaudio",
    hash = NULL,
    progress = TRUE
) {

  if(file.exists(path)) {
    rlang::inform(f("The local file ({path}) exists. Skipping the download.\n"))
  } else {
    rlang::inform(f("Downloading {key} to {path}.\n"))
    .download(key, path, storage_url, progress)
  }

  if(!is.null(hash)) {
    rlang::inform("Verifying the hash value.\n")
    digest = .get_hash(path)

    if(digest != hash) {
      value_error(f(
        "The hash value of the downloaded file ({path}), '{digest}' does not
        match the provided hash value, '{hash}'.\n"
      ))
    }

    rlang::inform("Hash validated.\n")
  }

  return(path)
}

.get_local_path <- function(key, dir = system.file(package = "torchaudio")) {
  path <- file.path(dir, key)
  .create_dir_if_not_exists(dirname(path))
  return(path)
}

.download <- function(
    file,
    path,
    storage_url,
    progress = TRUE
  ) {
  url = f("{storage_url}/{file}")
  utils::download.file(url = url, destfile = path, quiet = !progress, mode = "wb")
}

.get_hash <- function(path) {
  data <- file(path, "rb")
  hash <- as.character(openssl::sha256(data))
  close(data)

  return(hash)
}

.create_dir_if_not_exists <- function(dir) {
  if (!dir.exists(dir))
    dir.create(dir, recursive = TRUE, showWarnings = FALSE)

  return(invisible(dir))
}

torchaudio_models_url <- function() {
  "https://storage.googleapis.com/torchaudio-models/v1"
}
