#' @keywords internal
load_cmuarctic_item <- function(
  line,
  path,
  folder_audio,
  ext_audio
) {

  utterance_id_and_utterance = strsplit(strip(line), " ", )[[1]]
  lu = length(utterance_id_and_utterance)
  utterance_id = utterance_id_and_utterance[2]
  utterance = paste(utterance_id_and_utterance[-c(1, 2, lu)], collapse = " ")
  utterance = gsub("\"", "", utterance) # remove double quote

  file_audio = file.path(path, folder_audio, paste0(utterance_id, ext_audio))

  # Load audio
  audio_r <- torchaudio_loader(file_audio)
  waveform_and_sample_rate <- transform_to_tensor(audio_r)

  return(
    list(
      waveform = waveform_and_sample_rate[[1]],
      sample_rate = waveform_and_sample_rate[[2]],
      utterance = utterance,
      utterance_id = strsplit(utterance_id, "_")[[1]][2]
    )
  )
}



#' CMU Arctic Dataset
#'
#' Create a Dataset for CMU_ARCTIC.
#'
#' @param root  (str): Path to the directory where the dataset is found or downloaded.
#' @param url  (str, optional): The URL to download the dataset from or the type of the dataset to dowload.
#'            (default: ``"aew"``)
#'            Allowed type values are ``"aew"``, ``"ahw"``, ``"aup"``, ``"awb"``, ``"axb"``, ``"bdl"``,
#'            ``"clb"``, ``"eey"``, ``"fem"``, ``"gka"``, ``"jmk"``, ``"ksp"``, ``"ljm"``, ``"lnh"``,
#'            ``"rms"``, ``"rxr"``, ``"slp"`` or ``"slt"``.
#' @param folder_in_archive  (str, optional): The top-level directory of the dataset.  (default: ``"ARCTIC"``)
#' @param download  (bool, optional): Whether to download the dataset if it is not found at root path.  (default: ``FALSE``).
#'
#' @return a torch::dataset()
#'
#' @export
cmuarctic_dataset <- torch::dataset(
  "CMUArctic",
  .file_text = "txt.done.data",
  .folder_text = "etc",
  .ext_audio = ".wav",
  .folder_audio = "wav",

  .CHECKSUMS = list(
    "http://festvox.org/cmu_arctic/packed/cmu_us_aew_arctic.tar.bz2"=
      "4382b116efcc8339c37e01253cb56295",
    "http://festvox.org/cmu_arctic/packed/cmu_us_ahw_arctic.tar.bz2"=
      "b072d6e961e3f36a2473042d097d6da9",
    "http://festvox.org/cmu_arctic/packed/cmu_us_aup_arctic.tar.bz2"=
      "5301c7aee8919d2abd632e2667adfa7f",
    "http://festvox.org/cmu_arctic/packed/cmu_us_awb_arctic.tar.bz2"=
      "280fdff1e9857119d9a2c57b50e12db7",
    "http://festvox.org/cmu_arctic/packed/cmu_us_axb_arctic.tar.bz2"=
      "5e21cb26c6529c533df1d02ccde5a186",
    "http://festvox.org/cmu_arctic/packed/cmu_us_bdl_arctic.tar.bz2"=
      "b2c3e558f656af2e0a65da0ac0c3377a",
    "http://festvox.org/cmu_arctic/packed/cmu_us_clb_arctic.tar.bz2"=
      "3957c503748e3ce17a3b73c1b9861fb0",
    "http://festvox.org/cmu_arctic/packed/cmu_us_eey_arctic.tar.bz2"=
      "59708e932d27664f9eda3e8e6859969b",
    "http://festvox.org/cmu_arctic/packed/cmu_us_fem_arctic.tar.bz2"=
      "dba4f992ff023347c07c304bf72f4c73",
    "http://festvox.org/cmu_arctic/packed/cmu_us_gka_arctic.tar.bz2"=
      "24a876ea7335c1b0ff21460e1241340f",
    "http://festvox.org/cmu_arctic/packed/cmu_us_jmk_arctic.tar.bz2"=
      "afb69d95f02350537e8a28df5ab6004b",
    "http://festvox.org/cmu_arctic/packed/cmu_us_ksp_arctic.tar.bz2"=
      "4ce5b3b91a0a54b6b685b1b05aa0b3be",
    "http://festvox.org/cmu_arctic/packed/cmu_us_ljm_arctic.tar.bz2"=
      "6f45a3b2c86a4ed0465b353be291f77d",
    "http://festvox.org/cmu_arctic/packed/cmu_us_lnh_arctic.tar.bz2"=
      "c6a15abad5c14d27f4ee856502f0232f",
    "http://festvox.org/cmu_arctic/packed/cmu_us_rms_arctic.tar.bz2"=
      "71072c983df1e590d9e9519e2a621f6e",
    "http://festvox.org/cmu_arctic/packed/cmu_us_rxr_arctic.tar.bz2"=
      "3771ff03a2f5b5c3b53aa0a68b9ad0d5",
    "http://festvox.org/cmu_arctic/packed/cmu_us_slp_arctic.tar.bz2"=
      "9cbf984a832ea01b5058ba9a96862850",
    "http://festvox.org/cmu_arctic/packed/cmu_us_slt_arctic.tar.bz2"=
      "959eecb2cbbc4ac304c6b92269380c81"
  ),

  initialize = function(
    root,
    url = "aew",
    folder_in_archive = "ARCTIC",
    download = FALSE
  ) {

    if(url %in% c(
      "aew",
      "ahw",
      "aup",
      "awb",
      "axb",
      "bdl",
      "clb",
      "eey",
      "fem",
      "gka",
      "jmk",
      "ksp",
      "ljm",
      "lnh",
      "rms",
      "rxr",
      "slp",
      "slt"
    )) {
      url = paste0("cmu_us_", url, "_arctic")
      ext_archive = ".tar.bz2"
      base_url = "http://festvox.org/cmu_arctic/packed"
      url = file.path(base_url, paste0(url, ext_archive))
    }

    basename = basename(url)
    root = file.path(root, folder_in_archive)
    if(!dir.exists(root)) {
      dir.create(root)
    }
    archive = file.path(root, basename)

    basename = sub(ext_archive, "", basename, fixed = TRUE)
    self$.path = file.path(root, basename)

    if(download) {
      if(!fs::is_dir(self$.path)) {
        if(!fs::is_file(archive)){
          checksum = self$.CHECKSUMS[[url]]
          download_url(url = url, destfile = archive, checksum = checksum)
        }
        extract_archive(archive, fs::path_dir(archive))
      }
    }

    self$.text = file.path(self$.path, self$.folder_text, self$.file_text)
    self$.walker = readLines(self$.text)
  },

  .getitem = function(n) {
    force(n)
    if(length(n) != 1 || n <= 0) value_error("n should be a single positive integer.")
    line = self$.walker[n]
    return(load_cmuarctic_item(line, self$.path, self$.folder_audio, self$.ext_audio))
  },

  .length = function() {
    length(self$.walker)
  }
)


