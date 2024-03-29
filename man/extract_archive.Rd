% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dataset-utils.R
\name{extract_archive}
\alias{extract_archive}
\title{Extract Archive}
\usage{
extract_archive(from_path, to_path = NULL, overwrite = FALSE)
}
\arguments{
\item{from_path}{(str): the path of the archive.}

\item{to_path}{(str, optional): the root path of the extraced files (directory of from_path) (Default: \code{NULL})}

\item{overwrite}{(bool, optional): overwrite existing files (Default: \code{FALSE})}
}
\value{
list: List of paths to extracted files even if not overwritten.
}
\description{
Extract Archive
}
\examples{
if(torch::torch_is_installed()) {
url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
d <- fs::dir_create(tempdir(), "torchaudio")
from_path <- fs::path(d, basename(url))
utils::download.file(url = url, destfile = from_path)
torchaudio::extract_archive (from_path, d)
}

}
