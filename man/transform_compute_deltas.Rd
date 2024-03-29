% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/transforms.R
\name{transform_compute_deltas}
\alias{transform_compute_deltas}
\title{Delta Coefficients}
\usage{
transform_compute_deltas(win_length = 5, mode = "replicate")
}
\arguments{
\item{win_length}{(int): The window length used for computing delta. (Default: \code{5})}

\item{mode}{(str): Mode parameter passed to padding. (Default: \code{'replicate'})}
}
\value{
Tensor: Tensor of deltas of dimension (..., freq, time).
}
\description{
Compute delta coefficients of a tensor, usually a spectrogram.
}
\details{
forward param:
specgram  (Tensor): Tensor of audio of dimension (..., freq, time).

See \link{functional_compute_deltas} for more details.
}
