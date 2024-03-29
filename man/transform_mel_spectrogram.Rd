% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/transforms.R
\name{transform_mel_spectrogram}
\alias{transform_mel_spectrogram}
\title{Mel Spectrogram}
\usage{
transform_mel_spectrogram(
  sample_rate = 16000,
  n_fft = 400,
  win_length = NULL,
  hop_length = NULL,
  f_min = 0,
  f_max = NULL,
  pad = 0,
  n_mels = 128,
  window_fn = torch::torch_hann_window,
  power = 2,
  normalized = FALSE,
  ...
)
}
\arguments{
\item{sample_rate}{(int, optional): Sample rate of audio signal. (Default: \code{16000})}

\item{n_fft}{(int, optional): Size of FFT, creates \verb{n_fft // 2 + 1} bins. (Default: \code{400})}

\item{win_length}{(int or NULL, optional): Window size. (Default: \code{n_fft})}

\item{hop_length}{(int or NULL, optional): Length of hop between STFT windows. (Default: \verb{win_length // 2})}

\item{f_min}{(float, optional): Minimum frequency. (Default: \code{0.})}

\item{f_max}{(float or NULL, optional): Maximum frequency. (Default: \code{NULL})}

\item{pad}{(int, optional): Two sided padding of signal. (Default: \code{0})}

\item{n_mels}{(int, optional): Number of mel filterbanks. (Default: \code{128})}

\item{window_fn}{(function, optional): A function to create a window tensor
that is applied/multiplied to each frame/window. (Default: \code{torch_hann_window})}

\item{power}{(float, optional): Power of the norm. (Default: to \code{2.0})}

\item{normalized}{(logical): Whether to normalize by magnitude after stft (Default: \code{FALSE})}

\item{...}{(optional): Arguments for window function.}
}
\value{
\code{tensor}: Mel frequency spectrogram of size (..., \code{n_mels}, time).
}
\description{
Create MelSpectrogram for a raw audio signal. This is a composition of Spectrogram
and MelScale.
}
\details{
forward param:
waveform (Tensor): Tensor of audio of dimension (..., time).
}
\section{Sources}{

\itemize{
\item \url{https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe}
\item \url{https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html}
\item \url{https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html}
}
}

\examples{
#'   Example
\dontrun{

if(torch::torch_is_installed()) {
mp3_path <- system.file("sample_audio_1.mp3", package = "torchaudio")
sample_mp3 <- transform_to_tensor(tuneR_loader(mp3_path))
# (channel, n_mels, time)
mel_specgram <- transform_mel_spectrogram(sample_rate = sample_mp3[[2]])(sample_mp3[[1]])
}
}

}
