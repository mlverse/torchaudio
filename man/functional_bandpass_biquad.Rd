% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/functional.R
\name{functional_bandpass_biquad}
\alias{functional_bandpass_biquad}
\title{Band-pass Biquad Filter (functional)}
\usage{
functional_bandpass_biquad(
  waveform,
  sample_rate,
  central_freq,
  Q = 0.707,
  const_skirt_gain = FALSE
)
}
\arguments{
\item{waveform}{(Tensor): audio waveform of dimension of \verb{(..., time)}}

\item{sample_rate}{(int): sampling rate of the waveform, e.g. 44100 (Hz)}

\item{central_freq}{(float): central frequency (in Hz)}

\item{Q}{(float, optional): \url{https://en.wikipedia.org/wiki/Q_factor} (Default: \code{0.707})}

\item{const_skirt_gain}{(bool, optional) : If \code{TRUE}, uses a constant skirt gain (peak gain = Q).
If \code{FALSE}, uses a constant 0dB peak gain.  (Default: \code{FALSE})}
}
\value{
Tensor: Waveform of dimension of \verb{(..., time)}
}
\description{
Design two-pole band-pass filter.  Similar to SoX implementation.
}
\references{
\itemize{
\item \url{https://sox.sourceforge.net/sox.html}
\item \url{https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html}
}
}
