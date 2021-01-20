
# torchaudio <a href='https://curso-r.github.io/torchaudio/'><img src='man/figures/torchaudio.png' align="right" height="139" /></a>

<!-- README.md is generated from README.Rmd. Please edit that file -->

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://www.tidyverse.org/lifecycle/#experimental)
[![R build
status](https://github.com/curso-r/torchaudio/workflows/R-CMD-check/badge.svg)](https://github.com/curso-r/torchaudio/actions)
[![CRAN
status](https://www.r-pkg.org/badges/version/torchaudio)](https://CRAN.R-project.org/package=torchaudio)
[![](https://cranlogs.r-pkg.org/badges/torchaudio)](https://cran.r-project.org/package=torchaudio)
<!-- badges: end -->

torchaudio is an extension for [torch](https://github.com/mlverse/torch)
providing audio loading, transformations, common architectures for
signal processing, pre-trained weights and access to commonly used
datasets. An almost literal translation from [PyTorch’s
Torchaudio](https://pytorch.org/audio/stable/index.html) library to R.

## Installation

The CRAN release can be installed with:

``` r
install.packages("torchaudio")
```

You can install the development version from GitHub with:

``` r
remotes::install_github("curso-r/torchaudio")
```

## A Waveform

`torchaudio` also supports loading sound files in the wav and mp3
format. We call waveform the resulting raw audio signal.

``` r
library(torchaudio)

url = "https://pytorch.org/tutorials/_static/img/steam-train-whistle-daniel_simon-converted-from-mp3.wav"
filename = tempfile(fileext = ".wav")
r = httr::GET(url, httr::write_disk(filename, overwrite = TRUE))

waveform_and_sample_rate = transform_to_tensor(tuneR_loader(filename))
waveform = waveform_and_sample_rate[[1]]
sample_rate = waveform_and_sample_rate[[2]]

paste("Shape of waveform: ", paste(dim(waveform), collapse = " "))
#> [1] "Shape of waveform:  2 276859"
paste("Sample rate of waveform: ", sample_rate)
#> [1] "Sample rate of waveform:  44100"

plot(waveform[1], col = "royalblue", type = "l")
lines(waveform[2], col = "orange")
```

<img src="man/figures/README-unnamed-chunk-4-1.png" width="100%" />

## A Spectrogram

``` r
specgram <- transform_spectrogram()(waveform)

paste("Shape of spectrogram: ", paste(dim(specgram), collapse = " "))
#> [1] "Shape of spectrogram:  2 201 1385"

specgram_as_array <- as.array(specgram$log2()[1]$t())
image(specgram_as_array[,ncol(specgram_as_array):1], col = viridis::viridis(n = 257,  option = "magma"))
```

<img src="man/figures/README-unnamed-chunk-5-1.png" width="100%" />

## Datasets ([go to issue](https://github.com/curso-r/torchaudio/issues/17))

  - [x] CMUARCTIC
  - [ ] COMMONVOICE
  - [ ] GTZAN
  - [ ] LIBRISPEECH
  - [ ] LIBRITTS
  - [ ] LJSPEECH
  - [x] SPEECHCOMMANDS
  - [ ] TEDLIUM
  - [ ] VCTK
  - [ ] VCTK\_092
  - [x] YESNO

## Models ([go to issue](https://github.com/curso-r/torchaudio/issues/19))

  - [ ] ConvTasNet
  - [ ] Wav2Letter
  - [x] WaveRNN
  - [ ] (what else? novel structures are very welcome\!)

## I/O Backend

  - [x] {tuneR}

## Code of Conduct

Please note that the torchaudio project is released with a [Contributor
Code of
Conduct](https://contributor-covenant.org/version/2/0/CODE_OF_CONDUCT.html).
By contributing to this project, you agree to abide by its terms.
