## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup, message=FALSE, warning=FALSE--------------------------------------
library(torchaudio)
library(viridis)

## ---- fig.height=4, fig.width=6-----------------------------------------------
url = "https://pytorch.org/tutorials/_static/img/steam-train-whistle-daniel_simon-converted-from-mp3.wav"
filename = "steam-train-whistle-daniel_simon-converted-from-mp3.wav"
r = httr::GET(url, httr::write_disk(filename, overwrite = TRUE))

waveform_and_sample_rate = torchaudio_load(filename)
waveform = waveform_and_sample_rate[[1]]
sample_rate = waveform_and_sample_rate[[2]]

paste("Shape of waveform: ", paste(dim(waveform), collapse = " "))
paste("Sample rate of waveform: ", sample_rate)

plot(waveform[1], col = "royalblue", type = "l")
lines(waveform[2], col = "orange")

## ---- fig.height=3, fig.width=6-----------------------------------------------
specgram <- transform_spectrogram()(waveform)

paste("Shape of spectrogram: ", paste(dim(specgram), collapse = " "))

specgram_as_array <- as.array(specgram$log2()[1]$t())
image(specgram_as_array[,ncol(specgram_as_array):1], col = viridis(n = 257,  option = "magma"))

## ---- fig.height=3, fig.width=6-----------------------------------------------
specgram <- transform_mel_spectrogram()(waveform)

paste("Shape of spectrogram: ", paste(dim(specgram), collapse = " "))

specgram_as_array <- as.array(specgram$log2()[1]$t())
image(specgram_as_array[,ncol(specgram_as_array):1], col = viridis(n = 257,  option = "magma"))

## ---- fig.height=4, fig.width=6-----------------------------------------------
new_sample_rate <- sample_rate/10

# Since Resample applies to a single channel, we resample first channel here
channel <- 1
transformed <- transform_resample(sample_rate, new_sample_rate)(waveform[channel, ]$view(c(1,-1)))

paste("Shape of transformed waveform: ", paste(dim(transformed), collapse = " "))

plot(transformed[1], col = "royalblue", type = "l")

## -----------------------------------------------------------------------------
# Let's check if the tensor is in the interval [-1,1]
cat(sprintf("Min of waveform: %f \nMax of waveform: %f \nMean of waveform: %f", as.numeric(waveform$min()), as.numeric(waveform$max()), as.numeric(waveform$mean())))

## -----------------------------------------------------------------------------
normalize <- function(tensor) {
 # Subtract the mean, and scale to the interval [-1,1]
 tensor_minusmean <- tensor - tensor.mean()
 return(tensor_minusmean/tensor_minusmean$abs()$max())
}

# Let's normalize to the full interval [-1,1]
# waveform = normalize(waveform)

## ---- fig.height=4, fig.width=6-----------------------------------------------
transformed <- transform_mu_law_encoding()(waveform)

paste("Shape of transformed waveform: ", paste(dim(transformed), collapse = " "))

plot(transformed[1], col = "royalblue", type = "l")

## ---- fig.height=4, fig.width=6-----------------------------------------------
reconstructed <- transform_mu_law_decoding()(transformed)

paste("Shape of recovered waveform: ", paste(dim(reconstructed), collapse = " "))

plot(reconstructed[1], col = "royalblue", type = "l")

## -----------------------------------------------------------------------------
# Compute median relative difference
err <- as.numeric(((waveform - reconstructed)$abs() / waveform$abs())$median())

paste("Median relative difference between original and MuLaw reconstucted signals:", scales::percent(err, accuracy = 0.01))

## ---- fig.height=4, fig.width=6-----------------------------------------------
mu_law_encoding_waveform <- functional_mu_law_encoding(waveform, quantization_channels = 256)

paste("Shape of transformed waveform: ", paste(dim(mu_law_encoding_waveform), collapse = " "))

plot(mu_law_encoding_waveform[1], col = "royalblue", type = "l")

## ---- fig.height=3, fig.width=6-----------------------------------------------
computed <- functional_compute_deltas(specgram$contiguous(), win_length=3)

paste("Shape of computed deltas: ", paste(dim(computed), collapse = " "))

computed_as_array <- as.array(computed[1]$t())
image(computed_as_array[,ncol(computed_as_array):1], col = viridis(n = 257,  option = "magma"))

## -----------------------------------------------------------------------------
gain_waveform <- as.numeric(functional_gain(waveform, gain_db=5.0))
cat(sprintf("Min of gain_waveform: %f\nMax of gain_waveform: %f\nMean of gain_waveform: %f", min(gain_waveform), max(gain_waveform), mean(gain_waveform)))

dither_waveform <- as.numeric(functional_dither(waveform))
cat(sprintf("Min of dither_waveform: %f\nMax of dither_waveform: %f\nMean of dither_waveform: %f", min(dither_waveform), max(dither_waveform), mean(dither_waveform)))

## ----lowpass, cache=TRUE------------------------------------------------------
lowpass_waveform <- as.array(functional_lowpass_biquad(waveform, sample_rate, cutoff_freq=3000))

cat(sprintf("Min of lowpass_waveform: %f\nMax of lowpass_waveform: %f\nMean of lowpass_waveform: %f", min(lowpass_waveform), max(lowpass_waveform), mean(lowpass_waveform)))

plot(lowpass_waveform[1,], col = "royalblue", type = "l")
lines(lowpass_waveform[2,], col = "orange")

## ----highpass, cache=TRUE-----------------------------------------------------
highpass_waveform <- as.array(functional_highpass_biquad(waveform, sample_rate, cutoff_freq=3000))

cat(sprintf("Min of highpass_waveform: %f\nMax of highpass_waveform: %f\nMean of highpass_waveform: %f", min(highpass_waveform), max(highpass_waveform), mean(highpass_waveform)))

plot(highpass_waveform[1,], col = "royalblue", type = "l")
lines(highpass_waveform[2,], col = "orange")

## ---- fig.height=4, fig.width=6-----------------------------------------------
yesno_data <- yesno_dataset('./', download=TRUE)

# A data point in Yesno is a list (waveform, sample_rate, labels) where labels is a list of integers with 1 for yes and 0 for no.

# Pick data point number 3 to see an example of the the yesno_data:
n <- 3
sample <- yesno_data[n]
sample

plot(sample[[1]][1], col = "royalblue", type = "l")

