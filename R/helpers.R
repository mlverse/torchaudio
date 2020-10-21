#' Linear to mel frequency
#'
#' Converts frequencies from the linear scale to mel scale.
#'
#' @param frequency_in_hertz (numeric) tensor of frequencies in hertz to be converted to mel scale.
#' @param mel_break_frequency_hertz (numeric) scalar. (Default to 2595.0)
#' @param mel_high_frequency_q (numeric) scalar. (Default to 700.0)
#'
#' @export
linear_to_mel_frequency <- function(frequency_in_hertz, mel_break_frequency_hertz = 2595.0, mel_high_frequency_q = 700.0) {
  mel_break_frequency_hertz * log10(1.0 + (frequency_in_hertz / mel_high_frequency_q))
}

#' Mel to linear frequency
#'
#' Converts frequencies from the mel scale to linear scale.
#'
#' @param frequency_in_mel (numeric) tensor of frequencies in mel to be converted to linear scale.
#' @param mel_break_frequency_hertz (numeric) scalar. (Default to 2595.0)
#' @param mel_high_frequency_q (numeric) scalar. (Default to 700.0)
#'
#' @export
mel_to_linear_frequency <- function(frequency_in_mel, mel_break_frequency_linear = 2595.0, mel_high_frequency_q = 700.0) {
  mel_high_frequency_q * (10^(frequency_in_mel/mel_break_frequency_linear) - 1.0)
}
