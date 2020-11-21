if(!requireNamespace("purrr", quietly = TRUE)) stop("purrr package required.")

library(purrr)
library(testthat)
library(torchaudio)

test_check("torchaudio")
