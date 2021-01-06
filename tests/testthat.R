
library(testthat)
library(torchaudio)

if (Sys.getenv("TORCH_TEST", unset = 0) == 1)
  test_check("torchaudio")
