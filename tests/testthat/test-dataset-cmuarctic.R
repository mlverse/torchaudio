cmuarctic_dir <- system.file("", package = "torchaudio")

test_that("load_cmuarctics_item", {
  # load
  cmuarctic_ds <- cmuarctic_dataset(cmuarctic_dir)
  expect_length(cmuarctic_ds, 3)

  one_item <- cmuarctic_ds[1]
  expect_length(one_item, 4)
  expect_tensor(one_item[[1]])
})


