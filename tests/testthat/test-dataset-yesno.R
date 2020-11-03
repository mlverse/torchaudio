yesno_dir <- system.file("", package = "torchaudio")

test_that("load_yesnos_item", {
  # load
  yesno_ds <- yesno_dataset(yesno_dir, download = FALSE)
  expect_length(yesno_ds, 4)

  one_item <- yesno_ds[1]
  expect_length(one_item, 3)
  expect_tensor(one_item[[1]])
  expect_length(one_item[[3]], 8)
})

