speechcommand_dir <- system.file("SpeechCommands", package = "torchaudio")

test_that("load_speechcommands_item", {
  # load
  speechcommand_ds <- speechcommand_dataset(speechcommand_dir)
  expect_length(speechcommand_ds, 0)
  speechcommand_ds <- speechcommand_dataset(speechcommand_dir, folder_in_archive = "")
  expect_length(speechcommand_ds, 7)

  one_item <- speechcommand_ds[1]
  expect_length(one_item, 5)
  expect_tensor(one_item[[1]])
})


