waveform <- torch::torch_randn(2, 1, 100)

test_that("model_conv_block", {
  convblock <- model_conv_block(io_channels = 1, hidden_channels = 2, kernel_size = 2, padding = 0)
  expect_equal(1, 2)
})

test_that("model_mask_generator", {
  mask_generator <- model_mask_generator()
  expect_equal(1, 2)
})

test_that("model_conv_tasnet", {
  convtasnet <- model_conv_tasnet()
  expect_no_error(x <- convtasnet(waveform))
  expect_tensor(x)
  expect_true(x$requires_grad)
  expect_equal(dim(x), c(2, 2, 100))
})
