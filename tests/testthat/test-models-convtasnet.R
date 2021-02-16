waveform <- torch::torch_randn(2, 1, 100)

test_that("model_convblock", {
  convblock <- model_convblock()
  expect_equal(1, 2)
})

test_that("model_mask_generator", {
  mask_generator <- model_mask_generator()
  expect_equal(1, 2)
})

test_that("model_convtasnet", {
  convtasnet <- model_convtasnet()
  expect_no_error(x <- convtasnet(waveform))
  expect_tensor(x)
  expect_true(x$requires_grad)
  expect_equal(dim(x), c(2, 2, 100))
})
