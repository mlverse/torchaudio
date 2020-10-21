test_that("linear_to_mel_frequency and linear_to_mel_frequency", {
  x <- torch_tensor(matrix(1:6, 2))
  x_mel <- linear_to_mel_frequency(x)
  x_linear <- mel_to_linear_frequency(x_mel)

  expect_equal(x, x_linear)
  expect_tensor(x_mel)
  expect_tensor(x_linear)
})
