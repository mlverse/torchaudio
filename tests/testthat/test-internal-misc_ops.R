test_that("internal__normalize_audio", {
  x = torch::torch_tensor(c(2, 3, 4)^31)
  internal__normalize_audio(x)
  expect_tensor(x)
  expect_equal(as.numeric(x), as.numeric(torch::torch_tensor(c(2, 3, 4)^31)/(2^31)))

  x = torch::torch_tensor(c(2, 3, 4)^31)
  internal__normalize_audio(x, 2)
  expect_tensor(x)
  expect_equal(as.numeric(x), as.numeric(torch::torch_tensor(c(2, 3, 4)^31)/2))

  x = torch::torch_tensor(c(2, 3, 4)^31)
  internal__normalize_audio(x, function(x) 1)
  expect_tensor(x)
  expect_equal(as.numeric(x), as.numeric(torch::torch_tensor(c(2, 3, 4)^31)/1))
})
