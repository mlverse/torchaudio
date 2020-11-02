test_that("%||%", {
  expect_equal(NULL %||% 1, 1)
  expect_equal(list() %||% 1, 1)
  expect_equal(c() %||% 1, 1)
  expect_equal(numeric(0) %||% 1, 1)
  expect_equal(2 %||% 1, 2)
  expect_equal(list(3) %||% 1, list(3))
})
