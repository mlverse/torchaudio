#' Strip
#'
#' removes any leading (spaces at the beginning) and trailing (spaces at the end) characters.
#' Analog to strip() string method from Python.
#'
#' @keywords internal
strip <- function(str) gsub("^\\s+|\\s+$", "", str)

#' If x is `NULL`, return y, otherwise return x
#'
#' @param x,y Two elements to test, one potentially `NULL`
#'
#' @noRd
#'
#' @keywords internal
#' @examples
#' NULL %||% 1
"%||%" <- function(x, y){
  if (is.null(x) || length(x) == 0) {
    y
  } else {
    x
  }
}
