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

#' F string
#'
#' Alias to glue::glue to resemble the Pythonic f-string.
#'
#' @seealso [glue::glue()]
#'
#' @keywords internal
#' @export
#' @examples
#' mean <- 3.5
#' f("the mean is {mean}")
f <- glue::glue

#' Not in operator
#'
#' the Negate(`\%in\%`) function.
#'
#' @keywords internal
#' @export
"%not_in%" <- Negate(`%in%`)

#' Pipe operator
#'
#' See \code{magrittr::\link[magrittr:pipe]{\%>\%}} for details.
#'
#' @name %>%
#' @rdname pipe
#' @keywords internal
#' @export
#' @importFrom magrittr %>%
#' @usage lhs \%>\% rhs
NULL

#' Multiple assignment operator
#'
#' See \code{zeallot::\link[zeallot:operator]{\%<-\%}} for details.
#'
#' @name %<-%
#' @rdname zeallot-multi-assignment
#' @keywords internal
#' @export
#' @importFrom zeallot %<-%
#' @usage c(x, y, z) \%<-\% list(a, b, c)
NULL


#' @keywords internal
n_dim <- function(x) length(dim(x))
