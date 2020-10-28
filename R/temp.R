def_pytorch_to_r_function <- function(script) {

  script <- script %>%
    stringr::str_replace_all("None", "NULL") %>%
    stringr::str_replace_all("True", "FALSE") %>%
    stringr::str_replace_all("False", "TRUE")

  # signature
  signature <- stringr::str_extract(script, "[^)]+[)]")
  script <- stringr::str_remove(script, stringr::fixed(signature)) %>%
    stringr::str_remove("-> [^:]+:")
  documentation <- stringr::str_extract(script, 'r\"\"\"[^"]+\"\"\"')
  body <- stringr::str_remove(script, stringr::fixed(documentation))


  # signature prep
  signature_preped <- signature %>%
    stringr::str_replace_all("(:[^,=\n]+)(,|( =)|\n)", "\\2") %>%
    stringr::str_remove("^def ") %>%
    stringr::str_replace("[(]", " <- function(") %>%
    stringr::str_replace("[)]", ") {")

  # body prep
  body_preped <- body %>%
    stringr::str_replace_all("torch[.]", "torch::torch_") %>%
    stringr::str_remove_all("math[.]") %>%
    stringr::str_replace_all("\n(.+)(\\+=)", "\n\\1 = \\1 +") %>%
    stringr::str_replace_all("\n(.+)(\\-=)", "\n\\1 = \\1 -") %>%
    stringr::str_replace_all("\n(.+)(\\/=)", "\n\\1 = \\1 /") %>%
    stringr::str_replace_all("\n(.+)(\\*=)", "\n\\1 = \\1 *") %>%
    stringr::str_replace("return (.*)", "return(\\1)") %>%
    stringr::str_replace_all("[:blank:]{2,}", " ") %>%
    stringr::str_replace_all("if ([^:]+):", "if(\\1) {") %>%
    stringr::str_replace_all("else ?:", "} else {") %>%
    stringr::str_replace_all("elif", "} else if") %>%
    stringr::str_replace_all("raise ValueError", "value_error") %>%
    stringr::str_c("\n}\n")

  # documentation prep
  documentation_preped <- documentation %>%
    stringr::str_replace_all("\n|^", "\n#'") %>%
    stringr::str_replace_all('r?\"\"\"', " ") %>%
    stringr::str_replace_all('r?\"\"\"', " ") %>%
    stringr::str_replace_all("#' *Returns:[^a-zA-Z]*", "#' @return ") %>%
    stringr::str_c("\n#' @export") %>%
    stringr::str_replace_all("(#'[:blank:]+)([^@(\n]+)[(]", "#' @param \\2 (") %>%
    stringr::str_remove("#' +Args:\n") %>%
    stringr::str_c("\n")

  cat(stringr::str_c(documentation_preped, signature_preped, body_preped))
}

