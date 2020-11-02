def_pytorch_to_r_function <- function(script) {
  library(magrittr)
  script <- script %>%
    stringr::str_replace_all("None", "NULL") %>%
    stringr::str_replace_all("True", "TRUE") %>%
    stringr::str_replace_all("true", "TRUE") %>%
    stringr::str_replace_all("False", "FALSE") %>%
    stringr::str_replace_all("false", "FALSE") %>%
    stringr::str_replace_all(stringr::fixed("self."), "self$")

  # signature
  signature <- stringr::str_extract(script, "[^)]+[)]")
  script <- stringr::str_remove(script, stringr::fixed(signature)) %>%
    stringr::str_remove("-> [^:]+:") %>%
    stringr::str_replace_all(stringr::fixed('"""'), "@@@@@@")
  documentation <- stringr::str_extract(script, '@@@@@[^@]+@@@@@')

  body <- stringr::str_remove(script, stringr::fixed(ifelse(is.na(documentation), "NAORETIRARNADA", documentation)))


  # signature prep
  signature_preped <- signature %>%
    stringr::str_replace_all("(:[^,=\n]+)(,|( =)|\n)", "\\2") %>%
    stringr::str_remove("^def ") %>%
    stringr::str_replace("[(]", " <- function(") %>%
    stringr::str_replace("[)]", ") {") %>%
    stringr::str_replace(stringr::fixed("__init__ <-"), "initialize =") %>%
    stringr::str_remove(stringr::fixed("self,")) %>%
    stringr::str_replace(stringr::fixed("function(torch.nn.Module) {"), "torch::nn_module(")


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
    stringr::str_remove("^:") %>%
    stringr::str_remove_all("r@@") %>%
    stringr::str_c("\n}\n")

  # documentation prep
  documentation_preped <- documentation %>%
    stringr::str_replace_all("\n|^", "\n#'") %>%
    stringr::str_replace_all('r@@@@@', " ") %>%
    stringr::str_replace_all('@@@@@', " ") %>%
    stringr::str_replace_all("#' *Returns:[^a-zA-Z]*", "#' @return ") %>%
    stringr::str_c("\n#' @export") %>%
    stringr::str_replace_all("(#'[:blank:]+)([^@(\n]+)[(]", "#' @param \\2 (") %>%
    stringr::str_remove("#' +Args:\n") %>%
    stringr::str_c("\n")

  cat(stringr::str_c(
    ifelse(is.na(documentation_preped), "", documentation_preped),
    ifelse(is.na(signature_preped), "", signature_preped),
    ifelse(is.na(body_preped), "", body_preped)
    )
  )
}

