
#' Fit a SVDE generalizef linear model
#'
#' @param input_matrix 
#' @param model_matrix 
#' @param ncounts 
#' @param inference_method 
#' @param method_specific_args 
#'
#' @return
#' @export
#'
#' @importFrom magrittr %>%
#' @examples

fit_linear_model <- function(
  input_matrix,model_matrix,ncounts, 
  inference_method = "SVI",
  method_specific_args = list()
) {
  
  
  reticulate::source_python(system.file("python/functions_DE.py", package = "SVDE"))
  
  if(inference_method == "SVI") {
    
    if(length(method_specific_args) == 0) method_specific_args <- SVI_default_args()
    
    method_specific_args$input_matrix <- input_matrix
    method_specific_args$model_matrix <- model_matrix
    method_specific_args$ncounts <- as.array(ncounts)

    ret <- do.call(py$run_SVDE, method_specific_args)
    
    names(ret$params$theta) <- rownames(input_matrix)
    colnames(ret$params$eta) <- rownames(input_matrix)
    rownames(ret$params$eta) <- colnames(input_matrix)
    colnames(ret$residuals) <- rownames(input_matrix)
    rownames(ret$residuals) <- colnames(input_matrix)
    rownames(ret$params$beta) <- rownames(input_matrix)

  } else if (inference_method == "HMC") {
    
    if(length(method_specific_args) == 0) method_specific_args <- HMC_default_args()
    

    
    method_specific_args$input_matrix <- input_matrix
    method_specific_args$model_matrix <- model_matrix
    method_specific_args$ncounts <- ncounts
    
    ret <- do.call(py$run_HMC, method_specific_args)
    
    dimnames(ret$params$beta) <- list(paste0("sample", 1:dim(ret$params$beta)[1]),rownames(input_matrix), colnames(model_matrix))
    dimnames(ret$params$theta) <- list(paste0("sample", 1:dim(ret$params$beta)[1]),rownames(input_matrix))
    
  } else if(inference_method == "MLE") {
    cli::cli_alert_danger("This inference method is not yet supported, have a look at the documentation for implemented algorithms or open an issue to request a new feature!")
  } else {
    cli::cli_alert_danger("This inference method is not yet supported, have a look at the documentation for implemented algorithms or open an issue to request a new feature!")
    stop()
  }
  
  ret$run_params <- method_specific_args
  ret$input_params$model_matrix <- model_matrix
  ret$input_params$ncounts <- ncounts


  
  
  class(ret) <- "SVDE"
  
  return(ret)
  
}