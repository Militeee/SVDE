
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
  method_specific_args = list(steps = 350L, lr = 0.1,
                              gamma_lr = 0.1,
  cuda = TRUE, jit_compile = FALSE,
  full_cov = TRUE, batch_size = 10240L  , 
  prior_loc = 0.1, 
  theta_bounds = c(1e-6, 1e6),
  init_loc = 0.1, init_theta = 1000)
) {
  
  
  reticulate::source_python(system.file("python/functions_DE.py", package = "SVDE"))
  
  if(inference_method == "SVI") {
    method_specific_args$input_matrix <- input_matrix
    method_specific_args$model_matrix <- model_matrix
    method_specific_args$ncounts <- ncounts
    
    ret <- do.call(py$run_SVDE, method_specific_args)
  } else {
    cli::cli_alert_danger("This inference method is not yet supported, have a look at the documentation for implemented algorithms or open an issue to request a new feature!")
    stop()
  }
  
  ret$run_params <- method_specific_args
  ret$input_params$model_matrix <- model_matrix
  ret$input_params$ncounts <- ncounts
  names(ret$params$theta) <- rownames(input_matrix)
  colnames(ret$params$eta) <- rownames(input_matrix)
  rownames(ret$params$eta) <- colnames(input_matrix)
  colnames(ret$residuals) <- rownames(input_matrix)
  rownames(ret$residuals) <- colnames(input_matrix)
  rownames(ret$params$beta) <- rownames(input_matrix)

  
  
  class(ret) <- "SVDE"
  
  return(ret)
  
}