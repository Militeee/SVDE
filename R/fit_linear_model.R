fit_linear_model <- function(
  input_matrix,model_matrix,ncounts, 
  inference_method = "SVI",
  method_specific_args = list(steps = 100, lr = 0.1,
  cuda = False, jit_compile = True,
  full_cov = False, batch_size = 5120, 
  prior_loc = 0.1, prior_bias = c(0.,1.),
  dropout_beta_concentrations = c(1.0, 10.), theta_mean_variance = c(100, 300),
  init_loc = 0.1, init_theta = 128, init_delta = 0.1)
) {
  
  reticulate::source_python(system.file("python", "functions_DE.py", package = "SVDE"))
  
  if(inference_method == "SVI") {
    ret <- py$run_SVDE(method_specific_args)
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