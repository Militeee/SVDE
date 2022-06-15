plot_loss <- function(model_obj, log = FALSE) {
  
  p1 <- ggplot2::ggplot() + ggplot2::geom_line(ggplot2::aes(y = model_obj$loss, x = 1:length(model_obj$loss)), size = 1.2, col = "blue" ) + ggplot2::xlab("steps") +ggplot2::ylab( "ELBO scaled") +  ggplot2::theme_classic() 
  if(log)
    p1 <- p1 + ggplot2::scale_y_log10()
  return(p1)
}

plot_dropout <- function(model_obj, log = FALSE) {
  
  p1 <- ggplot2::ggplot() + ggplot2::geom_jitter(ggplot2::aes(y = model_obj$params$delta, x = log(matrixStats::colMedians(model_obj$params$eta))), size = 0.5, alpha = 0.5) + ggplot2::xlab("median expression") +ggplot2::ylab( "dropout_rate") +  ggplot2::theme_classic() 
  if(log)
    p1 <- p1 + ggplot2::scale_y_log10()
  return(p1)
}


plot_overdispersion <- function(model_obj, log = FALSE) {
  
  p1 <- ggplot2::ggplot() + ggplot2::geom_jitter(ggplot2::aes(y = model_obj$params$theta, x = log(matrixStats::colMedians(model_obj$params$eta))), size = 0.5, alpha = 0.5) + ggplot2::xlab("Median expression") +ggplot2::ylab( "Overdispersion") +  ggplot2::theme_classic() 
  if(log)
    p1 <- p1 + ggplot2::scale_y_log10()
  return(p1)
}