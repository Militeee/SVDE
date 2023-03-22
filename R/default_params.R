SVI_default_args <- function() {
  list(steps = 350L, lr = 0.1,
                         gamma_lr = 0.1,
                         cuda = TRUE, jit_compile = FALSE,
                         full_cov = TRUE, batch_size = 10240L  , 
                         prior_loc = 0.1, 
                         theta_bounds = c(1e-6, 1e6),
                         init_loc = 0.1, init_theta = 1000)
}

HMC_default_args <- function() {
  
  list(
       num_samples = 1000L, nchains = 4L, warmup_steps = 200L,
       cuda = FALSE, jit_compile = FALSE,
       full_cov = FALSE,
       prior_loc = 0.1,
       theta_bounds = c(1, 100))
  
}
