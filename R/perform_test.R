#' Title
#'
#' @param fit_object 
#' @param input_matrix 
#' @param model_matrix 
#' @param group1 
#' @param group2 
#' @param pval 
#' @param LFC 
#' @param pct 
#' @param filter 
#'
#' @return
#' @export
#'
#' @examples
perform_test <- function(fit_object,input_matrix,model_matrix,  group1, group2, pval = 0.05, LFC = 0.5, pct = 0.3, filter = TRUE) {
  
  
  idx_1 <- grepl(group1, colnames(model_matrix))
  idx_2 <- grepl(group2, colnames(model_matrix))
  
  group1_ids <- which(model_matrix[,idx_1] == 1)
  group2_ids <- which(model_matrix[,idx_2] == 1)
  
  
  pct1 <- rowSums(input_matrix[,group1_ids] > 0) / ncol(input_matrix[,group1_ids])
  pct2 <- rowSums(input_matrix[,group2_ids] > 0) / ncol(input_matrix[,group2_ids])
  
  
  mu_1 <- fit_object$params$beta[,idx_1, drop = T]
  if(is.na(mu_1[1])) mu_1 <- 0
  mu_2 <- fit_object$params$beta[,idx_2, drop = T]
  if(is.na(mu_2[1])) mu_2 <- 0 
  
  log_FC <- mu_1 - mu_2
  variance <- fit_object$params$variance
  
  
  variance1 <- variance[,which(idx_1), which(idx_1) ]
  if(is.na(variance1[1])) variance1 <- 0
  variance2 <- variance[,which(idx_2), which(idx_2) ]
  if(is.na(variance2[1])) variance2 <- 0
  covariance12 <- variance[,which(idx_2), which(idx_1) ]
  if(is.na(covariance12[1])) covariance12 <- 0
  
  
  p_vals <- (pnorm(abs(log_FC) * -1 , mean = 0, sd = sqrt(variance2 + variance1 +  2*covariance12)) ) * 2
  
  p_vals_adj <- p.adjust(p = p_vals, method = "fdr")
  
  ret <- data.frame(gene = rownames(fit_object$params$beta), log_FC = log_FC, p_value_adjusted = p_vals_adj, p_value = p_vals, pct1 = pct1, pct2 = pct2) %>% arrange(p_vals_adj)
  
  if(filter) ret <- ret %>% filter(p_value_adjusted < pval, abs(log_FC) > LFC, (pct1 > pct) | (pct2 > pct))
  return(ret)
  
  
}