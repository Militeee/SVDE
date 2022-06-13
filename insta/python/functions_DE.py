import pandas as pd
import numpy as np
import torch

import pyro 
import pyro.distributions as dist
from torch.distributions import constraints
from tqdm import trange


import seaborn as sns
import matplotlib.pyplot as plt

import pyro
import torch
from pyro.infer import SVI, Trace_ELBO, JitTrace_ELBO, TraceGraph_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal
import sys


def model(input_matrix, model_matrix, UMI, full_cov = False,
prior_loc = 0.1, prior_bias = (0.,1.), batch_size = 5120, 
dropout_beta_concentrations = (1.0, 10.), theta_mean_variance = (100, 300),
init_loc = 0.1, init_theta = 128, init_delta = 0.1):
  
  n_cells = input_matrix.shape[1]
  n_genes = input_matrix.shape[0]
  n_features = model_matrix.shape[1]
  
  with pyro.plate("genes", n_genes):
    
    bias = pyro.sample("bias", dist.Normal(prior_bias[0], prior_bias[1]))
    theta = pyro.sample("theta", dist.HalfNormal(theta_mean_variance[0], theta_mean_variance[1])).unsqueeze(0)
    dropout = pyro.sample("delta", dist.Beta(dropout_beta_concentrations[0], dropout_beta_concentrations[1]))
    beta_prior_mu = torch.zeros(n_features)
    
    if full_cov:
      beta_prior_sigma = torch.eye(n_features, n_features) * prior_loc
      beta = pyro.sample("beta", dist.MultivariateNormal(beta_prior_mu, scale_tril=beta_prior_sigma))
    else:
      beta_prior_sigma = torch.ones(n_features)
      beta = pyro.sample("beta", dist.Normal(beta_prior_mu, beta_prior_sigma).to_event(1))
      
    
    
    
    with pyro.plate("data", n_cells, subsample_size=1024 * 5,dim = -2) as ind:
    
      eta = torch.exp(torch.matmul(model_matrix[ind,:], beta.t())  + bias.unsqueeze(0) + torch.log(UMI[ind]).unsqueeze(1) )
      print((eta).min(), flush = True)
      print((eta).max(), flush = True)
      pyro.sample("obs", dist.ZeroInflatedNegativeBinomial(logits = torch.logit(torch.clamp(eta / (eta + theta), 1e-12,0.999)),
      total_count= theta,
      gate_logits = torch.logit(dropout)), obs = input_matrix[:,ind].t() )
      
      
      
def guide(input_matrix, model_matrix, UMI, steps = 200,   full_cov = False,
prior_loc = 0.1, prior_bias = (0.,1.), batch_size = 5120, 
dropout_beta_concentrations = (1.0, 10.), theta_mean_variance = (100, 300),
init_loc = 0.1, init_theta = 128, init_delta = 0.1):
  
    n_cells = input_matrix.shape[1]
    n_genes = input_matrix.shape[0]
    n_features = model_matrix.shape[1]
    
    beta_mean = pyro.param("beta_mean", torch.zeros(n_genes, n_features), constraint=constraints.real)
    
    if full_cov:
      beta_loc = pyro.param("beta_loc", (torch.eye(n_features, n_features).repeat([n_genes,1,1]) * init_loc).tril(), constraint=constraints.corr_cholesky)
    else:
      beta_loc = pyro.param("beta_loc", torch.ones(n_genes, n_features), constraint=constraints.positive)
    
    theta_p = pyro.param("theta_p", torch.ones(n_genes), constraint=constraints.positive) * init_theta
    bias_p = pyro.param("bias_p", torch.ones(n_genes), constraint=constraints.real)
    gate_p = pyro.param("delta_p", torch.ones(n_genes) * init_delta, constraint=constraints.simplex)
    
    with pyro.plate("genes", n_genes):
      
      pyro.sample("bias", dist.Delta(bias_p))
      pyro.sample("theta", dist.Delta(theta_p)).unsqueeze(0)
      pyro.sample("delta", dist.Delta(gate_p))
      
      if full_cov:
        pyro.sample("beta", dist.MultivariateNormal(beta_mean,scale_tril = beta_loc, validate_args=False))
      else:
        pyro.sample("beta", dist.Normal(beta_mean, beta_loc).to_event(1))
        
        

def run_SVDE(input_matrix,model_matrix,ncounts, 
            steps = 100, lr = 0.1,
            cuda = False, jit_compile = True,
            full_cov = False, batch_size = 5120, 
            prior_loc = 0.1, prior_bias = (0.,1.),
            dropout_beta_concentrations = (1.0, 10.), theta_mean_variance = (100, 300),
            init_loc = 0.1, init_theta = 128, init_delta = 0.1
):
  
  input_matrix, model_matrix, UMI = torch.tensor(input_matrix.todense()).float(), torch.tensor(model_matrix).float(), torch.tensor(ncounts).float()

  
  if cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
  else:
    torch.set_default_tensor_type(t=torch.FloatTensor)
  
  if jit_compile and not cuda:
    loss = JitTrace_ELBO
  else:
    loss = Trace_ELBO
    
  
  svi = SVI(model,
            guide,
            pyro.optim.ClippedAdam({"lr": lr}),
            loss=loss())
            
  elbo_list = [] 
  
  pyro.clear_param_store()
  
  norm_ind = input_matrix.shape[0] * input_matrix.shape[1]
  
  t = trange(steps, desc='Bar desc', leave = True)
  
  for i in t:
    elbo = svi.step(input_matrix, model_matrix, UMI, full_cov = full_cov, batch_size = batch_size,
    prior_loc = prior_loc, prior_bias = prior_bias,
            dropout_beta_concentrations = dropout_beta_concentrations, theta_mean_variance = theta_mean_variance,
            init_loc = init_loc, init_theta = init_theta, init_delta = init_delta)
            
    elbo_list.append(elbo / norm_ind)
    t.set_description('ELBO: {:.5f}  '.format(elbo / norm_ind))
    t.refresh()
    
    
  coeff = pyro.param("beta_mean")
  loc = pyro.param("beta_loc")
  bias = pyro.param("bias_p")
  if cuda: 
    overdispersion = pyro.param("theta_p").cpu().detach().numpy()
    dropout = pyro.param("delta_p").cpu().detach().numpy()
    etas = torch.exp(torch.matmul(model_matrix, coeff.t())  + bias + torch.unsqueeze(torch.log(UMI), 1) ).cpu().detach().numpy()
    coeff = coeff.cpu().detach().numpy()
    loc = loc.cpu().detach().numpy()
    bias = bias.cpu().detach().numpy()
  else:
    overdispersion = pyro.param("theta_p").detach().numpy()
    dropout = pyro.param("delta_p").detach().numpy()
    etas = np.exp(np.matmul(model_matrix, coeff.transpose())  + bias + np.expand_dims(torch.log(UMI), axis = 1))
  
  return {"loss" : elbo_list, "params" : {
  "theta" : overdispersion,
  "delta" : dropout,
  "bias" : bias,
  "beta" : coeff,
  "eta" : etas,
  "variance" : loc
  }}
