import pandas as pd
import numpy as np
import torch

import pyro 
import pyro.distributions as dist
from torch.distributions import constraints
from pyro.infer import SVI, Trace_ELBO, JitTrace_ELBO, TraceGraph_ELBO, NUTS, MCMC
from pyro.infer.autoguide import AutoMultivariateNormal

import seaborn as sns
import matplotlib.pyplot as plt

import sys
from tqdm import trange



def model(input_matrix, model_matrix, UMI, group_matrix = None, full_cov = False,SVI = False,
amortize = False,
prior_loc = 10, batch_size = 5120, 
theta_bounds = (0.1, 10000),
init_loc = 0.1, init_theta = 1):
  
  n_cells = input_matrix.shape[1]
  n_genes = input_matrix.shape[0]
  n_features = model_matrix.shape[1]
  
  
  with pyro.plate("genes", n_genes, dim = -1):
    theta = pyro.sample("theta", dist.Uniform(theta_bounds[0],theta_bounds[1])).unsqueeze(0)
    beta_prior_mu = torch.zeros(n_features)
    if group_matrix is not None:
      n_groups = group_matrix.shape[1]
      ### This is one possible implementation as a hierarchical model ###
      if full_cov:
        zeta = pyro.sample("zeta", dist.MultivariateNormal(torch.zeros(n_genes, n_groups), scale_tril=torch.eye(n_groups, n_groups) * prior_loc, validate_args=False))
      else:
        zeta = pyro.sample("zeta", dist.Normal(torch.zeros(n_genes, n_groups), torch.ones(n_groups) * prior_loc).to_event(1))
        
    if full_cov:
      beta = pyro.sample("beta", dist.MultivariateNormal(beta_prior_mu, scale_tril=torch.eye(n_features, n_features) * prior_loc, validate_args=False))
    else:
      beta = pyro.sample("beta", dist.Normal(beta_prior_mu, torch.ones(n_features) * prior_loc).to_event(1))
    if SVI:
      with pyro.plate("data", n_cells, subsample_size= batch_size ,dim = -2) as ind:
        eta = torch.matmul(model_matrix[ind,:], beta.t())  + torch.log(UMI[ind]).unsqueeze(1)
        if group_matrix is not None:
          eta_zeta = torch.matmul(group_matrix[ind,:], zeta.t())
          eta = eta + eta_zeta
        pyro.sample("obs", dist.NegativeBinomial(logits = eta - torch.log(theta) ,
        total_count= torch.clamp(theta, 1e-9,1e9)), obs = input_matrix[:,ind].t() )
    else:
      with pyro.plate("data", n_cells ,dim = -2):
        eta = torch.matmul(model_matrix, beta.t())  + torch.log(UMI).unsqueeze(1)
        if group_matrix is not None:
          eta_zeta = torch.matmul(group_matrix, zeta.t())
          eta = eta + eta_zeta
        pyro.sample("obs", dist.NegativeBinomial(logits = eta - torch.log(theta) ,
        total_count= torch.clamp(theta, 1e-9,1e9)), obs = input_matrix.t())

      
      
      
def guide(input_matrix, model_matrix, UMI,  group_matrix = None, full_cov = False,SVI = False,
prior_loc = 10, batch_size = 5120, 
theta_bounds = (0.1, 10000),
init_loc = 0.1, init_theta = 1):
  
    n_cells = input_matrix.shape[1]
    n_genes = input_matrix.shape[0]
    n_features = model_matrix.shape[1]
    
    beta_mean = pyro.param("beta_mean", torch.zeros(n_genes, n_features), constraint=constraints.real)
    
    if group_matrix is not None:
      n_groups = group_matrix.shape[1]
      ### This is one possible implementation as a hierarchical model ###
      if full_cov:
        zeta_loc = pyro.param("zeta_loc", (torch.eye(n_groups, n_groups).repeat([n_genes,1,1]) * init_loc), constraint=constraints.lower_cholesky)
      else:
        zeta_loc = pyro.param("zeta_loc", torch.ones(n_genes, n_groups) * init_loc, constraint=constraints.positive) 
        
    if full_cov:
      beta_loc = pyro.param("beta_loc", (torch.eye(n_features, n_features).repeat([n_genes,1,1]) * init_loc), constraint=constraints.lower_cholesky)
    else:
      beta_loc = pyro.param("beta_loc", torch.ones(n_genes, n_features) * init_loc, constraint=constraints.positive) 
    
    theta_p = pyro.param("theta_p", torch.ones(n_genes)* init_theta, constraint=constraints.positive) 
    with pyro.plate("genes", n_genes, dim = -1):
      
      pyro.sample("theta", dist.Delta(theta_p)).unsqueeze(1)
      
      if group_matrix is not None:
        if full_cov:
          zeta = pyro.sample("zeta", dist.MultivariateNormal(torch.zeros(n_genes, n_groups), scale_tril=zeta_loc, validate_args=False))
        else:
          zeta = pyro.sample("zeta", dist.Normal(torch.zeros(n_genes, n_groups), zeta_loc).to_event(1))
      
      if full_cov:
        pyro.sample("beta", dist.MultivariateNormal(beta_mean,scale_tril = beta_loc, validate_args=False))
      else:
        pyro.sample("beta", dist.Normal(beta_mean, beta_loc).to_event(1))
        
        
def guide_mle(*args, **kargs):
  pass
        
        

def run_SVDE(input_matrix,model_matrix, ncounts, 
            group_matrix = None,
            steps = 100, lr = 0.1,gamma_lr = 0.1,
            cuda = False, jit_compile = False,
            full_cov = False, batch_size = 5120, 
            prior_loc = 0.1,
            theta_bounds = (1, 1000),
            init_loc = 0.1, init_theta = 128
):
  
  if cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
  else:
    torch.set_default_tensor_type(t=torch.FloatTensor)
  
  if jit_compile and not cuda:
    loss = JitTrace_ELBO
  else:
    loss = Trace_ELBO
    
  input_matrix, model_matrix, UMI = torch.tensor(input_matrix).float(), torch.tensor(model_matrix).float(), torch.tensor(ncounts).float()
  

  lrd = gamma_lr ** (1 / steps)
    
  
  svi = SVI(model,
            guide,
            pyro.optim.ClippedAdam({"lr": lr, "lrd" : lrd}),
            loss=loss())
            
  elbo_list = [] 
  
  pyro.clear_param_store()
  
  norm_ind = input_matrix.shape[0] * input_matrix.shape[1]
  
  t = trange(steps, desc='Bar desc', leave = True)
  
  for i in t:
    elbo = svi.step(input_matrix, model_matrix, UMI,group_matrix = group_matrix, full_cov = full_cov, SVI = True, batch_size = batch_size,
    prior_loc = prior_loc,
            theta_bounds = theta_bounds,
            init_loc = init_loc, init_theta = init_theta)
            
    elbo_list.append(elbo / batch_size)
    t.set_description('ELBO: {:.5f}  '.format(elbo / norm_ind))
    t.refresh()
    
    
  n_features = model_matrix.shape[1]
  coeff = pyro.param("beta_mean")
  overdispersion = pyro.param("theta_p")
  
  if full_cov and n_features > 1:
    loc = torch.bmm(pyro.param("beta_loc"),pyro.param("beta_loc").permute(0,2,1))
  else:
    loc = pyro.param("beta_loc")
    
  eta = torch.exp(torch.matmul(model_matrix, coeff.t()) + torch.unsqueeze(torch.log(UMI), 1) )
  lk = dist.NegativeBinomial(logits = eta - torch.log(overdispersion) ,
        total_count= torch.clamp(overdispersion, 1e-9,1e9)).log_prob(input_matrix.t()).sum(dim = 0)
  
  if cuda: 
    input_matrix = input_matrix.cpu().detach().numpy() 
    overdispersion = overdispersion.cpu().detach().numpy() 
    eta = eta.cpu().detach().numpy()
    coeff = coeff.cpu().detach().numpy()
    loc = loc.cpu().detach().numpy()
    lk = lk.cpu().detach().numpy()
  else:
    input_matrix = input_matrix.detach().numpy() 
    overdispersion = overdispersion.detach().numpy() 
    eta = eta.detach().numpy()
    coeff = coeff.detach().numpy()
    loc = loc.detach().numpy()
    lk = lk.detach().numpy()
  
  variance =  eta + eta**2 / overdispersion
  
  return {"loss" : elbo_list, "params" : {
  "theta" : overdispersion,
  "lk" : lk,
  "beta" : coeff,
  "eta" : eta,
  "variance" : loc
  }, "residuals" : input_matrix.transpose(1,0) - eta / np.sqrt(variance) }
  
  
def run_HMC(input_matrix,model_matrix,ncounts,
            group_matrix = None,
            num_samples = 1000, nchains = 4, warmup_steps = 200,
            cuda = False, jit_compile = True,
            full_cov = False,
            prior_loc = 0.1,
            theta_bounds = (1, 1000)):
              
              
  if cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
  else:
    torch.set_default_tensor_type(t=torch.FloatTensor)
  
  if jit_compile and not cuda:
    loss = JitTrace_ELBO
  else:
    loss = Trace_ELBO
    
  input_matrix, model_matrix, UMI = torch.tensor(input_matrix).float(), torch.tensor(model_matrix).float(), torch.tensor(ncounts).float()
  
  nuts_kernel = NUTS(model)

  mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
  mcmc.run(input_matrix, model_matrix, UMI, group_matrix = group_matrix, full_cov = full_cov, SVI = False)
  if cuda: 
    hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
  else:
    hmc_samples = {k: v.detach().numpy() for k, v in mcmc.get_samples().items()}

  
  return  {"params" :  hmc_samples }

