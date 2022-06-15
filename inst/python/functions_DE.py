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


def model(input_matrix, model_matrix, UMI, full_cov = False,SVI = False,
prior_loc = 10, batch_size = 5120, 
theta_bounds = (0.1, 10000),
init_loc = 0.1, init_theta = 1):
  
  n_cells = input_matrix.shape[1]
  n_genes = input_matrix.shape[0]
  n_features = model_matrix.shape[1]
  
  with pyro.plate("genes", n_genes):
    theta = pyro.sample("theta", dist.Uniform(theta_bounds[0],theta_bounds[1])).unsqueeze(0)
    beta_prior_mu = torch.zeros(n_features)
    if full_cov:
      beta = pyro.sample("beta", dist.MultivariateNormal(beta_prior_mu, scale_tril=torch.eye(n_features, n_features) * prior_loc))
    else:
      beta = pyro.sample("beta", dist.Normal(beta_prior_mu, torch.ones(n_features) * prior_loc).to_event(1))
    if SVI:
      with pyro.plate("data", n_cells, subsample_size= batch_size ,dim = -2) as ind:
        eta = torch.matmul(model_matrix[ind,:], beta.t())  + torch.log(UMI[ind]).unsqueeze(1)
        pyro.sample("obs", dist.NegativeBinomial(logits = eta - torch.log(theta) ,
        total_count= torch.clamp(theta, 1e-9,1e9)), obs = input_matrix[:,ind].t() )
    else:
      with pyro.plate("data", n_cells ,dim = -2):
        eta = torch.matmul(model_matrix, beta.t())  + torch.log(UMI).unsqueeze(1)
        pyro.sample("obs", dist.NegativeBinomial(logits = eta - torch.log(theta) ,
        total_count= torch.clamp(theta, 1e-9,1e9)), obs = input_matrix.t())
      
      
      
def guide(input_matrix, model_matrix, UMI, full_cov = False,SVI = False,
prior_loc = 10, batch_size = 5120, 
theta_bounds = (0.1, 10000),
init_loc = 0.1, init_theta = 1):
  
    n_cells = input_matrix.shape[1]
    n_genes = input_matrix.shape[0]
    n_features = model_matrix.shape[1]
    
    beta_mean = pyro.param("beta_mean", torch.zeros(n_genes, n_features), constraint=constraints.real)
    
    if full_cov:
      beta_loc = pyro.param("beta_loc", (torch.eye(n_features, n_features).repeat([n_genes,1,1]) * init_loc), constraint=constraints.lower_cholesky)
    else:
      beta_loc = pyro.param("beta_loc", torch.ones(n_genes, n_features) * init_loc, constraint=constraints.positive) 
    
    theta_p = pyro.param("theta_p", torch.ones(n_genes)* init_theta, constraint=constraints.positive) 

    with pyro.plate("genes", n_genes):
      
      pyro.sample("theta", dist.Delta(theta_p)).unsqueeze(0)

      if full_cov:
        pyro.sample("beta", dist.MultivariateNormal(beta_mean,scale_tril = beta_loc, validate_args=False))
      else:
        pyro.sample("beta", dist.Normal(beta_mean, beta_loc).to_event(1))
        
        

def run_SVDE(input_matrix,model_matrix,ncounts, 
            steps = 100, lr = 0.1,gamma_lr = 0.1,
            cuda = False, jit_compile = True,
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
    elbo = svi.step(input_matrix, model_matrix, UMI, full_cov = full_cov, batch_size = batch_size,
    prior_loc = prior_loc,
            theta_bounds = theta_bounds,
            init_loc = init_loc, init_theta = init_theta)
            
    elbo_list.append(elbo / batch_size)
    t.set_description('ELBO: {:.5f}  '.format(elbo / norm_ind))
    t.refresh()
    
    
  coeff = pyro.param("beta_mean")
  overdispersion = pyro.param("theta_p")
  
  if full_cov:
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
  
  variance =  eta + eta**2 / overdispersion
  
  return {"loss" : elbo_list, "params" : {
  "theta" : overdispersion,
  "lk" : lk,
  "beta" : coeff,
  "eta" : eta,
  "variance" : loc
  }, "residuals" : input_matrix.transpose() - eta / np.sqrt(variance) }
  
  
def dump_tensors(gpu_only=True):
	"""Prints a list of the Tensors being tracked by the garbage collector."""
	import gc
	total_size = 0
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj):
				if not gpu_only or obj.is_cuda:
					print("%s:%s%s %s" % (type(obj).__name__, 
										  " GPU" if obj.is_cuda else "",
										  " pinned" if obj.is_pinned else "",
										  pretty_size(obj.size())))
					total_size += obj.numel()
			elif hasattr(obj, "data") and torch.is_tensor(obj.data):
				if not gpu_only or obj.is_cuda:
					print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
												   type(obj.data).__name__, 
												   " GPU" if obj.is_cuda else "",
												   " pinned" if obj.data.is_pinned else "",
												   " grad" if obj.requires_grad else "", 
												   " volatile" if obj.volatile else "",
												   pretty_size(obj.data.size())))
					total_size += obj.data.numel()
		except Exception as e:
			pass        
	print("Total size:", total_size)

