import math
import torch

def make_generator(seed):
    return torch.Generator().manual_seed(seed)

def skewed_cov_diag(d,pattern=(0.25,0.5,1.0,2.0,4.0)):
    reps = math.ceil(d/len(pattern))
    return torch.tensor(pattern*reps)[:d]

def make_task_batch(num_tasks,d,k,*,generator=None,x_scale=1.0,cov_diag=None,noise_sigma=0.0):
    w=torch.randn(num_tasks,d,generator=generator)/math.sqrt(d)
    z=torch.randn(num_tasks,k+1,d,generator=generator)
    if cov_diag is not None:
        z=z*cov_diag.sqrt()
    x=z*x_scale

    y=torch.einsum("td,tnd->tn",w,x)

    if noise_sigma > 0:
        eps = torch.randn(num_tasks,k,generator=generator)*noise_sigma
        y[:,:k]=y[:,:k]+eps 

    return x,y