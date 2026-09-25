import math
import torch

def make_generator(seed):
    return torch.Generator().manual_seed(seed)

def make_task_batch(num_tasks,d,k,*,generator=None):
    w=torch.randn(num_tasks,d,generator=generator)/math.sqrt(d)
    z=torch.randn(num_tasks,k+1,d,generator=generator)
    x=z

    y=torch.einsum("td,tnd->tn",w,x)

    return x,y
