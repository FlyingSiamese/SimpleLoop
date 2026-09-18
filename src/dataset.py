"""加载.pt数据集，并按照context length 分配"""

import torch 
from torch.utils.data import Dataset

class LinearRegressionDataset(Dataset):
    """
    线性回归dataset
    x:[k+1,d]
    y:[k+1]
    前k个context，最后一个query
    """

    def __init__(self,path,context_length=None):
        blob = torch.load(path,map_location="cpu")
        self.x=blob["x"]
        self.y=blob["y"]

        self.k_total = self.x.shape[1]-1
        if context_length is None:
            context_length = self.k_total

        if not 1 <= context_length <=self.k_total:
            raise ValueError(f"context_length 必须在 1..{self.k_total},收到{context_length}")
        self.context_length = context_length
        self._index = torch.cat([torch.arange(context_length),torch.tensor([self.k_total]),])

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self,i):
        if self.context_length ==self.k_total:
            return self.x[i],self.y[i]
        return self.x[i,self._index],self.y[i,self._index]

def split_context_query(x,y):
    return x[:,:-1],y[:,:-1],x[:, -1], y[:, -1]