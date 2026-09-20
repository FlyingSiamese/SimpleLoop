import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self,hidden_size,ffn_hidden_size):
        super().__init__()
        self.w_gate = nn.Linear(hidden_size,ffn_hidden_size,bias = False)
        self.w_up = nn.Linear(hidden_size,ffn_hidden_size,bias = False)
        self.w_down = nn.Linear(ffn_hidden_size,hidden_size,bias = False)

    def forward(self,x):
        return self.w_down(F.silu(self.w_gate(x))*self.w_up(x))