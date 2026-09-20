import torch.nn as nn

from src.attention import CausalSelfAttention
from src.rmsnorm import RMSNorm
from src.swiglu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self,hidden_size,num_heads,ffn_hidden_size,dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size,num_heads,dropout)
        self.norm2 = RMSNorm(hidden_size)
        self.ffn = SwiGLU(hidden_size,ffn_hidden_size)

    def forward(self,x,cos,sin):
        x=x+self.attn(self.norm1(x),cos,sin)
        x=x+self.ffn(self.norm2(x))
        return x