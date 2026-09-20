import torch
import torch.nn as nn
import torch.nn.functional as F
from src.rope import apply_rope

class CausalSelfAttention(nn.Module):
    def __init__(self,hidden_size,num_heads,dropout=0.0):
        super().__init__()
        assert hidden_size % num_heads ==0,"hidden_size must 被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_size,hidden_size,bias=False)
        self.k_proj = nn.Linear(hidden_size,hidden_size,bias=False)
        self.v_proj = nn.Linear(hidden_size,hidden_size,bias=False)
        self.o_proj = nn.Linear(hidden_size,hidden_size,bias=False)

    def forward(self,x,cos,sin):
        B,L,D = x.shape
        H,Dh = self.num_heads,self.head_dim

        q=self.q_proj(x).view(B,L,H,Dh).transpose(1,2)
        k=self.k_proj(x).view(B,L,H,Dh).transpose(1,2)
        v=self.v_proj(x).view(B,L,H,Dh).transpose(1,2)

        q,k = apply_rope(q,k,cos,sin)

        out = F.scaled_dot_product_attention(
            q,k,v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        out = out.transpose(1,2).reshape(B,L,D)
        return self.o_proj(out)