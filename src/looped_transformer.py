import torch
import torch.nn as nn

from src.rmsnorm import RMSNorm
from src.rope import build_rope_cache
from src.transformer_stack import TransformerStack

class LoopedTransformer(nn.Module):
    def __init__(self,d,hidden_size,num_layers,num_heads,ffn_hidden_size,train_loops=8,loss_window=4,input_injection=True,dropout=0.0):
        super().__init__()
        self.train_loops = train_loops
        self.loss_window = loss_window
        self.input_injection = input_injection
        self.head_dim = hidden_size//num_heads

        self.x_proj = nn.Linear(d,hidden_size)
        self.y_proj = nn.Linear(1,hidden_size)
        self.stack = TransformerStack(hidden_size,num_layers,num_heads,ffn_hidden_size,dropout)
        self.final_norm = RMSNorm(hidden_size)
        self.output_head = nn.Linear(hidden_size,1)

    @classmethod
    def from_config(cls,cfg,**overrides):
        kwargs = dict(
            d=cfg.data.d,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.loop_layers,
            num_heads=cfg.model.num_heads,
            ffn_hidden_size=cfg.model.ffn_hidden_size,
            train_loops=cfg.loop.train_loops,
            loss_window=cfg.loop.loss_window,
            input_injection=cfg.loop.input_injection,
            dropout=cfg.model.dropout,
        )

        kwargs.update(overrides)
        return cls(**kwargs)

    def embed_prompt(self,x,y):
        k=x.shape[1]-1
        x_ctx,x_q = x[:,:k],x[:,k]
        y_ctx = y[:,:k]

        ex = self.x_proj(x_ctx)
        ey = self.y_proj(y_ctx.unsqueeze(-1))
        eq = self.x_proj(x_q)

        B,k,D = ex.shape
        prompt = torch.stack([ex,ey],dim = 2).reshape(B,2*k,D)

        return torch.cat([prompt,eq.unsqueeze(1)],dim=1)

    def forward(self,x,y,num_loops=None):
        if num_loops is None:
            num_loops=self.train_loops

        P = self.embed_prompt(x,y)
        B,L,D=P.shape
        cos,sin=build_rope_cache(L,self.head_dim,device=P.device)
        if self.input_injection:
            H = torch.zeros_like(P)
        else:
            H = P

        predictions = []
        for _ in range(num_loops):
            if self.input_injection:
                H = self.stack(H+P,cos,sin)
            else:
                H = self.stack(H,cos,sin)
            h_q = self.final_norm(H[:,-1])
            predictions.append(self.output_head(h_q).squeeze(-1))

        return torch.stack(predictions)
