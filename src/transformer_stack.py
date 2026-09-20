import torch.nn as nn

from src.transformer_block import TransformerBlock

class TransformerStack(nn.Module):
    def __init__(self,hidden_size,num_layers,num_heads,ffn_hidden_size,dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_size,num_heads,ffn_hidden_size,dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self,x,cos,sin):
        for layer in self.layers:
            x=layer(x,cos,sin)
        return x