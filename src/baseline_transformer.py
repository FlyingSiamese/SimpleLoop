from src.looped_transformer import LoopedTransformer


class BaselineTransformer(LoopedTransformer):

    def __init__(self, d, hidden_size, num_layers, num_heads, ffn_hidden_size, dropout=0.0):
        super().__init__(
            d=d, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads,
            ffn_hidden_size=ffn_hidden_size,
            train_loops=1, loss_window=1, input_injection=True, dropout=dropout,
        )

    @classmethod
    def from_config(cls, cfg, **overrides):
        kwargs = dict(
            d=cfg.data.d,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.baseline_layers,
            num_heads=cfg.model.num_heads,
            ffn_hidden_size=cfg.model.ffn_hidden_size,
            dropout=cfg.model.dropout,
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def forward(self, x, y):
        return super().forward(x, y, num_loops=1)[0]      # [B]