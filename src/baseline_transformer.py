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

    def forward(self, x, y, num_loops=None, truncated_bptt=False, all_positions=False):
        # num_loops / truncated_bptt 仅为与 LoopedTransformer 接口统一而接受，
        # baseline 恒为 1 次前向
        return super().forward(x, y, num_loops=1,
                               all_positions=all_positions)[0]