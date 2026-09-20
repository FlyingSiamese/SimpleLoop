"""RoPE：旋转位置编码。"""

import torch


def build_rope_cache(seq_len, head_dim, base=10000.0, device=None):
    """预计算每个位置的 cos / sin。

    返回 cos, sin: [seq_len, head_dim]
    """
    half = head_dim // 2
    # 每个"配对"一个频率：theta_i = base^(-i/half)
    inv_freq = 1.0 / (base ** (torch.arange(half, device=device).float() / half))

    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)          # [seq_len, half]

    # 复制一份，与 rotate_half 的配对方式对齐
    emb = torch.cat([freqs, freqs], dim=-1)     # [seq_len, head_dim]
    return emb.cos(), emb.sin()


def rotate_half(x):
    """最后一维劈两半，交换位置并给前半取负。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    """对 q/k 施加旋转。q, k: [B, H, L, Dh]"""
    cos = cos.to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)   # [1, 1, L, Dh]
    sin = sin.to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot