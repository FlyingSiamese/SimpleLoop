"""通用工具：随机种子、设备、环境信息。"""

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    """固定所有随机源，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def environment_info():
    """记录训练环境，便于复现。"""
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None),
        "git_commit": git_commit(),
    }


def to_loop_predictions(out, ndim=2):
    """补上 loop 维，让 baseline 和 looped 的返回形状一致。

    LoopedTransformer 返回 [num_loops, B]（或 [num_loops, B, k+1]）；
    BaselineTransformer 返回 [B]（或 [B, k+1]），缺一个 loop 维。
    """
    while out.dim() < ndim:
        out = out.unsqueeze(0)
    return out


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
