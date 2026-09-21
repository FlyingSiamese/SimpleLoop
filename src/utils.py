"""通用工具：随机种子、设备、环境信息。"""

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    """固定所有随机源（计划第 34 节）。"""
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
    """计划第 34 节要求记录的环境信息。"""
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None),
        "git_commit": git_commit(),
    }


def to_loop_predictions(out):
    """统一成 [num_loops, B]。

    LoopedTransformer 返回 [num_loops, B]；BaselineTransformer 返回 [B]
    （它没有 loop 维度），补一个长度 1 的维度后两者接口一致。
    """
    return out.unsqueeze(0) if out.dim() == 1 else out


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
