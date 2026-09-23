"""回归指标。"""

import torch


def regression_metrics(pred, target):
    """pred, target: [N]。返回 mse / rmse / mae / r2。"""
    pred = pred.double().reshape(-1)
    target = target.double().reshape(-1)
    err = pred - target
    mse = err.pow(2).mean()
    ss_res = err.pow(2).sum()
    ss_tot = (target - target.mean()).pow(2).sum()
    r2 = 1.0 - (ss_res / ss_tot).item() if ss_tot > 0 else 0.0
    return {
        "mse": mse.item(),
        "rmse": mse.sqrt().item(),
        "mae": err.abs().mean().item(),
        "r2": r2,
    }
