"""OOD 评测：输入尺度、斜协方差、带噪标签。"""

import argparse
from pathlib import Path

from torch.utils.data import TensorDataset

from src.configs.config import load_config
from src.data import make_generator, make_task_batch, skewed_cov_diag
from src.evaluate import least_squares_predict, load_model, predict_loops
from src.utils import get_device, save_json


def score(model, x, y, k_total, num_eval_loops, device, batch_size):
    """返回 (模型 MSE, 最小二乘 MSE)，均取 query 位置。"""
    preds = predict_loops(model, TensorDataset(x, y), num_eval_loops, device, batch_size)
    target = y[:, k_total]
    model_mse = ((preds[-1] - target) ** 2).mean().item()
    ls_pred, _ = least_squares_predict(x, y, k_total)
    ls_mse = ((ls_pred - target) ** 2).mean().item()
    return model_mse, ls_mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", default="auto", choices=["auto", "looped", "baseline"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-tasks", type=int, default=5000)
    parser.add_argument("--num-loops", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    run_dir = Path(args.checkpoint).resolve().parent.parent
    model, arch, _ = load_model(args.checkpoint, cfg, args.arch, device)

    d, k = cfg.data.d, cfg.data.k
    n = args.num_tasks
    num_eval_loops = (args.num_loops if args.num_loops is not None
                      else cfg.evaluation.max_eval_loops) if arch == "looped" else 1
    if num_eval_loops < 1:
        parser.error("--num-loops 必须大于 0")
    # 每个设置都用同一个种子：w 和 x 完全相同，只有 OOD 开关不同
    seed = cfg.seed + 100

    def run(**kw):
        x, y = make_task_batch(n, d, k, generator=make_generator(seed), **kw)
        return score(model, x, y, k, num_eval_loops, device, args.batch_size)

    print(f"checkpoint : {args.checkpoint}   架构: {arch}   样本数: {n}")

    # ---- 第 21 节：输入尺度 ----
    scaling = {}
    print("\n输入尺度 s:")
    for s in cfg.evaluation.scaling_factors:
        m, ls = run(x_scale=s)
        scaling[str(s)] = {"model_mse": m, "least_squares_mse": ls}
        print(f"  s={s:<4} model {m:.6f}   最小二乘 {ls:.6f}")

    # ---- 第 22 节：斜协方差 ----
    covariance = {}
    print("\n协方差:")
    for name, kw in [("identity", {}), ("skewed", {"cov_diag": skewed_cov_diag(d)})]:
        m, ls = run(**kw)
        covariance[name] = {"model_mse": m, "least_squares_mse": ls}
        print(f"  {name:<9} model {m:.6f}   最小二乘 {ls:.6f}")

    # ---- 第 23 节：context 标签噪声 ----
    noise = {}
    print("\ncontext 标签噪声 sigma:")
    for s in cfg.evaluation.noise_sigmas:
        m, ls = run(noise_sigma=s)
        noise[str(s)] = {"model_mse": m, "least_squares_mse": ls}
        print(f"  sigma={s:<4} model {m:.6f}   最小二乘 {ls:.6f}")

    output_path = run_dir / ("ood.json" if num_eval_loops == cfg.evaluation.max_eval_loops
                             or arch == "baseline" else f"ood_{num_eval_loops}.json")
    save_json({
        "run_name": run_dir.name,
        "architecture": arch,
        "num_eval_loops": num_eval_loops,
        "num_tasks": n,
        "scaling": scaling,
        "covariance": covariance,
        "noise": noise,
    }, output_path)
    print(f"\n已写入 {output_path}")


if __name__ == "__main__":
    main()
