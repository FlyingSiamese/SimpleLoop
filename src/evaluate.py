"""ID 评测：loop 外推、context length、最小二乘基线。"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.baseline_transformer import BaselineTransformer
from src.configs.config import load_config
from src.dataset import LinearRegressionDataset
from src.looped_transformer import LoopedTransformer
from src.metrics import regression_metrics
from src.utils import get_device, save_json, to_loop_predictions


def infer_num_layers(state_dict):
    """从 state_dict 数出 stack 有多少层。"""
    idx = {int(k.split(".")[2]) for k in state_dict if k.startswith("stack.layers.")}
    return len(idx)


def load_model(checkpoint, cfg, arch, device):
    """arch 为 auto / looped / baseline。"""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if arch == "auto":
        n_layers = infer_num_layers(ckpt["model"])
        arch = ("baseline" if n_layers == cfg.model.baseline_layers
                and cfg.model.baseline_layers != cfg.model.loop_layers else "looped")
    cls = BaselineTransformer if arch == "baseline" else LoopedTransformer
    model = cls.from_config(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, arch, ckpt


@torch.no_grad()
def predict_loops(model, dataset, num_loops, device, batch_size):
    """返回 [num_loops, N] 的预测。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    chunks = []
    for x, y in loader:
        preds = to_loop_predictions(model(x.to(device), y.to(device), num_loops=num_loops))
        chunks.append(preds.float().cpu())
    return torch.cat(chunks, dim=1)


def least_squares_predict(x, y, context_length):
    """x: [N, k_total+1, d]，y: [N, k_total+1]。返回 pred, target，均为 [N]。

    query 永远在最后一个位置（k_total），与 dataset 的切片约定一致。
    """
    k_total = x.shape[1] - 1
    xc, yc = x[:, :context_length], y[:, :context_length]
    xq, yq = x[:, k_total], y[:, k_total]
    w_hat = torch.linalg.lstsq(xc, yc.unsqueeze(-1)).solution      # [N, d, 1]
    pred = (xq.unsqueeze(1) @ w_hat).reshape(xq.shape[0])          # [N]
    assert pred.shape == yq.shape, f"{pred.shape} vs {yq.shape}"
    return pred, yq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", default="auto", choices=["auto", "looped", "baseline"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    run_dir = Path(args.checkpoint).resolve().parent.parent
    model, arch, ckpt = load_model(args.checkpoint, cfg, args.arch, device)

    test_path = str(Path(args.data_dir) / "test.pt")
    test = LinearRegressionDataset(test_path)
    k_total = test.k_total
    target = test.y[:, k_total]                                    # [N]，query 目标
    num_eval_loops = cfg.evaluation.max_eval_loops if arch == "looped" else 1

    print(f"checkpoint : {args.checkpoint}")
    print(f"架构       : {arch}  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"测试样本   : {len(test)}  loop 数: {num_eval_loops}")

    # ---- ID 测试 + loop 外推 ----
    preds = predict_loops(model, test, num_eval_loops, device, args.batch_size)
    mse_by_loop = [((preds[t] - target) ** 2).mean().item() for t in range(num_eval_loops)]
    final_metrics = regression_metrics(preds[-1], target)
    print(f"\n最终 loop MSE : {final_metrics['mse']:.6f}  "
          f"RMSE {final_metrics['rmse']:.6f}  MAE {final_metrics['mae']:.6f}  "
          f"R² {final_metrics['r2']:.6f}")
    if arch == "looped":
        print("每轮 loop MSE:")
        for t in range(0, num_eval_loops, 4):
            chunk = "  ".join(f"{t + i + 1}:{mse_by_loop[t + i]:.5f}"
                              for i in range(min(4, num_eval_loops - t)))
            print(f"  {chunk}")

    # ---- 最小二乘基线 ----
    ls_pred, _ = least_squares_predict(test.x, test.y, k_total)
    ls_metrics = regression_metrics(ls_pred, target)
    print(f"\n最小二乘 MSE  : {ls_metrics['mse']:.3e}  R² {ls_metrics['r2']:.6f}")

    # ---- context length ----
    context_length = {}
    print("\ncontext length:")
    for k in cfg.evaluation.context_lengths:
        ds_k = LinearRegressionDataset(test_path, context_length=k)
        p = predict_loops(model, ds_k, num_eval_loops, device, args.batch_size)
        model_mse = ((p[-1] - target) ** 2).mean().item()
        ls_k, _ = least_squares_predict(test.x, test.y, k)
        ls_mse = ((ls_k - target) ** 2).mean().item()
        context_length[str(k)] = {"model_mse": model_mse, "least_squares_mse": ls_mse}
        print(f"  k={k:<3} model {model_mse:.6f}   最小二乘 {ls_mse:.3e}")

    save_json({
        "run_name": run_dir.name,
        "architecture": arch,
        "checkpoint": str(args.checkpoint),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_eval_loops": num_eval_loops,
        "mse_by_loop": mse_by_loop,
        "metrics_at_final_loop": final_metrics,
        "least_squares": ls_metrics,
        "context_length": context_length,
    }, run_dir / "evaluation.json")
    print(f"\n已写入 {run_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
