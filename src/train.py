"""训练 Looped Transformer。"""

import argparse
import math
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.checkpoint import load_checkpoint, save_checkpoint
from src.configs.config import load_config, load_config_raw
from src.dataset import LinearRegressionDataset
from src.logger import RunLogger
from src.losses import loop_window_loss
from src.looped_transformer import LoopedTransformer
from src.utils import (environment_info, get_device, set_seed,
                       to_loop_predictions)

LOG_EVERY = 50


def build_model(cfg):
    return LoopedTransformer.from_config(cfg)


def make_optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        betas=tuple(cfg.training.betas),
        weight_decay=cfg.training.weight_decay,
    )


def make_scheduler(optimizer, cfg):
    """线性 warmup -> cosine 衰减（计划第 14 节）。"""
    warmup = cfg.training.warmup_steps
    total = cfg.training.max_steps

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parameter_stats(model, title):
    """计划第 28 节：按模块统计参数量。"""
    total = sum(p.numel() for p in model.parameters())
    print(f"[{title}] 总参数量: {total:,}")
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        if n:
            print(f"    {name:<12} {n:>12,}")
    return total


@torch.no_grad()
def evaluate(model, loader, device, num_loops):
    """返回每一轮 loop 的 MSE 列表，长度 num_loops。"""
    model.eval()
    sq = torch.zeros(num_loops, dtype=torch.float64)
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = to_loop_predictions(model(x, y, num_loops=num_loops))  # [L, B]
        target = y[:, -1].double()                                     # [B]
        sq += ((preds.double() - target.unsqueeze(0)) ** 2).sum(dim=1).cpu()
        n += x.shape[0]
    model.train()
    return (sq / n).tolist()


def train(cfg, model, run_name, config_path, resume=None):
    device = get_device()
    set_seed(cfg.seed)

    run_dir = Path("outputs") / run_name
    logger = RunLogger(run_dir)
    shutil.copy(config_path, run_dir / "config.yaml")

    print(f"运行目录 : {run_dir}")
    print(f"设备     : {device}")
    for key, val in environment_info().items():
        print(f"  {key}: {val}")
    n_params = parameter_stats(model, run_name)
    model.to(device)

    train_loader = DataLoader(LinearRegressionDataset("data/train.pt"),
                              batch_size=cfg.training.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(LinearRegressionDataset("data/val.pt"),
                            batch_size=cfg.training.batch_size,
                            shuffle=False, num_workers=0)

    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)

    step = 0
    best_val_loss = float("inf")
    if resume:
        ckpt = load_checkpoint(resume, model, optimizer, scheduler, restore_rng=True)
        step = ckpt["step"]
        best_val_loss = ckpt["best_val_loss"]
        print(f"从 {resume} 恢复，step={step}, best_val_loss={best_val_loss:.6f}")

    use_bf16 = (cfg.training.precision == "bf16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"精度     : {'bf16 autocast' if use_bf16 else 'fp32'}")

    # 短跑也要有曲线：logging 间隔随总步数自适应
    log_every = max(1, min(LOG_EVERY, cfg.training.max_steps // 20))

    # 模型自己的 loop 数与 loss window：looped=(8,4)，baseline=(1,1)
    num_loops = model.train_loops
    window = model.loss_window
    raw_config = load_config_raw(config_path)

    model.train()
    t0 = time.time()
    running_loss, running_n = 0.0, 0

    while step < cfg.training.max_steps:
        for x, y in train_loader:
            if step >= cfg.training.max_steps:
                break
            x, y = x.to(device), y.to(device)
            target = y[:, -1]

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=use_bf16):
                preds = to_loop_predictions(model(
                    x, y, num_loops=num_loops,
                    truncated_bptt=cfg.loop.truncated_bptt))
                loss = loop_window_loss(preds, target, window)

            # 形状断言：广播不会报错，只能主动检查
            assert preds.shape == (num_loops, x.shape[0]), \
                f"predictions {tuple(preds.shape)} != ({num_loops}, {x.shape[0]})"

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       cfg.training.grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_n += 1
            step += 1

            if step % log_every == 0:
                elapsed = time.time() - t0
                logger.log({
                    "step": step,
                    "train_loss": running_loss / running_n,
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "samples_per_sec": cfg.training.batch_size * running_n / elapsed,
                })
                running_loss, running_n, t0 = 0.0, 0, time.time()

            if step % cfg.training.eval_every == 0 or step >= cfg.training.max_steps:
                val_mses = evaluate(model, val_loader, device, num_loops)
                val_loss = sum(val_mses[num_loops - window:]) / window
                record = {"step": step, "val_loss": val_loss}
                for i, mse in enumerate(val_mses, 1):
                    record[f"val_mse_loop_{i}"] = mse
                logger.log(record)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(run_dir / "checkpoints" / "best.pt", model,
                                    optimizer, scheduler, step, best_val_loss,
                                    raw_config)
                    print(f"    -> 新 best_val_loss = {best_val_loss:.6f}")

            if step % cfg.training.save_every == 0:
                save_checkpoint(run_dir / "checkpoints" / f"step_{step:06d}.pt",
                                model, optimizer, scheduler, step, best_val_loss,
                                raw_config)

    logger.save_summary({
        "run_name": run_name,
        "final_step": step,
        "best_val_loss": best_val_loss,
        "num_parameters": n_params,
        "train_loops": num_loops,
        "loss_window": window,
        "environment": environment_info(),
    })
    print(f"完成。best_val_loss = {best_val_loss:.6f}")
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/configs/base.yaml")
    parser.add_argument("--run-name", default="main")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, build_model(cfg), args.run_name, args.config, resume=args.resume)


if __name__ == "__main__":
    main()
