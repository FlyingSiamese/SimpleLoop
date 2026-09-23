"""出图：从 metrics.jsonl / evaluation.json / ood.json / ablation.json 生成全部图。

图内文字统一用英文，避免中文字体缺失。
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.configs.config import load_config

LS_STYLE = dict(color="black", linestyle=":", linewidth=1.5,
                label="Least Squares")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


# ---------------------------------------------------------------- 训练曲线

def plot_training_curves(rows, fig_dir):
    """Gradient norm 曲线。"""
    made = []
    train = [r for r in rows if "train_loss" in r]
    val = [r for r in rows if "val_loss" in r]

    if train:
        for logscale in (False, True):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot([r["step"] for r in train], [r["train_loss"] for r in train],
                    linewidth=1.2, label="train loss")
            if val:
                ax.plot([r["step"] for r in val], [r["val_loss"] for r in val],
                        linewidth=1.5, marker="o", markersize=3, label="val loss")
            if logscale:
                ax.set_yscale("log")
            ax.set_xlabel("training step")
            ax.set_ylabel("MSE")
            ax.set_title("Train / Validation Loss" + (" (log scale)" if logscale else ""))
            ax.grid(alpha=0.3)
            ax.legend()
            name = "train_val_loss_log.png" if logscale else "train_val_loss.png"
            made.append(save(fig, fig_dir / name))

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot([r["step"] for r in train], [r["grad_norm"] for r in train],
                linewidth=1.2, color="tab:red")
        ax.set_xlabel("training step")
        ax.set_ylabel("grad norm (before clipping)")
        ax.set_title("Gradient Norm")
        ax.grid(alpha=0.3)
        made.append(save(fig, fig_dir / "grad_norm.png"))

    return made


# ------------------------------------------------------- loop / context

def plot_mse_vs_loop(ev, cfg, outputs, fig_dir):
    """整个项目最核心的图：误差随 loop 迭代的变化。"""
    made = []
    mse = ev["mse_by_loop"]
    fig, ax = plt.subplots(figsize=(7, 4.5))

    if len(mse) > 1:
        ax.plot(range(1, len(mse) + 1), mse, marker="o", markersize=4,
                linewidth=1.5, label=f"{ev['architecture']}")
        n_train = cfg.loop.train_loops
        if 1 <= n_train <= len(mse):
            ax.axvline(n_train, color="gray", linestyle="--", linewidth=1.2,
                       label=f"train loops = {n_train}")
    else:
        ax.axhline(mse[0], linewidth=1.5, label=f"{ev['architecture']}")

    ax.axhline(ev["least_squares"]["mse"], **LS_STYLE)

    base_ckpt = outputs / "baseline" / "evaluation.json"
    if base_ckpt.exists():
        base_mse = load_json(base_ckpt)["mse_by_loop"][0]
        ax.axhline(base_mse, color="tab:green", linestyle="-.", linewidth=1.5,
                   label="16-layer baseline")

    ax.set_yscale("log")
    ax.set_xlabel("loop iteration")
    ax.set_ylabel("test MSE (log scale)")
    ax.set_title("Test MSE vs Loop Iteration")
    ax.grid(alpha=0.3)
    ax.legend()
    made.append(save(fig, fig_dir / "mse_vs_loop.png"))
    return made


def plot_context_length(ev, fig_dir):
    ctx = ev["context_length"]
    ks = sorted(int(k) for k in ctx)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, [ctx[str(k)]["model_mse"] for k in ks], marker="o",
            linewidth=1.5, label=ev["architecture"])
    ax.plot(ks, [ctx[str(k)]["least_squares_mse"] for k in ks], marker="s",
            **LS_STYLE)
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("context length k")
    ax.set_ylabel("test MSE (log scale)")
    ax.set_title("Test MSE vs Context Length")
    ax.grid(alpha=0.3)
    ax.legend()
    return [save(fig, fig_dir / "mse_vs_context_length.png")]


# ------------------------------------------------------------------- OOD

def plot_ood(ood, fig_dir):
    made = []
    specs = [
        ("scaling", "ood_scaling.png", "input scale s", True),
        ("noise", "ood_noise.png", "context label noise sigma", False),
    ]
    for key, fname, xlabel, logx in specs:
        data = ood[key]
        xs = sorted(float(k) for k in data)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(xs, [data[str(x) if str(x) in data else x]["model_mse"] for x in xs],
                marker="o", linewidth=1.5, label=ood["architecture"])
        ax.plot(xs, [data[str(x) if str(x) in data else x]["least_squares_mse"] for x in xs],
                marker="s", **LS_STYLE)
        ax.set_yscale("log")
        if logx:
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("test MSE (log scale)")
        ax.set_title(f"OOD: {xlabel}")
        ax.grid(alpha=0.3)
        ax.legend()
        made.append(save(fig, fig_dir / fname))

    cov = ood["covariance"]
    labels = list(cov.keys())
    fig, ax = plt.subplots(figsize=(6, 4.5))
    width = 0.35
    xs = range(len(labels))
    ax.bar([x - width / 2 for x in xs], [cov[l]["model_mse"] for l in labels],
           width, label=ood["architecture"])
    ax.bar([x + width / 2 for x in xs], [cov[l]["least_squares_mse"] for l in labels],
           width, label="Least Squares")
    ax.set_yscale("log")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylabel("test MSE (log scale)")
    ax.set_title("OOD: skewed covariance")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    made.append(save(fig, fig_dir / "ood_covariance.png"))
    return made


# -------------------------------------------------------------- ablation

def plot_ablation(abl, fig_dir):
    made = []
    titles = {
        "input_injection": ("input injection", "input_injection_ablation.png"),
        "loop_count": ("train loop count", "loop_count_ablation.png"),
        "loss_window": ("loss window T", "loss_window_ablation.png"),
    }
    for group, members in abl["groups"].items():
        if not members:
            continue
        xlabel, fname = titles[group]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label, res in members.items():
            mse = res["mse_by_loop"]
            if len(mse) > 1:
                ax.plot(range(1, len(mse) + 1), mse, marker="o", markersize=4,
                        linewidth=1.5, label=f"{xlabel} = {label}")
            else:
                ax.axhline(mse[0], linewidth=1.5, label=f"{xlabel} = {label}")
        ax.axhline(abl["least_squares_mse"], **LS_STYLE)
        ax.set_yscale("log")
        ax.set_xlabel("loop iteration")
        ax.set_ylabel("test MSE (log scale)")
        ax.set_title(f"Ablation: {xlabel}")
        ax.grid(alpha=0.3)
        ax.legend()
        made.append(save(fig, fig_dir / fname))
    return made


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="outputs/main")
    args = parser.parse_args()

    run_dir = Path(args.run)
    outputs = run_dir.parent
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    made, skipped = [], []

    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        curves = plot_training_curves(load_jsonl(metrics_path), fig_dir)
        if curves:
            made += curves
        else:
            skipped.append("metrics.jsonl 中没有 train_loss 记录")
    else:
        skipped.append("metrics.jsonl")

    ev_path = run_dir / "evaluation.json"
    if ev_path.exists():
        ev = load_json(ev_path)
        cfg = load_config(run_dir / "config.yaml")
        made += plot_mse_vs_loop(ev, cfg, outputs, fig_dir)
        made += plot_context_length(ev, fig_dir)
    else:
        skipped.append("evaluation.json")

    ood_path = run_dir / "ood.json"
    if ood_path.exists():
        made += plot_ood(load_json(ood_path), fig_dir)
    else:
        skipped.append("ood.json")

    abl_path = outputs / "ablation.json"
    if abl_path.exists():
        made += plot_ablation(load_json(abl_path), fig_dir)
    else:
        skipped.append("ablation.json")

    print(f"输出目录: {fig_dir}")
    for name in made:
        print(f"  ✓ {name}")
    for name in skipped:
        print(f"  - 跳过（缺少 {name}）")
    print(f"共生成 {len(made)} 张图")


if __name__ == "__main__":
    main()
