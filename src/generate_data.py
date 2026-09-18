import argparse
import json
from pathlib import Path

import torch

from src.configs.config import load_config
from src.data import make_generator,make_task_batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",default="src/configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    d,k = cfg.data.d,cfg.data.k

    out_dir = Path("data")
    out_dir.mkdir(parents=True,exist_ok=True)

    splits = [
        ("train",cfg.data.train_size,cfg.seed+0),
        ("val",cfg.data.val_size,cfg.seed+1),
        ("test",cfg.data.test_size,cfg.seed+2),
    ]

    for name, size, seed in splits:
        x, y = make_task_batch(size, d, k, generator=make_generator(seed))
        path = out_dir / f"{name}.pt"
        torch.save({"x": x, "y": y}, path)
        size_mb = path.stat().st_size / 1e6
        print(f"{name:<5} x={tuple(x.shape)} y={tuple(y.shape)} -> {path} ({size_mb:.1f} MB)")

    metadata = {
        "seed": cfg.seed,
        "d": d,
        "k": k,
        "train_size": cfg.data.train_size,
        "val_size": cfg.data.val_size,
        "test_size": cfg.data.test_size,
        "x_distribution": "N(0, I)",
        "w_distribution": "N(0, I/d)",
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"metadata -> {meta_path}")


if __name__ == "__main__":
    main()