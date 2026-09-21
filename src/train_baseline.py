"""训练算力对齐的普通 Transformer baseline。"""

import argparse

from src.baseline_transformer import BaselineTransformer
from src.configs.config import load_config
from src.train import train


def build_model(cfg):
    return BaselineTransformer.from_config(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/configs/base.yaml")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, build_model(cfg), args.run_name, args.config, resume=args.resume)


if __name__ == "__main__":
    main()
