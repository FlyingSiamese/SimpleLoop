"""消融实验汇总：input injection / loop count / loss window。

run 名统一取"配置文件名去掉 .yaml"，base 除外（它叫 main）。
main 同时充当三组消融的对照点。
"""

import argparse
from pathlib import Path

from src.configs.config import load_config
from src.dataset import LinearRegressionDataset
from src.evaluate import least_squares_predict, load_model, predict_loops
from src.utils import get_device, save_json

GROUPS = {
    "input_injection": [
        ("true", "main", "src/configs/base.yaml"),
        ("false", "ablation_no_injection", "src/configs/ablation_no_injection.yaml"),
    ],
    "loop_count": [
        ("4", "ablation_loop_4", "src/configs/ablation_loop_4.yaml"),
        ("8", "main", "src/configs/base.yaml"),
        ("12", "ablation_loop_12", "src/configs/ablation_loop_12.yaml"),
    ],
    "loss_window": [
        ("1", "ablation_window_1", "src/configs/ablation_window_1.yaml"),
        ("4", "main", "src/configs/base.yaml"),
        ("8", "ablation_window_8", "src/configs/ablation_window_8.yaml"),
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-loops", type=int, default=32)
    args = parser.parse_args()

    device = get_device()
    outputs = Path(args.outputs)
    test_path = str(Path(args.data_dir) / "test.pt")
    test = LinearRegressionDataset(test_path)
    target = test.y[:, test.k_total]

    ls_pred, _ = least_squares_predict(test.x, test.y, test.k_total)
    ls_mse = ((ls_pred - target) ** 2).mean().item()

    cache = {}

    def collect(run_name, config_path):
        if run_name in cache:
            return cache[run_name]
        ckpt = outputs / run_name / "checkpoints" / "best.pt"
        if not ckpt.exists():
            print(f"  [跳过] {run_name}：找不到 {ckpt}")
            cache[run_name] = None
            return None
        cfg = load_config(config_path)
        model, arch, _ = load_model(str(ckpt), cfg, "auto", device)
        loops = args.num_loops if arch == "looped" else 1
        preds = predict_loops(model, test, loops, device, args.batch_size)
        by_loop = [((preds[t] - target) ** 2).mean().item() for t in range(loops)]
        cache[run_name] = {
            "architecture": arch,
            "train_loops": cfg.loop.train_loops,
            "loss_window": cfg.loop.loss_window,
            "input_injection": bool(cfg.loop.input_injection),
            "mse_by_loop": by_loop,
        }
        print(f"  {run_name:<24} {arch:<9} 最终 loop MSE {by_loop[-1]:.6f}")
        return cache[run_name]

    groups = {}
    for group, members in GROUPS.items():
        print(f"\n[{group}]")
        entry = {}
        for label, run_name, config_path in members:
            result = collect(run_name, config_path)
            if result is not None:
                entry[label] = {"run_name": run_name, **result}
        groups[group] = entry

    print("\n[baseline]")
    baseline = collect("baseline", "src/configs/base.yaml")

    result = {
        "num_eval_loops": args.num_loops,
        "least_squares_mse": ls_mse,
        "groups": groups,
    }
    if baseline is not None:
        result["baseline"] = {"run_name": "baseline", **baseline}

    save_json(result, outputs / "ablation.json")
    print(f"\n已写入 {outputs / 'ablation.json'}")


if __name__ == "__main__":
    main()
