"""训练日志：metrics.jsonl 记录流水，metrics.json 存汇总。"""

import json
from pathlib import Path


class RunLogger:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "figures").mkdir(exist_ok=True)
        self.jsonl_path = self.run_dir / "metrics.jsonl"

    def log(self, record):
        """追加一行流水，并打印到控制台。"""
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        parts = []
        for key, val in record.items():
            parts.append(f"{key}={val:.4g}" if isinstance(val, float) else f"{key}={val}")
        print("  " + "  ".join(parts), flush=True)

    def save_summary(self, summary):
        with open(self.run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
