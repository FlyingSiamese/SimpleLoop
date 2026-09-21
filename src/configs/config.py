from pathlib import Path
from types import SimpleNamespace

import yaml

def _to_namespace(obj):
    if isinstance(obj,dict):
        return SimpleNamespace(**{k:_to_namespace(v) for k,v in obj.items()})
    if isinstance(obj,list):
        return [_to_namespace(v) for v in obj]
    return obj
def load_config_raw(path):
    """读取 YAML 原始 dict（用于写入 checkpoint 和 config.yaml）。"""
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path):
    """读取 YAML 并转成可用点号访问的对象。"""
    return _to_namespace(load_config_raw(path))