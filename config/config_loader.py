from __future__ import annotations
from pathlib import Path
import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pinn_full_config(cfg_path):
    cfg_path = Path(cfg_path).resolve()
    raw_cfg = load_yaml(cfg_path)

    # 兼容新旧两种写法
    include_cfg = raw_cfg.get("include", {})
    physics_rel = include_cfg.get("physics", raw_cfg.get("physics_config"))
    if physics_rel is None:
        raise ValueError("Missing physics config: use include.physics or physics_config")

    physics_path = (cfg_path.parent.parent / physics_rel).resolve()
    physics_cfg = load_yaml(physics_path)

    # 去掉 include / physics_config，其余全部视作 train 部分
    train_cfg = {
        k: v for k, v in raw_cfg.items()
        if k not in {"include", "physics_config"}
    }

    return {
        "physics": physics_cfg,
        "train": train_cfg,
    }