from __future__ import annotations
from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pinn_full_config(pinn_cfg_path: str | Path) -> dict:
    """
    返回结构：
    {
        "physics": {...},   # teukolsky_radial.yaml
        "train":   {...},   # pinn_config.yaml 除 physics_config 外的内容
    }
    """
    pinn_cfg_path = Path(pinn_cfg_path).resolve()
    pinn_cfg = load_yaml(pinn_cfg_path)

    physics_cfg_rel = pinn_cfg["physics_config"]
    physics_cfg_path = (pinn_cfg_path.parent.parent / physics_cfg_rel).resolve()

    physics_cfg = load_yaml(physics_cfg_path)

    train_cfg = {k: v for k, v in pinn_cfg.items() if k != "physics_config"}

    return {
        "physics": physics_cfg,
        "train": train_cfg,
    }