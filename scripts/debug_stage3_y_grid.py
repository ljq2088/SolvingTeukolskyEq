#!/usr/bin/env python3
"""Debug y-grid for Stage-3 training configs.

Usage:
  python scripts/debug_stage3_y_grid.py \\
    --config config/autoencoder_stage3_ubasis_nearinf.yaml
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from config.config_loader import load_pinn_full_config
from scripts.train_autoencoder_stage3_ubasis import _build_y_grid


def _get_dtype(dtype_name):
    return torch.float32 if dtype_name == "float32" else torch.float64


def main():
    parser = argparse.ArgumentParser(description="Debug Stage-3 y-grid")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    full_cfg = load_pinn_full_config(args.config)
    runtime_cfg = full_cfg.get("runtime", {})
    dtype_name = runtime_cfg.get("dtype", "float64")
    dtype = _get_dtype(dtype_name)

    stage3_cfg = full_cfg.get("train", {}).get("stage3", full_cfg.get("stage3", {}))
    samp_cfg = stage3_cfg.get("sampling", {})

    n_interior = int(samp_cfg.get("n_interior", 128))
    n_near_inf = int(samp_cfg.get("n_near_infinity", 0))
    y_inf_min = float(samp_cfg.get("y_inf_min", -1.0))
    y_inf_max = float(samp_cfg.get("y_inf_max", -0.95))
    y_eps = float(samp_cfg.get("y_eps", 1e-3))
    y_min = float(samp_cfg.get("y_min", -1.0))
    y_max = float(samp_cfg.get("y_max", 1.0))
    y_strategy = str(samp_cfg.get("y_strategy", "full_domain"))

    device = torch.device(args.device)

    # ---- Build base grid (no extra) to see its range ----
    y_base = _build_y_grid(n_interior, 0, y_inf_min, y_inf_max,
                           device, dtype, y_eps=y_eps,
                           y_min=y_min, y_max=y_max,
                           y_strategy=y_strategy)

    # ---- Build full grid ----
    y_grid = _build_y_grid(n_interior, n_near_inf, y_inf_min, y_inf_max,
                           device, dtype, y_eps=y_eps,
                           y_min=y_min, y_max=y_max,
                           y_strategy=y_strategy)

    y_np = y_grid.cpu().numpy()

    print("=" * 60)
    print("Stage-3 y-grid Debug")
    print("=" * 60)
    print(f"  y_strategy:      {y_strategy}")
    print(f"  y_eps:           {y_eps}")
    print(f"  n_interior:      {n_interior}")
    print(f"  n_near_infinity: {n_near_inf}")
    print(f"  base grid min:   {y_base.min().item():.8f}")
    print(f"  base grid max:   {y_base.max().item():.8f}")
    if n_near_inf > 0:
        print(f"  near-inf range:  [{y_inf_min}, {y_inf_max}]")
    print(f"  final grid min:  {y_np.min():.8f}")
    print(f"  final grid max:  {y_np.max():.8f}")
    print(f"  final grid size: {len(y_np)}")
    print(f"  first 10 y:      {y_np[:10]}")
    print(f"  last 10 y:       {y_np[-10:]}")

    # ---- Validation ----
    print()
    errors = []
    if y_np.min() < y_min + y_eps - 1e-10:
        errors.append(f"y_min {y_np.min():.8f} < allowed {y_min+y_eps:.8f}")
    if y_np.max() > y_max + 1e-10:
        errors.append(f"y_max {y_np.max():.8f} > allowed {y_max:.8f}")

    if y_strategy == "near_infinity_only":
        if y_np.max() > 0.0 + 1e-10:
            errors.append("near_infinity_only: y_max > 0 — grid leaks into positive y!")
        if abs(y_np.min() - (-1.0 + y_eps)) > 1e-3:
            errors.append(f"near_infinity_only: y_min {y_np.min():.8f} far from -1+y_eps={-1+y_eps:.8f}")

    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("OK: y-grid validates against config.")


if __name__ == "__main__":
    main()
