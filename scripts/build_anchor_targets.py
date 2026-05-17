#!/usr/bin/env python3
"""Build spectral anchor targets for Stage-3 anchor loss.

Loads spectral basis profiles, interpolates u_up/u_down to the training
y-grid, and saves as a .pt file for use during training.

Usage:
  python scripts/build_anchor_targets.py \
    --points-json outputs/spectral_calibration/patch_000_logw_v2/patch0_calibration_points.json \
    --profiles-dir outputs/spectral_calibration/patch_000_logw_v2/basis_profiles \
    --output outputs/spectral_calibration/patch_000_logw_v2/anchor_targets.pt \
    --n-interior 128 --n-near-inf 64 --y-eps 1e-3 --y-max 0.0 \
    --labels center low_omega high_omega
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from scipy.interpolate import CubicSpline


def build_y_grid(n_interior, n_near_inf, y_inf_min=-1.0, y_inf_max=-0.95,
                 y_eps=1e-3, y_min=-1.0, y_max=0.0):
    """Replicate the training y-grid construction."""
    k = torch.arange(n_interior, dtype=torch.float64)
    y_base = -torch.cos(torch.pi * k / (n_interior - 1))
    y_base = (y_base + 1.0) / 2.0 * (y_max - y_min - 2 * y_eps) + y_min + y_eps
    y_base = y_base.clamp(y_min + y_eps, y_max - y_eps)

    if n_near_inf > 0:
        y_extra = torch.linspace(y_inf_min, y_inf_max, n_near_inf, dtype=torch.float64)
        y_extra = y_extra.clamp(-1.0 + y_eps, 1.0 - y_eps)
        y_all = torch.cat([y_base, y_extra])
        y_all = torch.unique(y_all)
    else:
        y_all = y_base

    return y_all.numpy()


def main():
    parser = argparse.ArgumentParser(description="Build spectral anchor targets")
    parser.add_argument("--points-json", type=str, required=True)
    parser.add_argument("--profiles-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-interior", type=int, default=128)
    parser.add_argument("--n-near-inf", type=int, default=64)
    parser.add_argument("--y-eps", type=float, default=1e-3)
    parser.add_argument("--y-inf-min", type=float, default=-1.0)
    parser.add_argument("--y-inf-max", type=float, default=-0.95)
    parser.add_argument("--y-max", type=float, default=0.0)
    parser.add_argument("--labels", type=str, nargs="+",
                        default=["center", "low_omega", "high_omega"])
    args = parser.parse_args()

    profiles_dir = Path(args.profiles_dir)

    with open(args.points_json) as f:
        points = json.load(f)
    point_map = {p["label"]: p for p in points}

    y_grid = build_y_grid(
        n_interior=args.n_interior, n_near_inf=args.n_near_inf,
        y_inf_min=args.y_inf_min, y_inf_max=args.y_inf_max,
        y_eps=args.y_eps, y_max=args.y_max,
    )
    print(f"[anchor-build] y_grid: {len(y_grid)} points, range=[{y_grid[0]:.6f}, {y_grid[-1]:.6f}]")

    targets = {}
    for label in args.labels:
        if label not in point_map:
            print(f"[anchor-build] WARNING: '{label}' not in points JSON, skipping")
            continue

        pt = point_map[label]
        profile_path = profiles_dir / f"point_{label}_N80.npz"
        if not profile_path.exists():
            print(f"[anchor-build] WARNING: {profile_path} not found, skipping")
            continue

        prof = np.load(profile_path)
        y_prof = prof["y"]  # y values from spectral profile
        u_down_prof = prof["u_down"]
        u_up_prof = prof["u_up"]

        # Interpolate spectral u to training y-grid using cubic spline
        # y_prof is monotonically increasing from -1 to 2*z1-1
        mask = np.isfinite(u_down_prof) & np.isfinite(u_up_prof)
        y_clean = y_prof[mask]
        u_d_clean = u_down_prof[mask]
        u_u_clean = u_up_prof[mask]

        # Interpolate to training y-grid (clamp to profile range)
        y_valid = y_grid[(y_grid >= y_clean[0]) & (y_grid <= y_clean[-1])]
        if len(y_valid) < 2:
            print(f"[anchor-build] WARNING: '{label}' has <2 valid y points")
            continue

        cs_down = CubicSpline(y_clean, u_d_clean)
        cs_up = CubicSpline(y_clean, u_u_clean)

        u_down_interp = cs_down(y_valid)
        u_up_interp = cs_up(y_valid)

        targets[label] = {
            "a": float(pt["a"]),
            "omega": float(pt["omega"]),
            "u": float(pt.get("u", 0.5)),
            "v": float(pt.get("v", 0.5)),
            "lambda": float(pt.get("lambda_real", pt.get("lam", 0))),
            "y": y_valid.astype(np.float64),
            "u_down": u_down_interp.astype(np.complex128),
            "u_up": u_up_interp.astype(np.complex128),
        }
        print(f"[anchor-build] {label}: {len(y_valid)} y-points, "
              f"u_down range=[{np.abs(u_down_interp).min():.4f},{np.abs(u_down_interp).max():.4f}], "
              f"u_up range=[{np.abs(u_up_interp).min():.4f},{np.abs(u_up_interp).max():.4f}]")

    torch.save(targets, args.output)
    print(f"[anchor-build] Saved {len(targets)} anchor targets to {args.output}")


if __name__ == "__main__":
    main()
