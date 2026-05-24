#!/usr/bin/env python3
"""Pre-compute f(y) anchor targets for Stage-1 training from pybhpt.

For each (a, omega) in the training pool, compute the true f(y) at a fixed set
of anchor y-points. Saves as a dict for fast lookup during training.

Usage:
  python scripts/build_stage1_f_anchors.py \
    --probe-json outputs/domain/probe_l2_m2_logw.json \
    --patch-json outputs/domain/patch_cover_l2_m2_logw.json \
    --atlas-json outputs/domain/atlas_l2_m2_logw.json \
    --patch-id 0 \
    --n-anchors 5 \
    --output outputs/stage1_retrain/f_anchors_patch0.pt
"""
import argparse, json, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from physical_ansatz.transform_y import h_factor, g_factor, h1_factor, horizon_regularity_slope
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives
from physical_ansatz.mapping import r_plus
from utils.compute_lambda_usage import compute_lambda
from domain.patch_cover import load_patch_cover, load_valid_chart_points


def compute_true_f(a, omega, lam, y_anchors, M=1.0, m=2, s=-2):
    """Compute true f(y) at anchor points using pybhpt."""
    from pybhpt_usage.compute_solution import compute_pybhpt_solution

    dtype = torch.float64
    cdtype = torch.complex128
    a_t = torch.tensor([a], dtype=dtype)
    omega_t = torch.tensor([omega], dtype=dtype)
    lam_t = torch.tensor([lam], dtype=cdtype)

    rp = float(r_plus(a_t, M))
    y_np = np.atleast_1d(np.asarray(y_anchors, dtype=np.float64))
    x_np = 0.5 * (y_np + 1.0)
    r_np = rp / np.clip(x_np, 1e-12, None)

    _, R_ref = compute_pybhpt_solution(a, omega, ell=2, m=m, r_grid=r_np, timeout=30.0)
    R_ref = np.asarray(R_ref, dtype=np.complex128)

    r_t = torch.tensor(r_np, dtype=dtype)
    rp_t = r_plus(a_t, M)
    _, rm_t, _, _, _ = build_prefactor_primitives(r_t, a_t, M=M, need_rs=False)
    P, _, _ = Leaver_prefactors(r_t, a_t, omega_t, m=m, M=M, s=s, rp=rp_t, rm=rm_t)
    h2 = h_factor(a_t, omega_t, m=m, M=M, s=s)
    S_true = torch.tensor(R_ref, dtype=cdtype) / (P * h2)

    slope = horizon_regularity_slope(a=a_t, omega=omega_t, lambda_=lam_t, m=m, M=M, s=s)
    x_t = torch.tensor(x_np, dtype=dtype)
    h1, _, _ = h1_factor(x_t)
    g, _, _ = g_factor(x_t, slope.squeeze())
    f_true = (S_true.squeeze() - g.squeeze() - 1.0) / (g.squeeze() * h1.squeeze())

    return f_true.detach().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-json", required=True)
    parser.add_argument("--patch-json", required=True)
    parser.add_argument("--atlas-json", required=True)
    parser.add_argument("--patch-id", type=int, required=True)
    parser.add_argument("--n-anchors", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    patch_cover = load_patch_cover(args.patch_json)
    patch = [p for p in patch_cover.patches if p.patch_id == args.patch_id][0]

    uv_points, aw_points = load_valid_chart_points(
        probe_json=args.probe_json,
        atlas_json=args.atlas_json,
        component_id=patch.component_id,
        omega_chart_mode=patch_cover.meta.get("omega_chart_mode", "linear"),
    )
    mask = (
        (np.abs(uv_points[:, 0] - patch.u_center) <= patch.h_u) &
        (np.abs(uv_points[:, 1] - patch.v_center) <= patch.h_v)
    )
    patch_aw = aw_points[mask]
    patch_uv = uv_points[mask]
    print(f"Patch {args.patch_id}: {len(patch_aw)} training points")

    # Build fixed anchor y-points: Chebyshev distribution in [-0.99, 0.99]
    # Avoid exact y=1 (horizon, W=0) and y=-1 (infinity, singular)
    k = np.arange(args.n_anchors)
    y_anchors = -np.cos(np.pi * k / (args.n_anchors - 1))
    y_anchors = 0.99 * y_anchors  # Scale to avoid exact endpoints
    # Ensure anchors are in the far half (y<0.5) where W is non-negligible
    y_anchors = np.clip(y_anchors, -0.99, 0.49)
    print(f"Anchor y-points: {y_anchors}")

    targets = {}
    M, m_mode, s_val = 1.0, 2, -2

    for i in tqdm(range(len(patch_aw)), desc="Building f-anchors"):
        a = float(patch_aw[i, 0])
        omega = float(patch_aw[i, 1])
        u = float(patch_uv[i, 0])
        v = float(patch_uv[i, 1])
        lam = compute_lambda(a, omega, 2, m_mode, s=s_val)

        key = f"{a:.12f}_{omega:.12f}"
        try:
            f_vals = compute_true_f(a, omega, lam, y_anchors, M=M, m=m_mode, s=s_val)
            targets[key] = {
                "a": a, "omega": omega, "u": u, "v": v,
                "lambda": complex(lam),
                "y": y_anchors.tolist(),
                "f": [complex(v) for v in f_vals],
            }
        except Exception as e:
            print(f"  FAILED [{key}]: {e}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(targets, str(output_path))
    print(f"Saved {len(targets)} anchor targets to {output_path}")


if __name__ == "__main__":
    main()
