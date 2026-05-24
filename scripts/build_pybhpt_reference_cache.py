#!/usr/bin/env python3
"""Pre-compute pybhpt R_in/P references for Stage-4 supervised training.

For each sampled (a, omega) point from the rpred_cache, calls pybhpt to get
R_in(r), then divides by the Leaver prefactor P(r) to get R_in/P on the
training y-grid.

Usage:
  python scripts/build_pybhpt_reference_cache.py \
    --rpred-cache outputs/pre_stage4_bundle/patch_000_logw_v2/rpred_cache.npz \
    --n-points 16 \
    --y-min -0.999 --y-max -0.5 \
    --n-y-near 96 --n-y-transition 32 \
    --timeout 30.0 \
    --max-workers 4 \
    --output-dir outputs/pybhpt_reference_cache/patch_000_logw_v2
"""
import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from physical_ansatz.mapping import r_plus, r_minus
from physical_ansatz.prefactor import Leaver_prefactors
from pybhpt_usage.compute_solution import compute_pybhpt_solution


def compute_one_point(args_tuple):
    """Compute pybhpt R_in/P for one (a, omega) point. Runs in subprocess."""
    idx, a_val, omega_val, y_grid, timeout = args_tuple
    try:
        rp = 1.0 + np.sqrt(1.0 - a_val * a_val)
        x_np = (y_grid + 1.0) / 2.0
        r_np = rp / np.clip(x_np, 1e-12, None)
        r_sorted = np.sort(np.unique(r_np))

        r_vals, R_in = compute_pybhpt_solution(
            a_val, omega_val, ell=2, m=2, r_grid=r_sorted, timeout=timeout)

        # Interpolate R_in to training r points
        R_in_interp = (np.interp(r_np, r_vals, R_in.real)
                       + 1j * np.interp(r_np, r_vals, R_in.imag))

        # Compute P(r)
        a_t = torch.tensor([a_val], dtype=torch.float64)
        omega_t = torch.tensor([omega_val], dtype=torch.float64)
        r_t = torch.tensor(r_np, dtype=torch.float64)
        rp_t = r_plus(a_t, 1.0)
        rm_t = r_minus(a_t, 1.0)
        P_np, _, _ = Leaver_prefactors(r_t, a_t, omega_t, m=2, M=1.0, s=-2,
                                       rp=rp_t, rm=rm_t)

        R_over_P = R_in_interp / P_np.numpy()

        success = np.all(np.isfinite(R_over_P))
        return idx, a_val, omega_val, y_grid, R_over_P, success, None
    except Exception as e:
        return idx, a_val, omega_val, None, None, False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpred-cache", type=str, required=True)
    parser.add_argument("--n-points", type=int, default=16)
    parser.add_argument("--y-min", type=float, default=-0.999)
    parser.add_argument("--y-max", type=float, default=-0.5)
    parser.add_argument("--n-y-near", type=int, default=96,
                        help="Points in near-inf region [-1, -0.95]")
    parser.add_argument("--n-y-transition", type=int, default=32,
                        help="Points in transition region [-0.95, y_max]")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    # Load rpred cache
    cache = np.load(args.rpred_cache)
    a_all = cache["a"]
    omega_all = cache["omega"]
    u_all = cache["u"]
    v_all = cache["v"]
    lam_all = cache["lambda_"]

    valid = np.isfinite(a_all) & np.isfinite(omega_all) & np.isfinite(lam_all)
    a_all = a_all[valid]
    omega_all = omega_all[valid]
    u_all = u_all[valid]
    v_all = v_all[valid]
    lam_all = lam_all[valid]

    n_total = len(a_all)
    print(f"Parameter pool: {n_total} valid points")
    print(f"  a: [{a_all.min():.4f}, {a_all.max():.4f}]")
    print(f"  omega: [{omega_all.min():.6e}, {omega_all.max():.6e}]")

    # Sample points
    rng = np.random.default_rng(args.seed)
    n_sample = min(args.n_points, n_total)
    idx = rng.choice(n_total, n_sample, replace=False)
    idx = np.sort(idx)

    # Build y-grid: dense near infinity
    y_near = np.linspace(-0.999, -0.95, args.n_y_near)
    y_transition = np.linspace(-0.95, args.y_max, args.n_y_transition)
    y_grid = np.concatenate([y_near, y_transition])
    n_y = len(y_grid)
    print(f"y-grid: [{y_grid[0]:.4f}, {y_grid[-1]:.4f}], {n_y} points "
          f"({args.n_y_near} near-inf + {args.n_y_transition} transition)")

    # Prepare jobs
    jobs = []
    for i, pi in enumerate(idx):
        jobs.append((i, float(a_all[pi]), float(omega_all[pi]),
                     y_grid.copy(), args.timeout))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parallel pybhpt computation
    print(f"\nComputing pybhpt for {len(jobs)} points with {args.max_workers} workers ...")
    refs = {}
    n_ok = 0
    n_fail = 0

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(compute_one_point, job): job[0] for job in jobs}
        for future in as_completed(futures):
            i, a_val, omega_val, yg, R_over_P, success, err = future.result()
            refs[i] = {
                "a": a_val, "omega": omega_val,
                "R_over_P": R_over_P, "y": yg,
            }
            if success:
                n_ok += 1
                print(f"  [{n_ok + n_fail}/{len(jobs)}] a={a_val:.4f} ω={omega_val:.6e} OK "
                      f"|R/P|∈[{np.min(np.abs(R_over_P)):.2e}, {np.max(np.abs(R_over_P)):.2e}]")
            else:
                n_fail += 1
                print(f"  [{n_ok + n_fail}/{len(jobs)}] a={a_val:.4f} ω={omega_val:.6e} FAIL: {err}")

    print(f"\nDone: {n_ok} OK, {n_fail} failed")

    # Save
    a_refs = np.array([refs[i]["a"] for i in range(len(jobs)) if refs[i]["R_over_P"] is not None])
    omega_refs = np.array([refs[i]["omega"] for i in range(len(jobs)) if refs[i]["R_over_P"] is not None])
    R_over_P_refs = np.array([refs[i]["R_over_P"] for i in range(len(jobs)) if refs[i]["R_over_P"] is not None])

    # Also save u, v, lambda for the sampled points
    u_refs = u_all[idx[:n_ok]] if n_ok > 0 else np.array([])
    v_refs = v_all[idx[:n_ok]] if n_ok > 0 else np.array([])
    lam_refs = lam_all[idx[:n_ok]] if n_ok > 0 else np.array([])

    out_path = out_dir / "pybhpt_references.npz"
    np.savez(out_path,
             a=a_refs, omega=omega_refs,
             u=u_refs, v=v_refs, lam=lam_refs,
             y=y_grid, R_over_P=R_over_P_refs,
             n_points=n_ok, n_y=n_y)

    # Config summary
    with open(out_dir / "build_config.json", "w") as f:
        json.dump({
            "n_points_requested": args.n_points,
            "n_ok": n_ok, "n_failed": n_fail,
            "y_min": args.y_min, "y_max": args.y_max,
            "n_y_near": args.n_y_near, "n_y_transition": args.n_y_transition,
            "timeout": args.timeout, "max_workers": args.max_workers,
        }, f, indent=2)

    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
