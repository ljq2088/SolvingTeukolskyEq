#!/usr/bin/env python3
"""Build fixed spectral u_up/u_down basis cache for Stage-4 consistency loss.

Solves the Teukolsky ODE via Chebyshev collocation (utils/amplitude.py) on
[0, z_b] for each training parameter point. Stores u_up(z), u_down(z) on a
common y-grid for use in Stage-4.

The cache is FIXED — no autograd, no training. Stage-4 only reads from it.

Usage:
  # Pilot: 16 points
  python scripts/build_stage4_spectral_basis_cache.py \
    --stage1-artifact outputs/stage1_artifacts/patch_000_logw_v2 \
    --calibration-dir outputs/spectral_calibration/patch_000_logw_v2 \
    --n-param 16 --y-min -0.999 --y-max -0.95 --n-y 128 \
    --N-out 80 --z-b 0.05 \
    --output-dir outputs/stage4_spectral_basis_cache/patch_000_logw_v2

  # Full: 64 points
  python scripts/build_stage4_spectral_basis_cache.py \
    --stage1-artifact outputs/stage1_artifacts/patch_000_logw_v2 \
    --calibration-dir outputs/spectral_calibration/patch_000_logw_v2 \
    --n-param 64 --y-min -0.999 --y-max -0.95 --n-y 128 \
    --N-out 80 --z-b 0.05 \
    --output-dir outputs/stage4_spectral_basis_cache/patch_000_logw_v2
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Build Stage-4 spectral basis cache")
    parser.add_argument("--stage1-artifact", type=str, required=True)
    parser.add_argument("--calibration-dir", type=str,
                        default="outputs/spectral_calibration/patch_000_logw_v2")
    parser.add_argument("--n-param", type=int, default=64)
    parser.add_argument("--y-min", type=float, default=-0.999)
    parser.add_argument("--y-max", type=float, default=-0.95)
    parser.add_argument("--n-y", type=int, default=128)
    parser.add_argument("--N-out", type=int, default=80,
                        help="Chebyshev N for outer domain [0, z_b]")
    parser.add_argument("--z-b", type=float, default=0.05,
                        help="Right boundary of spectral domain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    stage1_dir = Path(args.stage1_artifact)
    rpred_path = stage1_dir / "rpred_cache.npz"
    if not rpred_path.exists():
        raise FileNotFoundError(f"rpred_cache.npz not found in {stage1_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load parameter pool ----
    pool = np.load(rpred_path)
    valid = np.isfinite(pool["a"]) & np.isfinite(pool["omega"]) & np.isfinite(pool["lambda_"])
    a_all = pool["a"][valid]
    omega_all = pool["omega"][valid]
    u_all = pool["u"][valid]
    v_all = pool["v"][valid]
    lam_all = pool["lambda_"][valid]

    n_total = len(a_all)
    print(f"[cache-build] Parameter pool: {n_total} valid points")
    print(f"  a: [{a_all.min():.4f}, {a_all.max():.4f}]")
    print(f"  omega: [{omega_all.min():.6e}, {omega_all.max():.6e}]")

    # Sample points
    rng = np.random.default_rng(args.seed)
    n_sample = min(args.n_param, n_total)
    idx = rng.choice(n_total, n_sample, replace=False)
    idx = np.sort(idx)

    a_samp = a_all[idx]
    omega_samp = omega_all[idx]
    u_samp = u_all[idx]
    v_samp = v_all[idx]
    lam_samp = lam_all[idx]

    print(f"[cache-build] Sampled {n_sample} points")

    # ---- Build y grid (near-infinity) ----
    y_grid = np.linspace(args.y_min, args.y_max, args.n_y)
    z_grid = (y_grid + 1.0) / 2.0
    print(f"[cache-build] y-grid: [{y_grid[0]:.6f}, {y_grid[-1]:.6f}], "
          f"z-grid: [{z_grid[0]:.6f}, {z_grid[-1]:.6f}], n={args.n_y}")

    z_b = args.z_b
    if z_grid[-1] > z_b:
        print(f"[cache-build] WARNING: y_max={args.y_max} -> z={z_grid[-1]:.4f} > z_b={z_b}. "
              f"Reducing y_max to z_b={z_b}.")
        z_grid = z_grid[z_grid <= z_b]
        y_grid = 2.0 * z_grid - 1.0
        if len(y_grid) < 2:
            raise ValueError("Too few y points after clipping to z_b")

    # ---- Solve spectral basis for each point ----
    from utils.mode import KerrMode
    from utils.amplitude import solve_basis_domain

    # Storage
    u_up_cache = np.zeros((n_sample, len(y_grid)), dtype=np.complex128)
    u_down_cache = np.zeros((n_sample, len(y_grid)), dtype=np.complex128)

    for i in range(n_sample):
        a_i = float(a_samp[i])
        omega_i = float(omega_samp[i])
        lam_i = complex(float(lam_samp[i]), 0)

        mode = KerrMode(M=1.0, a=a_i, omega=omega_i, ell=2, m=2, s=-2, lam=lam_i)

        sol_down = solve_basis_domain(mode, "down", args.N_out, 0.0, z_b, "left")
        sol_up = solve_basis_domain(mode, "up", args.N_out, 0.0, z_b, "left")

        # Interpolate spectral solutions to common y-grid
        u_down_interp = np.interp(z_grid, sol_down["z"][::-1],
                                  sol_down["u"][::-1].real) + \
                        1j * np.interp(z_grid, sol_down["z"][::-1],
                                       sol_down["u"][::-1].imag)
        u_up_interp = np.interp(z_grid, sol_up["z"][::-1],
                                sol_up["u"][::-1].real) + \
                      1j * np.interp(z_grid, sol_up["z"][::-1],
                                     sol_up["u"][::-1].imag)

        u_down_cache[i, :] = u_down_interp
        u_up_cache[i, :] = u_up_interp

        if (i + 1) % max(1, n_sample // 10) == 0:
            print(f"  [{i+1}/{n_sample}] a={a_i:.4f}, omega={omega_i:.6e}")

    # ---- Verify boundary conditions ----
    u_up_at_0 = u_up_cache[:, 0]
    u_down_at_0 = u_down_cache[:, 0]
    max_dev_up = np.max(np.abs(u_up_at_0 - 1.0))
    max_dev_down = np.max(np.abs(u_down_at_0 - 1.0))
    print(f"[cache-build] BC check: max|u_up(inf)-1| = {max_dev_up:.2e}, "
          f"max|u_down(inf)-1| = {max_dev_down:.2e}")

    # ---- Save ----
    npz_path = out_dir / "spectral_basis_cache.npz"
    np.savez_compressed(
        npz_path,
        a=a_samp,
        omega=omega_samp,
        u=u_samp,
        v=v_samp,
        lambda_=lam_samp,
        y=y_grid,
        z=z_grid,
        u_up=u_up_cache,
        u_down=u_down_cache,
        N_out=args.N_out,
        z_b=z_b,
        basis_convention="u(infinity)=1, u_z(0)=boundary_du_exact (z-space)",
    )
    print(f"[cache-build] Saved to {npz_path}")
    print(f"  shapes: u_up={u_up_cache.shape}, u_down={u_down_cache.shape}")

    # ---- Metadata ----
    import subprocess
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        git_commit = "unknown"

    metadata = {
        "git_commit": git_commit,
        "built_at": datetime.now().isoformat(),
        "stage1_artifact": str(stage1_dir.resolve()),
        "calibration_dir": str(Path(args.calibration_dir).resolve()),
        "n_param": int(n_sample),
        "n_total_pool": int(n_total),
        "y_min": args.y_min,
        "y_max": args.y_max,
        "n_y": args.n_y,
        "N_out": args.N_out,
        "z_b": z_b,
        "bc_check_max_dev_up": float(max_dev_up),
        "bc_check_max_dev_down": float(max_dev_down),
        "basis_convention": "u(infinity)=1, left BC at z=0",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ---- Summary markdown ----
    lines = [
        "# Stage-4 Spectral Basis Cache",
        "",
        f"- **Parameters**: {n_sample} points sampled from {n_total} total",
        f"- **y range**: [{args.y_min}, {args.y_max}]",
        f"- **z range**: [{z_grid[0]:.6f}, {z_grid[-1]:.6f}]",
        f"- **n_y**: {args.n_y}",
        f"- **N_out**: {args.N_out}",
        f"- **z_b**: {z_b}",
        f"- **BC check**: max|u_up(inf)-1|={max_dev_up:.2e}, max|u_down(inf)-1|={max_dev_down:.2e}",
        "",
        "## Convention",
        "- u_up(z=0) = u_down(z=0) = 1",
        "- z = r_+/r, z=0 at infinity",
        "- Left BC at z=0: u(0)=1, u'(0) from analytic boundary condition",
    ]
    with open(out_dir / "spectral_basis_cache_summary.md", "w") as f:
        f.write("\n".join(lines))

    print(f"[cache-build] Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
