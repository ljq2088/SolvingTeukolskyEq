#!/usr/bin/env python3
"""Check if spectral basis solutions satisfy Stage-3 u-equation residual.

For each calibration point, substitutes the spectral u_up/u_down into:
    C2 * u_yy + C1 * u_y + C0 * u
using Chebyshev derivatives (no autograd), and reports the residual.

Usage:
  python scripts/check_spectral_solution_in_u_residual.py
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from physical_ansatz.u_residual import compute_u_equation_coefficients


def cheb_D(N, a, b):
    """Chebyshev differentiation matrix on [a,b] with N+1 points."""
    xi = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = c[-1] = 2.0
    D = np.zeros((N + 1, N + 1), dtype=float)
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1) ** (i + j)) / (xi[i] - xi[j])
    D[np.diag_indices(N + 1)] = -np.sum(D, axis=1)
    D = (-2.0 / (b - a)) * D
    return D


def main():
    pts_json = PROJECT_ROOT / "outputs/spectral_calibration/patch_000_logw_v2/patch0_calibration_points.json"
    profiles_dir = PROJECT_ROOT / "outputs/spectral_calibration/patch_000_logw_v2/basis_profiles"

    with open(pts_json) as f:
        points = json.load(f)

    bases = ["up", "down"]
    all_results = []

    for pt in points:
        label = pt["label"]
        profile_path = profiles_dir / f"point_{label}_N80.npz"
        if not profile_path.exists():
            print(f"  SKIP {label}: no profile")
            continue

        prof = np.load(profile_path)
        z_full = prof["z"]
        N = int(prof["N"])
        a_val = float(prof["a"])
        omega_val = float(prof["omega"])
        lam_val = complex(float(prof["lambda_"]), 0)

        # Chebyshev D matrix on [0, z1]
        z1 = float(prof["z1"])
        D = cheb_D(N, 0.0, z1)

        # y = 2z - 1 (full grid)
        y_full = 2.0 * z_full - 1.0

        # Convert to torch tensors for coefficient computation
        a_t = torch.tensor([[a_val]], dtype=torch.float64)
        omega_t = torch.tensor([[omega_val]], dtype=torch.float64)
        lam_t = torch.tensor([[lam_val]], dtype=torch.complex128)

        for basis in bases:
            u_key = f"u_{basis}"
            uz_key = f"uz_{basis}"
            u_full = prof[u_key]
            uz_full = prof[uz_key]

            # u_zz via Chebyshev D2: u_zz = D @ uz
            u_zz_full = D @ uz_full

            # Convert to y-space derivatives: dy/dz = 2
            u_y_full = uz_full / 2.0
            u_yy_full = u_zz_full / 4.0

            # Compute C2, C1, C0 at each y point (full grid)
            y_t = torch.tensor(y_full.reshape(1, -1), dtype=torch.float64)
            C2, C1, C0 = compute_u_equation_coefficients(
                a_t, omega_t, y_t, lam_t, M=1.0, s=-2, m=2, basis=basis)
            C2_np = C2[0, :].detach().cpu().numpy()
            C1_np = C1[0, :].detach().cpu().numpy()
            C0_np = C0[0, :].detach().cpu().numpy()

            # Residual: C2*u_yy + C1*u_y + C0*u
            residual_full = C2_np * u_yy_full + C1_np * u_y_full + C0_np * u_full

            # Exclude z=0 (horizon) — coefficients diverge as 1/x
            mask = z_full > 1e-15
            residual = residual_full[mask]
            u_m = u_full[mask]
            u_y_m = u_y_full[mask]
            u_yy_m = u_yy_full[mask]
            C0_m = C0_np[mask]
            C1_m = C1_np[mask]
            C2_m = C2_np[mask]
            z_m = z_full[mask]

            # Relative residual
            denom = (np.abs(C0_m * u_m) + np.abs(C1_m * u_y_m)
                     + np.abs(C2_m * u_yy_m) + 1e-30)
            rel_res = np.abs(residual) / denom

            res_abs = np.abs(residual)
            info = {
                "label": label,
                "basis": basis,
                "z_range": [float(z_m[0]), float(z_m[-1])],
                "y_range": [float(2*z_m[0]-1), float(2*z_m[-1]-1)],
                "n_points": len(z_m),
                "residual_median_abs": float(np.median(res_abs)),
                "residual_max_abs": float(np.max(res_abs)),
                "relative_residual_median": float(np.median(rel_res)),
                "relative_residual_max": float(np.max(rel_res)),
            }
            all_results.append(info)

            print(f"{label:12s} {basis:6s}  "
                  f"abs_res: med={info['residual_median_abs']:.2e} max={info['residual_max_abs']:.2e}  "
                  f"rel_res: med={info['relative_residual_median']:.2e} max={info['relative_residual_max']:.2e}")

    # Summary
    print(f"\n--- Summary ---")
    max_abs = max(r["residual_max_abs"] for r in all_results)
    max_rel = max(r["relative_residual_max"] for r in all_results)
    med_abs = np.median([r["residual_median_abs"] for r in all_results])
    med_rel = np.median([r["relative_residual_median"] for r in all_results])
    print(f"  Max abs residual:   {max_abs:.3e}")
    print(f"  Max rel residual:   {max_rel:.3e}")
    print(f"  Median abs residual: {med_abs:.3e}")
    print(f"  Median rel residual: {med_rel:.3e}")

    threshold = 1e-3  # relative residual; absolute scales with u (up to 1e6)
    all_ok = max_rel < threshold
    print(f"\n  PASS (max_rel < {threshold}): {all_ok}")

    # Save
    out_dir = PROJECT_ROOT / "outputs/spectral_calibration/patch_000_logw_v2"
    with open(out_dir / "spectral_in_u_residual_check.json", "w") as f:
        json.dump({"results": all_results, "summary": {
            "max_abs_residual": max_abs,
            "max_rel_residual": max_rel,
            "median_abs_residual": med_abs,
            "median_rel_residual": med_rel,
            "pass": all_ok,
        }}, f, indent=2)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
