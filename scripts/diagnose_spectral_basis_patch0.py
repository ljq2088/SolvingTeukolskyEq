#!/usr/bin/env python3
"""Diagnose spectral basis functions (u_down, u_up) for patch 0 calibration points.

Checks boundary conditions, computes u-equation residuals, and saves basis profiles
for later Stage-3 decoder comparison.

Usage:
  python scripts/diagnose_spectral_basis_patch0.py \
    --points-json outputs/spectral_calibration/patch_000_logw_v2/patch0_calibration_points.json \
    --output-dir outputs/spectral_calibration/patch_000_logw_v2
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from utils.mode import KerrMode
from utils.amplitude import solve_basis_domain, boundary_du_exact
from physical_ansatz.mapping import r_plus
from physical_ansatz.prefactor import delta, delta_r, V_of_r
from physical_ansatz.u_residual import compute_q_terms


def _get_dtype(dtype_name="float64"):
    return torch.float64 if dtype_name == "float64" else torch.float32


def compute_u_eq_residual_on_z(mode, z_vals, u_vals, basis, device="cpu", dtype=torch.float64):
    """Compute u-equation residual C2*u_zz + C1*u_z + C0*u on a z grid.

    Uses finite differences on z-grid for derivatives.
    Returns residual_norm = mean(|residual|^2).
    """
    rp = r_plus(torch.tensor([[mode.a]], dtype=dtype), mode.M)
    z_t = torch.tensor(z_vals, dtype=dtype, device=device)
    a_t = torch.tensor([[mode.a]], dtype=dtype, device=device)
    omega_t = torch.tensor([[mode.omega]], dtype=dtype, device=device)
    u_t = torch.tensor(u_vals.reshape(1, -1), dtype=torch.complex128, device=device)

    # r = rp / z
    r_t = rp / z_t.unsqueeze(0)

    # y = 2*z - 1
    y_t = 2.0 * z_t.unsqueeze(0) - 1.0

    # u_y = u_z * dz/dy = u_z / 2
    # u_yy = u_zz / 4
    # Use finite differences for u_z, u_zz
    dz = z_vals[1] - z_vals[0] if len(z_vals) > 1 else 1e-3
    du_dz = np.gradient(u_vals, dz, edge_order=2)
    d2u_dz2 = np.gradient(du_dz, dz, edge_order=2)

    u_z_t = torch.tensor(du_dz.reshape(1, -1), dtype=torch.complex128, device=device)
    u_zz_t = torch.tensor(d2u_dz2.reshape(1, -1), dtype=torch.complex128, device=device)

    # Convert to y-space derivatives
    u_y = u_z_t / 2.0
    u_yy = u_zz_t / 4.0

    # Compute C2, C1, C0 in y-space
    y1 = y_t + 1.0
    y_r = -y1 ** 2 / (2.0 * rp)
    y_rr = y1 ** 3 / (2.0 * rp ** 2)

    Delta = delta(r_t, a_t, mode.M)
    Delta_r_v = delta_r(r_t, mode.M)
    V = V_of_r(r_t, a_t, omega_t, mode.m, mode.s,
               torch.tensor([[mode.lam]], dtype=torch.complex128, device=device), mode.M)

    q_r, q_rr = compute_q_terms(r_t, a_t, omega_t, basis, mode.M)

    C2 = Delta * (y_r ** 2)
    C1 = Delta * y_rr + (2.0 * Delta * q_r + (mode.s + 1.0) * Delta_r_v) * y_r
    C0 = Delta * (q_rr + q_r ** 2) + (mode.s + 1.0) * Delta_r_v * q_r + V

    residual = C2 * u_yy + C1 * u_y + C0 * u_t
    pw = torch.abs(residual) ** 2
    return float(pw.mean().item()), float(pw.max().item()), float(pw.min().item())


def main():
    parser = argparse.ArgumentParser(description="Diagnose spectral basis on patch 0")
    parser.add_argument("--points-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--N", type=int, default=80)
    parser.add_argument("--z1", type=float, default=0.625)
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    args = parser.parse_args()

    with open(args.points_json) as f:
        points = json.load(f)
    if args.labels is not None:
        points = [p for p in points if p["label"] in args.labels]
    print(f"[basis-diag] {len(points)} points, N={args.N} z1={args.z1}")

    out_dir = Path(args.output_dir)
    profiles_dir = out_dir / "basis_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu"

    summary_rows = []
    for pt in points:
        label = pt["label"]
        a_val = pt["a"]
        omega_val = pt["omega"]
        lam_val = pt.get("lambda_real", pt.get("lam"))

        print(f"\n{'='*60}")
        print(f"[basis-diag] {label}: a={a_val:.6f} omega={omega_val:.6e} lam={lam_val}")

        mode = KerrMode(M=1.0, a=a_val, omega=omega_val, ell=2, m=2, s=-2,
                        lam=complex(lam_val) if lam_val is not None else None)

        # Solve basis functions on left patch [0, z1]
        sol_down = solve_basis_domain(mode, "down", args.N, 0.0, args.z1, "left")
        sol_up = solve_basis_domain(mode, "up", args.N, 0.0, args.z1, "left")

        z_down = sol_down["z"]
        u_down = sol_down["u"]
        z_up = sol_up["z"]
        u_up = sol_up["u"]

        # Boundary checks at z=0 (infinity)
        u_down_0 = u_down[0]
        u_up_0 = u_up[0]
        du_dz_exact_down = boundary_du_exact(mode, "down", "left")
        du_dz_exact_up = boundary_du_exact(mode, "up", "left")

        # Numerical derivative at z=0
        dz0 = z_down[1] - z_down[0]
        du_dz_num_down = (u_down[1] - u_down[0]) / dz0
        du_dz_num_up = (u_up[1] - u_up[0]) / dz0

        err_val_down = abs(u_down_0 - 1.0)
        err_val_up = abs(u_up_0 - 1.0)
        err_deriv_down = abs(du_dz_num_down - du_dz_exact_down)
        err_deriv_up = abs(du_dz_num_up - du_dz_exact_up)

        # u-equation residual
        res_down_mean, res_down_max, res_down_min = compute_u_eq_residual_on_z(
            mode, z_down, u_down, "down", device=device)
        res_up_mean, res_up_max, res_up_min = compute_u_eq_residual_on_z(
            mode, z_up, u_up, "up", device=device)

        print(f"  Boundary: u_down(0)={u_down_0:.12f}  err={err_val_down:.2e}")
        print(f"            u_up(0)  ={u_up_0:.12f}  err={err_val_up:.2e}")
        print(f"            du_down/dz err={err_deriv_down:.2e}  "
              f"exact={du_dz_exact_down:.6e}")
        print(f"            du_up/dz   err={err_deriv_up:.2e}  "
              f"exact={du_dz_exact_up:.6e}")
        print(f"  u-eq residual: down mean={res_down_mean:.4e} max={res_down_max:.4e}")
        print(f"                 up   mean={res_up_mean:.4e} max={res_up_max:.4e}")

        # Save basis profile
        y_vals = 2.0 * z_down - 1.0
        np.savez(
            profiles_dir / f"point_{label}_N{args.N}.npz",
            z=z_down,
            y=y_vals,
            u_down=u_down,
            u_up=u_up,
            uz_down=sol_down.get("uz", np.zeros_like(z_down)),
            uz_up=sol_up.get("uz", np.zeros_like(z_up)),
            a=a_val,
            omega=omega_val,
            u_chart=pt.get("u", 0.0),
            v_chart=pt.get("v", 0.0),
            lambda_=lam_val,
            N=args.N,
            z1=args.z1,
        )
        print(f"  Saved: {profiles_dir / f'point_{label}_N{args.N}.npz'}")

        summary_rows.append({
            "label": label,
            "a": a_val, "omega": omega_val,
            "u_down_0": float(u_down_0.real) + 1j * float(u_down_0.imag),
            "u_up_0": float(u_up_0.real) + 1j * float(u_up_0.imag),
            "err_val_down": float(err_val_down),
            "err_val_up": float(err_val_up),
            "err_deriv_down": float(err_deriv_down),
            "err_deriv_up": float(err_deriv_up),
            "du_dz_exact_down": str(du_dz_exact_down),
            "du_dz_exact_up": str(du_dz_exact_up),
            "residual_down_mean": res_down_mean,
            "residual_up_mean": res_up_mean,
        })

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Label':12s} {'u(0) err':10s} {'du/dz err':10s} {'res_mean':12s}")
    print(f"{'='*80}")
    for r in summary_rows:
        print(f"{r['label']:12s} down:{r['err_val_down']:.2e} up:{r['err_val_up']:.2e} | "
              f"down:{r['err_deriv_down']:.2e} up:{r['err_deriv_up']:.2e} | "
              f"down:{r['residual_down_mean']:.4e} up:{r['residual_up_mean']:.4e}")


if __name__ == "__main__":
    main()
