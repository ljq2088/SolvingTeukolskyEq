#!/usr/bin/env python3
"""Near-horizon u_in comparison: spectral vs pybhpt.

R_in(r) = u_in(r) * A_in(r)  where  A_in(r) = Delta(r)^2 * exp(-i*k_hor*r_star(r))
k_hor = omega - m*a/(2*M*r_+)

Spectral: solve u_in ODE via Chebyshev collocation (basis='in', bc_side='right')
  with BC: u_in(r_+) = 1, u'_in(r_+) from regularity condition
pybhpt:  R_in / A_in

Usage:
  python scripts/redraw_u_in_horizon_single_point.py \
    --a 0.5127 --omega 0.0198 \
    --output-dir outputs/stage1_nearinf_bottleneck_diagnosis/patch_000_logw_v2
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.mode import KerrMode
from utils.amplitude import solve_basis_domain, A_in, r_of_z
from utils.compute_lambda_usage import compute_lambda
from pybhpt_usage.compute_solution import compute_pybhpt_solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=float, default=0.5127)
    parser.add_argument("--omega", type=float, default=0.0198)
    parser.add_argument("--M", type=float, default=1.0)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--N", type=int, default=80, help="Chebyshev order")
    parser.add_argument("--z-min", type=float, default=0.3,
                        help="Left boundary of z-domain (z=r_+/r, so z<1 is outside horizon)")
    parser.add_argument("--z-max", type=float, default=1.0,
                        help="Right boundary = horizon")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/stage1_nearinf_bottleneck_diagnosis/patch_000_logw_v2")
    parser.add_argument("--pybhpt-timeout", type=float, default=30.0)
    parser.add_argument("--no-pybhpt", action="store_true")
    args = parser.parse_args()

    a_val = args.a
    omega_val = args.omega
    M = args.M
    s = args.s
    l = args.l
    m = args.m

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Compute lambda ----
    lam_val = compute_lambda(a_val, omega_val, l, m, s=s)
    mode = KerrMode(M=M, a=a_val, omega=omega_val, ell=l, m=m, s=s, lam=lam_val)

    print(f"a={a_val:.6g}, omega={omega_val:.6g}, M={M}, s={s}, l={l}, m={m}")
    print(f"lambda = {lam_val:.10f}")
    print(f"r_+ = {mode.rp:.10f}, r_- = {mode.rm:.10f}")
    print(f"k_hor = {mode.k_hor:.10f}  (= omega - m*Omega_H)")
    print(f"Omega_H = {mode.Omega_H:.10f}")

    # ---- Spectral solve for u_in near horizon ----
    print(f"\nSolving u_in ODE (Chebyshev N={args.N}, z∈[{args.z_min},{args.z_max}], bc_side='right') ...")
    sol_in = solve_basis_domain(mode, 'in', args.N, args.z_min, args.z_max, 'right')
    z_spec = sol_in['z']
    u_in_spec = sol_in['u']
    uz_in_spec = sol_in['uz']

    print(f"  z range: [{z_spec[0]:.6f}, {z_spec[-1]:.6f}] (N+1={args.N+1} points)")
    print(f"  u_in(z=1) = {u_in_spec[-1]:.10f}  (BC: should be 1)")
    print(f"  u_z_in(z=1) = {uz_in_spec[-1]:.10f}")
    print(f"  |u_in| range: [{np.min(np.abs(u_in_spec)):.6f}, {np.max(np.abs(u_in_spec)):.6f}]")

    # Map z to y (y = 2*z - 1 since z = x = r_+/r and y = 2x-1)
    y_spec = 2.0 * z_spec - 1.0
    r_spec = mode.rp / z_spec

    # Compute A_in factor for reference (skip z=1 where r_star is singular)
    A_in_vals = np.zeros_like(z_spec, dtype=np.complex128)
    r_for_ain = np.where(np.abs(z_spec - 1.0) < 1e-14, mode.rp + 1e-12, r_spec)
    A_in_vals = A_in(r_for_ain, mode)

    # ---- pybhpt ----
    u_in_pybhpt = None
    r_pybhpt = None
    y_pybhpt = None
    B_trans_est = None
    if not args.no_pybhpt:
        print(f"\nComputing pybhpt solution ...")
        try:
            # Build pybhpt r-grid that covers the spectral r-range with safe margin
            r_spec_min = r_of_z(np.max(z_spec[:-1]), mode)  # largest z = smallest r
            r_for_pybhpt = np.linspace(r_spec_min, r_spec.max(), 400)
            r_vals, R_in = compute_pybhpt_solution(a_val, omega_val, ell=l, m=m,
                                                    r_grid=r_for_pybhpt, timeout=args.pybhpt_timeout)
            # Evaluate A_in and ratio on pybhpt's own r-grid
            A_in_py = A_in(r_vals, mode)
            u_in_pybhpt_raw = R_in / A_in_py

            # Normalize to 1 at the innermost (closest to horizon) r
            B_trans_est = u_in_pybhpt_raw[0]  # closest r to horizon
            u_in_pybhpt = u_in_pybhpt_raw / B_trans_est

            r_pybhpt = r_vals
            y_pybhpt = 2.0 * (mode.rp / r_vals) - 1.0
            print(f"  pybhpt: {len(r_vals)} r-points, r∈[{r_vals[0]:.4f},{r_vals[-1]:.4f}]")
            print(f"  B_trans (est) = {B_trans_est:.6e}")
            print(f"  |u_in_pybhpt_norm| range: [{np.min(np.abs(u_in_pybhpt)):.6f}, "
                  f"{np.max(np.abs(u_in_pybhpt)):.6f}]")
        except Exception as e:
            print(f"  pybhpt FAILED: {e}")
            import traceback; traceback.print_exc()
            u_in_pybhpt = None

    # ---- Plot ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Spectral u_in on full grid (including z=1), pybhpt only on non-horizon grid
    y_spec_plot = y_spec
    u_spec_plot = u_in_spec
    r_spec_plot = r_spec
    if u_in_pybhpt is not None:
        y_py_plot = y_pybhpt
        u_py_plot = u_in_pybhpt
        r_py_plot = r_pybhpt

    # --- |u_in| vs y ---
    ax_mag = axes[0, 0]
    ax_mag.plot(y_spec_plot, np.abs(u_spec_plot), 'C0.-', label="Spectral u_in", linewidth=1.5, markersize=3)
    if u_in_pybhpt is not None:
        ax_mag.plot(y_py_plot, np.abs(u_py_plot), 'C1--', label="pybhpt / A_in (norm)", linewidth=1.5)
    ax_mag.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax_mag.set_xlabel("y")
    ax_mag.set_ylabel("|u_in|")
    ax_mag.set_title(f"|u_in| near horizon  a={a_val:.4f}, $\omega$={omega_val:.6e}")
    ax_mag.legend()
    ax_mag.grid(True, alpha=0.3)

    # --- Phase(u_in) vs y ---
    ax_phase = axes[1, 0]
    phase_spec = np.unwrap(np.angle(u_spec_plot))
    ax_phase.plot(y_spec_plot, phase_spec, 'C0.-', label="Spectral u_in", linewidth=1.5, markersize=3)
    if u_in_pybhpt is not None:
        phase_py = np.unwrap(np.angle(u_py_plot))
        ax_phase.plot(y_py_plot, phase_py, 'C1--', label="pybhpt / A_in (norm)", linewidth=1.5)
    ax_phase.set_xlabel("y")
    ax_phase.set_ylabel("Phase(u_in)")
    ax_phase.set_title(f"Phase(u_in) near horizon")
    ax_phase.legend()
    ax_phase.grid(True, alpha=0.3)

    # --- |u_in| vs r ---
    ax_mag_r = axes[0, 1]
    ax_mag_r.plot(r_spec, np.abs(u_spec_plot), 'C0.-', label="Spectral u_in", linewidth=1.5, markersize=3)
    if u_in_pybhpt is not None:
        ax_mag_r.plot(r_py_plot, np.abs(u_py_plot), 'C1--', label="pybhpt / A_in (norm)", linewidth=1.5)
    ax_mag_r.axvline(x=mode.rp, color='gray', linestyle=':', alpha=0.5, label=f"r_+={mode.rp:.3f}")
    ax_mag_r.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
    ax_mag_r.set_xlabel("r")
    ax_mag_r.set_ylabel("|u_in|")
    ax_mag_r.set_title("|u_in| vs r")
    ax_mag_r.legend()
    ax_mag_r.grid(True, alpha=0.3)

    # --- Phase(u_in) vs r ---
    ax_phase_r = axes[1, 1]
    ax_phase_r.plot(r_spec, phase_spec, 'C0.-', label="Spectral u_in", linewidth=1.5, markersize=3)
    if u_in_pybhpt is not None:
        ax_phase_r.plot(r_py_plot, phase_py, 'C1--', label="pybhpt / A_in (norm)", linewidth=1.5)
    ax_phase_r.axvline(x=mode.rp, color='gray', linestyle=':', alpha=0.5)
    ax_phase_r.set_xlabel("r")
    ax_phase_r.set_ylabel("Phase(u_in)")
    ax_phase_r.set_title("Phase(u_in) vs r")
    ax_phase_r.legend()
    ax_phase_r.grid(True, alpha=0.3)

    # --- |u_in - u_pybhpt| / |u_pybhpt| ---
    ax_err = axes[0, 2]
    if u_in_pybhpt is not None:
        # Match grids: interpolate spectral to pybhpt y-grid
        u_spec_interp = (np.interp(y_py_plot, y_spec_plot, u_spec_plot.real)
                         + 1j * np.interp(y_py_plot, y_spec_plot, u_spec_plot.imag))
        rel_err = np.abs(u_spec_interp - u_py_plot) / (np.abs(u_py_plot) + 1e-12)
        ax_err.semilogy(y_py_plot, rel_err, 'C3.-', linewidth=1.5, markersize=3)
        ax_err.axhline(y=np.median(rel_err), color='gray', linestyle=':',
                       alpha=0.7, label=f"median={np.median(rel_err):.4e}")
        ax_err.legend()
    ax_err.set_xlabel("y")
    ax_err.set_ylabel("|u_spec - u_pybhpt| / |u_pybhpt|")
    ax_err.set_title("Relative error: spectral vs pybhpt (both norm to 1)")
    ax_err.grid(True, alpha=0.3)

    # --- |A_in(r)| reference ---
    ax_ain = axes[1, 2]
    r_for_ain_plot = np.where(np.abs(z_spec - 1.0) < 1e-14, mode.rp + 1e-12, r_spec)
    A_in_plot = A_in(r_for_ain_plot, mode)
    ax_ain.semilogy(r_spec, np.abs(A_in_plot), 'C4-', linewidth=1.5)
    ax_ain.axvline(x=mode.rp, color='gray', linestyle=':', alpha=0.5)
    ax_ain.set_xlabel("r")
    ax_ain.set_ylabel("|A_in| = |Δ² exp(-i k_hor r_star)|")
    ax_ain.set_title("Horizon prefactor |A_in(r)|")
    ax_ain.grid(True, alpha=0.3)

    fig.suptitle(
        f"Near-Horizon u_in: R = u_in * Δ² * exp(-i·k_hor·r_*)\n"
        f"a={a_val:.6g}, ω={omega_val:.6g}, s={s}, ℓ={l}, m={m}, "
        f"k_hor={mode.k_hor:.6f}, N={args.N}",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    plot_path = out_dir / "u_in_horizon_single_point.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {plot_path}")

    # ---- Boundary condition derivation summary ----
    print(f"\n========================================")
    print(f"Boundary condition derivation:")
    print(f"  Factor: R(r) = u(r) * A_in(r)")
    print(f"  A_in(r) = Δ(r)² * exp(-i·k_hor·r_*(r))")
    print(f"  k_hor = ω - m·a/(2·M·r_+) = {mode.k_hor:.10f}")
    print(f"")
    print(f"  At r=r_+ (y=1, z=1):")
    print(f"    BC1: u(r_+) = 1.0  (regular at horizon)")
    print(f"    BC2: u'(r_+) from ODE regularity")
    print(f"         (coefficient of 1/Δ term must vanish as Δ→0)")
    print(f"    u'_z(r_+) = {sol_in['uz'][-1]:.10f}")
    print(f"")
    print(f"  ODE for u(z), z = r_+/r ∈ [{args.z_min}, {args.z_max}]:")
    print(f"    B2(z)·u_zz + B1(z)·u_z + B0(z)·u = 0")
    print(f"    B2 = D·(dz/dr)²")
    print(f"    B1 = D·(d²z/dr² + 2q·dz/dr) - D'·(dz/dr)")
    print(f"    B0 = D·(q'+q²) - D'·q + V")
    print(f"    q(r) = 2·Δ'/Δ - i·k_hor·dr_*/dr")
    print(f"========================================")


if __name__ == "__main__":
    main()
