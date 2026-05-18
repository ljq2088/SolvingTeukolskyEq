#!/usr/bin/env python3
"""Redraw R_in/P comparison for a single (a, omega) point from patch 0.

R_in/P = (B_ref * r^3 * exp(i*omega*r_star) * u_up
        + B_inc * r^(-1) * exp(-i*omega*r_star) * u_down) / P(r)

Three sources:
  - MMA+Spectral: spectral u_up/u_down (Chebyshev N=80, z∈[0,0.05]) × MMA B
  - pybhpt: numerical ODE reference
  - MMA (pure): Mathematica's own R_in/P if available

Usage:
  python scripts/redraw_R_over_P_single_point.py \
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
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from physical_ansatz.mapping import r_plus, r_minus
from physical_ansatz.prefactor import Leaver_prefactors
from physical_ansatz.u_basis import A_up, A_down
from utils.mode import KerrMode
from utils.amplitude import solve_basis_domain
from utils.compute_lambda_usage import compute_lambda
from pybhpt_usage.compute_solution import compute_pybhpt_solution
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr


def Leaver_P(r, a, omega, m=2, M=1.0, s=-2):
    rp = r_plus(a, M)
    rm = r_minus(a, M)
    sigma_p = (2.0 * omega * rp - m * a) / (rp - rm)
    pp = -s - 1j * sigma_p
    pm = -1 - s + 2j * omega + 1j * sigma_p
    return ((r - rp) ** pp) * ((r - rm) ** pm) * torch.exp(1j * omega * r)


def _wl_complex_to_py(wl_val):
    if hasattr(wl_val, 'args') and len(wl_val.args) == 2:
        return complex(float(wl_val.args[0]), float(wl_val.args[1]))
    return complex(wl_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=float, default=0.5127)
    parser.add_argument("--omega", type=float, default=0.0198)
    parser.add_argument("--M", type=float, default=1.0)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--N-out", type=int, default=80)
    parser.add_argument("--z-b", type=float, default=0.05)
    parser.add_argument("--y-min", type=float, default=-0.999)
    parser.add_argument("--y-max", type=float, default=-0.95)
    parser.add_argument("--n-y", type=int, default=128)
    parser.add_argument("--output-dir", type=str,
                        default="outputs/stage1_nearinf_bottleneck_diagnosis/patch_000_logw_v2")
    parser.add_argument("--mma-wl", type=str, default="mma/Radial_Function.wl")
    parser.add_argument("--pybhpt-timeout", type=float, default=30.0)
    parser.add_argument("--no-mma", action="store_true")
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

    print(f"Point: a={a_val:.6g}, omega={omega_val:.6g}, M={M}, s={s}, l={l}, m={m}")

    # ---- Compute lambda ----
    lam_val = compute_lambda(a_val, omega_val, l, m, s=s)
    print(f"lambda = {lam_val:.10f}")

    # ---- Build y-grid ----
    y_grid = np.linspace(args.y_min, args.y_max, args.n_y)
    z_grid = (y_grid + 1.0) / 2.0
    print(f"y: [{y_grid[0]:.6f}, {y_grid[-1]:.6f}], z: [{z_grid[0]:.6f}, {z_grid[-1]:.6f}]")

    # ---- Spectral solve for u_up, u_down ----
    print(f"Solving spectral basis (N={args.N_out}, z∈[0,{args.z_b}]) ...")
    mode = KerrMode(M=M, a=a_val, omega=omega_val, ell=l, m=m, s=s, lam=lam_val)

    sol_down = solve_basis_domain(mode, "down", args.N_out, 0.0, args.z_b, "left")
    sol_up = solve_basis_domain(mode, "up", args.N_out, 0.0, args.z_b, "left")

    # Interpolate to y-grid (z-grid)
    u_down_spec = (np.interp(z_grid, sol_down["z"], sol_down["u"].real)
                   + 1j * np.interp(z_grid, sol_down["z"], sol_down["u"].imag))
    u_up_spec = (np.interp(z_grid, sol_up["z"], sol_up["u"].real)
                 + 1j * np.interp(z_grid, sol_up["z"], sol_up["u"].imag))

    print(f"  u_up(inf) = {u_up_spec[0]:.10f}  (should be ~1)")
    print(f"  u_down(inf) = {u_down_spec[-1]:.10f}  (should be ~1)")
    print(f"  u_up range: [{np.min(np.abs(u_up_spec)):.4f}, {np.max(np.abs(u_up_spec)):.4f}]")
    print(f"  u_down range: [{np.min(np.abs(u_down_spec)):.4f}, {np.max(np.abs(u_down_spec)):.4f}]")

    # ---- Compute r_grid for R_in/P ----
    rp = float(1.0 + np.sqrt(1.0 - a_val * a_val))
    x_grid = (y_grid + 1.0) / 2.0
    r_grid = rp / np.clip(x_grid, 1e-12, None)

    # ---- MMA B amplitudes ----
    B_inc_mma = None
    B_ref_mma = None
    if not args.no_mma:
        kernel_path = "/mnt/f/mma/WolframKernel.exe"
        if not Path(kernel_path).exists():
            kernel_path = "/usr/local/Wolfram/Mathematica/14.0/Executables/WolframKernel"
        print(f"Mathematica kernel: {kernel_path}")
        session = WolframLanguageSession(kernel=kernel_path)
        session.start()
        wl_path = Path(args.mma_wl).resolve()
        session.evaluate(wlexpr(f'Get["{wl_path}"]'))
        print(f"  Loaded {wl_path}")

        expr = f"ComputeAmplitudes[{s}, {l}, {m}, {a_val:.16g}, {omega_val:.16g}]"
        print(f"  Calling: {expr}")
        result = session.evaluate(wlexpr(expr))
        try:
            B_inc_mma = _wl_complex_to_py(result["Incidence"])
            B_ref_mma = _wl_complex_to_py(result["Reflection"])
            print(f"  B_inc = {B_inc_mma:.6e}")
            print(f"  B_ref = {B_ref_mma:.6e}")
        except (TypeError, KeyError, IndexError) as e:
            print(f"  MMA parse error: {e}, raw={str(result)[:200]}")
            B_inc_mma = B_ref_mma = None
        session.stop()

    # ---- Compute R_in/P = (B_ref*A_up*u_up + B_inc*A_down*u_down) / P ----
    a_t = torch.tensor([a_val], dtype=torch.float64)
    omega_t = torch.tensor([omega_val], dtype=torch.float64)
    r_t = torch.tensor(r_grid, dtype=torch.float64)
    y_t = torch.tensor(y_grid, dtype=torch.float64)

    rp_t = r_plus(a_t, M)
    rm_t = r_minus(a_t, M)

    # P factor
    P_np, _, _ = Leaver_prefactors(r_t, a_t, omega_t, m=m, M=M, s=s, rp=rp_t, rm=rm_t)
    P = P_np.numpy()

    # MMA+Spectral R_in/P
    R_mma_spectral = None
    if B_ref_mma is not None:
        with torch.no_grad():
            A_u = A_up(r_t, a_t, omega_t, M).to(dtype=torch.complex128)
            A_d = A_down(r_t, a_t, omega_t, M).to(dtype=torch.complex128)
            B_ref_c = torch.tensor([[B_ref_mma]], dtype=torch.complex128)
            B_inc_c = torch.tensor([[B_inc_mma]], dtype=torch.complex128)
            u_up_t = torch.tensor(u_up_spec.reshape(1, -1), dtype=torch.complex128)
            u_down_t = torch.tensor(u_down_spec.reshape(1, -1), dtype=torch.complex128)
            R_combo = (B_ref_c * A_u * u_up_t + B_inc_c * A_d * u_down_t)
            R_mma_spectral = (R_combo / P_np.to(dtype=torch.complex128)).squeeze(0).numpy()
        print(f"  |R_mma_spectral/P| range: [{np.min(np.abs(R_mma_spectral)):.4e}, "
              f"{np.max(np.abs(R_mma_spectral)):.4e}]")

    # ---- pybhpt R_in/P ----
    R_pybhpt = None
    if not args.no_pybhpt:
        print(f"Computing pybhpt solution ...")
        try:
            r_sorted = np.sort(r_grid)
            r_vals, R_in = compute_pybhpt_solution(a_val, omega_val, ell=l, m=m,
                                                    r_grid=r_sorted, timeout=args.pybhpt_timeout)
            R_in_interp = (np.interp(r_grid, r_vals, R_in.real)
                           + 1j * np.interp(r_grid, r_vals, R_in.imag))
            R_pybhpt = R_in_interp / P
            print(f"  pybhpt: {len(r_vals)} r-points")
            print(f"  |R_pybhpt/P| range: [{np.min(np.abs(R_pybhpt)):.4e}, "
                  f"{np.max(np.abs(R_pybhpt)):.4e}]")
        except Exception as e:
            print(f"  pybhpt FAILED: {e}")
            R_pybhpt = None

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- |R_in/P| ---
    ax_abs = axes[0, 0]
    if R_mma_spectral is not None:
        ax_abs.semilogy(y_grid, np.abs(R_mma_spectral), 'C0-', label="MMA+Spectral", linewidth=1.5)
    if R_pybhpt is not None:
        ax_abs.semilogy(y_grid, np.abs(R_pybhpt), 'C1--', label="pybhpt (ODE)", linewidth=1.5)
    ax_abs.set_xlabel("y")
    ax_abs.set_ylabel("|R_in/P|")
    ax_abs.set_title(f"|R_in/P|  a={a_val:.4f}, $\omega$={omega_val:.6e}")
    ax_abs.legend()
    ax_abs.grid(True, alpha=0.3)

    # --- Phase(R_in/P) ---
    ax_phase = axes[1, 0]
    if R_mma_spectral is not None:
        phase_ms = np.unwrap(np.angle(R_mma_spectral))
        ax_phase.plot(y_grid, phase_ms, 'C0-', label="MMA+Spectral", linewidth=1.5)
    if R_pybhpt is not None:
        phase_py = np.unwrap(np.angle(R_pybhpt))
        ax_phase.plot(y_grid, phase_py, 'C1--', label="pybhpt (ODE)", linewidth=1.5)
    ax_phase.set_xlabel("y")
    ax_phase.set_ylabel("Phase(R_in/P)")
    ax_phase.set_title(f"Phase(R_in/P)  a={a_val:.4f}, $\omega$={omega_val:.6e}")
    ax_phase.legend()
    ax_phase.grid(True, alpha=0.3)

    # --- Relative error vs pybhpt ---
    ax_err = axes[0, 1]
    if R_mma_spectral is not None and R_pybhpt is not None:
        rel_err = np.abs(R_mma_spectral - R_pybhpt) / (np.abs(R_pybhpt) + 1e-12)
        ax_err.semilogy(y_grid, rel_err, 'C3-', linewidth=1.5)
        ax_err.axhline(y=np.median(rel_err), color='gray', linestyle=':', alpha=0.7,
                       label=f"median={np.median(rel_err):.4e}")
        ax_err.legend()
    ax_err.set_xlabel("y")
    ax_err.set_ylabel("|MMA+Spectral - pybhpt| / |pybhpt|")
    ax_err.set_title("Relative error: MMA+Spectral vs pybhpt")
    ax_err.grid(True, alpha=0.3)

    # --- Spectral u_up, u_down ---
    ax_u = axes[1, 1]
    ax_u.plot(y_grid, np.abs(u_up_spec), 'C0-', label="|u_up| spectral", linewidth=1.5)
    ax_u.plot(y_grid, np.abs(u_down_spec), 'C1-', label="|u_down| spectral", linewidth=1.5)
    ax_u.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax_u.set_xlabel("y")
    ax_u.set_ylabel("|u|")
    ax_u.set_title("Spectral u_up, u_down (N=80, z∈[0,0.05])")
    ax_u.legend()
    ax_u.grid(True, alpha=0.3)

    fig.suptitle(f"R_in/P Comparison near y=-1: MMA+Spectral vs pybhpt\n"
                 f"a={a_val:.6g}, $\omega$={omega_val:.6g}, s={s}, $\ell$={l}, m={m}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    plot_path = out_dir / "R_over_P_single_point.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {plot_path}")

    # ---- P-factor expression ----
    rp_val = float(1.0 + np.sqrt(1.0 - a_val * a_val))
    rm_val = float(1.0 - np.sqrt(1.0 - a_val * a_val))
    sigma_p = (2.0 * omega_val * rp_val - m * a_val) / (rp_val - rm_val)
    p_plus = -s - 1j * sigma_p
    p_minus = -1 - s + 2j * omega_val + 1j * sigma_p

    print(f"\n--- P-factor for (a={a_val:.6g}, omega={omega_val:.6g}) ---")
    print(f"  r_plus  = {rp_val:.10f}")
    print(f"  r_minus = {rm_val:.10f}")
    print(f"  sigma_p = {sigma_p:.10f}")
    print(f"  p_plus  = {p_plus:.10f}")
    print(f"  p_minus = {p_minus:.10f}")
    print(f"  P(r) = (r - r_plus)^(p_plus) * (r - r_minus)^(p_minus) * exp(i*omega*r)")
    print(f"\n  General formula:")
    print(f"  P(r) = (r - r_+)^(p_+) * (r - r_-)^(p_-) * exp(iωr)")
    print(f"  p_+ = -s - i*σ_+,  p_- = -1 - s + 2iω + i*σ_+")
    print(f"  σ_+ = (2ω*r_+ - m*a) / (r_+ - r_-)")
    print(f"  r_± = M ± sqrt(M² - a²)")
    print(f"  For s=-2, m=2:  p_+ = 2 - i*σ_+,  p_- = 1 + 2iω + i*σ_+")


if __name__ == "__main__":
    main()
