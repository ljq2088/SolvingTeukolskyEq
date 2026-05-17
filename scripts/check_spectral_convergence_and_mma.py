#!/usr/bin/env python3
"""Spectral convergence test on narrow near-infinity domain + MMA amplitude comparison.

1. Solve u_up/u_down on z∈[0, z1] (narrow, matching Stage-4 cache) with N=40,80,160,320.
   - Check boundary conditions, u-equation residuals (Chebyshev D matrices).
   - Check self-convergence (difference between successive N).
2. Extract B_inc/B_ref via TeukRadAmplitudeInWithAbelChecks for each N.
3. Compare with Mathematica ComputeAmplitudes.

Usage:
  python scripts/check_spectral_convergence_and_mma.py \
    --points-json outputs/spectral_calibration/patch_000_logw_v2/patch0_calibration_points.json \
    --output-dir outputs/spectral_calibration/patch_000_logw_v2
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from utils.mode import KerrMode
from utils.amplitude import (
    solve_basis_domain,
    boundary_du_exact,
    TeukRadAmplitudeInWithAbelChecks,
)
from physical_ansatz.mapping import r_plus
from physical_ansatz.prefactor import delta, delta_r, V_of_r
from physical_ansatz.u_residual import compute_q_terms


def cheb_D(N, a, b):
    """Chebyshev differentiation matrix on [a,b] with N+1 points."""
    xi = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1); c[0] = c[-1] = 2.0
    D = np.zeros((N + 1, N + 1), dtype=float)
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1) ** (i + j)) / (xi[i] - xi[j])
    D[np.diag_indices(N + 1)] = -np.sum(D, axis=1)
    D = (-2.0 / (b - a)) * D
    return D


def compute_u_eq_residual_cheb(mode, z, u, D, basis, mask=None):
    """Compute u-equation residual using Chebyshev differentiation on FULL grid.

    z, u: full Chebyshev grid arrays (length N+1)
    D: full (N+1, N+1) Chebyshev differentiation matrix
    mask: optional bool array for which points to return

    Returns (abs_res, rel_res) for masked points.
    rel_res = |C2*u_zz + C1*u_z + C0*u| / (|C2*u_zz| + |C1*u_z| + |C0*u|)
    """
    # derivatives on full grid
    uz = D @ u
    uzz = D @ uz

    rp = mode.rp
    a_t = torch.tensor([[mode.a]], dtype=torch.float64)
    omega_t = torch.tensor([[mode.omega]], dtype=torch.float64)
    lam_t = torch.tensor([[mode.lam]], dtype=torch.complex128)

    r = rp / z
    r_t = torch.tensor(r.reshape(1, -1), dtype=torch.float64)

    Delta = delta(r_t, a_t, mode.M)
    Delta_r_v = delta_r(r_t, mode.M)
    V = V_of_r(r_t, a_t, omega_t, mode.m, mode.s, lam_t, mode.M)

    q_r, q_rr = compute_q_terms(r_t, a_t, omega_t, basis, mode.M)

    # y-space conversion: y = 2z-1, y_r = dy/dr, y_rr = d²y/dr²
    y1 = 2.0 * z - 1.0 + 1.0  # = 2*z
    y_r = -y1 ** 2 / (2.0 * rp)
    y_rr = y1 ** 3 / (2.0 * rp ** 2)

    C2 = Delta * (y_r ** 2)
    C1 = Delta * y_rr + (2.0 * Delta * q_r + (mode.s + 1.0) * Delta_r_v) * y_r
    C0 = Delta * (q_rr + q_r ** 2) + (mode.s + 1.0) * Delta_r_v * q_r + V

    C2_np = C2[0, :].detach().cpu().numpy()
    C1_np = C1[0, :].detach().cpu().numpy()
    C0_np = C0[0, :].detach().cpu().numpy()

    # u_y = u_z * dz/dy = u_z / 2,  u_yy = u_zz / 4
    u_y = uz / 2.0
    u_yy = uzz / 4.0

    residual = C2_np * u_yy + C1_np * u_y + C0_np * u
    denom = np.abs(C2_np * u_yy) + np.abs(C1_np * u_y) + np.abs(C0_np * u) + 1e-30

    if mask is not None:
        residual = residual[mask]
        denom = denom[mask]

    abs_res = np.abs(residual)
    rel_res = abs_res / denom
    return abs_res, rel_res


def rel_err(val, ref, eps=1e-30):
    return abs(val - ref) / max(abs(ref), eps)


# ---- Mathematica ----
def _try_import_wolfram():
    try:
        from wolframclient.evaluation import WolframLanguageSession
        from wolframclient.language import wlexpr
        return (WolframLanguageSession, wlexpr), None
    except ImportError:
        return None, "wolframclient not installed"


def _get_mma_kernel_path():
    import os
    env = os.environ.get("WOLFRAM_KERNEL")
    if env and Path(env).exists():
        return env
    candidates = [
        "/mnt/f/mma/WolframKernel.exe",
        "/usr/local/Wolfram/Mathematica/14.0/Executables/WolframKernel",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def _wl_complex_to_py(wl_val):
    if hasattr(wl_val, 'args') and len(wl_val.args) == 2:
        return complex(float(wl_val.args[0]), float(wl_val.args[1]))
    return complex(wl_val)


def compute_mma_amplitudes(session, wlexpr, s, l, m, a, omega):
    expr = f"ComputeAmplitudes[{s}, {l}, {m}, {a:.16g}, {omega:.16g}]"
    result = session.evaluate(wlexpr(expr))
    return {
        "B_inc": _wl_complex_to_py(result["Incidence"]),
        "B_ref": _wl_complex_to_py(result["Reflection"]),
        "B_trans": _wl_complex_to_py(result["Transmission"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--z1", type=float, default=0.05,
                        help="Narrow near-inf domain endpoint (default 0.05, matching Stage-4 cache)")
    parser.add_argument("--N-list", type=int, nargs="+", default=[40, 80, 160, 320],
                        help="Chebyshev N values for convergence test")
    parser.add_argument("--sm-N-list", type=int, nargs="+", default=[40, 80, 160],
                        help="N values for S-matrix B extraction")
    parser.add_argument("--zm", type=float, default=0.30,
                        help="Match point for S-matrix")
    parser.add_argument("--wl-path", type=str, default=str(PROJECT_ROOT / "mma" / "Radial_Function.wl"))
    parser.add_argument("--kernel-path", type=str, default=None)
    args = parser.parse_args()

    with open(args.points_json) as f:
        points = json.load(f)
    print(f"Points: {len(points)}, z1={args.z1}, N_list={args.N_list}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Part 1: u_up/u_down convergence on narrow domain [0, z1]
    # ================================================================
    print(f"\n{'='*80}")
    print("Part 1: u_up/u_down spectral convergence on z∈[0, {:.3f}]".format(args.z1))
    print(f"{'='*80}")

    conv_results = []

    for pt in points:
        label = pt["label"]
        a_val = pt["a"]
        omega_val = pt["omega"]
        lam_val = pt.get("lambda_real", pt.get("lam"))
        mode = KerrMode(M=1.0, a=a_val, omega=omega_val, ell=2, m=2, s=-2,
                        lam=complex(lam_val))

        print(f"\n--- {label}: a={a_val:.4f} ω={omega_val:.6e} ---")

        prev_u = {"down": None, "up": None}
        prev_N = None

        for N in args.N_list:
            D_mat = cheb_D(N, 0.0, args.z1)
            sol_down = solve_basis_domain(mode, "down", N, 0.0, args.z1, "left")
            sol_up = solve_basis_domain(mode, "up", N, 0.0, args.z1, "left")

            z_down = sol_down["z"]
            u_down = sol_down["u"]
            z_up = sol_up["z"]
            u_up = sol_up["u"]

            # BC errors
            du_dz_exact_down = boundary_du_exact(mode, "down", "left")
            du_dz_exact_up = boundary_du_exact(mode, "up", "left")
            dz0 = z_down[1] - z_down[0]
            err_val_down = abs(u_down[0] - 1.0)
            err_val_up = abs(u_up[0] - 1.0)
            err_deriv_down = abs((u_down[1] - u_down[0]) / dz0 - du_dz_exact_down)
            err_deriv_up = abs((u_up[1] - u_up[0]) / dz0 - du_dz_exact_up)

            # u-equation residuals (Chebyshev D) — compute on full grid, mask for stats
            mask = z_down > 1e-15
            abs_down, rel_down = compute_u_eq_residual_cheb(mode, z_down, u_down,
                                                             D_mat, "down", mask=mask)
            abs_up, rel_up = compute_u_eq_residual_cheb(mode, z_up, u_up,
                                                         D_mat, "up", mask=mask)

            # Self-convergence: interpolate prev solution onto current grid, compare
            conv_down = None
            conv_up = None
            if prev_u["down"] is not None:
                # Only compare on overlapping region
                u_down_interp = np.interp(z_down, prev_z_down, prev_u["down"],
                                          left=prev_u["down"][0], right=prev_u["down"][-1])
                u_up_interp = np.interp(z_up, prev_z_up, prev_u["up"],
                                        left=prev_u["up"][0], right=prev_u["up"][-1])
                conv_down = np.max(np.abs(u_down - u_down_interp))
                conv_up = np.max(np.abs(u_up - u_up_interp))

            prev_u = {"down": u_down, "up": u_up}
            prev_z_down = z_down
            prev_z_up = z_up
            prev_N = N

            print(f"  N={N:4d}: BC_err down={err_val_down:.2e}/{err_deriv_down:.2e} "
                  f"up={err_val_up:.2e}/{err_deriv_up:.2e} | "
                  f"u_eq rel down: med={np.median(rel_down):.2e} max={np.max(rel_down):.2e} | "
                  f"up: med={np.median(rel_up):.2e} max={np.max(rel_up):.2e}", end="")
            if conv_down is not None:
                print(f" | conv down={conv_down:.2e} up={conv_up:.2e}", end="")
            print()

            conv_results.append({
                "label": label, "a": a_val, "omega": omega_val, "N": N, "z1": args.z1,
                "err_val_down": float(err_val_down), "err_val_up": float(err_val_up),
                "err_deriv_down": float(err_deriv_down), "err_deriv_up": float(err_deriv_up),
                "u_eq_rel_down_med": float(np.median(rel_down)),
                "u_eq_rel_down_max": float(np.max(rel_down)),
                "u_eq_rel_up_med": float(np.median(rel_up)),
                "u_eq_rel_up_max": float(np.max(rel_up)),
                "conv_down": float(conv_down) if conv_down is not None else None,
                "conv_up": float(conv_up) if conv_up is not None else None,
            })

    # ================================================================
    # Part 2: B_inc/B_ref convergence + MMA comparison
    # ================================================================
    print(f"\n{'='*80}")
    print("Part 2: B_inc/B_ref spectral convergence + MMA comparison")
    print(f"{'='*80}")

    # Spectral B for each N
    spectral_B = {}
    for N in args.sm_N_list:
        print(f"\n--- S-matrix N_in=N_out={N}, z_m={args.zm} ---")
        for pt in points:
            label = pt["label"]
            a_val = pt["a"]
            omega_val = pt["omega"]
            lam_val = pt.get("lambda_real", pt.get("lam"))
            mode = KerrMode(M=1.0, a=a_val, omega=omega_val, ell=2, m=2, s=-2,
                            lam=complex(lam_val))
            try:
                amp = TeukRadAmplitudeInWithAbelChecks(mode, N_in=N, N_out=N, z_m=args.zm)
                key = (label, N)
                spectral_B[key] = {
                    "B_inc": amp.B_inc,
                    "B_ref": amp.B_ref,
                    "B_trans": amp.B_trans,
                    "outer_abel": amp.outer_abel_residual,
                    "inner_abel": amp.inner_abel_residual,
                    "detS": amp.detS_residual,
                }
                print(f"  {label:12s}: B_inc=({amp.B_inc.real:.6e},{amp.B_inc.imag:.6e}) "
                      f"B_ref=({amp.B_ref.real:.6e},{amp.B_ref.imag:.6e}) "
                      f"Abel_out={amp.outer_abel_residual:.2e} Abel_in={amp.inner_abel_residual:.2e}")
            except Exception as e:
                print(f"  {label:12s}: FAILED: {e}")
                spectral_B[(label, N)] = {"error": str(e)}

    # B convergence: compare successive N
    print(f"\n--- B_inc/B_ref self-convergence ---")
    for i in range(1, len(args.sm_N_list)):
        N_prev = args.sm_N_list[i - 1]
        N_curr = args.sm_N_list[i]
        print(f"  N {N_prev} → {N_curr}:")
        for pt in points:
            label = pt["label"]
            prev = spectral_B.get((label, N_prev), {})
            curr = spectral_B.get((label, N_curr), {})
            if "error" in prev or "error" in curr:
                print(f"    {label}: skip (error)")
                continue
            d_inc = rel_err(curr["B_inc"], prev["B_inc"])
            d_ref = rel_err(curr["B_ref"], prev["B_ref"])
            print(f"    {label}: ΔB_inc={d_inc:.2e} ΔB_ref={d_ref:.2e}")

    # MMA comparison
    WolframClient, mma_err = _try_import_wolfram()
    mma_results = {}

    if WolframClient is not None:
        WolframLanguageSession, wlexpr = WolframClient
        kernel_path = args.kernel_path or _get_mma_kernel_path()
        if kernel_path:
            print(f"\n--- Mathematica comparison ---")
            session = WolframLanguageSession(kernel=kernel_path)
            session.start()
            wl_path = str(Path(args.wl_path).resolve())
            session.evaluate(wlexpr(f'Get["{wl_path}"]'))
            print(f"Loaded {wl_path}")

            for pt in points:
                label = pt["label"]
                a_val = pt["a"]
                omega_val = pt["omega"]
                try:
                    amps = compute_mma_amplitudes(session, wlexpr, -2, 2, 2, a_val, omega_val)
                    mma_results[label] = amps
                except Exception as e:
                    print(f"  {label}: MMA FAILED: {e}")
                    mma_results[label] = {"error": str(e)}

            session.stop()

            # Compare: for each N, compute rel diff vs MMA
            print(f"\n--- B vs MMA (rel diff) ---")
            header = f"{'Label':12s}"
            for N in args.sm_N_list:
                header += f" {'N=' + str(N):>24s}"
            print(header)
            print(f"{'':12s}" + "  ".join(f"{'B_inc':>10s} {'B_ref':>10s}" for _ in args.sm_N_list))

            for pt in points:
                label = pt["label"]
                mma = mma_results.get(label, {})
                if "error" in mma:
                    print(f"{label:12s}  MMA error")
                    continue
                B_inc_mma = mma["B_inc"]
                B_ref_mma = mma["B_ref"]
                row = f"{label:12s}"
                for N in args.sm_N_list:
                    sp = spectral_B.get((label, N), {})
                    if "error" in sp:
                        row += f"  {'FAIL':>10s} {'FAIL':>10s}"
                    else:
                        d_inc = rel_err(sp["B_inc"], B_inc_mma)
                        d_ref = rel_err(sp["B_ref"], B_ref_mma)
                        row += f"  {d_inc:10.2e} {d_ref:10.2e}"
                print(row)

    # ================================================================
    # Summary JSON
    # ================================================================
    summary = {
        "z1": args.z1,
        "N_list": args.N_list,
        "sm_N_list": args.sm_N_list,
        "zm": args.zm,
        "u_convergence": conv_results,
        "spectral_B": {f"{label}_N{N}": {
            "B_inc": str(v.get("B_inc", v.get("error", ""))),
            "B_ref": str(v.get("B_ref", "")),
            "outer_abel": v.get("outer_abel"),
            "inner_abel": v.get("inner_abel"),
            "detS": v.get("detS"),
        } for (label, N), v in spectral_B.items()},
        "mma_B": {label: {k: str(v) for k, v in m.items()}
                  for label, m in mma_results.items()},
    }
    out_path = out_dir / "spectral_convergence_and_mma.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
