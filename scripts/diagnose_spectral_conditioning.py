#!/usr/bin/env python3
"""Diagnose spectral Teukolsky amplitude solver conditioning.

This script is intentionally benchmark-oriented: it does not train anything.
It scans selected `(a, omega, N, z_m)` cases for the current Chebyshev
collocation solver in `utils/amplitude.py` and records:

  - domain matrix condition numbers for down/up/in/out bases,
  - row/column equilibrated condition numbers,
  - match-matrix conditioning from the S-matrix solve,
  - Abel and determinant residuals,
  - amplitude dynamic ranges.

The output is designed to identify why the spectral method degrades at very
low or very high frequencies before using it as a PINN/surrogate teacher.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.amplitude import (  # noqa: E402
    _domain_D,
    _resolve_grid,
    boundary_du_exact,
    coeffs_numeric,
    compute_smatrix_with_abel,
)
from utils.mode import KerrMode  # noqa: E402


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def equilibrated_cond(A: np.ndarray, floor: float = 1.0e-300) -> dict[str, float]:
    A = np.asarray(A, dtype=np.complex128)
    row_norms = np.linalg.norm(A, axis=1)
    col_norms = np.linalg.norm(A, axis=0)
    row_scale = np.where(row_norms > floor, row_norms, 1.0)
    col_scale = np.where(col_norms > floor, col_norms, 1.0)
    A_row = A / row_scale[:, None]
    A_col = A / col_scale[None, :]
    A_row_col = A_row / np.linalg.norm(A_row, axis=0).clip(min=floor)[None, :]
    return {
        "cond_raw": float(np.linalg.cond(A)),
        "cond_row_scaled": float(np.linalg.cond(A_row)),
        "cond_col_scaled": float(np.linalg.cond(A_col)),
        "cond_row_col_scaled": float(np.linalg.cond(A_row_col)),
        "row_norm_min": float(np.min(row_norms)),
        "row_norm_max": float(np.max(row_norms)),
        "col_norm_min": float(np.min(col_norms)),
        "col_norm_max": float(np.max(col_norms)),
    }


def build_basis_matrix(
    mode: KerrMode,
    basis: str,
    N: int,
    z_a: float,
    z_b: float,
    bc_side: str,
    *,
    grid_kind: str,
    domain: str,
):
    D, z = _domain_D(mode, N, z_a, z_b, grid_kind=grid_kind, domain=domain)
    D2 = D @ D
    A = np.zeros((N + 1, N + 1), dtype=np.complex128)
    b = np.zeros(N + 1, dtype=np.complex128)

    if bc_side == "left":
        du0 = boundary_du_exact(mode, basis, "left")
        A[0, 0] = 1.0
        b[0] = 1.0
        A[1, :] = D[0, :]
        b[1] = du0
        idx = np.arange(2, N + 1)
        B2, B1, B0 = coeffs_numeric(z[idx], mode, basis)
        A[idx, :] = B2[:, None] * D2[idx, :] + B1[:, None] * D[idx, :]
        A[idx, idx] += B0
    elif bc_side == "right":
        idx = np.arange(0, N - 1)
        B2, B1, B0 = coeffs_numeric(z[idx], mode, basis)
        A[idx, :] = B2[:, None] * D2[idx, :] + B1[:, None] * D[idx, :]
        A[idx, idx] += B0
        du1 = boundary_du_exact(mode, basis, "right")
        A[-2, :] = D[-1, :]
        b[-2] = du1
        A[-1, -1] = 1.0
        b[-1] = 1.0
    else:
        raise ValueError(f"Unknown bc_side={bc_side}")

    return A, b, z


def complex_to_pair(value: complex) -> dict[str, float]:
    return {"re": float(np.real(value)), "im": float(np.imag(value)), "abs": float(abs(value))}


def run_case(
    a: float,
    omega: float,
    N: int,
    z_m: float | None,
    grid_kind: str,
    omega_mp_cut: float,
    mp_dps_loww: int,
):
    mode = KerrMode(M=1.0, a=float(a), omega=float(omega), ell=2, m=2, s=-2)
    resolved_z_m, resolved_grid = _resolve_grid(mode, z_m, grid_kind)
    lam = mode.lambda_value
    row: dict[str, object] = {
        "a": float(a),
        "omega": float(omega),
        "lambda_re": float(np.real(lam)),
        "lambda_im": float(np.imag(lam)),
        "N": int(N),
        "N_in": int(N),
        "N_out": int(N),
        "z_m": float(resolved_z_m),
        "requested_z_m": None if z_m is None else float(z_m),
        "grid_kind": resolved_grid,
        "k_hor": float(mode.k_hor),
        "r_plus": float(mode.rp),
        "delta_h": float(mode.delta_h),
    }

    basis_specs = [
        ("down", N, 0.0, resolved_z_m, "left", "outer"),
        ("up", N, 0.0, resolved_z_m, "left", "outer"),
        ("in", N, resolved_z_m, 1.0, "right", "inner"),
        ("out", N, resolved_z_m, 1.0, "right", "inner"),
    ]
    for basis, n_basis, z_a, z_b, bc_side, domain in basis_specs:
        A, b, _ = build_basis_matrix(
            mode,
            basis,
            n_basis,
            z_a,
            z_b,
            bc_side,
            grid_kind=resolved_grid,
            domain=domain,
        )
        cond = equilibrated_cond(A)
        for key, value in cond.items():
            row[f"{basis}_{key}"] = value
        row[f"{basis}_rhs_norm"] = float(np.linalg.norm(b))

    sm = compute_smatrix_with_abel(
        mode,
        N_in=N,
        N_out=N,
        z_m=resolved_z_m,
        grid_kind=resolved_grid,
        omega_mp_cut=omega_mp_cut,
        mp_dps_loww=mp_dps_loww,
    )
    for key in [
        "outer_abel_residual",
        "inner_abel_residual",
        "detS_residual",
        "solve_in_relres",
        "solve_out_relres",
        "solve_in_cond_raw",
        "solve_out_cond_raw",
        "solve_in_cond_scaled",
        "solve_out_cond_scaled",
        "use_mp_backend",
    ]:
        row[key] = sm[key]

    B_inc = complex(sm["B_inc"])
    B_ref = complex(sm["B_ref"])
    row["B_inc_re"] = float(B_inc.real)
    row["B_inc_im"] = float(B_inc.imag)
    row["B_inc_abs"] = float(abs(B_inc))
    row["B_ref_re"] = float(B_ref.real)
    row["B_ref_im"] = float(B_ref.imag)
    row["B_ref_abs"] = float(abs(B_ref))
    row["amp_abs_ratio_inc_over_ref"] = float(abs(B_inc) / max(abs(B_ref), 1.0e-300))
    row["amp_abs_ratio_ref_over_inc"] = float(abs(B_ref) / max(abs(B_inc), 1.0e-300))

    max_domain_cond = max(
        float(row[f"{basis}_cond_raw"]) for basis in ["down", "up", "in", "out"]
    )
    max_domain_cond_eq = max(
        float(row[f"{basis}_cond_row_col_scaled"]) for basis in ["down", "up", "in", "out"]
    )
    row["max_domain_cond_raw"] = max_domain_cond
    row["max_domain_cond_row_col_scaled"] = max_domain_cond_eq
    row["max_invariant_residual"] = max(
        float(row["outer_abel_residual"]),
        float(row["inner_abel_residual"]),
        float(row["detS_residual"]),
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-list", default="0.0,0.5,0.9,0.99")
    parser.add_argument("--omega-list", default="1e-4,1e-3,1e-2,0.1,1.0,10.0")
    parser.add_argument("--N-list", default="60,100,160")
    parser.add_argument("--z-m-list", default="0.2,0.3,0.5")
    parser.add_argument("--grid-kind", choices=["linear", "anmr", "auto"], default="linear")
    parser.add_argument("--omega-mp-cut", type=float, default=1.0e-2)
    parser.add_argument("--mp-dps-loww", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/spectral_conditioning_diagnostics")
    args = parser.parse_args()

    a_values = parse_float_list(args.a_list)
    omega_values = parse_float_list(args.omega_list)
    N_values = parse_int_list(args.N_list)
    z_m_values = [None] if args.z_m_list.strip().lower() == "auto" else parse_float_list(args.z_m_list)

    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    total = len(a_values) * len(omega_values) * len(N_values) * len(z_m_values)
    count = 0
    for a in a_values:
        for omega in omega_values:
            for N in N_values:
                for z_m in z_m_values:
                    count += 1
                    z_m_label = "auto" if z_m is None else f"{z_m:g}"
                    print(f"[{count}/{total}] a={a:g} omega={omega:g} N={N} z_m={z_m_label} grid={args.grid_kind}", flush=True)
                    try:
                        rows.append(
                            run_case(
                                a=a,
                                omega=omega,
                                N=N,
                                z_m=z_m,
                                grid_kind=args.grid_kind,
                                omega_mp_cut=args.omega_mp_cut,
                                mp_dps_loww=args.mp_dps_loww,
                            )
                        )
                    except Exception as exc:
                        failures.append({
                            "a": a,
                            "omega": omega,
                            "N": N,
                            "z_m": z_m,
                            "error": str(exc),
                        })
                        print(f"  FAILED: {exc}", flush=True)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_dir / "spectral_conditioning_cases.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "args": vars(args),
        "n_cases": len(rows),
        "n_failures": len(failures),
        "failures": failures,
    }
    if rows:
        finite_rows = rows
        summary["overall"] = {
            "max_domain_cond_raw": float(max(r["max_domain_cond_raw"] for r in finite_rows)),
            "max_domain_cond_row_col_scaled": float(max(r["max_domain_cond_row_col_scaled"] for r in finite_rows)),
            "max_invariant_residual": float(max(r["max_invariant_residual"] for r in finite_rows)),
            "max_B_inc_abs": float(max(r["B_inc_abs"] for r in finite_rows)),
            "min_B_inc_abs": float(min(r["B_inc_abs"] for r in finite_rows)),
            "max_B_ref_abs": float(max(r["B_ref_abs"] for r in finite_rows)),
            "min_B_ref_abs": float(min(r["B_ref_abs"] for r in finite_rows)),
        }
        summary["worst_by_invariant_residual"] = max(
            finite_rows, key=lambda r: r["max_invariant_residual"]
        )
        summary["worst_by_domain_cond"] = max(
            finite_rows, key=lambda r: r["max_domain_cond_raw"]
        )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "README.md", "w") as f:
        f.write("# Spectral conditioning diagnostics\n\n")
        f.write(f"- Cases: {len(rows)}\n")
        f.write(f"- Failures: {len(failures)}\n")
        if "overall" in summary:
            f.write(f"- Max raw domain cond: `{summary['overall']['max_domain_cond_raw']:.3e}`\n")
            f.write(f"- Max invariant residual: `{summary['overall']['max_invariant_residual']:.3e}`\n")
        f.write("\nSee `spectral_conditioning_cases.csv` and `summary.json`.\n")

    print(f"Saved diagnostics to {out_dir}")
    print(json.dumps(summary.get("overall", {}), indent=2))


if __name__ == "__main__":
    main()
