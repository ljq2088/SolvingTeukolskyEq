#!/usr/bin/env python3
"""Spectral-element marching plus far-field projection for Kerr s=-2 Rin.

This prototype is intentionally independent of pybhpt/MMA/GSN for the actual
amplitude extraction.  It uses:

1. the project Teukolsky ODE and horizon ingoing endpoint condition;
2. local Chebyshev collocation elements to march the same solution outward;
3. analytic infinity basis functions to project the resulting profile onto
   ``B_inc A_down + B_ref A_up``.

External solvers are used only when ``--compare-gsn`` is enabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.amplitude import (  # noqa: E402
    A_down,
    A_up,
    Delta,
    Delta_p,
    K_of_r,
    basis_values_at_match,
    compute_smatrix_with_abel,
    solve_basis_domain,
)
from utils.compute_lambda_usage import compute_lambda  # noqa: E402
from utils.matlcheb import cheb  # noqa: E402
from utils.mode import KerrMode  # noqa: E402


DEFAULT_GSN_PROJECT = Path("/home/ljq/code/GSN/GeneralizedSasakiNakamura.jl")


def cheb_D_r(N: int, r_a: float, r_b: float):
    Dy, y = cheb(N)
    r = 0.5 * (r_a + r_b) + 0.5 * (r_a - r_b) * y
    D1 = Dy / (0.5 * (r_a - r_b))
    return D1, D1 @ D1, r


def teuk_coeffs_r(r: np.ndarray, mode: KerrMode):
    D = Delta(r, mode)
    Dp = Delta_p(r, mode)
    K = K_of_r(r, mode)
    V = (K * K + 4j * (r - mode.M) * K) / D - 8j * mode.omega * r - mode.lambda_value
    return D.astype(np.complex128), (-Dp).astype(np.complex128), V.astype(np.complex128)


def solve_equilibrated(A: np.ndarray, b: np.ndarray, floor: float = 1.0e-300):
    row = np.linalg.norm(A, axis=1)
    row = np.where(row > floor, row, 1.0)
    Ar = A / row[:, None]
    br = b / row
    col = np.linalg.norm(Ar, axis=0)
    col = np.where(col > floor, col, 1.0)
    As = Ar / col[None, :]
    y = np.linalg.solve(As, br)
    x = y / col
    return x, float(np.linalg.cond(As))


def solve_raw_segment(mode: KerrMode, r_a: float, r_b: float, R_a: complex, Rr_a: complex, N: int):
    D1, D2, r = cheb_D_r(N, r_a, r_b)
    A = np.zeros((N + 1, N + 1), dtype=np.complex128)
    b = np.zeros(N + 1, dtype=np.complex128)
    A[0, 0] = 1.0
    b[0] = R_a
    A[1, :] = D1[0, :]
    b[1] = Rr_a
    idx = np.arange(2, N + 1)
    C2, C1, C0 = teuk_coeffs_r(r[idx], mode)
    A[idx, :] = C2[:, None] * D2[idx, :] + C1[:, None] * D1[idx, :]
    A[idx, idx] += C0
    R, cond = solve_equilibrated(A, b)
    Rr = D1 @ R
    term2 = np.zeros_like(R)
    term1 = np.zeros_like(R)
    term0 = np.zeros_like(R)
    all_idx = np.arange(1, N + 1)
    C2a, C1a, C0a = teuk_coeffs_r(r[all_idx], mode)
    term2[all_idx] = C2a * (D2 @ R)[all_idx]
    term1[all_idx] = C1a * (D1 @ R)[all_idx]
    term0[all_idx] = C0a * R[all_idx]
    den = np.maximum.reduce([np.abs(term2), np.abs(term1), np.abs(term0), np.full_like(R, 1.0e-300)])
    rel = np.abs(term2 + term1 + term0) / den
    return {
        "r": r,
        "R": R,
        "Rr": Rr,
        "cond": cond,
        "max_rel_residual": float(np.max(rel[1:])),
    }


def initial_horizon_data(mode: KerrMode, *, z_start: float, N_init: int):
    sol = solve_basis_domain(mode, "in", N_init, z_start, 1.0, "right", grid_kind="linear", domain="inner")
    R, Rr = basis_values_at_match(mode, "in", sol, "left")
    return float(mode.rp / z_start), complex(R), complex(Rr)


def initial_residual_down_data(mode: KerrMode, *, z_start: float, N_init: int, B_inc: complex):
    r0, R_in, Rr_in = initial_horizon_data(mode, z_start=z_start, N_init=N_init)
    sol_down = solve_basis_domain(mode, "down", N_init, 0.0, z_start, "left", grid_kind="linear", domain="outer")
    R_down, Rr_down = basis_values_at_match(mode, "down", sol_down, "right")
    return r0, complex(R_in - B_inc * R_down), complex(Rr_in - B_inc * Rr_down)


def march_profile(
    mode: KerrMode,
    *,
    r_start: float,
    R_start: complex,
    Rr_start: complex,
    r_max: float,
    dr: float,
    N_elem: int,
):
    r_left = float(r_start)
    R_left = complex(R_start)
    Rr_left = complex(Rr_start)
    samples_r: list[float] = []
    samples_R: list[complex] = []
    max_cond = 0.0
    max_res = 0.0
    n_elem = 0
    while r_left < r_max - 1.0e-12:
        r_right = min(r_left + dr, r_max)
        seg = solve_raw_segment(mode, r_left, r_right, R_left, Rr_left, N_elem)
        r_nodes = np.asarray(seg["r"], dtype=float)
        R_nodes = np.asarray(seg["R"], dtype=np.complex128)
        if samples_r:
            r_nodes = r_nodes[1:]
            R_nodes = R_nodes[1:]
        samples_r.extend(float(x) for x in r_nodes)
        samples_R.extend(complex(x) for x in R_nodes)
        R_left = complex(seg["R"][-1])
        Rr_left = complex(seg["Rr"][-1])
        r_left = r_right
        max_cond = max(max_cond, float(seg["cond"]))
        max_res = max(max_res, float(seg["max_rel_residual"]))
        n_elem += 1
    return {
        "r": np.asarray(samples_r, dtype=np.float64),
        "R": np.asarray(samples_R, dtype=np.complex128),
        "n_elem": n_elem,
        "max_segment_cond": max_cond,
        "max_segment_residual": max_res,
    }


def fit_farfield(mode: KerrMode, r: np.ndarray, R: np.ndarray, *, r_min: float, r_max: float, order: int, weight: str):
    mask = (r >= r_min) & (r <= r_max)
    r_fit = r[mask]
    R_fit = R[mask]
    if len(r_fit) < 4:
        raise ValueError("not enough far-field samples")
    Ad = A_down(r_fit, mode)
    Au = A_up(r_fit, mode)
    y = R_fit / Ad
    q = Au / Ad
    powers = [r_fit ** (-k) for k in range(order + 1)]
    X_down = np.column_stack(powers)
    X_up = np.column_stack([q * p for p in powers])
    X = np.column_stack([X_down, X_up])
    if weight == "none":
        w = np.ones_like(r_fit)
    elif weight == "unit-row":
        w = 1.0 / np.maximum(np.linalg.norm(X, axis=1), 1.0e-300)
    elif weight == "r-minus4":
        w = r_fit ** (-4)
    else:
        raise ValueError(weight)
    Xw = X * w[:, None]
    yw = y * w
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    residual = np.linalg.norm(X @ coef - y) / max(np.linalg.norm(y), 1.0e-300)
    return {
        "B_inc": complex(coef[0]),
        "B_ref": complex(coef[order + 1]),
        "fit_residual": float(residual),
        "fit_cond": float(np.linalg.cond(Xw)),
        "n_fit": int(len(r_fit)),
    }


def fit_farfield_fixed_binc(
    mode: KerrMode,
    r: np.ndarray,
    R: np.ndarray,
    *,
    r_min: float,
    r_max: float,
    order: int,
    weight: str,
    fixed_binc: complex,
):
    mask = (r >= r_min) & (r <= r_max)
    r_fit = r[mask]
    R_fit = R[mask]
    if len(r_fit) < 4:
        raise ValueError("not enough far-field samples")
    Ad = A_down(r_fit, mode)
    Au = A_up(r_fit, mode)
    y = R_fit / Ad - fixed_binc
    q = Au / Ad
    down_powers = [r_fit ** (-k) for k in range(1, order + 1)]
    up_powers = [r_fit ** (-k) for k in range(order + 1)]
    blocks = []
    if down_powers:
        blocks.append(np.column_stack(down_powers))
    up_start = sum(block.shape[1] for block in blocks)
    blocks.append(np.column_stack([q * p for p in up_powers]))
    X = np.column_stack(blocks)
    if weight == "none":
        w = np.ones_like(r_fit)
    elif weight == "unit-row":
        w = 1.0 / np.maximum(np.linalg.norm(X, axis=1), 1.0e-300)
    elif weight == "r-minus4":
        w = r_fit ** (-4)
    else:
        raise ValueError(weight)
    Xw = X * w[:, None]
    yw = y * w
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    residual = np.linalg.norm(X @ coef - y) / max(np.linalg.norm(y), 1.0e-300)
    return {
        "B_inc": complex(fixed_binc),
        "B_ref": complex(coef[up_start]),
        "fit_residual": float(residual),
        "fit_cond": float(np.linalg.cond(Xw)),
        "n_fit": int(len(r_fit)),
    }


def run_gsn(mode: KerrMode, *, work_dir: Path, project: Path, timeout: float):
    script = work_dir / "gsn_s2_amplitudes.jl"
    out = work_dir / "gsn_s2_amplitudes.csv"
    script.write_text(
        f"""
using GeneralizedSasakiNakamura
s = -2
l = {mode.ell}
m = {mode.m}
a = {mode.a:.17g}
omega = {mode.omega:.17g}
Rin = Teukolsky_radial(s, l, m, a, omega, IN)
open("{out}", "w") do io
    println(io, "B_inc_re,B_inc_im,B_ref_re,B_ref_im,B_trans_re,B_trans_im,lambda_re,lambda_im")
    println(io, join([
        real(Rin.incidence_amplitude),
        imag(Rin.incidence_amplitude),
        real(Rin.reflection_amplitude),
        imag(Rin.reflection_amplitude),
        real(Rin.transmission_amplitude),
        imag(Rin.transmission_amplitude),
        real(Rin.mode.lambda),
        imag(Rin.mode.lambda)
    ], ","))
end
"""
    )
    proc = subprocess.run(
        ["julia", f"--project={project}", str(script)],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        return None, proc.stderr.strip()[-500:]
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    row = rows[0]
    return {
        "B_inc": complex(float(row["B_inc_re"]), float(row["B_inc_im"])),
        "B_ref": complex(float(row["B_ref_re"]), float(row["B_ref_im"])),
        "B_trans": complex(float(row["B_trans_re"]), float(row["B_trans_im"])),
        "lambda": complex(float(row["lambda_re"]), float(row["lambda_im"])),
    }, "ok"


def relerr(value: complex, ref: complex):
    return float(abs(value - ref) / max(abs(ref), 1.0e-300))


def cdict(prefix: str, value: complex):
    return {f"{prefix}_re": value.real, f"{prefix}_im": value.imag, f"{prefix}_abs": abs(value)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--omega", type=float, default=10.0)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--z-start", type=float, default=0.7)
    parser.add_argument("--N-init", type=int, default=80)
    parser.add_argument("--N-elem", type=int, default=16)
    parser.add_argument("--dr-list", default="0.05,0.1,0.2")
    parser.add_argument("--r-max-list", default="30,50,80")
    parser.add_argument("--r-min-list", default="10,20,30")
    parser.add_argument("--order-list", default="0,1,2")
    parser.add_argument("--weight-list", default="none,unit-row,r-minus4")
    parser.add_argument("--fit-mode-list", default="free,fixed-spectral-binc")
    parser.add_argument("--march-target", choices=["in", "residual-down"], default="in")
    parser.add_argument("--spectral-binc-N", type=int, default=80)
    parser.add_argument("--spectral-binc-zm", type=float, default=0.3)
    parser.add_argument("--compare-gsn", action="store_true")
    parser.add_argument("--gsn-project", default=str(DEFAULT_GSN_PROJECT))
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--output-dir", default="outputs/spectral_marching_farfield_s2")
    args = parser.parse_args()

    mode = KerrMode(M=1.0, a=args.a, omega=args.omega, ell=args.ell, m=args.m, s=-2)
    mode = KerrMode(M=1.0, a=args.a, omega=args.omega, ell=args.ell, m=args.m, s=-2, lam=compute_lambda(args.a, args.omega, args.ell, args.m, s=-2))
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    gsn, gsn_status = (None, "skipped")
    if args.compare_gsn:
        gsn, gsn_status = run_gsn(mode, work_dir=run_dir, project=Path(args.gsn_project), timeout=args.timeout)

    spectral_binc = None
    if "fixed-spectral-binc" in args.fit_mode_list:
        sm = compute_smatrix_with_abel(
            mode,
            N_in=args.spectral_binc_N,
            N_out=args.spectral_binc_N,
            z_m=args.spectral_binc_zm,
            grid_kind="linear",
        )
        spectral_binc = complex(sm["B_inc"])
    if args.march_target == "in":
        r0, R0, Rr0 = initial_horizon_data(mode, z_start=args.z_start, N_init=args.N_init)
    elif args.march_target == "residual-down":
        if spectral_binc is None:
            sm = compute_smatrix_with_abel(
                mode,
                N_in=args.spectral_binc_N,
                N_out=args.spectral_binc_N,
                z_m=args.spectral_binc_zm,
                grid_kind="linear",
            )
            spectral_binc = complex(sm["B_inc"])
        r0, R0, Rr0 = initial_residual_down_data(
            mode, z_start=args.z_start, N_init=args.N_init, B_inc=spectral_binc
        )
    else:
        raise ValueError(args.march_target)
    rows = []
    for dr in [float(x) for x in args.dr_list.split(",") if x.strip()]:
        for r_max in [float(x) for x in args.r_max_list.split(",") if x.strip()]:
            if r_max <= r0:
                continue
            print(f"march dr={dr:g} r_max={r_max:g}", flush=True)
            profile = march_profile(mode, r_start=r0, R_start=R0, Rr_start=Rr0, r_max=r_max, dr=dr, N_elem=args.N_elem)
            np.savez_compressed(run_dir / f"profile_dr{dr:g}_rmax{r_max:g}.npz", r=profile["r"], R=profile["R"])
            for r_min in [float(x) for x in args.r_min_list.split(",") if x.strip()]:
                if not (r0 < r_min < r_max):
                    continue
                for order in [int(x) for x in args.order_list.split(",") if x.strip()]:
                    for weight in [x.strip() for x in args.weight_list.split(",") if x.strip()]:
                        for fit_mode in [x.strip() for x in args.fit_mode_list.split(",") if x.strip()]:
                            try:
                                if fit_mode == "free":
                                    fit = fit_farfield(mode, profile["r"], profile["R"], r_min=r_min, r_max=r_max, order=order, weight=weight)
                                elif fit_mode == "fixed-spectral-binc":
                                    if spectral_binc is None:
                                        raise ValueError("spectral_binc is not available")
                                    fixed_for_profile = 0.0 + 0.0j if args.march_target == "residual-down" else spectral_binc
                                    fit = fit_farfield_fixed_binc(
                                        mode,
                                        profile["r"],
                                        profile["R"],
                                        r_min=r_min,
                                        r_max=r_max,
                                        order=order,
                                        weight=weight,
                                        fixed_binc=fixed_for_profile,
                                    )
                                    if args.march_target == "residual-down":
                                        fit["B_inc"] = spectral_binc
                                else:
                                    raise ValueError(f"unknown fit_mode={fit_mode!r}")
                                row = {
                                    "a": args.a,
                                    "omega": args.omega,
                                    "ell": args.ell,
                                    "m": args.m,
                                    "s": -2,
                                    "lambda_re": mode.lambda_value.real,
                                    "lambda_im": mode.lambda_value.imag,
                                    "z_start": args.z_start,
                                    "r_start": r0,
                                    "march_target": args.march_target,
                                    "N_init": args.N_init,
                                    "N_elem": args.N_elem,
                                    "dr": dr,
                                    "r_min": r_min,
                                    "r_max": r_max,
                                    "order": order,
                                    "weight": weight,
                                    "fit_mode": fit_mode,
                                    "n_elem": profile["n_elem"],
                                    "max_segment_cond": profile["max_segment_cond"],
                                    "max_segment_residual": profile["max_segment_residual"],
                                    "fit_residual": fit["fit_residual"],
                                    "fit_cond": fit["fit_cond"],
                                    "n_fit": fit["n_fit"],
                                    "gsn_status": gsn_status,
                                }
                                row.update(cdict("B_inc_fit", fit["B_inc"]))
                                row.update(cdict("B_ref_fit", fit["B_ref"]))
                                if spectral_binc is not None:
                                    row.update(cdict("B_inc_spectral", spectral_binc))
                                if gsn is not None:
                                    row.update(cdict("B_inc_gsn", gsn["B_inc"]))
                                    row.update(cdict("B_ref_gsn", gsn["B_ref"]))
                                    row["B_inc_relerr"] = relerr(fit["B_inc"], gsn["B_inc"])
                                    row["B_ref_relerr"] = relerr(fit["B_ref"], gsn["B_ref"])
                                rows.append(row)
                            except Exception as exc:
                                rows.append({
                                    "a": args.a,
                                    "omega": args.omega,
                                    "dr": dr,
                                    "r_min": r_min,
                                    "r_max": r_max,
                                    "order": order,
                                    "weight": weight,
                                    "fit_mode": fit_mode,
                                    "error": str(exc),
                                    "gsn_status": gsn_status,
                                })

    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(run_dir / "marching_farfield_cases.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary = {"args": vars(args), "gsn_status": gsn_status, "n_cases": len(rows)}
    valid = [r for r in rows if "B_ref_relerr" in r]
    if valid:
        summary["best_joint"] = min(valid, key=lambda r: max(float(r["B_inc_relerr"]), float(r["B_ref_relerr"])))
        summary["best_B_ref"] = min(valid, key=lambda r: float(r["B_ref_relerr"]))
        summary["best_B_inc"] = min(valid, key=lambda r: float(r["B_inc_relerr"]))
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {run_dir}")
    if "best_joint" in summary:
        print(json.dumps(summary["best_joint"], indent=2))


if __name__ == "__main__":
    main()
