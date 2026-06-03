#!/usr/bin/env python3
"""Value-only amplitude matching using the physical in-basis u_in=R/A_in.

This is a pure spectral diagnostic.  It solves
    u_in   = R_in / A_in      on y in [y_in_left, 1],
    u_down = R_down / A_down  on y in [-1, y_outer_right],
    u_up   = R_up / A_up      on y in [-1, y_outer_right],
then fits on y in [y1, y2]:
    R_in = B_inc R_down + B_ref R_up.

The default fit space divides by R_down:
    R_in/R_down = B_inc + B_ref (R_up/R_down),
which improves scaling for the small high-frequency reflected channel.
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

from scripts.match_spectral_pinn_overlap_amplitudes import scaled_complex_lstsq, write_csv  # noqa: E402
from utils.amplitude import A_down, A_in, A_up, r_of_z, solve_basis_domain  # noqa: E402
from utils.matlcheb import real_to_cheb  # noqa: E402
from utils.mode import KerrMode  # noqa: E402


def cheb_eval(coeff: np.ndarray, y: np.ndarray, y_left: float, y_right: float) -> np.ndarray:
    xi = np.clip((y - 0.5 * (y_left + y_right)) / (0.5 * (y_right - y_left)), -1.0, 1.0)
    theta = np.arccos(xi)
    k = np.arange(coeff.shape[0])
    return np.cos(np.outer(theta, k)) @ coeff


def solve_branch_coeff(mode: KerrMode, basis: str, n: int, y_left: float, y_right: float) -> np.ndarray:
    z_left = 0.5 * (y_left + 1.0)
    z_right = 0.5 * (y_right + 1.0)
    if basis == "in":
        sol = solve_basis_domain(mode, "in", n, z_left, 1.0, "right")
    elif basis in {"down", "up"}:
        sol = solve_basis_domain(mode, basis, n, 0.0, z_right, "left")
    else:
        raise ValueError(basis)
    # utils.amplitude.cheb_D orders nodes from left to right, while
    # utils.matlcheb.real_to_cheb/cheb_eval here use xi=+1 at y_right.
    return real_to_cheb(sol["u"][::-1])


def branch_R(mode: KerrMode, basis: str, u: np.ndarray, y: np.ndarray) -> np.ndarray:
    r = r_of_z(0.5 * (y + 1.0), mode)
    if basis == "in":
        return A_in(r, mode) * u
    if basis == "down":
        return A_down(r, mode) * u
    if basis == "up":
        return A_up(r, mode) * u
    raise ValueError(basis)


def fit_window(
    mode: KerrMode,
    *,
    n_in: int,
    n_outer: int,
    y_in_left: float,
    y_outer_right: float,
    y1: float,
    y2: float,
    n_match: int,
    fit_space: str,
) -> dict:
    c_in = solve_branch_coeff(mode, "in", n_in, y_in_left, 1.0)
    c_down = solve_branch_coeff(mode, "down", n_outer, -1.0, y_outer_right)
    c_up = solve_branch_coeff(mode, "up", n_outer, -1.0, y_outer_right)
    y = np.linspace(y1, y2, n_match)
    u_in = cheb_eval(c_in, y, y_in_left, 1.0)
    u_down = cheb_eval(c_down, y, -1.0, y_outer_right)
    u_up = cheb_eval(c_up, y, -1.0, y_outer_right)
    R_in = branch_R(mode, "in", u_in, y)
    R_down = branch_R(mode, "down", u_down, y)
    R_up = branch_R(mode, "up", u_up, y)
    if fit_space == "down-ratio":
        mat = np.stack([np.ones_like(R_down), R_up / R_down], axis=1)
        rhs = R_in / R_down
    elif fit_space == "raw":
        mat = np.stack([R_down, R_up], axis=1)
        rhs = R_in
    else:
        raise ValueError(fit_space)
    coef, cond_scaled, fit_rel_res = scaled_complex_lstsq(mat, rhs)
    B_inc, B_ref = complex(coef[0]), complex(coef[1])
    R_fit = B_inc * R_down + B_ref * R_up
    point_rel = np.abs(R_fit - R_in) / np.maximum(np.abs(R_in), 1e-300)
    return {
        "y1": y1,
        "y2": y2,
        "y_center": 0.5 * (y1 + y2),
        "n_in": n_in,
        "n_outer": n_outer,
        "y_in_left": y_in_left,
        "y_outer_right": y_outer_right,
        "fit_space": fit_space,
        "B_inc": B_inc,
        "B_ref": B_ref,
        "B_ref_over_B_inc": B_ref / B_inc,
        "raw_cond": float(np.linalg.cond(mat)),
        "scaled_cond": cond_scaled,
        "fit_rel_res": fit_rel_res,
        "point_rel_median": float(np.median(point_rel)),
        "point_rel_max": float(np.max(point_rel)),
    }


def parse_windows(raw: str, width: float) -> list[tuple[float, float]]:
    centers = [float(x.strip()) for x in raw.split(",") if x.strip()]
    return [(c - 0.5 * width, c + 0.5 * width) for c in centers]


def cdict(value: complex):
    return {"re": value.real, "im": value.imag, "abs": abs(value)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--omega", type=float, default=10.0)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--n-in", type=int, default=160)
    parser.add_argument("--n-outer", type=int, default=160)
    parser.add_argument("--y-in-left", type=float, default=-0.8)
    parser.add_argument("--y-outer-right", type=float, default=0.0)
    parser.add_argument("--window-centers", default="-0.25,-0.4,-0.6")
    parser.add_argument("--window-width", type=float, default=0.08)
    parser.add_argument("--n-match", type=int, default=121)
    parser.add_argument("--fit-space", choices=["down-ratio", "raw"], default="down-ratio")
    parser.add_argument("--gsn-Binc-re", type=float, default=24.675049712429978)
    parser.add_argument("--gsn-Binc-im", type=float, default=-5.00013326071889)
    parser.add_argument("--gsn-Bref-re", type=float, default=-6.457956617151793e-09)
    parser.add_argument("--gsn-Bref-im", type=float, default=-8.144599106575082e-09)
    parser.add_argument("--output-dir", default="outputs/inA_spectral_overlap_amplitude_match")
    args = parser.parse_args()

    mode = KerrMode(M=1.0, a=args.a, omega=args.omega, ell=args.ell, m=args.m, s=args.s)
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    gsn_Binc = complex(args.gsn_Binc_re, args.gsn_Binc_im)
    gsn_Bref = complex(args.gsn_Bref_re, args.gsn_Bref_im)
    rows = []
    for y1, y2 in parse_windows(args.window_centers, args.window_width):
        res = fit_window(
            mode,
            n_in=args.n_in,
            n_outer=args.n_outer,
            y_in_left=args.y_in_left,
            y_outer_right=args.y_outer_right,
            y1=y1,
            y2=y2,
            n_match=args.n_match,
            fit_space=args.fit_space,
        )
        row = {
            **{k: v for k, v in res.items() if not isinstance(v, complex)},
            "B_inc_re": res["B_inc"].real,
            "B_inc_im": res["B_inc"].imag,
            "B_inc_abs": abs(res["B_inc"]),
            "B_ref_re": res["B_ref"].real,
            "B_ref_im": res["B_ref"].imag,
            "B_ref_abs": abs(res["B_ref"]),
            "B_ref_over_B_inc_abs": abs(res["B_ref_over_B_inc"]),
            "gsn_B_inc_re": gsn_Binc.real,
            "gsn_B_inc_im": gsn_Binc.imag,
            "gsn_B_ref_re": gsn_Bref.real,
            "gsn_B_ref_im": gsn_Bref.imag,
            "relerr_B_inc": abs(res["B_inc"] - gsn_Binc) / max(abs(gsn_Binc), 1e-300),
            "relerr_B_ref": abs(res["B_ref"] - gsn_Bref) / max(abs(gsn_Bref), 1e-300),
            "relerr_ratio": abs(res["B_ref_over_B_inc"] - gsn_Bref / gsn_Binc) / max(abs(gsn_Bref / gsn_Binc), 1e-300),
        }
        rows.append(row)
    write_csv(run_dir / "scan.csv", rows)
    best_inc = min(rows, key=lambda row: row["relerr_B_inc"])
    best_ref = min(rows, key=lambda row: row["relerr_B_ref"])
    summary = {
        "args": vars(args),
        "run_dir": str(run_dir),
        "gsn": {"B_inc": cdict(gsn_Binc), "B_ref": cdict(gsn_Bref)},
        "best_B_inc": best_inc,
        "best_B_ref": best_ref,
        "rows": rows,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
