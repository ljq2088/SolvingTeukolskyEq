#!/usr/bin/env python3
"""High-precision value-only overlap matching for u_in=R/A_in."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import mpmath as mp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "utils"))
sys.path.insert(0, str(PROJECT_ROOT))

from three_patch_mp_minimal import (  # noqa: E402
    KerrModeMP,
    basis_factor,
    r_of_z,
    resolve_lambda_value,
    solve_basis_domain_mp,
)


def mp_complex_dict(value):
    return {
        "re": float(mp.re(value)),
        "im": float(mp.im(value)),
        "abs": float(abs(value)),
    }


def cheb_lobatto_weights(n: int):
    weights = []
    for j in range(n + 1):
        c = mp.mpf("0.5") if j == 0 or j == n else mp.mpf(1)
        weights.append(((-1) ** j) * c)
    return weights


def bary_eval(z_nodes, values, weights, z):
    for node, value in zip(z_nodes, values):
        if abs(z - node) <= mp.eps * 100 * max(1, abs(z)):
            return value
    numer = mp.mpc(0)
    denom = mp.mpc(0)
    for node, value, weight in zip(z_nodes, values, weights):
        term = weight / (z - node)
        numer += term * value
        denom += term
    return numer / denom


def solve_scaled_lstsq_2col(col0, col1, rhs):
    s0 = mp.sqrt(mp.fsum(abs(x) ** 2 for x in col0))
    s1 = mp.sqrt(mp.fsum(abs(x) ** 2 for x in col1))
    if s0 == 0:
        s0 = mp.mpf(1)
    if s1 == 0:
        s1 = mp.mpf(1)
    a0 = [x / s0 for x in col0]
    a1 = [x / s1 for x in col1]
    g00 = mp.fsum(mp.conj(x) * x for x in a0)
    g01 = mp.fsum(mp.conj(x) * y for x, y in zip(a0, a1))
    g10 = mp.conj(g01)
    g11 = mp.fsum(mp.conj(x) * x for x in a1)
    h0 = mp.fsum(mp.conj(x) * y for x, y in zip(a0, rhs))
    h1 = mp.fsum(mp.conj(x) * y for x, y in zip(a1, rhs))
    det = g00 * g11 - g01 * g10
    c0s = (h0 * g11 - g01 * h1) / det
    c1s = (g00 * h1 - h0 * g10) / det
    c0 = c0s / s0
    c1 = c1s / s1
    residual = [c0 * x0 + c1 * x1 - b for x0, x1, b in zip(col0, col1, rhs)]
    rel = mp.sqrt(mp.fsum(abs(x) ** 2 for x in residual)) / max(
        mp.sqrt(mp.fsum(abs(x) ** 2 for x in rhs)), mp.mpf("1e-300")
    )
    return c0, c1, rel


def branch_R_values(mode, basis, sol, y_values):
    n = len(sol["z"]) - 1
    weights = cheb_lobatto_weights(n)
    out = []
    for y in y_values:
        z = (mp.mpf(y) + 1) / 2
        u = bary_eval(sol["z"], [sol["u"][i] for i in range(n + 1)], weights, z)
        r = r_of_z(z, mode)
        out.append(basis_factor(r, mode, basis) * u)
    return out


def fit_window(args, mode, y_center):
    half = mp.mpf(args.window_width) / 2
    y1 = mp.mpf(y_center) - half
    y2 = mp.mpf(y_center) + half
    z_in_left = (mp.mpf(args.y_in_left) + 1) / 2
    z_outer_right = (mp.mpf(args.y_outer_right) + 1) / 2
    sol_in = solve_basis_domain_mp(mode, "in", args.n_in, z_in_left, 1, "right")
    sol_down = solve_basis_domain_mp(mode, "down", args.n_outer, 0, z_outer_right, "left")
    sol_up = solve_basis_domain_mp(mode, "up", args.n_outer, 0, z_outer_right, "left")
    y_values = [y1 + (y2 - y1) * i / (args.n_match - 1) for i in range(args.n_match)]
    R_in = branch_R_values(mode, "in", sol_in, y_values)
    R_down = branch_R_values(mode, "down", sol_down, y_values)
    R_up = branch_R_values(mode, "up", sol_up, y_values)
    if args.fit_space == "down-ratio":
        col0 = [mp.mpc(1) for _ in y_values]
        col1 = [ru / rd for ru, rd in zip(R_up, R_down)]
        rhs = [ri / rd for ri, rd in zip(R_in, R_down)]
    else:
        col0 = R_down
        col1 = R_up
        rhs = R_in
    B_inc, B_ref, rel = solve_scaled_lstsq_2col(col0, col1, rhs)
    point_rel = [
        abs(B_inc * rd + B_ref * ru - ri) / max(abs(ri), mp.mpf("1e-300"))
        for ri, rd, ru in zip(R_in, R_down, R_up)
    ]
    return {
        "y1": float(y1),
        "y2": float(y2),
        "y_center": float(y_center),
        "B_inc": B_inc,
        "B_ref": B_ref,
        "B_ref_over_B_inc": B_ref / B_inc,
        "fit_rel_res": rel,
        "point_rel_median": sorted(point_rel)[len(point_rel) // 2],
        "point_rel_max": max(point_rel),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="0.5")
    parser.add_argument("--omega", default="10.0")
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--lam", default="auto")
    parser.add_argument("--dps", type=int, default=80)
    parser.add_argument("--n-in", type=int, default=120)
    parser.add_argument("--n-outer", type=int, default=120)
    parser.add_argument("--y-in-left", default="-0.995")
    parser.add_argument("--y-outer-right", default="-0.9")
    parser.add_argument("--window-centers", default="-0.9,-0.95,-0.98")
    parser.add_argument("--window-width", default="0.02")
    parser.add_argument("--n-match", type=int, default=61)
    parser.add_argument("--fit-space", choices=["down-ratio", "raw"], default="down-ratio")
    parser.add_argument("--gsn-Binc-re", default="24.675049712429978")
    parser.add_argument("--gsn-Binc-im", default="-5.00013326071889")
    parser.add_argument("--gsn-Bref-re", default="-6.457956617151793e-09")
    parser.add_argument("--gsn-Bref-im", default="-8.144599106575082e-09")
    parser.add_argument("--output-dir", default="outputs/inA_mp_overlap_amplitude_match")
    args = parser.parse_args()

    mp.mp.dps = args.dps
    lam_arg = None if str(args.lam).strip().lower() == "auto" else args.lam
    lam = resolve_lambda_value(mp.mpf(args.a), mp.mpf(args.omega), args.ell, args.m, args.s, lam_arg)
    mode = KerrModeMP(
        M=mp.mpf(1),
        a=mp.mpf(args.a),
        omega=mp.mpf(args.omega),
        ell=args.ell,
        m=args.m,
        lam=lam,
        s=args.s,
    )
    gsn_Binc = mp.mpc(args.gsn_Binc_re, args.gsn_Binc_im)
    gsn_Bref = mp.mpc(args.gsn_Bref_re, args.gsn_Bref_im)
    rows = []
    for raw_center in args.window_centers.split(","):
        if not raw_center.strip():
            continue
        res = fit_window(args, mode, mp.mpf(raw_center.strip()))
        row = {
            "y1": res["y1"],
            "y2": res["y2"],
            "y_center": res["y_center"],
            "B_inc_re": float(mp.re(res["B_inc"])),
            "B_inc_im": float(mp.im(res["B_inc"])),
            "B_inc_abs": float(abs(res["B_inc"])),
            "B_ref_re": float(mp.re(res["B_ref"])),
            "B_ref_im": float(mp.im(res["B_ref"])),
            "B_ref_abs": float(abs(res["B_ref"])),
            "B_ref_over_B_inc_abs": float(abs(res["B_ref_over_B_inc"])),
            "fit_rel_res": float(res["fit_rel_res"]),
            "point_rel_median": float(res["point_rel_median"]),
            "point_rel_max": float(res["point_rel_max"]),
            "relerr_B_inc": float(abs(res["B_inc"] - gsn_Binc) / abs(gsn_Binc)),
            "relerr_B_ref": float(abs(res["B_ref"] - gsn_Bref) / abs(gsn_Bref)),
            "relerr_ratio": float(abs(res["B_ref_over_B_inc"] - gsn_Bref / gsn_Binc) / abs(gsn_Bref / gsn_Binc)),
        }
        rows.append(row)

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "scan.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "args": vars(args),
        "run_dir": str(run_dir),
        "lambda": [float(mp.re(lam)), float(mp.im(lam))],
        "gsn": {"B_inc": mp_complex_dict(gsn_Binc), "B_ref": mp_complex_dict(gsn_Bref)},
        "best_B_ref": min(rows, key=lambda row: row["relerr_B_ref"]),
        "best_B_inc": min(rows, key=lambda row: row["relerr_B_inc"]),
        "rows": rows,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
