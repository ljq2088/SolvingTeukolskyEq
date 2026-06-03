#!/usr/bin/env python3
"""Three-patch high-precision spectral value fit for high-frequency B_ref.

Patch layout in z=(y+1)/2:
  left:   [0, z_fit_right]          solve down/up bases
  middle: [z1, z2]                  propagate raw R from in-state at z2 to z1
  right:  [z2, 1]                   solve in basis u_in=R/A_in

Amplitude extraction is value-only on y in [y1, y1+fit_width]:
    R_mid/R_down = B_inc + B_ref * R_up/R_down.
"""

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
    Delta,
    Delta_p,
    basis_factor,
    basis_values_at_match_mp,
    cheb_D_mp,
    coeffs_raw_R,
    d2z_dr2,
    d_dr_drstar,
    drstar_dr,
    dz_dr,
    middle_raw_transport_mp,
    r_of_z,
    resolve_lambda_value,
    solve_2x2_mp,
    solve_basis_domain_mp,
    solve_middle_raw_basis_mp,
    V_teuk,
)


def cheb_lobatto_weights(n: int):
    return [(((-1) ** j) * (mp.mpf("0.5") if j in (0, n) else mp.mpf(1))) for j in range(n + 1)]


def bary_eval(nodes, values, weights, x):
    for node, value in zip(nodes, values):
        if abs(x - node) <= mp.eps * 100 * max(1, abs(x)):
            return value
    num = mp.mpc(0)
    den = mp.mpc(0)
    for node, value, weight in zip(nodes, values, weights):
        term = weight / (x - node)
        num += term * value
        den += term
    return num / den


def solve_scaled_lstsq_2col(col0, col1, rhs):
    s0 = mp.sqrt(mp.fsum(abs(x) ** 2 for x in col0)) or mp.mpf(1)
    s1 = mp.sqrt(mp.fsum(abs(x) ** 2 for x in col1)) or mp.mpf(1)
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


def eval_basis_R(mode, basis: str, sol, z_values):
    n = len(sol["z"]) - 1
    weights = cheb_lobatto_weights(n)
    out = []
    values = [sol["u"][i] for i in range(n + 1)]
    for z in z_values:
        u = bary_eval(sol["z"], values, weights, z)
        out.append(basis_factor(r_of_z(z, mode), mode, basis) * u)
    return out


def eval_raw_R(sol_raw, z_values):
    n = len(sol_raw["z"]) - 1
    weights = cheb_lobatto_weights(n)
    values = [sol_raw["R"][i] for i in range(n + 1)]
    return [bary_eval(sol_raw["z"], values, weights, z) for z in z_values]


def solve_middle_raw_from_right_mp(mode, N: int, z_a, z_b, R_bc, Rr_bc):
    """Solve raw R on [z_a,z_b] using right boundary state at z_b."""
    D, z = cheb_D_mp(N, z_a, z_b)
    D2 = D * D
    A = mp.matrix(N + 1, N + 1)
    b = mp.matrix(N + 1, 1)
    for i in range(N + 1):
        b[i] = 0
        for j in range(N + 1):
            A[i, j] = 0

    for i in range(0, N - 1):
        B2, B1, B0 = coeffs_raw_R(z[i], mode)
        for j in range(N + 1):
            A[i, j] = B2 * D2[i, j] + B1 * D[i, j]
        A[i, i] += B0

    for j in range(N + 1):
        A[N - 1, j] = D[N, j]
    b[N - 1] = Rr_bc / dz_dr(z[N], mode)
    A[N, N] = 1
    b[N] = R_bc

    R = mp.lu_solve(A, b)
    Rz = D * R
    return {"z": z, "R": R, "Rz": Rz}


def phase_factor(r, mode, sign=1):
    return mp.e ** (1j * sign * mode.omega * r_star_local(r, mode))


def r_star_local(r, mode):
    d = mode.delta_h
    rp = mode.rp
    rm = mode.rm
    return r + 2 * rp / d * mp.log((r - rp) / 2) - 2 * rm / d * mp.log((r - rm) / 2)


def coeffs_phase_u(z, mode, sign=1):
    """Coefficients for R=exp(sign*i omega r*) U in z coordinate."""
    r = r_of_z(z, mode)
    zr = dz_dr(z, mode)
    zrr = d2z_dr2(z, mode)
    D = Delta(r, mode)
    Dp = Delta_p(r, mode)
    V = V_teuk(r, mode)
    q = 1j * sign * mode.omega * drstar_dr(r, mode)
    qp = 1j * sign * mode.omega * d_dr_drstar(r, mode)
    B2 = D * zr * zr
    B1 = D * (zrr + 2 * q * zr) - Dp * zr
    B0 = D * (qp + q * q) - Dp * q + V
    return B2, B1, B0


def solve_middle_phase_from_right_mp(mode, N: int, z_a, z_b, R_bc, Rr_bc, sign=1):
    """Solve U on [z_a,z_b] with R=exp(sign*i omega r*) U and right R state."""
    D, z = cheb_D_mp(N, z_a, z_b)
    D2 = D * D
    A = mp.matrix(N + 1, N + 1)
    b = mp.matrix(N + 1, 1)
    for i in range(N + 1):
        b[i] = 0
        for j in range(N + 1):
            A[i, j] = 0

    for i in range(0, N - 1):
        B2, B1, B0 = coeffs_phase_u(z[i], mode, sign)
        for j in range(N + 1):
            A[i, j] = B2 * D2[i, j] + B1 * D[i, j]
        A[i, i] += B0

    rb = r_of_z(z[N], mode)
    Fb = phase_factor(rb, mode, sign)
    q_b = 1j * sign * mode.omega * drstar_dr(rb, mode)
    U_bc = R_bc / Fb
    Uz_bc = (Rr_bc / Fb - q_b * U_bc) / dz_dr(z[N], mode)
    for j in range(N + 1):
        A[N - 1, j] = D[N, j]
    b[N - 1] = Uz_bc
    A[N, N] = 1
    b[N] = U_bc

    U = mp.lu_solve(A, b)
    Uz = D * U
    R = mp.matrix(N + 1, 1)
    Rz = mp.matrix(N + 1, 1)
    for i in range(N + 1):
        ri = r_of_z(z[i], mode)
        Fi = phase_factor(ri, mode, sign)
        qi = 1j * sign * mode.omega * drstar_dr(ri, mode)
        R[i] = Fi * U[i]
        Rz[i] = Fi * (Uz[i] + qi * U[i] / dz_dr(z[i], mode))
    return {"z": z, "R": R, "Rz": Rz, "U": U, "Uz": Uz}


def fit_case(args, mode, y2: mp.mpf):
    y1 = mp.mpf(args.y1)
    y_fit_right = y1 + mp.mpf(args.fit_width)
    z1 = (y1 + 1) / 2
    z2 = (y2 + 1) / 2
    z_fit_right = (y_fit_right + 1) / 2
    if not (0 < z1 < z_fit_right <= z2 < 1):
        raise ValueError(f"Invalid patches: z1={z1}, z_fit_right={z_fit_right}, z2={z2}")

    sol_down = solve_basis_domain_mp(mode, "down", args.n_left, 0, z_fit_right, "left")
    sol_up = solve_basis_domain_mp(mode, "up", args.n_left, 0, z_fit_right, "left")

    sol_in = solve_basis_domain_mp(mode, "in", args.n_right, z2, 1, "right")
    R_z2, Rr_z2 = basis_values_at_match_mp(mode, "in", sol_in, "left")
    state_z2 = mp.matrix([R_z2, Rr_z2])

    if args.middle_method == "right-bvp":
        sol_raw = solve_middle_raw_from_right_mp(mode, args.n_mid_profile, z1, z2, state_z2[0], state_z2[1])
    elif args.middle_method == "phase-right-bvp":
        sol_raw = solve_middle_phase_from_right_mp(
            mode, args.n_mid_profile, z1, z2, state_z2[0], state_z2[1], args.phase_sign
        )
    else:
        transport = middle_raw_transport_mp(mode, args.n_mid_transport, z1, z2)
        state_z1 = solve_2x2_mp(transport, state_z2)
        sol_raw = solve_middle_raw_basis_mp(mode, args.n_mid_profile, z1, z2, state_z1[0], state_z1[1])

    z_values = [
        z1 + (z_fit_right - z1) * i / (args.n_match - 1)
        for i in range(args.n_match)
    ]
    R_mid = eval_raw_R(sol_raw, z_values)
    R_down = eval_basis_R(mode, "down", sol_down, z_values)
    R_up = eval_basis_R(mode, "up", sol_up, z_values)

    col0 = [mp.mpc(1) for _ in z_values]
    col1 = [ru / rd for ru, rd in zip(R_up, R_down)]
    rhs = [rm / rd for rm, rd in zip(R_mid, R_down)]
    B_inc, B_ref, fit_rel = solve_scaled_lstsq_2col(col0, col1, rhs)
    point_rel = [
        abs(B_inc * rd + B_ref * ru - rm) / max(abs(rm), mp.mpf("1e-300"))
        for rm, rd, ru in zip(R_mid, R_down, R_up)
    ]
    return {
        "y1": float(y1),
        "y_fit_right": float(y_fit_right),
        "y2": float(y2),
        "z1": float(z1),
        "z_fit_right": float(z_fit_right),
        "z2": float(z2),
        "B_inc": B_inc,
        "B_ref": B_ref,
        "B_ref_over_B_inc": B_ref / B_inc,
        "fit_rel_res": fit_rel,
        "point_rel_median": sorted(point_rel)[len(point_rel) // 2],
        "point_rel_max": max(point_rel),
    }


def row_from_result(res, gsn_Binc, gsn_Bref):
    return {
        "y1": res["y1"],
        "y_fit_right": res["y_fit_right"],
        "y2": res["y2"],
        "z1": res["z1"],
        "z_fit_right": res["z_fit_right"],
        "z2": res["z2"],
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="0.5")
    parser.add_argument("--omega", default="10.0")
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--lam", default="auto")
    parser.add_argument("--dps", type=int, default=100)
    parser.add_argument("--y1", default="-0.99")
    parser.add_argument("--fit-width", default="0.01")
    parser.add_argument("--y2-list", default="-0.2,0.0,0.25,0.4")
    parser.add_argument("--n-left", type=int, default=100)
    parser.add_argument("--n-right", type=int, default=100)
    parser.add_argument("--n-mid-transport", type=int, default=80)
    parser.add_argument("--n-mid-profile", type=int, default=100)
    parser.add_argument("--n-match", type=int, default=61)
    parser.add_argument("--middle-method", choices=["right-bvp", "phase-right-bvp", "transport"], default="right-bvp")
    parser.add_argument("--phase-sign", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--gsn-Binc-re", default="24.675049712429978")
    parser.add_argument("--gsn-Binc-im", default="-5.00013326071889")
    parser.add_argument("--gsn-Bref-re", default="-6.457956617151793e-09")
    parser.add_argument("--gsn-Bref-im", default="-8.144599106575082e-09")
    parser.add_argument("--output-dir", default="outputs/three_patch_raw_mp_value_fit")
    args = parser.parse_args()

    mp.mp.dps = args.dps
    lam_arg = None if str(args.lam).strip().lower() == "auto" else args.lam
    lam = resolve_lambda_value(mp.mpf(args.a), mp.mpf(args.omega), args.ell, args.m, args.s, lam_arg)
    mode = KerrModeMP(M=mp.mpf(1), a=mp.mpf(args.a), omega=mp.mpf(args.omega), ell=args.ell, m=args.m, lam=lam, s=args.s)
    gsn_Binc = mp.mpc(args.gsn_Binc_re, args.gsn_Binc_im)
    gsn_Bref = mp.mpc(args.gsn_Bref_re, args.gsn_Bref_im)
    rows = []
    for raw_y2 in args.y2_list.split(","):
        if raw_y2.strip():
            res = fit_case(args, mode, mp.mpf(raw_y2.strip()))
            rows.append(row_from_result(res, gsn_Binc, gsn_Bref))

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
        "best_B_ref": min(rows, key=lambda row: row["relerr_B_ref"]),
        "best_B_inc": min(rows, key=lambda row: row["relerr_B_inc"]),
        "rows": rows,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
