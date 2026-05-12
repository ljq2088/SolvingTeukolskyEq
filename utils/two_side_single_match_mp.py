#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
two_side_single_match_mp.py

Two-side single-point matching for the s=-2 Kerr Teukolsky radial equation.

This script compares two constructions of the smooth basis functions:
  1. spectral (Chebyshev collocation)
  2. high-precision direct ODE integration with mpmath

On the left [0, z_match]:
  - down
  - up

On the right [z_match, 1]:
  - in

At the same matching point z_match, the script prints:
  - u, u_z, R, R_r for down/up/in
  - basis deltas between spectral and integrated solutions
  - matching matrices and solved amplitudes for both methods
"""
from __future__ import annotations

import argparse
import mpmath as mp

try:
    from .three_patch_mp_minimal import (
        KerrModeMP,
        basis_factor,
        boundary_du_exact,
        cheb_D_mp,
        coeffs_for_u,
        cond_2x2_mp,
        dz_dr,
        edge_diagnostics_mp,
        print_edge_block,
        q_and_qp,
        r_of_z,
        resolve_lambda_value,
        solve_2x2_mp,
        solve_basis_domain_mp,
    )
except ImportError:
    from three_patch_mp_minimal import (
        KerrModeMP,
        basis_factor,
        boundary_du_exact,
        cheb_D_mp,
        coeffs_for_u,
        cond_2x2_mp,
        dz_dr,
        edge_diagnostics_mp,
        print_edge_block,
        q_and_qp,
        r_of_z,
        resolve_lambda_value,
        solve_2x2_mp,
        solve_basis_domain_mp,
    )


def mp_abs(z):
    return abs(z)


def mp_fmt(z, ndig: int = 24) -> str:
    return f"{mp.nstr(mp.re(z), ndig)} {mp.nstr(mp.im(z), ndig, show_zero_exponent=True)}j"


def make_mode(args):
    lam_val = resolve_lambda_value(
        mp.mpf(args.a),
        mp.mpf(args.omega),
        args.ell,
        args.m,
        args.s,
        args.lam,
    )
    return KerrModeMP(
        M=mp.mpf(args.M),
        a=mp.mpf(args.a),
        omega=mp.mpf(args.omega),
        ell=args.ell,
        m=args.m,
        lam=lam_val,
        s=args.s,
    )


def coeff_triplet(mode: KerrModeMP, basis: str, z):
    return coeffs_for_u(z, mode, basis)


def coeff_derivs(mode: KerrModeMP, basis: str, zb):
    f2 = lambda zz: coeff_triplet(mode, basis, zz)[0]
    f1 = lambda zz: coeff_triplet(mode, basis, zz)[1]
    f0 = lambda zz: coeff_triplet(mode, basis, zz)[2]
    B2 = f2(zb)
    B1 = f1(zb)
    B0 = f0(zb)
    B2p = mp.diff(f2, zb, 1)
    B1p = mp.diff(f1, zb, 1)
    B0p = mp.diff(f0, zb, 1)
    B2pp = mp.diff(f2, zb, 2)
    B1pp = mp.diff(f1, zb, 2)
    B0pp = mp.diff(f0, zb, 2)
    return B2, B1, B0, B2p, B1p, B0p, B2pp, B1pp, B0pp


def boundary_taylor_state(mode: KerrModeMP, basis: str, side: str, eps: str, bc_order: int):
    eps = mp.mpf(eps)
    if side == "left":
        z_ref = eps / 2
        z0 = eps
        t = z0 - z_ref
        u1 = boundary_du_exact(mode, basis, "left")
        u0 = mp.mpc(1) + u1 * z_ref
    elif side == "right":
        z_ref = mp.mpf("1.0") - eps / 2
        z0 = mp.mpf("1.0") - eps
        t = z0 - z_ref
        u1 = boundary_du_exact(mode, basis, "right")
        u0 = mp.mpc(1) + u1 * (z_ref - 1)
    else:
        raise ValueError("side must be left/right")

    if bc_order <= 1:
        u_eps = u0 + u1 * t
        uz_eps = u1
        return (z0, u_eps, uz_eps, {"u0": u0, "u1": u1, "z_ref": z_ref})

    B2, B1, B0, B2p, B1p, B0p, B2pp, B1pp, B0pp = coeff_derivs(mode, basis, z_ref)
    u2 = -((B1p + B0) * u1 + B0p * u0) / (B2p + B1)

    if bc_order <= 2:
        u_eps = u0 + u1 * t + mp.mpf("0.5") * u2 * t * t
        uz_eps = u1 + u2 * t
        return (z0, u_eps, uz_eps, {"u0": u0, "u1": u1, "u2": u2, "z_ref": z_ref})

    u3 = -((B2pp + 2 * B1p + B0) * u2 + (B1pp + 2 * B0p) * u1 + B0pp * u0) / (2 * B2p + B1)
    u_eps = u0 + u1 * t + mp.mpf("0.5") * u2 * t * t + mp.mpf("1") / 6 * u3 * t**3
    uz_eps = u1 + u2 * t + mp.mpf("0.5") * u3 * t * t
    return (z0, u_eps, uz_eps, {"u0": u0, "u1": u1, "u2": u2, "u3": u3, "z_ref": z_ref})


def diag_from_u_uz(mode: KerrModeMP, basis: str, z, u, uz, side: str):
    r = r_of_z(z, mode)
    F = basis_factor(r, mode, basis)
    q, _ = q_and_qp(r, mode, basis)
    R = F * u
    Rr = F * (dz_dr(z, mode) * uz + q * u)
    return {
        "basis": basis,
        "side": side,
        "z": z,
        "r": r,
        "u": u,
        "uz": uz,
        "R": R,
        "Rr": Rr,
    }


def residual_diag(mode: KerrModeMP, basis: str, z, u, uz, uzz):
    B2, B1, B0 = coeffs_for_u(z, mode, basis)
    lhs = B2 * uzz + B1 * uz + B0 * u
    scale = mp_abs(B2 * uzz) + mp_abs(B1 * uz) + mp_abs(B0 * u) + mp.mpf("1e-80")
    return {
        "lhs": lhs,
        "abs": mp_abs(lhs),
        "rel": mp_abs(lhs) / scale,
        "B2": B2,
        "B1": B1,
        "B0": B0,
        "scale": scale,
    }


def solve_basis_domain_mp_bc_order(
    mode: KerrModeMP,
    basis: str,
    N: int,
    z_match,
    side: str,
    eps: str,
    bc_order: int,
):
    z_match = mp.mpf(z_match)
    z0, u_bc, uz_bc, jet = boundary_taylor_state(mode, basis, side, eps, bc_order)

    if side == "left":
        z_a, z_b = z0, z_match
        D, z = cheb_D_mp(N, z_a, z_b)
        D2 = D * D
        A = mp.matrix(N + 1, N + 1)
        b = mp.matrix(N + 1, 1)
        for i in range(N + 1):
            for j in range(N + 1):
                A[i, j] = mp.mpc(0)
            b[i] = mp.mpc(0)
        A[0, 0] = 1
        b[0] = u_bc
        for j in range(N + 1):
            A[1, j] = D[0, j]
        b[1] = uz_bc
        for i in range(2, N + 1):
            B2, B1, B0 = coeffs_for_u(z[i], mode, basis)
            for j in range(N + 1):
                A[i, j] = B2 * D2[i, j] + B1 * D[i, j]
            A[i, i] += B0
    elif side == "right":
        z_a, z_b = z_match, z0
        D, z = cheb_D_mp(N, z_a, z_b)
        D2 = D * D
        A = mp.matrix(N + 1, N + 1)
        b = mp.matrix(N + 1, 1)
        for i in range(N + 1):
            for j in range(N + 1):
                A[i, j] = mp.mpc(0)
            b[i] = mp.mpc(0)
        for i in range(0, N - 1):
            B2, B1, B0 = coeffs_for_u(z[i], mode, basis)
            for j in range(N + 1):
                A[i, j] = B2 * D2[i, j] + B1 * D[i, j]
            A[i, i] += B0
        for j in range(N + 1):
            A[N - 1, j] = D[N, j]
        b[N - 1] = uz_bc
        A[N, N] = 1
        b[N] = u_bc
    else:
        raise ValueError("side must be left/right")

    u = mp.lu_solve(A, b)
    uz = D * u
    uzz = D2 * u
    return {"z": z, "u": u, "uz": uz, "uzz": uzz, "jet": jet, "z_bc": z0}


def u_rhs(mode: KerrModeMP, basis: str, z, y):
    u, uz = y[0], y[1]
    B2, B1, B0 = coeffs_for_u(z, mode, basis)
    uzz = -(B1 * uz + B0 * u) / B2
    return [uz, uzz]


def rk4_step(mode: KerrModeMP, basis: str, z, y, h):
    k1 = u_rhs(mode, basis, z, y)
    y2 = [y[i] + h * k1[i] / 2 for i in range(2)]
    k2 = u_rhs(mode, basis, z + h / 2, y2)
    y3 = [y[i] + h * k2[i] / 2 for i in range(2)]
    k3 = u_rhs(mode, basis, z + h / 2, y3)
    y4 = [y[i] + h * k3[i] for i in range(2)]
    k4 = u_rhs(mode, basis, z + h, y4)
    return [
        y[i] + h * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6
        for i in range(2)
    ]


def integrate_basis_to_match(mode: KerrModeMP, basis: str, z_match, eps: str, n_steps: int):
    z_match = mp.mpf(z_match)
    eps = mp.mpf(eps)

    if basis in ("down", "up"):
        side = "left"
        z0 = eps
        du = boundary_du_exact(mode, basis, "left")
        y0 = [mp.mpc(1) + du * z0, du]
    elif basis == "in":
        side = "right"
        z0 = mp.mpf(1) - eps
        du = boundary_du_exact(mode, basis, "right")
        y0 = [mp.mpc(1) + du * (z0 - 1), du]
    else:
        raise ValueError(f"unsupported basis={basis}")

    h = (z_match - z0) / n_steps
    z = z0
    y = y0
    for _ in range(n_steps):
        y = rk4_step(mode, basis, z, y, h)
        z += h
    return diag_from_u_uz(mode, basis, z_match, y[0], y[1], "right" if basis in ("down", "up") else "left")


def solve_from_diags(diag_down, diag_up, diag_in):
    M_match = mp.matrix(
        [
            [diag_down["R"], diag_up["R"]],
            [diag_down["Rr"], diag_up["Rr"]],
        ]
    )
    y_in = mp.matrix([diag_in["R"], diag_in["Rr"]])
    r_match = diag_in["r"]
    inc_scale = r_match
    ref_scale = r_match**3
    M_match_scaled = mp.matrix(
        [
            [diag_down["R"] * inc_scale, diag_up["R"] / ref_scale],
            [diag_down["Rr"] * inc_scale, diag_up["Rr"] / ref_scale],
        ]
    )
    coef_scaled = solve_2x2_mp(M_match_scaled, y_in)
    B_inc_scaled = coef_scaled[0]
    B_ref_scaled = coef_scaled[1]
    B_inc = B_inc_scaled * inc_scale
    B_ref = B_ref_scaled / ref_scale
    return {
        "r_match": r_match,
        "inc_scale": inc_scale,
        "ref_scale": ref_scale,
        "M_match": M_match,
        "M_match_scaled": M_match_scaled,
        "y_in": y_in,
        "coef_scaled": coef_scaled,
        "B_inc_scaled": B_inc_scaled,
        "B_inc": B_inc,
        "B_ref_scaled": B_ref_scaled,
        "B_ref": B_ref,
        "B_ref_over_B_inc": B_ref / B_inc,
    }


def scale_diag(diag: dict, scale) -> dict:
    out = dict(diag)
    out["u"] = diag["u"] * scale
    out["uz"] = diag["uz"] * scale
    out["R"] = diag["R"] * scale
    out["Rr"] = diag["Rr"] * scale
    return out


def match_two_side_single_point_spectral(mode: KerrModeMP, N_left: int, N_right: int, z_match, eps: str, bc_order: int):
    z_match = mp.mpf(z_match)
    sol_down = solve_basis_domain_mp_bc_order(mode, "down", N_left, z_match, "left", eps, bc_order)
    sol_up = solve_basis_domain_mp_bc_order(mode, "up", N_left, z_match, "left", eps, bc_order)
    sol_in = solve_basis_domain_mp_bc_order(mode, "in", N_right, z_match, "right", eps, bc_order)

    diag_down = edge_diagnostics_mp(mode, "down", sol_down, "right")
    diag_up = edge_diagnostics_mp(mode, "up", sol_up, "right")
    diag_in = edge_diagnostics_mp(mode, "in", sol_in, "left")
    j_left = len(sol_down["z"]) - 1
    j_right = 0
    residuals = {
        "down": residual_diag(mode, "down", z_match, sol_down["u"][j_left], sol_down["uz"][j_left], sol_down["uzz"][j_left]),
        "up": residual_diag(mode, "up", z_match, sol_up["u"][j_left], sol_up["uz"][j_left], sol_up["uzz"][j_left]),
        "in": residual_diag(mode, "in", z_match, sol_in["u"][j_right], sol_in["uz"][j_right], sol_in["uzz"][j_right]),
    }
    out = {
        "diag_down": diag_down,
        "diag_up": diag_up,
        "diag_in": diag_in,
        "residuals": residuals,
    }
    out.update(solve_from_diags(diag_down, diag_up, diag_in))
    return out


def match_two_side_single_point_integrated(mode: KerrModeMP, z_match, eps: str, n_steps: int):
    diag_down = integrate_basis_to_match(mode, "down", z_match, eps, n_steps)
    diag_up = integrate_basis_to_match(mode, "up", z_match, eps, n_steps)
    diag_in = integrate_basis_to_match(mode, "in", z_match, eps, n_steps)
    out = {
        "diag_down": diag_down,
        "diag_up": diag_up,
        "diag_in": diag_in,
    }
    out.update(solve_from_diags(diag_down, diag_up, diag_in))
    return out


def cheb_coeffs_from_lobatto_values(vals):
    N = len(vals) - 1
    coeffs = []
    for k in range(N + 1):
        s = mp.mpc(0)
        for j, v in enumerate(vals):
            w = mp.mpf("0.5") if (j == 0 or j == N) else mp.mpf("1.0")
            s += w * v * mp.cos(mp.pi * k * j / N)
        ck = (mp.mpf(2) / N) * s
        if k == 0 or k == N:
            ck *= mp.mpf("0.5")
        coeffs.append(ck)
    return coeffs


def basis_coeff_summary(sol, tail=8):
    coeffs = cheb_coeffs_from_lobatto_values([sol["u"][i] for i in range(len(sol["u"]))])
    mags = [mp_abs(c) for c in coeffs]
    t = min(tail, len(coeffs))
    tail_mags = mags[-t:]
    return {
        "coeffs": coeffs,
        "mags": mags,
        "max_mag": max(mags),
        "tail_max": max(tail_mags),
        "tail_last": tail_mags[-1],
        "tail_mags": tail_mags,
    }


def print_matrix_block(title: str, M):
    print(f"\n=== {title} ===")
    print(f"[0,0] = {mp_fmt(M[0,0])}  |.|={mp.nstr(mp_abs(M[0,0]), 24)}")
    print(f"[0,1] = {mp_fmt(M[0,1])}  |.|={mp.nstr(mp_abs(M[0,1]), 24)}")
    print(f"[1,0] = {mp_fmt(M[1,0])}  |.|={mp.nstr(mp_abs(M[1,0]), 24)}")
    print(f"[1,1] = {mp_fmt(M[1,1])}  |.|={mp.nstr(mp_abs(M[1,1]), 24)}")
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    print(f"det = {mp_fmt(det)}  |.|={mp.nstr(mp_abs(det), 24)}")
    print(f"cond_inf = {mp.nstr(cond_2x2_mp(M), 24)}")


def print_vector_block(title: str, v):
    print(f"\n=== {title} ===")
    for i in range(len(v)):
        print(f"[{i}] = {mp_fmt(v[i])}  |.|={mp.nstr(mp_abs(v[i]), 24)}")


def print_match_result(label: str, res):
    print(f"\n##### {label} matching #####")
    print(f"\n=== scaling ===")
    print(f"r_match = {mp.nstr(res['r_match'], 24)}")
    print(f"inc_scale = r_match = {mp.nstr(res['inc_scale'], 24)}")
    print(f"ref_scale = r_match^3 = {mp.nstr(res['ref_scale'], 24)}")
    print_matrix_block("Single-point matching matrix M_match (unscaled)", res["M_match"])
    print_matrix_block(
        "Single-point matching matrix M_match_scaled (down-column * r_match, up-column / r_match^3)",
        res["M_match_scaled"],
    )
    print_vector_block("Right in-state Y_in(z_match)", res["y_in"])
    print_vector_block("Solved scaled coefficient vector [B_inc_scaled, B_ref_scaled]", res["coef_scaled"])

    print("\n=== amplitudes ===")
    print(f"B_inc_scaled= {mp_fmt(res['B_inc_scaled'])}  |.|={mp.nstr(mp_abs(res['B_inc_scaled']), 24)}")
    print(f"B_inc       = {mp_fmt(res['B_inc'])}  |.|={mp.nstr(mp_abs(res['B_inc']), 24)}")
    print(f"B_ref_scaled= {mp_fmt(res['B_ref_scaled'])}  |.|={mp.nstr(mp_abs(res['B_ref_scaled']), 24)}")
    print(f"B_ref       = {mp_fmt(res['B_ref'])}  |.|={mp.nstr(mp_abs(res['B_ref']), 24)}")
    print(
        f"B_ref/B_inc = {mp_fmt(res['B_ref_over_B_inc'])}  "
        f"|.|={mp.nstr(mp_abs(res['B_ref_over_B_inc']), 24)}"
    )


def print_coeff_scan_row(label: str, N: int, summary: dict):
    tail_str = ", ".join(mp.nstr(v, 6) for v in summary["tail_mags"])
    print(
        f"{label:>4s} N={N:3d}  "
        f"max|c|={mp.nstr(summary['max_mag'], 8):>12s}  "
        f"tail_max={mp.nstr(summary['tail_max'], 8):>12s}  "
        f"last={mp.nstr(summary['tail_last'], 8):>12s}  "
        f"tail=[{tail_str}]"
    )


def print_residual_block(residuals: dict):
    print("\n=== spectral ODE residuals at z_match ===")
    for key in ("down", "up", "in"):
        r = residuals[key]
        print(
            f"[{key}] "
            f"|lhs|={mp.nstr(r['abs'], 24)}  "
            f"rel={mp.nstr(r['rel'], 24)}  "
            f"lhs={mp_fmt(r['lhs'])}"
        )


def run_scan_spectral(args):
    mp.mp.dps = args.dps
    mode = make_mode(args)
    N_list = [int(x.strip()) for x in args.N_list.split(",") if x.strip()]
    z_match = mp.mpf(args.z_match)

    print("=== spectral coefficient scan ===")
    print(
        f"mode: a={args.a}, omega={args.omega}, ell={args.ell}, m={args.m}, "
        f"lambda={mode.lambda_value}, z_match={z_match}"
    )
    print("\n-- varying N_left for left bases down/up --")
    for N in N_list:
        sol_down = solve_basis_domain_mp_bc_order(mode, "down", N, z_match, "left", args.bc_eps, args.bc_order)
        sol_up = solve_basis_domain_mp_bc_order(mode, "up", N, z_match, "left", args.bc_eps, args.bc_order)
        print_coeff_scan_row("down", N, basis_coeff_summary(sol_down, tail=args.tail))
        print_coeff_scan_row("up", N, basis_coeff_summary(sol_up, tail=args.tail))
    print("\n-- varying N_right for right basis in --")
    for N in N_list:
        sol_in = solve_basis_domain_mp_bc_order(mode, "in", N, z_match, "right", args.bc_eps, args.bc_order)
        print_coeff_scan_row("in", N, basis_coeff_summary(sol_in, tail=args.tail))


def run_single(args):
    mp.mp.dps = args.dps
    mode = make_mode(args)
    spec = match_two_side_single_point_spectral(mode, args.N_left, args.N_right, args.z_match, args.bc_eps, args.bc_order)
    print("=== two-side single-point matching mp result ===")
    print(f"dps={args.dps}, N_left={args.N_left}, N_right={args.N_right}, z_match={args.z_match}")
    print(
        f"mode: M={args.M}, a={args.a}, s={args.s}, ell={args.ell}, m={args.m}, "
        f"omega={args.omega}, lambda={mode.lambda_value}"
    )
    print(
        f"bc_eps={args.bc_eps}, bc_order={args.bc_order}, "
        f"run_integration={args.run_integration}, ode_eps={args.ode_eps}, ode_steps={args.ode_steps}"
    )

    print_edge_block(
        "Spectral basis values at matching point z_match",
        [spec["diag_down"], spec["diag_up"], spec["diag_in"]],
    )
    print_residual_block(spec["residuals"])
    print_match_result("spectral", spec)

    if args.run_integration:
        integ = match_two_side_single_point_integrated(mode, args.z_match, args.ode_eps, args.ode_steps)
        print_edge_block(
            "Integrated basis values at matching point z_match",
            [integ["diag_down"], integ["diag_up"], integ["diag_in"]],
        )

        print("\n=== spectral vs integrated basis deltas ===")
        for key in ("down", "up", "in"):
            ds = spec[f"diag_{key}"]
            di = integ[f"diag_{key}"]
            print(
                f"[{key}] "
                f"|du|={mp.nstr(mp_abs(di['u'] - ds['u']), 24)}  "
                f"|du_z|={mp.nstr(mp_abs(di['uz'] - ds['uz']), 24)}  "
                f"|dR|={mp.nstr(mp_abs(di['R'] - ds['R']), 24)}  "
                f"|dR_r|={mp.nstr(mp_abs(di['Rr'] - ds['Rr']), 24)}"
            )
        print_match_result("integrated", integ)

    in_scale = mp.mpf(args.in_scale)
    if in_scale != 1:
        scaled_in_diag = scale_diag(spec["diag_in"], in_scale)
        scaled_spec = {
            "diag_down": spec["diag_down"],
            "diag_up": spec["diag_up"],
            "diag_in": scaled_in_diag,
        }
        scaled_spec.update(solve_from_diags(spec["diag_down"], spec["diag_up"], scaled_in_diag))
        print(f"\n##### spectral with scaled horizon BC (B_trans={mp.nstr(in_scale, 24)}) #####")
        print_match_result("spectral_scaled_in", scaled_spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=str, default="1.0")
    parser.add_argument("--a", type=str, default="0.1")
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--omega", type=str, default="0.1")
    parser.add_argument("--lam", type=str, default=None)
    parser.add_argument("--N-left", type=int, default=32)
    parser.add_argument("--N-right", type=int, default=32)
    parser.add_argument("--z-match", type=str, default="0.4")
    parser.add_argument("--dps", type=int, default=70)
    parser.add_argument("--bc-eps", type=str, default="1e-6")
    parser.add_argument("--bc-order", type=int, default=1)
    parser.add_argument("--ode-eps", type=str, default="1e-6")
    parser.add_argument("--ode-steps", type=int, default=20000)
    parser.add_argument("--in-scale", type=str, default="1.0")
    parser.add_argument("--run-integration", action="store_true")
    parser.add_argument("--scan-spectral", action="store_true")
    parser.add_argument("--N-list", type=str, default="20,28,40,60")
    parser.add_argument("--tail", type=int, default=8)
    args = parser.parse_args()
    if args.scan_spectral:
        run_scan_spectral(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
