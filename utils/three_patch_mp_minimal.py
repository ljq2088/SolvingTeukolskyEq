#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
three_patch_mp_minimal.py

A minimal three-patch high-precision spectral solver for the s=-2 Kerr
Teukolsky radial equation.

Patches
-------
  left   : [0, z1]   infinity bases  down/up
  middle : [z1, z2]  raw Teukolsky R transport
  right  : [z2, 1]   horizon bases   in/out

The in-mode amplitudes are obtained by
  1. computing the right in-solution state at z2;
  2. propagating this raw state back to z1 through the middle transport;
  3. decomposing the z1 state on the left down/up basis.

Everything in the spectral construction, ODE coefficients, matrices, and
linear solves is done with mpmath arbitrary precision. No value-only fit,
no custom middle ansatz, no diagnostics beyond convergence-relevant output.

Example
-------
  python three_patch_mp_minimal.py --omega 0.1 --lam 3.933359729602348 \
      --N-left 32 --N-mid 32 --N-right 32 --z1 0.2 --z2 0.4 --dps 80

Low-frequency convergence test
------------------------------
  python three_patch_mp_minimal.py --low-test --dps 70 --z1 0.2 --z2 0.4 \
      --N-list 20,24,28,32
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable
import mpmath as mp

try:
    from .compute_lambda import compute_lambda
except ImportError:
    from compute_lambda import compute_lambda


@dataclass(frozen=True)
class KerrModeMP:
    M: mp.mpf
    a: mp.mpf
    omega: mp.mpf
    ell: int
    m: int
    lam: mp.mpc
    s: int = -2

    @property
    def rp(self):
        return self.M + mp.sqrt(self.M * self.M - self.a * self.a)

    @property
    def rm(self):
        return self.M - mp.sqrt(self.M * self.M - self.a * self.a)

    @property
    def delta_h(self):
        return self.rp - self.rm

    @property
    def Omega_H(self):
        return self.a / (self.rp * self.rp + self.a * self.a)

    @property
    def k_hor(self):
        return self.omega - self.m * self.Omega_H

    @property
    def lambda_value(self):
        return self.lam


def resolve_lambda_value(a: float, omega: float, ell: int, m: int, s: int, lam_arg: str | None):
    if lam_arg is None:
        lam_val = compute_lambda(a, omega, ell, m, s)
        return mp.mpc(str(lam_val))
    return mp.mpc(str(lam_arg))


def cabs(z):
    return abs(complex(z))


def cfmt(z, ndig: int = 16) -> str:
    zc = complex(z)
    return f"{zc.real:+.{ndig}e} {zc.imag:+.{ndig}e}j"


def Delta(r, mode: KerrModeMP):
    return (r - mode.rp) * (r - mode.rm)


def Delta_p(r, mode: KerrModeMP):
    return 2 * r - 2 * mode.M


def K_of_r(r, mode: KerrModeMP):
    return (r * r + mode.a * mode.a) * mode.omega - mode.a * mode.m


def r_star(r, mode: KerrModeMP):
    d = mode.delta_h
    return (
        r
        + 2 * mode.rp / d * mp.log((r - mode.rp) / 2)
        - 2 * mode.rm / d * mp.log((r - mode.rm) / 2)
    )


def drstar_dr(r, mode: KerrModeMP):
    return (r * r + mode.a * mode.a) / Delta(r, mode)


def d_dr_drstar(r, mode: KerrModeMP):
    P = r * r + mode.a * mode.a
    D = Delta(r, mode)
    Dp = Delta_p(r, mode)
    return (2 * r * D - P * Dp) / (D * D)


def r_of_z(z, mode: KerrModeMP):
    return mode.rp / z


def dz_dr(z, mode: KerrModeMP):
    return -(z * z) / mode.rp


def d2z_dr2(z, mode: KerrModeMP):
    return 2 * z**3 / (mode.rp**2)


def A_down(r, mode: KerrModeMP):
    return r ** (-1) * mp.e ** (-1j * mode.omega * r_star(r, mode))


def A_up(r, mode: KerrModeMP):
    return r ** 3 * mp.e ** (+1j * mode.omega * r_star(r, mode))


def A_in(r, mode: KerrModeMP):
    return Delta(r, mode) ** 2 * mp.e ** (-1j * mode.k_hor * r_star(r, mode))


def A_out(r, mode: KerrModeMP):
    return mp.e ** (+1j * mode.k_hor * r_star(r, mode))


def q_and_qp(r, mode: KerrModeMP, basis: str):
    rst = drstar_dr(r, mode)
    rst_p = d_dr_drstar(r, mode)
    D = Delta(r, mode)
    Dp = Delta_p(r, mode)

    if basis == "down":
        q = -1 / r - 1j * mode.omega * rst
        qp = 1 / (r * r) - 1j * mode.omega * rst_p
    elif basis == "up":
        q = 3 / r + 1j * mode.omega * rst
        qp = -3 / (r * r) + 1j * mode.omega * rst_p
    elif basis == "in":
        q = 2 * Dp / D - 1j * mode.k_hor * rst
        qp = 2 * (2 * D - Dp * Dp) / (D * D) - 1j * mode.k_hor * rst_p
    elif basis == "out":
        q = 1j * mode.k_hor * rst
        qp = 1j * mode.k_hor * rst_p
    else:
        raise ValueError(f"unknown basis={basis}")
    return q, qp


def basis_factor(r, mode: KerrModeMP, basis: str):
    if basis == "down":
        return A_down(r, mode)
    if basis == "up":
        return A_up(r, mode)
    if basis == "in":
        return A_in(r, mode)
    if basis == "out":
        return A_out(r, mode)
    raise ValueError(f"unknown basis={basis}")


def boundary_du_exact(mode: KerrModeMP, basis: str, side: str):
    rp = mode.rp
    rm = mode.rm
    d = mode.delta_h
    Pp = rp * rp + mode.a * mode.a
    om = mode.omega
    lam = mode.lambda_value
    am = mode.a * mode.m
    kH = mode.k_hor

    if side == "left":
        if basis == "down":
            return -2 * (rp + rm) / rp + 1j * (2 - am * om - mp.mpf("0.5") * lam) / (om * rp)
        if basis == "up":
            return 1j * (am * om + mp.mpf("0.5") * lam) / (om * rp)
        raise ValueError("left boundary only supports down/up")

    if side == "right":
        if basis == "in":
            return (
                rp
                * (d * (lam - 4 + 1j * (6 * kH * rp + 4 * om * rp)) - 4 * am * kH * rp)
                / (d * (2j * kH * Pp - 3 * d))
            )
        if basis == "out":
            return (
                rp
                * (4 * am * kH * rp - d * (lam + 1j * (2 * kH * rp + 4 * om * rp)))
                / (d * (2j * kH * Pp - d))
            )
        raise ValueError("right boundary only supports in/out")

    raise ValueError("side must be left/right")


def V_teuk(r, mode: KerrModeMP):
    D = Delta(r, mode)
    K = K_of_r(r, mode)
    return (K * K + 4j * (r - mode.M) * K) / D - 8j * mode.omega * r - mode.lambda_value


def coeffs_for_u(z, mode: KerrModeMP, basis: str):
    r = r_of_z(z, mode)
    zr = dz_dr(z, mode)
    zrr = d2z_dr2(z, mode)
    D = Delta(r, mode)
    Dp = Delta_p(r, mode)
    V = V_teuk(r, mode)
    q, qp = q_and_qp(r, mode, basis)
    B2 = D * zr * zr
    B1 = D * (zrr + 2 * q * zr) - Dp * zr
    B0 = D * (qp + q * q) - Dp * q + V
    return B2, B1, B0


def coeffs_raw_R(z, mode: KerrModeMP):
    r = r_of_z(z, mode)
    zr = dz_dr(z, mode)
    zrr = d2z_dr2(z, mode)
    D = Delta(r, mode)
    Dp = Delta_p(r, mode)
    V = V_teuk(r, mode)
    B2 = D * zr * zr
    B1 = D * zrr - Dp * zr
    B0 = V
    return B2, B1, B0


def cheb_D_mp(N: int, a, b):
    a = mp.mpf(a)
    b = mp.mpf(b)
    xi = [mp.cos(mp.pi * i / N) for i in range(N + 1)]
    c = [mp.mpf(1) for _ in range(N + 1)]
    c[0] = c[-1] = mp.mpf(2)
    D = mp.matrix(N + 1, N + 1)
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1) ** (i + j)) / (xi[i] - xi[j])
            else:
                D[i, j] = 0
    for i in range(N + 1):
        D[i, i] = -mp.fsum(D[i, j] for j in range(N + 1) if j != i)
    z = [a + (b - a) * (1 - x) / 2 for x in xi]
    D = (-2 / (b - a)) * D
    return D, z


def matmul(A, B):
    return A * B


def vector_norm(v):
    return mp.sqrt(mp.fsum(abs(v[i]) ** 2 for i in range(len(v))))


def solve_mp(A, b):
    return mp.lu_solve(A, b)


def solve_basis_domain_mp(mode: KerrModeMP, basis: str, N: int, z_a, z_b, bc_side: str):
    D, z = cheb_D_mp(N, z_a, z_b)
    D2 = D * D
    A = mp.matrix(N + 1, N + 1)
    b = mp.matrix(N + 1, 1)
    for i in range(N + 1):
        b[i] = 0
        for j in range(N + 1):
            A[i, j] = 0

    if bc_side == "left":
        du0 = boundary_du_exact(mode, basis, "left")
        A[0, 0] = 1
        b[0] = 1
        for j in range(N + 1):
            A[1, j] = D[0, j]
        b[1] = du0
        for i in range(2, N + 1):
            B2, B1, B0 = coeffs_for_u(z[i], mode, basis)
            for j in range(N + 1):
                A[i, j] = B2 * D2[i, j] + B1 * D[i, j]
            A[i, i] += B0
    elif bc_side == "right":
        du1 = boundary_du_exact(mode, basis, "right")
        for i in range(0, N - 1):
            B2, B1, B0 = coeffs_for_u(z[i], mode, basis)
            for j in range(N + 1):
                A[i, j] = B2 * D2[i, j] + B1 * D[i, j]
            A[i, i] += B0
        for j in range(N + 1):
            A[N - 1, j] = D[N, j]
        b[N - 1] = du1
        A[N, N] = 1
        b[N] = 1
    else:
        raise ValueError("bc_side must be left/right")

    u = solve_mp(A, b)
    uz = D * u
    return {"z": z, "u": u, "uz": uz}


def basis_values_at_match_mp(mode: KerrModeMP, basis: str, sol, side: str):
    if side == "left":
        j = 0
    elif side == "right":
        j = len(sol["z"]) - 1
    else:
        raise ValueError
    z = sol["z"][j]
    u = sol["u"][j]
    uz = sol["uz"][j]
    r = r_of_z(z, mode)
    F = basis_factor(r, mode, basis)
    q, _ = q_and_qp(r, mode, basis)
    R = F * u
    Rr = F * (dz_dr(z, mode) * uz + q * u)
    return R, Rr


def edge_diagnostics_mp(mode: KerrModeMP, basis: str, sol, side: str):
    if side == "left":
        j = 0
    elif side == "right":
        j = len(sol["z"]) - 1
    else:
        raise ValueError

    z = sol["z"][j]
    r = r_of_z(z, mode)

    if basis == "raw":
        u = sol["R"][j]
        uz = sol["Rz"][j]
        R = u
        Rr = dz_dr(z, mode) * uz
    else:
        u = sol["u"][j]
        uz = sol["uz"][j]
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


def solve_middle_raw_basis_mp(mode: KerrModeMP, N: int, z_a, z_b, R_bc, Rr_bc):
    """Solve raw R on [z_a,z_b] from left state [R,R_r] at z_a."""
    D, z = cheb_D_mp(N, z_a, z_b)
    D2 = D * D
    A = mp.matrix(N + 1, N + 1)
    b = mp.matrix(N + 1, 1)
    for i in range(N + 1):
        b[i] = 0
        for j in range(N + 1):
            A[i, j] = 0

    # R(z_a)=R_bc
    A[0, 0] = 1
    b[0] = R_bc
    # R_z(z_a)=R_r/z_r
    for j in range(N + 1):
        A[1, j] = D[0, j]
    b[1] = Rr_bc / dz_dr(z[0], mode)

    for i in range(2, N + 1):
        B2, B1, B0 = coeffs_raw_R(z[i], mode)
        for j in range(N + 1):
            A[i, j] = B2 * D2[i, j] + B1 * D[i, j]
        A[i, i] += B0

    R = solve_mp(A, b)
    Rz = D * R
    return {"z": z, "R": R, "Rz": Rz}


def raw_state_at_edge(mode: KerrModeMP, sol, side: str):
    if side == "left":
        j = 0
    elif side == "right":
        j = len(sol["z"]) - 1
    else:
        raise ValueError
    z = sol["z"][j]
    R = sol["R"][j]
    Rr = dz_dr(z, mode) * sol["Rz"][j]
    return mp.matrix([R, Rr])


def middle_raw_transport_mp(mode: KerrModeMP, N: int, z1, z2):
    sol0 = solve_middle_raw_basis_mp(mode, N, z1, z2, 1, 0)
    sol1 = solve_middle_raw_basis_mp(mode, N, z1, z2, 0, 1)
    y0 = raw_state_at_edge(mode, sol0, "right")
    y1 = raw_state_at_edge(mode, sol1, "right")
    T = mp.matrix([[y0[0], y1[0]], [y0[1], y1[1]]])
    return T


def solve_2x2_mp(M, y):
    return mp.lu_solve(M, y)


def compute_three_patch_mp(mode: KerrModeMP, N_left: int, N_mid: int, N_right: int, z1, z2):
    z1 = mp.mpf(z1)
    z2 = mp.mpf(z2)
    sol_down = solve_basis_domain_mp(mode, "down", N_left, 0, z1, "left")
    sol_up = solve_basis_domain_mp(mode, "up", N_left, 0, z1, "left")
    Rd, Rrd = basis_values_at_match_mp(mode, "down", sol_down, "right")
    Ru, Rru = basis_values_at_match_mp(mode, "up", sol_up, "right")

    M_left = mp.matrix([[Rd, Ru], [Rrd, Rru]])

    sol_in = solve_basis_domain_mp(mode, "in", N_right, z2, 1, "right")
    sol_out = solve_basis_domain_mp(mode, "out", N_right, z2, 1, "right")
    Ri, Rri = basis_values_at_match_mp(mode, "in", sol_in, "left")
    Ro, Rro = basis_values_at_match_mp(mode, "out", sol_out, "left")
    y_in_z2 = mp.matrix([Ri, Rri])
    y_out_z2 = mp.matrix([Ro, Rro])

    T_mid = middle_raw_transport_mp(mode, N_mid, z1, z2)
    y_in_z1 = solve_2x2_mp(T_mid, y_in_z2)
    y_out_z1 = solve_2x2_mp(T_mid, y_out_z2)

    coef_in = solve_2x2_mp(M_left, y_in_z1)
    coef_out = solve_2x2_mp(M_left, y_out_z1)

    B_inc = coef_in[0]
    B_ref = coef_in[1]
    B_trans = mp.mpc(1)
    return {
        "B_inc": B_inc,
        "B_ref": B_ref,
        "B_trans": B_trans,
        "B_ref_over_B_inc": B_ref / B_inc,
        "B_trans_over_B_inc": B_trans / B_inc,
        "Cout_down": coef_out[0],
        "Cout_up": coef_out[1],
        "M_left": M_left,
        "y_in_z2": y_in_z2,
        "y_out_z2": y_out_z2,
        "y_in_z1": y_in_z1,
        "y_out_z1": y_out_z1,
        "coef_in": coef_in,
        "coef_out": coef_out,
        "sol_down": sol_down,
        "sol_up": sol_up,
        "sol_in": sol_in,
        "sol_out": sol_out,
        "T_mid": T_mid,
    }


def print_edge_block(title: str, rows: list[dict]):
    print(f"\n=== {title} ===")
    for d in rows:
        print(f"[{d['basis']}] side={d['side']}  z={d['z']}  r={d['r']}")
        print(f"  u   = {cfmt(d['u'])}  |.|={cabs(d['u']):.16e}")
        print(f"  u_z = {cfmt(d['uz'])}  |.|={cabs(d['uz']):.16e}")
        print(f"  R   = {cfmt(d['R'])}  |.|={cabs(d['R']):.16e}")
        print(f"  R_r = {cfmt(d['Rr'])}  |.|={cabs(d['Rr']):.16e}")


def cond_2x2_mp(M) -> mp.mpf:
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    det = a * d - b * c
    if det == 0:
        return mp.inf

    # simple induced infinity-norm condition number
    norm_M = max(abs(a) + abs(b), abs(c) + abs(d))
    inv00 = d / det
    inv01 = -b / det
    inv10 = -c / det
    inv11 = a / det
    norm_Minv = max(abs(inv00) + abs(inv01), abs(inv10) + abs(inv11))
    return norm_M * norm_Minv


def print_matrix_block(title: str, M):
    print(f"\n=== {title} ===")
    print(f"[0,0] = {cfmt(M[0,0])}  |.|={cabs(M[0,0]):.16e}")
    print(f"[0,1] = {cfmt(M[0,1])}  |.|={cabs(M[0,1]):.16e}")
    print(f"[1,0] = {cfmt(M[1,0])}  |.|={cabs(M[1,0]):.16e}")
    print(f"[1,1] = {cfmt(M[1,1])}  |.|={cabs(M[1,1]):.16e}")
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    print(f"det(T_mid) = {cfmt(det)}  |.|={cabs(det):.16e}")
    print(f"cond_inf(T_mid) = {float(cond_2x2_mp(M)):.16e}")


def print_vector_block(title: str, v):
    print(f"\n=== {title} ===")
    for i in range(len(v)):
        print(f"[{i}] = {cfmt(v[i])}  |.|={cabs(v[i]):.16e}")


def make_mode(args):
    lam_val = resolve_lambda_value(args.a, args.omega, args.ell, args.m, args.s, args.lam)
    return KerrModeMP(
        M=mp.mpf(str(args.M)),
        a=mp.mpf(str(args.a)),
        omega=mp.mpf(str(args.omega)),
        ell=args.ell,
        m=args.m,
        lam=lam_val,
        s=args.s,
    )


def run_single(args):
    mp.mp.dps = args.dps
    mode = make_mode(args)
    res = compute_three_patch_mp(mode, args.N_left, args.N_mid, args.N_right, args.z1, args.z2)
    print("=== minimal three-patch mp result ===")
    print(f"dps={args.dps}, N=({args.N_left},{args.N_mid},{args.N_right}), z1={args.z1}, z2={args.z2}")
    print(f"mode: M={args.M}, a={args.a}, s={args.s}, ell={args.ell}, m={args.m}, omega={args.omega}, lambda={mode.lambda_value}")
    print(f"B_inc              = {cfmt(res['B_inc'])}  |.|={cabs(res['B_inc']):.16e}")
    print(f"B_ref              = {cfmt(res['B_ref'])}  |.|={cabs(res['B_ref']):.16e}")
    print(f"B_ref/B_inc        = {cfmt(res['B_ref_over_B_inc'])}  |.|={cabs(res['B_ref_over_B_inc']):.16e}")
    print(f"B_trans/B_inc      = {cfmt(res['B_trans_over_B_inc'])}  |.|={cabs(res['B_trans_over_B_inc']):.16e}")
    print_matrix_block("Middle transport matrix T_mid", res["T_mid"])
    print_matrix_block("Left decomposition matrix M_left", res["M_left"])
    print_vector_block("Right in-state at z2: Y_in(z2)", res["y_in_z2"])
    print_vector_block("Right out-state at z2: Y_out(z2)", res["y_out_z2"])
    print_vector_block("Back-propagated in-state at z1: Y_in(z1)=T_mid^{-1}Y_in(z2)", res["y_in_z1"])
    print_vector_block("Back-propagated out-state at z1: Y_out(z1)=T_mid^{-1}Y_out(z2)", res["y_out_z1"])
    print_vector_block("Solved coefficient vector for in-mode [B_inc, B_ref]", res["coef_in"])
    print_vector_block("Solved coefficient vector for out-column [Cout_down, Cout_up]", res["coef_out"])

    z1 = mp.mpf(args.z1)
    z2 = mp.mpf(args.z2)
    sol_mid_0 = solve_middle_raw_basis_mp(mode, args.N_mid, z1, z2, 1, 0)
    sol_mid_1 = solve_middle_raw_basis_mp(mode, args.N_mid, z1, z2, 0, 1)

    z1_rows = [
        edge_diagnostics_mp(mode, "down", res["sol_down"], "right"),
        edge_diagnostics_mp(mode, "up", res["sol_up"], "right"),
        edge_diagnostics_mp(mode, "raw", sol_mid_0, "left"),
        edge_diagnostics_mp(mode, "raw", sol_mid_1, "left"),
    ]
    z2_rows = [
        edge_diagnostics_mp(mode, "raw", sol_mid_0, "right"),
        edge_diagnostics_mp(mode, "raw", sol_mid_1, "right"),
        edge_diagnostics_mp(mode, "in", res["sol_in"], "left"),
        edge_diagnostics_mp(mode, "out", res["sol_out"], "left"),
    ]

    print_edge_block("Basis values at z1", z1_rows)
    print_edge_block("Basis values at z2", z2_rows)


def parse_N_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run_low_test(args):
    # Lambda values supplied/used in the project conversation for a=0.1,l=m=2,s=-2.
    tests = [
        (mp.mpf("0.1"), mp.mpc("3.933359729602348")),
        (mp.mpf("0.2"), mp.mpc("3.86677")),
        (mp.mpf("0.3"), mp.mpc("3.80024")),
        (mp.mpf("0.5"), mp.mpc("3.66732")),
        (mp.mpf("0.7"), mp.mpc("3.53461")),
    ]
    N_list = parse_N_list(args.N_list)
    print("=== low-frequency convergence test: minimal three-patch mp ===")
    print(f"dps={args.dps}, z1={args.z1}, z2={args.z2}, N-list={N_list}")
    print("omega      N      |B_ref/B_inc|          rel.diff from previous")
    print("----------------------------------------------------------------")
    for omega, lam in tests:
        prev = None
        for N in N_list:
            mode = KerrModeMP(
                M=mp.mpf(str(args.M)),
                a=mp.mpf(str(args.a)),
                omega=omega,
                ell=args.ell,
                m=args.m,
                lam=lam,
                s=args.s,
            )
            res = compute_three_patch_mp(mode, N, N, N, args.z1, args.z2)
            ratio = res["B_ref_over_B_inc"]
            if prev is None:
                rd = mp.nan
                rd_s = "-"
            else:
                rd = abs(ratio - prev) / max(abs(prev), mp.mpf("1e-80"))
                rd_s = f"{float(rd):.3e}"
            print(f"{float(omega):7.3f}  {N:5d}  {abs(complex(ratio)):.16e}  {rd_s}")
            prev = ratio
        print("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--omega", type=float, default=0.1)
    parser.add_argument("--lam", type=str, default=None)
    parser.add_argument("--N-left", type=int, default=32)
    parser.add_argument("--N-mid", type=int, default=32)
    parser.add_argument("--N-right", type=int, default=32)
    parser.add_argument("--z1", type=str, default="0.2")
    parser.add_argument("--z2", type=str, default="0.4")
    parser.add_argument("--dps", type=int, default=70)
    parser.add_argument("--low-test", action="store_true")
    parser.add_argument("--N-list", type=str, default="18,22,26,30")
    args = parser.parse_args()
    mp.mp.dps = args.dps
    if args.low_test:
        run_low_test(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
