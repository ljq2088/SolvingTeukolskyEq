#!/usr/bin/env python3
"""Endpoint-regular spectral matching for Kerr scalar (s=0) radial Teukolsky.

This is an exploratory pure-spectral prototype inspired by ``GF_adaptive_match``:

  * compact coordinate ``z = r_+/r``;
  * two Chebyshev domains on the full interval ``z in [0, 1]``;
  * leading asymptotic factors are extracted at infinity/horizon;
  * endpoint derivatives are imposed from the degenerate transformed ODE with
    ``u(endpoint)=1``;
  * 2x2 matching at an interior point.

It targets the scalar ``s=0`` case first because the radial equation is simpler
and Mathematica's Teukolsky package can provide an independent amplitude
benchmark via ``ComputeAmplitudes[0,l,m,a,omega]``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.compute_lambda_usage import compute_lambda  # noqa: E402
from utils.matlcheb import cheb  # noqa: E402


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def Delta(r, rp, rm):
    return (r - rp) * (r - rm)


def Delta_p(r):
    return 2.0 * r - 2.0


def K_of_r(r, a, m, omega):
    return (r * r + a * a) * omega - a * m


def r_star(r, a, rp, rm):
    delta_h = rp - rm
    return r + 2.0 * rp / delta_h * np.log((r - rp) / 2.0) - 2.0 * rm / delta_h * np.log((r - rm) / 2.0)


def drstar_dr(r, a, rp, rm):
    return (r * r + a * a) / Delta(r, rp, rm)


def d_dr_drstar(r, a, rp, rm):
    D = Delta(r, rp, rm)
    Dp = Delta_p(r)
    P = r * r + a * a
    return (2.0 * r * D - P * Dp) / (D * D)


def z_derivatives(z, rp):
    zr = -(z * z) / rp
    zrr = 2.0 * z**3 / (rp * rp)
    return zr, zrr


def domain_D(N: int, z_a: float, z_b: float):
    Dy, y = cheb(N)
    z = 0.5 * (z_a + z_b) + 0.5 * (z_a - z_b) * y
    D1 = Dy / (0.5 * (z_a - z_b))
    D2 = D1 @ D1
    return D2, D1, z


def domain_D_sinh(N: int, z_a: float, z_b: float, kappa: float):
    if kappa <= 1.0e-14:
        return domain_D(N, z_a, z_b)
    Dy, y = cheb(N)
    t = 0.5 * (1.0 - y)
    sinh_k = math.sinh(kappa)
    z = z_a + (z_b - z_a) * np.sinh(kappa * t) / sinh_k
    dz_dy = -(z_b - z_a) * kappa * np.cosh(kappa * t) / (2.0 * sinh_k)
    D1 = (1.0 / dz_dy)[:, None] * Dy
    D2 = D1 @ D1
    return D2, D1, z


def resolve_domain_D(N: int, z_a: float, z_b: float, *, omega: float, grid_kind: str, domain: str):
    if grid_kind == "auto":
        grid_kind = "anmr" if abs(float(omega)) < 1.0e-1 else "linear"
    if grid_kind == "linear":
        return domain_D(N, z_a, z_b)
    if grid_kind != "anmr":
        raise ValueError(f"unknown grid_kind={grid_kind!r}")
    kappa = abs(math.log(max(abs(float(omega)), 1.0e-300)))
    if domain == "inner":
        kappa *= 0.5
    return domain_D_sinh(N, z_a, z_b, kappa)


def adaptive_match_z(omega: float, rp: float) -> float:
    r_match = 3.0 + max(abs(float(omega)), 1.0e-300) ** (-0.5)
    r_match = max(r_match, rp * (1.0 + 1.0e-8))
    return float(np.clip(rp / r_match, 1.0e-6, 0.85))


def solve_equilibrated_2x2(M: np.ndarray, y: np.ndarray, floor: float = 1.0e-300):
    M = np.asarray(M, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128)
    row = np.linalg.norm(M, axis=1)
    row = np.where(row > floor, row, 1.0)
    Mr = M / row[:, None]
    yr = y / row
    col = np.linalg.norm(Mr, axis=0)
    col = np.where(col > floor, col, 1.0)
    Ms = Mr / col[None, :]
    xs = np.linalg.solve(Ms, yr)
    x = xs / col
    relres = np.linalg.norm(M @ x - y) / max(np.linalg.norm(y), floor)
    Mc = M / np.where(np.linalg.norm(M, axis=0) > floor, np.linalg.norm(M, axis=0), 1.0)[None, :]
    return x, {
        "relres": float(relres),
        "cond_raw": float(np.linalg.cond(M)),
        "cond_col_scaled": float(np.linalg.cond(Mc)),
        "cond_row_col_scaled": float(np.linalg.cond(Ms)),
        "row_norm_0": float(row[0]),
        "row_norm_1": float(row[1]),
        "col_norm_0": float(col[0]),
        "col_norm_1": float(col[1]),
    }


def basis_q_qp(r, *, a, m, omega, rp, rm, spin: int, basis: str):
    rst = drstar_dr(r, a, rp, rm)
    rst_p = d_dr_drstar(r, a, rp, rm)
    D = Delta(r, rp, rm)
    Dp = Delta_p(r)
    if basis == "down":
        q = -1.0 / r - 1j * omega * rst
        qp = 1.0 / (r * r) - 1j * omega * rst_p
    elif basis == "up":
        power = -2.0 * spin - 1.0
        q = power / r + 1j * omega * rst
        qp = -power / (r * r) + 1j * omega * rst_p
    elif basis == "in":
        k_hor = omega - m * a / (rp * rp + a * a)
        q = -spin * Dp / D - 1j * k_hor * rst
        qp = -spin * (2.0 * D - Dp * Dp) / (D * D) - 1j * k_hor * rst_p
    elif basis == "out":
        k_hor = omega - m * a / (rp * rp + a * a)
        q = 1j * k_hor * rst
        qp = 1j * k_hor * rst_p
    else:
        raise ValueError(basis)
    return q, qp


def basis_A(r, *, a, m, omega, rp, rm, spin: int, basis: str):
    rs = r_star(r, a, rp, rm)
    if basis == "down":
        return r ** (-1.0) * np.exp(-1j * omega * rs)
    if basis == "up":
        return r ** (-2.0 * spin - 1.0) * np.exp(1j * omega * rs)
    if basis == "in":
        k_hor = omega - m * a / (rp * rp + a * a)
        return Delta(r, rp, rm) ** (-spin) * np.exp(-1j * k_hor * rs)
    if basis == "out":
        k_hor = omega - m * a / (rp * rp + a * a)
        return np.exp(1j * k_hor * rs)
    raise ValueError(basis)


def coeffs_spin(z, *, a, m, omega, lam, rp, rm, spin: int, basis: str):
    r = rp / z
    zr, zrr = z_derivatives(z, rp)
    D = Delta(r, rp, rm)
    Dp = Delta_p(r)
    K = K_of_r(r, a, m, omega)
    V = (K * K - 2j * spin * (r - 1.0) * K) / D + 4j * spin * omega * r - lam
    q, qp = basis_q_qp(r, a=a, m=m, omega=omega, rp=rp, rm=rm, spin=spin, basis=basis)
    B2 = D * zr * zr
    B1 = D * (zrr + 2.0 * q * zr) + (spin + 1.0) * Dp * zr
    B0 = D * (qp + q * q) + (spin + 1.0) * Dp * q + V
    return B2.astype(complex), B1.astype(complex), B0.astype(complex)


def endpoint_du(*, a, m, omega, lam, rp, rm, spin: int, basis: str, side: str) -> complex:
    """Endpoint slope from the transformed ODE with B2=0 and u=1.

    The algebraic condition is ``B1 u_z + B0 u = 0`` at the regular singular
    endpoint after the leading asymptotic factor has been removed.  We evaluate
    the limiting ratio ``-B0/B1`` by short polynomial extrapolation in ``z`` or
    ``1-z``; this keeps the implementation generic in ``spin`` and avoids
    copying a hand-derived formula for each basis.
    """
    if side == "left":
        x = np.array([1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5], dtype=float)
        z = x
    elif side == "right":
        x = np.array([1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4], dtype=float)
        z = 1.0 - x
    else:
        raise ValueError(side)
    _, B1, B0 = coeffs_spin(z, a=a, m=m, omega=omega, lam=lam, rp=rp, rm=rm, spin=spin, basis=basis)
    vals = -B0 / B1
    deg = min(2, len(x) - 1)
    re = np.polyfit(x, vals.real, deg)[-1]
    im = np.polyfit(x, vals.imag, deg)[-1]
    return complex(re, im)


def solve_basis(*, N, z_a, z_b, a, m, omega, lam, rp, rm, spin, basis, bc_side, grid_kind, domain):
    D2, D1, z = resolve_domain_D(N, z_a, z_b, omega=omega, grid_kind=grid_kind, domain=domain)
    A = np.zeros((N + 1, N + 1), dtype=np.complex128)
    b = np.zeros(N + 1, dtype=np.complex128)
    if bc_side == "left":
        du = endpoint_du(a=a, m=m, omega=omega, lam=lam, rp=rp, rm=rm, spin=spin, basis=basis, side="left")
        A[0, 0] = 1.0
        b[0] = 1.0
        A[1, :] = D1[0, :]
        b[1] = du
        idx = np.arange(2, N + 1)
    elif bc_side == "right":
        du = endpoint_du(a=a, m=m, omega=omega, lam=lam, rp=rp, rm=rm, spin=spin, basis=basis, side="right")
        idx = np.arange(0, N - 1)
        A[-2, :] = D1[-1, :]
        b[-2] = du
        A[-1, -1] = 1.0
        b[-1] = 1.0
    else:
        raise ValueError(bc_side)
    B2, B1, B0 = coeffs_spin(z[idx], a=a, m=m, omega=omega, lam=lam, rp=rp, rm=rm, spin=spin, basis=basis)
    A[idx, :] = B2[:, None] * D2[idx, :] + B1[:, None] * D1[idx, :]
    A[idx, idx] += B0
    row = np.linalg.norm(A, axis=1)
    row = np.where(row > 1.0e-300, row, 1.0)
    Ar = A / row[:, None]
    br = b / row
    col = np.linalg.norm(Ar, axis=0)
    col = np.where(col > 1.0e-300, col, 1.0)
    As = Ar / col[None, :]
    y = np.linalg.solve(As, br)
    u = y / col
    uz = D1 @ u
    return {"z": z, "u": u, "uz": uz, "D1": D1, "D2": D2, "cond": float(np.linalg.cond(As)), "du_endpoint": du}


def basis_values_at_match(sol, *, side, a, m, omega, rp, rm, spin, basis):
    if side == "left":
        idx = 0
    else:
        idx = -1
    z = sol["z"][idx]
    u = sol["u"][idx]
    uz = sol["uz"][idx]
    r = rp / z
    zr, _ = z_derivatives(z, rp)
    F = basis_A(r, a=a, m=m, omega=omega, rp=rp, rm=rm, spin=spin, basis=basis)
    q, _ = basis_q_qp(r, a=a, m=m, omega=omega, rp=rp, rm=rm, spin=spin, basis=basis)
    R = F * u
    Rr = F * (zr * uz + q * u)
    return complex(R), complex(Rr)


def residual_spin(sol, *, a, m, omega, lam, rp, rm, spin, basis, skip_boundary: tuple[bool, bool]):
    z = sol["z"]
    D1 = sol["D1"]
    D2 = sol["D2"]
    u = sol["u"]
    interior = np.ones_like(z, dtype=bool)
    interior[0] = False
    interior[-1] = False
    B2 = np.zeros_like(z, dtype=np.complex128)
    B1 = np.zeros_like(z, dtype=np.complex128)
    B0 = np.zeros_like(z, dtype=np.complex128)
    B2[interior], B1[interior], B0[interior] = coeffs_spin(
        z[interior], a=a, m=m, omega=omega, lam=lam, rp=rp, rm=rm, spin=spin, basis=basis
    )
    term2 = B2 * (D2 @ u)
    term1 = B1 * (D1 @ u)
    term0 = B0 * u
    res = term2 + term1 + term0
    den = np.maximum.reduce([np.abs(term2), np.abs(term1), np.abs(term0), np.full_like(z, 1.0e-300)])
    rel = np.abs(res) / den
    mask = np.ones_like(z, dtype=bool)
    if skip_boundary[0]:
        mask[0:2] = False
    if skip_boundary[1]:
        mask[-2:] = False
    return rel[mask]


def solve_case(*, a, omega, ell, m, spin, N, z_out, z_h, z_m, grid_kind):
    rp = 1.0 + np.sqrt(1.0 - a * a)
    rm = 1.0 - np.sqrt(1.0 - a * a)
    if z_m is None:
        z_m = adaptive_match_z(omega, rp)
    lam = complex(compute_lambda(a, omega, ell, m, spin))
    sol_down = solve_basis(N=N, z_a=z_out, z_b=z_m, a=a, m=m, omega=omega, lam=lam, rp=rp, rm=rm, spin=spin, basis="down", bc_side="left", grid_kind=grid_kind, domain="outer")
    sol_up = solve_basis(N=N, z_a=z_out, z_b=z_m, a=a, m=m, omega=omega, lam=lam, rp=rp, rm=rm, spin=spin, basis="up", bc_side="left", grid_kind=grid_kind, domain="outer")
    sol_in = solve_basis(N=N, z_a=z_m, z_b=z_h, a=a, m=m, omega=omega, lam=lam, rp=rp, rm=rm, spin=spin, basis="in", bc_side="right", grid_kind=grid_kind, domain="inner")
    sol_out = solve_basis(N=N, z_a=z_m, z_b=z_h, a=a, m=m, omega=omega, lam=lam, rp=rp, rm=rm, spin=spin, basis="out", bc_side="right", grid_kind=grid_kind, domain="inner")
    Rd, Rrd = basis_values_at_match(sol_down, side="right", a=a, m=m, omega=omega, rp=rp, rm=rm, spin=spin, basis="down")
    Ru, Rru = basis_values_at_match(sol_up, side="right", a=a, m=m, omega=omega, rp=rp, rm=rm, spin=spin, basis="up")
    Ri, Rri = basis_values_at_match(sol_in, side="left", a=a, m=m, omega=omega, rp=rp, rm=rm, spin=spin, basis="in")
    Ro, Rro = basis_values_at_match(sol_out, side="left", a=a, m=m, omega=omega, rp=rp, rm=rm, spin=spin, basis="out")
    Mmat = np.array([[Rd, Ru], [Rrd, Rru]], dtype=np.complex128)
    rhs = np.array([Ri, Rri], dtype=np.complex128)
    coef_in, diag = solve_equilibrated_2x2(Mmat, rhs)
    B_inc, B_ref = coef_in
    res_vals = []
    for sol, basis, skip in [
        (sol_down, "down", (True, False)),
        (sol_up, "up", (True, False)),
        (sol_in, "in", (False, True)),
        (sol_out, "out", (False, True)),
    ]:
        res_vals.append(residual_spin(sol, a=a, m=m, omega=omega, lam=lam, rp=rp, rm=rm, spin=spin, basis=basis, skip_boundary=skip))
    max_res = max(float(np.max(x)) for x in res_vals)
    med_res = float(np.median(np.concatenate(res_vals)))
    return {
        "a": a,
        "omega": omega,
        "ell": ell,
        "m": m,
        "spin": spin,
        "N": N,
        "z_out": z_out,
        "z_h": z_h,
        "z_m": z_m,
        "grid_kind": grid_kind,
        "r_out": float("inf") if z_out == 0.0 else rp / z_out,
        "r_h_cut": rp / z_h,
        "lambda_re": lam.real,
        "lambda_im": lam.imag,
        "B_inc": complex(B_inc),
        "B_ref": complex(B_ref),
        "B_trans": 1.0 + 0.0j,
        "match_cond_raw": diag["cond_raw"],
        "match_cond_col_scaled": diag["cond_col_scaled"],
        "match_cond_row_col_scaled": diag["cond_row_col_scaled"],
        "match_relres": diag["relres"],
        "max_residual": max_res,
        "median_residual": med_res,
        "domain_cond_max": max(sol_down["cond"], sol_up["cond"], sol_in["cond"], sol_out["cond"]),
    }


def cfields(prefix, value):
    return {
        f"{prefix}_re": float(value.real),
        f"{prefix}_im": float(value.imag),
        f"{prefix}_abs": float(abs(value)),
    }


def run_mma_amplitudes(cases, *, work_dir: Path, kernel_path: str | None, wl_path_win: str, timeout: float):
    kernel = kernel_path or os.environ.get("WOLFRAM_KERNEL") or "/mnt/f/mma/WolframKernel.exe"
    if not Path(kernel).exists():
        return {}, f"kernel-not-found:{kernel}"
    work_dir.mkdir(parents=True, exist_ok=True)
    script_path = work_dir / "mma_s0_amplitudes.wls"
    csv_path = work_dir / "mma_s0_amplitudes.csv"
    case_rows = ",".join(f"{{{a:.17g},{omega:.17g},{ell},{m}}}" for a, omega, ell, m in cases)
    script_path.write_text(f"""
Get["{wl_path_win}"];
cases = {{{case_rows}}};
headers = {{"a","omega","ell","m","status","B_inc_re","B_inc_im","B_ref_re","B_ref_im","B_trans_re","B_trans_im"}};
rows = Table[
  a = c[[1]]; omega = c[[2]]; ell = Round[c[[3]]]; mm = Round[c[[4]]];
  amp = Quiet[Check[TimeConstrained[ComputeAmplitudes[0, ell, mm, a, omega], {timeout:.17g}, $Aborted], $Failed]];
  If[AssociationQ[amp],
    {{a, omega, ell, mm, "ok", Re[N[amp["Incidence"]]], Im[N[amp["Incidence"]]], Re[N[amp["Reflection"]]], Im[N[amp["Reflection"]]], Re[N[amp["Transmission"]]], Im[N[amp["Transmission"]]]}},
    {{a, omega, ell, mm, ToString[amp, InputForm], "", "", "", "", "", ""}}
  ],
  {{c, cases}}
];
Export["{csv_path}", Prepend[rows, headers], "CSV"];
Quit[];
""")
    proc = subprocess.run([kernel, "-script", str(script_path)], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return {}, f"mma-exit-{proc.returncode}:{proc.stderr.strip()[:240]}"
    if not csv_path.exists():
        return {}, "mma-no-output"
    out = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (float(row["a"]), float(row["omega"]), int(float(row["ell"])), int(float(row["m"])))
            if row["status"] != "ok":
                out[key] = (None, row["status"])
            else:
                out[key] = ({
                    "B_inc": complex(float(row["B_inc_re"]), float(row["B_inc_im"])),
                    "B_ref": complex(float(row["B_ref_re"]), float(row["B_ref_im"])),
                    "B_trans": complex(float(row["B_trans_re"]), float(row["B_trans_im"])),
                }, "ok")
    return out, "ok"


def relerr(a, b):
    return float(abs(a - b) / max(abs(b), 1.0e-300))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-list", default="0.5")
    parser.add_argument("--omega-list", default="0.1,1.0")
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--N-list", default="60,80,100")
    parser.add_argument("--z-out-list", default="0")
    parser.add_argument("--z-h-list", default="1")
    parser.add_argument("--z-m-list", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--grid-kind", choices=["linear", "anmr", "auto"], default="linear")
    parser.add_argument("--skip-mma", action="store_true")
    parser.add_argument("--mma-kernel", default=None)
    parser.add_argument("--mma-wl-win", default="F:/EMRI/Radial_flow/Radial_Function.wl")
    parser.add_argument("--mma-timeout", type=float, default=180.0)
    parser.add_argument("--output-dir", default="outputs/kerr_s0_spectral_match")
    args = parser.parse_args()
    a_values = parse_float_list(args.a_list)
    omega_values = parse_float_list(args.omega_list)
    N_values = [int(x) for x in args.N_list.split(",") if x.strip()]
    z_out_values = parse_float_list(args.z_out_list)
    z_h_values = parse_float_list(args.z_h_list)
    z_m_values = [None if item.strip().lower() == "auto" else float(item.strip()) for item in args.z_m_list.split(",") if item.strip()]
    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    mma_results = {}
    mma_status = "skipped"
    cases_for_mma = [(a, w, args.ell, args.m) for a in a_values for w in omega_values]
    if args.spin != 0 and not args.skip_mma:
        raise ValueError("This MMA batch helper currently calls ComputeAmplitudes[0,...]; use --skip-mma for spin != 0.")
    if not args.skip_mma:
        mma_results, mma_status = run_mma_amplitudes(cases_for_mma, work_dir=out_dir, kernel_path=args.mma_kernel, wl_path_win=args.mma_wl_win, timeout=args.mma_timeout)
    rows = []
    failures = []
    total = len(a_values) * len(omega_values) * len(N_values) * len(z_out_values) * len(z_h_values) * len(z_m_values)
    count = 0
    for a in a_values:
        for omega in omega_values:
            for N in N_values:
                for z_out in z_out_values:
                    for z_h in z_h_values:
                        for z_m in z_m_values:
                            count += 1
                            if z_m is not None and not (z_out < z_m < z_h):
                                continue
                            z_m_label = "auto" if z_m is None else f"{z_m:g}"
                            print(f"[{count}/{total}] a={a:g} w={omega:g} N={N} zout={z_out:g} zh={z_h:g} zm={z_m_label} grid={args.grid_kind}", flush=True)
                            try:
                                result = solve_case(a=a, omega=omega, ell=args.ell, m=args.m, spin=args.spin, N=N, z_out=z_out, z_h=z_h, z_m=z_m, grid_kind=args.grid_kind)
                                row = {k: v for k, v in result.items() if k not in {"B_inc", "B_ref", "B_trans"}}
                                row.update(cfields("spectral_B_inc", result["B_inc"]))
                                row.update(cfields("spectral_B_ref", result["B_ref"]))
                                row.update(cfields("spectral_B_trans", result["B_trans"]))
                                mma, status = mma_results.get((a, omega, args.ell, args.m), (None, mma_status))
                                row["mma_status"] = status
                                if mma is not None:
                                    row.update(cfields("mma_B_inc", mma["B_inc"]))
                                    row.update(cfields("mma_B_ref", mma["B_ref"]))
                                    row["B_inc_relerr"] = relerr(result["B_inc"], mma["B_inc"])
                                    row["B_ref_relerr"] = relerr(result["B_ref"], mma["B_ref"])
                                rows.append(row)
                            except Exception as exc:
                                failures.append({"a": a, "omega": omega, "N": N, "z_out": z_out, "z_h": z_h, "z_m": z_m, "error": str(exc)})
                                print(f"  FAILED: {exc}", flush=True)
    if rows:
        with open(out_dir / "kerr_s0_spectral_cases.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = {"args": vars(args), "mma_status": mma_status, "n_cases": len(rows), "failures": failures}
    if rows:
        valid = [r for r in rows if "B_inc_relerr" in r]
        if valid:
            summary["best_by_Binc"] = min(valid, key=lambda r: float(r["B_inc_relerr"]))
            summary["best_by_Bref"] = min(valid, key=lambda r: float(r["B_ref_relerr"]))
            summary["best_joint"] = min(valid, key=lambda r: max(float(r["B_inc_relerr"]), float(r["B_ref_relerr"])))
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved Kerr s=0 spectral match to {out_dir}")
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith('best')}, indent=2))
    if "best_joint" in summary:
        print("Best joint:")
        print(json.dumps(summary["best_joint"], indent=2))


if __name__ == "__main__":
    main()
