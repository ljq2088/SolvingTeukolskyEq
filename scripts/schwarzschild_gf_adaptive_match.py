#!/usr/bin/env python3
"""Python port of ``utils/GF_adaptive_match.m``.

This is a Schwarzschild/Bondi-coordinate toy scattering solver, not the Kerr
Teukolsky solver used elsewhere in the project.  It is useful because it
implements the same algorithmic ideas we want to reuse:

  * two-domain Chebyshev collocation,
  * low-frequency analytic mesh refinement,
  * matching at ``r_p = 3M + omega^{-1/2}``,
  * extraction of scattering coefficients from a 2x2 interface system.
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

from utils.matlcheb import cheb, real_to_cheb, cheb_tail_ratio  # noqa: E402


def bondi_matrix(z: np.ndarray, D1: np.ndarray, D2: np.ndarray, ell: int, sigma: complex) -> np.ndarray:
    """Matrix used by ``GF_adaptive_match.m``/``calculate_residual``."""
    a2 = z**2 * (1.0 - z)
    a1 = z * (2.0 - 3.0 * z) - sigma
    a0 = -(ell * (ell + 1.0) + z)
    return a2[:, None] * D2 + a1[:, None] * D1 + np.diag(a0.astype(complex))


def relative_residual(z: np.ndarray, D1: np.ndarray, D2: np.ndarray, ell: int, sigma: complex, phi: np.ndarray) -> np.ndarray:
    a2 = z**2 * (1.0 - z)
    a1 = z * (2.0 - 3.0 * z) - sigma
    a0 = -(ell * (ell + 1.0) + z)
    term2 = a2 * (D2 @ phi)
    term1 = a1 * (D1 @ phi)
    term0 = a0 * phi
    res = term2 + term1 + term0
    den = np.maximum.reduce([np.abs(term2), np.abs(term1), np.abs(term0), np.full_like(z, 1.0e-300)])
    return np.abs(res) / den


def linear_domain(N: int, z_left: float, z_right: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Dy, y = cheb(N)
    z = (z_right - z_left) * (y + 1.0) / 2.0 + z_left
    D1 = Dy / ((z_right - z_left) / 2.0)
    D2 = D1 @ D1
    return D2, D1, z


def sinh_domain(N: int, z_left: float, z_right: float, kappa: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Dy, y = cheb(N)
    if abs(kappa) < 1.0e-14:
        return linear_domain(N, z_left, z_right)
    z = (z_right - z_left) * np.sinh(kappa * (y + 1.0) / 2.0) / np.sinh(kappa) + z_left
    dzdy = (z_right - z_left) * kappa / 2.0 * np.cosh(kappa * (y + 1.0) / 2.0) / np.sinh(kappa)
    D1 = Dy / dzdy[:, None]
    D2 = D1 @ D1
    return D2, D1, z


def solve_with_boundary(A: np.ndarray, bc_row: int, bc_col: int) -> np.ndarray:
    """Replace one row by a normalization condition and solve."""
    mat = np.array(A, dtype=np.complex128, copy=True)
    rhs = np.zeros(mat.shape[0], dtype=np.complex128)
    mat[bc_row, :] = 0.0
    mat[bc_row, bc_col] = 1.0
    rhs[bc_row] = 1.0
    return np.linalg.solve(mat, rhs)


def solve_case(M: float, ell: int, omega: float, N: int, use_anmr: bool | None = None) -> dict[str, object]:
    rp = 3.0 * M + omega ** (-0.5)
    zp = 2.0 * M / rp
    xp = 2.0 * M * (1.0 / zp + np.log(1.0 - zp) - np.log(zp))
    if use_anmr is None:
        use_anmr = omega < 1.0e-1

    if not use_anmr:
        D2z1, Dz1, z1 = linear_domain(N, zp, 1.0)
        D2z0, Dz0, z0 = linear_domain(N, 0.0, zp)
    else:
        kappa = abs(np.log(omega * 2.0 * M))
        D2z1, Dz1, z1 = sinh_domain(N, zp, 1.0, 0.5 * kappa)
        D2z0, Dz0, z0 = sinh_domain(N, 0.0, zp, kappa)

    sigma = -1j * omega * 4.0 * M
    rh = 2.0 * M

    B1 = bondi_matrix(z1, Dz1, D2z1, ell, sigma)
    phi_in = solve_with_boundary(B1, bc_row=N, bc_col=N)
    dphi_in = Dz1 @ phi_in

    B0 = bondi_matrix(z0, Dz0, D2z0, ell, sigma)
    phi_down = solve_with_boundary(B0, bc_row=0, bc_col=N)
    dphi_down = Dz0 @ phi_down

    phase = np.exp(-1j * omega * xp)
    dxp = rh / (zp**2 * (zp - 1.0))
    GFM11 = phase * phi_down[0]
    GFM21 = phase * (dphi_down[0] + phi_down[0] * (-1j * omega) * dxp)
    GFM = np.array([[GFM11, np.conj(GFM11)], [GFM21, np.conj(GFM21)]], dtype=np.complex128)
    rhs = np.array([
        phi_in[-1] * phase,
        phase * (dphi_in[-1] + phi_in[-1] * (-1j * omega) * dxp),
    ], dtype=np.complex128)
    Cid, Ciu = np.linalg.solve(GFM, rhs)

    res_in = relative_residual(z1, Dz1, D2z1, ell, sigma, phi_in)[:-1]
    res_down = relative_residual(z0, Dz0, D2z0, ell, sigma, phi_down)[1:-1]
    coeff_in = real_to_cheb(phi_in)
    coeff_down = real_to_cheb(phi_down)
    return {
        "M": float(M),
        "ell": int(ell),
        "omega": float(omega),
        "N": int(N),
        "use_anmr": bool(use_anmr),
        "rp_match": float(rp),
        "zp_match": float(zp),
        "Cid_re": float(Cid.real),
        "Cid_im": float(Cid.imag),
        "Cid_abs": float(abs(Cid)),
        "Ciu_re": float(Ciu.real),
        "Ciu_im": float(Ciu.imag),
        "Ciu_abs": float(abs(Ciu)),
        "T": float(1.0 / abs(Cid) ** 2),
        "R": float(abs(Ciu) ** 2 / abs(Cid) ** 2),
        "T_plus_R": float((1.0 + abs(Ciu) ** 2) / abs(Cid) ** 2),
        "match_cond": float(np.linalg.cond(GFM)),
        "max_res_in": float(np.max(res_in)),
        "median_res_in": float(np.median(res_in)),
        "max_res_down": float(np.max(res_down)),
        "median_res_down": float(np.median(res_down)),
        "tail_in": cheb_tail_ratio(coeff_in),
        "tail_down": cheb_tail_ratio(coeff_down),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omega-list", default="1e-4,1e-3,1e-2,0.1,1.0")
    parser.add_argument("--N-list", default="64,96,128")
    parser.add_argument("--ell", type=int, default=0)
    parser.add_argument("--M", type=float, default=1.0)
    parser.add_argument("--force-linear", action="store_true")
    parser.add_argument("--force-anmr", action="store_true")
    parser.add_argument("--output-dir", default="outputs/schwarzschild_gf_adaptive_match")
    args = parser.parse_args()

    omega_values = [float(item) for item in args.omega_list.split(",") if item.strip()]
    N_values = [int(item) for item in args.N_list.split(",") if item.strip()]
    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    for omega in omega_values:
        for N in N_values:
            try:
                use_anmr = True if args.force_anmr else False if args.force_linear else None
                print(f"omega={omega:g} N={N} anmr={use_anmr}", flush=True)
                rows.append(solve_case(args.M, args.ell, omega, N, use_anmr=use_anmr))
            except Exception as exc:
                failures.append({"omega": omega, "N": N, "error": str(exc)})
                print(f"  FAILED: {exc}", flush=True)

    if rows:
        with open(out_dir / "schwarzschild_gf_cases.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "args": vars(args),
        "n_cases": len(rows),
        "failures": failures,
    }
    if rows:
        summary["max_residual"] = max(max(row["max_res_in"], row["max_res_down"]) for row in rows)
        summary["max_unitarity_error"] = max(abs(row["T_plus_R"] - 1.0) for row in rows)
        summary["best_by_omega"] = {
            str(omega): min(
                [row for row in rows if row["omega"] == omega],
                key=lambda row: max(row["max_res_in"], row["max_res_down"]),
            )
            for omega in omega_values
            if any(row["omega"] == omega for row in rows)
        }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved Schwarzschild GF diagnostics to {out_dir}")
    print(json.dumps({k: v for k, v in summary.items() if k != "best_by_omega"}, indent=2))


if __name__ == "__main__":
    main()
