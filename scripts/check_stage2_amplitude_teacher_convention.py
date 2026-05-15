#!/usr/bin/env python3
"""Check Stage-2 amplitude teacher convention.

For each sample in amplitude_teacher.npz, solves pybhpt radial solution,
then reconstructs R(r) using two conventions at large r:

  Convention A (r_*):  R = B_inc * A_down_rs + B_ref * A_up_rs
    A_up_rs   = r^3 exp(+i ω r_*(r,a))
    A_down_rs = r^{-1} exp(-i ω r_*(r,a))

  Convention B (r):    R = B_inc * A_down_r + B_ref * A_up_r
    A_up_r   = r^3 exp(+i ω r)
    A_down_r = r^{-1} exp(-i ω r)

Compares reconstruction relative error against pybhpt reference in the
large-r tail. Reports which convention matches better.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def r_plus(a, M=1.0):
    return M + np.sqrt(M**2 - a**2)


def r_minus(a, M=1.0):
    return M - np.sqrt(M**2 - a**2)


def r_star_np(r, a, M=1.0):
    """Tortoise coordinate matching physical_ansatz.prefactor.r_star."""
    rp = r_plus(a, M)
    rm = r_minus(a, M)
    denom = rp - rm
    return (r
            + (rp**2 + a**2) / denom * np.log(np.abs(r - rp) / (2 * M))
            - (rm**2 + a**2) / denom * np.log(np.abs(r - rm) / (2 * M)))


def A_up_rs(r, a, omega):
    return r**3 * np.exp(1j * omega * r_star_np(r, a))


def A_down_rs(r, a, omega):
    return r**(-1) * np.exp(-1j * omega * r_star_np(r, a))


def A_up_r(r, omega):
    return r**3 * np.exp(1j * omega * r)


def A_down_r(r, omega):
    return r**(-1) * np.exp(-1j * omega * r)


def main():
    parser = argparse.ArgumentParser(description="Check amplitude teacher convention")
    parser.add_argument("--teacher-npz", type=str, required=True)
    parser.add_argument("--n-check", type=int, default=8)
    parser.add_argument("--r-max", type=float, default=1000.0)
    parser.add_argument("--n-r", type=int, default=512)
    args = parser.parse_args()

    data = np.load(args.teacher_npz)
    a_vals = data["a"]
    omega_vals = data["omega"]
    B_inc = data["B_inc"]
    B_ref = data["B_ref"]

    n = min(args.n_check, len(a_vals))
    idx = np.linspace(0, len(a_vals) - 1, n, dtype=int)

    err_rs_list = []
    err_r_list = []
    err_tail_rs_list = []
    err_tail_r_list = []

    for i, j in enumerate(idx):
        a = float(a_vals[j])
        omega = float(omega_vals[j])
        b_inc = complex(B_inc[j])
        b_ref = complex(B_ref[j])

        # Solve pybhpt at large r
        from pybhpt.radial import RadialTeukolsky
        r_h = 1.0 + np.sqrt(1.0 - a**2)
        r_min = max(r_h + 1e-3, 2.0)
        r = np.linspace(r_min, args.r_max, args.n_r)

        rad = RadialTeukolsky(s=-2, j=2, m=2, a=a, omega=omega, r=r)
        rad.solve()
        R_ref = np.asarray(rad.radialsolutions("In"))

        # Reconstruct with both conventions
        R_rs = b_inc * A_down_rs(r, a, omega) + b_ref * A_up_rs(r, a, omega)
        R_r = b_inc * A_down_r(r, omega) + b_ref * A_up_r(r, omega)

        # Relative errors over all r
        rel_rs = np.abs(R_rs - R_ref) / np.maximum(np.abs(R_ref), 1e-14)
        rel_r = np.abs(R_r - R_ref) / np.maximum(np.abs(R_ref), 1e-14)

        # Relative errors over tail (last 20%)
        n_tail = max(20, args.n_r // 5)
        rel_rs_tail = rel_rs[-n_tail:]
        rel_r_tail = rel_r[-n_tail:]

        med_rs = float(np.median(rel_rs))
        med_r = float(np.median(rel_r))
        med_rs_tail = float(np.median(rel_rs_tail))
        med_r_tail = float(np.median(rel_r_tail))

        err_rs_list.append(med_rs)
        err_r_list.append(med_r)
        err_tail_rs_list.append(med_rs_tail)
        err_tail_r_list.append(med_r_tail)

        print(f"[{i+1}/{n}] a={a:.4f} omega={omega:.6e}")
        print(f"  r_* conv: median={med_rs:.4e} tail={med_rs_tail:.4e}")
        print(f"  r   conv: median={med_r:.4e} tail={med_r_tail:.4e}")
        winner = "r_*" if med_rs_tail < med_r_tail else "r"
        ratio = max(med_rs_tail, med_r_tail) / max(min(med_rs_tail, med_r_tail), 1e-30)
        print(f"  WINNER: {winner} (ratio={ratio:.2e})")

    print(f"\n{'='*60}")
    print(f"SUMMARY over {n} samples:")
    print(f"  r_* tail median range: [{np.min(err_tail_rs_list):.4e}, {np.max(err_tail_rs_list):.4e}]")
    print(f"  r   tail median range: [{np.min(err_tail_r_list):.4e}, {np.max(err_tail_r_list):.4e}]")
    print(f"  r_* tail mean: {np.mean(err_tail_rs_list):.4e}")
    print(f"  r   tail mean: {np.mean(err_tail_r_list):.4e}")

    n_rs_wins = sum(1 for rs, r_ in zip(err_tail_rs_list, err_tail_r_list) if rs < r_)
    n_r_wins = n - n_rs_wins
    print(f"  r_* wins: {n_rs_wins}/{n}")
    print(f"  r   wins: {n_r_wins}/{n}")

    if n_rs_wins > n_r_wins:
        print(f"\n=> r_* convention is BETTER for reconstruction.")
        print(f"   Current teacher uses r convention — NEEDS FIX.")
    elif n_r_wins > n_rs_wins:
        print(f"\n=> r convention happens to match better (unexpected).")
    else:
        print(f"\n=> Ambiguous — both similar.")

    if n_rs_wins > n_r_wins or np.mean(err_tail_rs_list) < np.mean(err_tail_r_list):
        sys.exit(1)  # signal that fix is needed
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
