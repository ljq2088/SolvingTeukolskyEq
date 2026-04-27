from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.mode import KerrMode
from utils.amplitude_three_patch import compute_smatrix_three_patch_with_abel


def format_complex(z: complex) -> str:
    z = complex(z)
    return f"{z.real:+.10e}{z.imag:+.10e}j"


def safe_rel_change(x: complex, x0: complex, floor: float = 1.0e-30) -> float:
    denom = max(abs(x0), floor)
    return float(abs(x - x0) / denom)


def build_side_points(z0: float, n_total: int, lo_delta: float, hi_delta: float) -> list[float]:
    if n_total < 2:
        raise ValueError("n_total must be at least 2")

    n_left = n_total // 2
    n_right = n_total - n_left

    left = []
    right = []
    if n_left > 0:
        left = list(z0 - hi_delta + (hi_delta - lo_delta) * (i / max(n_left - 1, 1)) for i in range(n_left))
    if n_right > 0:
        right = list(z0 + lo_delta + (hi_delta - lo_delta) * (i / max(n_right - 1, 1)) for i in range(n_right))

    return [float(x) for x in left + right]


def compute_row(
    mode: KerrMode,
    N_left: int,
    N_mid: int,
    N_right: int,
    z1: float,
    z2: float,
    omega_mp_cut: float,
    mp_dps_loww: int,
):
    sm = compute_smatrix_three_patch_with_abel(
        mode,
        N_left=N_left,
        N_mid=N_mid,
        N_right=N_right,
        z1=z1,
        z2=z2,
        return_details=False,
        omega_mp_cut=omega_mp_cut,
        mp_dps_loww=mp_dps_loww,
    )
    return {
        "z1": float(z1),
        "z2": float(z2),
        "B_inc": complex(sm["B_inc"]),
        "B_ref": complex(sm["B_ref"]),
        "B_trans": complex(sm["B_trans"]),
        "ratio_ref_over_inc": sm["ratio_ref_over_inc"],
        "outer_abel_residual": float(sm["outer_abel_residual"]),
        "inner_abel_residual": float(sm["inner_abel_residual"]),
        "detS_residual": float(sm["detS_residual"]),
        "cond_M_left": float(sm["cond_M_left"]),
        "cond_T_mid": float(sm["cond_T_mid"]),
        "cond_M_outer_at_z2": float(sm["cond_M_outer_at_z2"]),
        "state_in_z1_relerr": float(sm["state_in_z1_relerr"]),
        "state_out_z1_relerr": float(sm["state_out_z1_relerr"]),
        "use_mp_backend": bool(sm["use_mp_backend"]),
    }


def print_section(title: str):
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def print_baseline(row: dict):
    print_section("Baseline three-patch amplitudes")
    print(f"z1 = {row['z1']:.8f}, z2 = {row['z2']:.8f}")
    print(f"B_inc   = {format_complex(row['B_inc'])}")
    print(f"B_ref   = {format_complex(row['B_ref'])}")
    print(f"B_trans = {format_complex(row['B_trans'])}")
    print(f"B_ref/B_inc = {format_complex(row['ratio_ref_over_inc']) if row['ratio_ref_over_inc'] is not None else 'None'}")
    print(f"outer_abel_residual = {row['outer_abel_residual']:.3e}")
    print(f"inner_abel_residual = {row['inner_abel_residual']:.3e}")
    print(f"detS_residual       = {row['detS_residual']:.3e}")
    print(f"cond_M_left         = {row['cond_M_left']:.3e}")
    print(f"cond_T_mid          = {row['cond_T_mid']:.3e}")
    print(f"cond_M_outer_at_z2  = {row['cond_M_outer_at_z2']:.3e}")
    print(f"state_in_z1_relerr  = {row['state_in_z1_relerr']:.3e}")
    print(f"state_out_z1_relerr = {row['state_out_z1_relerr']:.3e}")
    print(f"use_mp_backend      = {row['use_mp_backend']}")


def print_scan(scan_name: str, rows: list[dict], base: dict, vary_key: str):
    print_section(scan_name)
    header = (
        f"{vary_key:>12}  {'z1':>12}  {'z2':>12}  "
        f"{'|dBinc|/|Binc0|':>16}  {'|dBref|/|Bref0|':>16}  "
        f"{'outer_abel':>12}  {'inner_abel':>12}  {'detS':>12}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        val = row[vary_key]
        d_inc = safe_rel_change(row["B_inc"], base["B_inc"])
        d_ref = safe_rel_change(row["B_ref"], base["B_ref"])
        print(
            f"{val:12.8f}  {row['z1']:12.8f}  {row['z2']:12.8f}  "
            f"{d_inc:16.3e}  {d_ref:16.3e}  "
            f"{row['outer_abel_residual']:12.3e}  {row['inner_abel_residual']:12.3e}  {row['detS_residual']:12.3e}"
        )
        print(f"  B_inc        = {format_complex(row['B_inc'])}")
        print(f"  B_ref        = {format_complex(row['B_ref'])}")
        print(f"  B_ref/B_inc  = {format_complex(row['ratio_ref_over_inc']) if row['ratio_ref_over_inc'] is not None else 'None'}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Use the native amplitude_three_patch.py three-patch solver, "
            "but scan interface locations around z1 and z2 to inspect amplitude sensitivity. "
            "The scan uses 10 sample points around each nominal interface, covering ±0.01 and ±0.05 windows."
        )
    )
    parser.add_argument("--M", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--omega", type=float, default=0.2)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--lam", type=float, default=None)

    parser.add_argument("--N-left", type=int, default=64)
    parser.add_argument("--N-mid", type=int, default=96)
    parser.add_argument("--N-right", type=int, default=64)

    parser.add_argument("--z1", type=float, default=0.1)
    parser.add_argument("--z2", type=float, default=0.6)
    parser.add_argument("--n-scan-points", type=int, default=10)
    parser.add_argument("--delta-small", type=float, default=0.01)
    parser.add_argument("--delta-large", type=float, default=0.05)

    parser.add_argument("--omega-mp-cut", type=float, default=1.0e-2)
    parser.add_argument("--mp-dps-loww", type=int, default=80)

    args = parser.parse_args()

    if not (0.0 < args.z1 < args.z2 < 1.0):
        raise ValueError("Require 0 < z1 < z2 < 1.")
    if not (0.0 < args.delta_small < args.delta_large):
        raise ValueError("Require 0 < delta_small < delta_large.")

    mode = KerrMode(
        M=args.M,
        a=args.a,
        omega=args.omega,
        ell=args.ell,
        m=args.m,
        lam=args.lam,
        s=args.s,
    )

    print_section("Mode")
    print(
        f"M={args.M}, a={args.a}, omega={args.omega}, s={args.s}, ell={args.ell}, m={args.m}, "
        f"lambda={mode.lambda_value}"
    )
    print(
        f"N_left={args.N_left}, N_mid={args.N_mid}, N_right={args.N_right}, "
        f"z1={args.z1}, z2={args.z2}"
    )

    base = compute_row(
        mode=mode,
        N_left=args.N_left,
        N_mid=args.N_mid,
        N_right=args.N_right,
        z1=args.z1,
        z2=args.z2,
        omega_mp_cut=args.omega_mp_cut,
        mp_dps_loww=args.mp_dps_loww,
    )
    print_baseline(base)

    z1_points = build_side_points(args.z1, args.n_scan_points, args.delta_small, args.delta_large)
    z2_points = build_side_points(args.z2, args.n_scan_points, args.delta_small, args.delta_large)

    z1_rows = []
    for z1v in z1_points:
        if not (0.0 < z1v < args.z2):
            continue
        row = compute_row(
            mode=mode,
            N_left=args.N_left,
            N_mid=args.N_mid,
            N_right=args.N_right,
            z1=z1v,
            z2=args.z2,
            omega_mp_cut=args.omega_mp_cut,
            mp_dps_loww=args.mp_dps_loww,
        )
        z1_rows.append(row)

    z2_rows = []
    for z2v in z2_points:
        if not (args.z1 < z2v < 1.0):
            continue
        row = compute_row(
            mode=mode,
            N_left=args.N_left,
            N_mid=args.N_mid,
            N_right=args.N_right,
            z1=args.z1,
            z2=z2v,
            omega_mp_cut=args.omega_mp_cut,
            mp_dps_loww=args.mp_dps_loww,
        )
        z2_rows.append(row)

    print_scan("Scan z1 with z2 fixed", z1_rows, base, "z1")
    print_scan("Scan z2 with z1 fixed", z2_rows, base, "z2")


if __name__ == "__main__":
    main()
