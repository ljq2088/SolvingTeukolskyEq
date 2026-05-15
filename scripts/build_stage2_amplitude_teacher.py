#!/usr/bin/env python3
"""Build Stage-2 amplitude teacher from rpred_cache using pybhpt spectral solver.

For each (a, omega) in the Stage-1 rpred_cache, uses pybhpt to solve the
homogeneous Teukolsky equation and extract asymptotic amplitudes B_inc, B_ref.

Uses pybhpt's standard normalization: B_inc ≈ 1 (unit ingoing at infinity).

Output: outputs/stage1_artifacts/patch_XXX_YYY/amplitude_teacher.npz
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from amplitude_teacher import AmplitudeTeacher


def main():
    parser = argparse.ArgumentParser(description="Build Stage-2 amplitude teacher")
    parser.add_argument("--cache-dir", type=str, required=True,
                        help="Directory containing rpred_cache.npz and metadata.json")
    parser.add_argument("--method", type=str, default="pybhpt", choices=["pybhpt", "gsn"])
    parser.add_argument("--n-max", type=int, default=0,
                        help="Max samples to process (0=all)")
    parser.add_argument("--start", type=int, default=0,
                        help="Starting sample index")
    parser.add_argument("--r-max", type=float, default=500.0)
    parser.add_argument("--n-r", type=int, default=256)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    npz_path = cache_dir / "rpred_cache.npz"
    meta_path = cache_dir / "metadata.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"rpred_cache.npz not found in {cache_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {cache_dir}")

    data = np.load(npz_path)
    with open(meta_path) as f:
        meta = json.load(f)

    a_all = data["a"]
    omega_all = data["omega"]
    n_total = len(a_all)

    end = min(args.start + args.n_max, n_total) if args.n_max > 0 else n_total
    a_vals = a_all[args.start:end]
    omega_vals = omega_all[args.start:end]

    physics = meta.get("physics", meta.get("phyics", {}))
    s = int(physics.get("s", -2))
    l = int(physics.get("l", 2))
    m = int(physics.get("m", 2))

    n = len(a_vals)
    print(f"Building amplitude teacher: {n} / {n_total} samples (indices {args.start}:{end})")
    print(f"  (s,l,m) = ({s},{l},{m})")
    print(f"  a range: [{a_vals.min():.4f}, {a_vals.max():.4f}]")
    print(f"  omega range: [{omega_vals.min():.6e}, {omega_vals.max():.6e}]")

    if args.method == "gsn":
        from amplitude_teacher.gsn_teacher import compute_amplitudes_gsn
        B_inc, B_ref = compute_amplitudes_gsn(
            a_vals, omega_vals, s=s, l=l, m=m,
            r_max=args.r_max, n_r=args.n_r, verbose=True,
        )
    else:
        from amplitude_teacher.spectral_teacher import compute_amplitudes_pybhpt
        B_inc, B_ref = compute_amplitudes_pybhpt(
            a_vals, omega_vals, s=s, l=l, m=m,
            r_max=args.r_max, n_r=args.n_r, verbose=True,
        )

    n_nan = int(np.sum(np.isnan(B_inc)))
    if n_nan > 0:
        print(f"WARNING: {n_nan}/{n} samples have NaN B_inc")
    n_finite = int(np.sum(np.isfinite(B_inc) & np.isfinite(B_ref)))
    print(f"Finite: {n_finite}/{n}")

    teacher = AmplitudeTeacher(
        a_vals=a_vals,
        omega_vals=omega_vals,
        B_inc=B_inc,
        B_ref=B_ref,
        metadata={
            "created": datetime.now().isoformat(),
            "cache_dir": str(cache_dir.resolve()),
            "method": args.method,
            "s": s, "l": l, "m": m,
            "r_max": args.r_max,
            "n_r": args.n_r,
            "n_total": int(n),
            "n_finite": int(n_finite),
            "start_index": int(args.start),
            "scattering_convention": "R_in = B_ref * u_up * A_up + B_inc * u_down * A_down",
            "normalization": "pybhpt standard: unit ingoing amplitude at infinity",
        },
    )

    out_path = cache_dir / "amplitude_teacher.npz"
    teacher.save(out_path)
    print(f"Saved {out_path}")
    print(f"  B_inc mag range: [{np.nanmin(np.abs(B_inc)):.4e}, {np.nanmax(np.abs(B_inc)):.4e}]")
    print(f"  B_ref mag range: [{np.nanmin(np.abs(B_ref)):.4e}, {np.nanmax(np.abs(B_ref)):.4e}]")


if __name__ == "__main__":
    main()
