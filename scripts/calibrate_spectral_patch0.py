#!/usr/bin/env python3
"""Calibrate spectral method convergence on patch 0 parameter points.

Scans N (spectral order) and z_m (match point) for each calibration point,
computing B_inc, B_ref, and Abel diagnostics to assess convergence.

Usage:
  python scripts/calibrate_spectral_patch0.py \
    --points-json outputs/spectral_calibration/patch_000_logw_v2/patch0_calibration_points.json \
    --output-dir outputs/spectral_calibration/patch_000_logw_v2
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from utils.mode import KerrMode
from utils.amplitude import TeukRadAmplitudeInWithAbelChecks


def main():
    parser = argparse.ArgumentParser(description="Calibrate spectral convergence on patch 0")
    parser.add_argument("--points-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--N-vals", type=int, nargs="+", default=[24, 32, 48, 64, 80, 96])
    parser.add_argument("--zm-vals", type=float, nargs="+", default=[0.20, 0.30, 0.40, 0.50])
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Specific point labels to run (default: all)")
    args = parser.parse_args()

    with open(args.points_json) as f:
        points = json.load(f)

    if args.labels is not None:
        points = [p for p in points if p["label"] in args.labels]
    print(f"[calibrate] {len(points)} calibration points")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    all_rows = []

    for pt in points:
        label = pt["label"]
        a_val = pt["a"]
        omega_val = pt["omega"]
        lam_val = pt.get("lambda_real", pt.get("lam"))
        print(f"\n{'='*60}")
        print(f"[calibrate] Point: {label}  a={a_val:.6f} omega={omega_val:.6e} lam={lam_val}")

        for N in args.N_vals:
            for z_m in args.zm_vals:
                t0 = time.time()
                try:
                    mode = KerrMode(M=1.0, a=a_val, omega=omega_val, ell=2, m=2, s=-2,
                                    lam=complex(lam_val) if lam_val is not None else None)
                    amp = TeukRadAmplitudeInWithAbelChecks(
                        mode, N_in=N, N_out=N, z_m=z_m,
                        omega_mp_cut=1.0e-2, mp_dps_loww=80,
                    )
                    B_inc = amp.B_inc
                    B_ref = amp.B_ref
                    B_trans = amp.B_trans
                    outer_abel = amp.outer_abel_residual
                    inner_abel = amp.inner_abel_residual
                    detS = amp.detS_residual
                    elapsed = time.time() - t0

                    row = {
                        "label": label, "a": a_val, "omega": omega_val,
                        "N": N, "z_m": z_m,
                        "B_inc_real": float(B_inc.real),
                        "B_inc_imag": float(B_inc.imag),
                        "B_ref_real": float(B_ref.real),
                        "B_ref_imag": float(B_ref.imag),
                        "B_trans_real": float(B_trans.real),
                        "B_trans_imag": float(B_trans.imag),
                        "outer_abel_residual": float(outer_abel),
                        "inner_abel_residual": float(inner_abel),
                        "detS_residual": float(detS),
                        "elapsed_s": elapsed,
                    }
                    all_rows.append(row)
                    status = "OK" if np.isfinite(outer_abel) and outer_abel < 1e-3 else "HIGH_ABEL"
                    print(f"  N={N:3d} zm={z_m:.2f} | "
                          f"B_inc=({row['B_inc_real']:.6e},{row['B_inc_imag']:.6e}) "
                          f"B_ref=({row['B_ref_real']:.6e},{row['B_ref_imag']:.6e}) | "
                          f"Abel_out={outer_abel:.2e} detS={detS:.2e} {elapsed:.1f}s {status}")
                except Exception as e:
                    print(f"  N={N:3d} zm={z_m:.2f} | FAILED: {e}")
                    all_rows.append({
                        "label": label, "a": a_val, "omega": omega_val,
                        "N": N, "z_m": z_m,
                        "error": str(e),
                    })

    # ---- Save CSV ----
    csv_path = out_dir / "spectral_convergence_cases.csv"
    import csv
    fieldnames = ["label", "a", "omega", "N", "z_m",
                  "B_inc_real", "B_inc_imag", "B_ref_real", "B_ref_imag",
                  "B_trans_real", "B_trans_imag",
                  "outer_abel_residual", "inner_abel_residual", "detS_residual",
                  "elapsed_s", "error"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[calibrate] CSV saved to {csv_path}")

    # ---- Convergence summary ----
    summary = {}
    for pt in points:
        label = pt["label"]
        pt_rows = [r for r in all_rows if r["label"] == label and "error" not in r]
        if not pt_rows:
            continue
        # Find best (lowest outer_abel) at highest N
        best = max(pt_rows, key=lambda r: (r["N"], -r["outer_abel_residual"]))
        summary[label] = {
            "a": pt["a"], "omega": pt["omega"],
            "best_N": best["N"], "best_z_m": best["z_m"],
            "B_inc": f"({best['B_inc_real']:.8e},{best['B_inc_imag']:.8e})",
            "B_ref": f"({best['B_ref_real']:.8e},{best['B_ref_imag']:.8e})",
            "B_trans": f"({best['B_trans_real']:.8e},{best['B_trans_imag']:.8e})",
            "outer_abel_residual": best["outer_abel_residual"],
            "inner_abel_residual": best["inner_abel_residual"],
            "detS_residual": best["detS_residual"],
        }

    with open(out_dir / "spectral_convergence_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[calibrate] Summary saved to {out_dir / 'spectral_convergence_summary.json'}")

    # ---- Quick convergence check ----
    print(f"\n{'='*60}")
    print("Convergence Summary")
    print(f"{'='*60}")
    for label, s in summary.items():
        print(f"  {label:12s}: N={s['best_N']} zm={s['best_z_m']:.2f} "
              f"Abel_out={s['outer_abel_residual']:.2e} detS={s['detS_residual']:.2e}")
        print(f"              B_inc={s['B_inc']}  B_ref={s['B_ref']}")


if __name__ == "__main__":
    main()
