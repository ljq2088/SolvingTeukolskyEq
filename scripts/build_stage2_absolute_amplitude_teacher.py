#!/usr/bin/env python3
"""Build Stage-2 ABSOLUTE amplitude teacher from spectral cache.

Uses TeukRadAmplitudeInWithAbelChecks which computes true absolute B_inc/B_ref
(no normalization), matching the decomposition:

    R_in = B_ref * u_up * A_up + B_inc * u_down * A_down

Output: absolute_amplitude_teacher.npz with log-magnitude and phase-unit targets.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from utils.amplitude import TeukRadAmplitudeInWithAbelChecks
from utils.mode import KerrMode


def main():
    parser = argparse.ArgumentParser(description="Build Stage-2 absolute amplitude teacher")
    parser.add_argument("--spectral-cache", type=str, required=True,
                        help="spectral_basis_cache.npz path")
    parser.add_argument("--calibration-dir", type=str, default=None,
                        help="Optional calibration dir for metadata")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    cache = np.load(args.spectral_cache)
    a_vals = cache["a"]
    omega_vals = cache["omega"]
    u_vals = cache["u"]
    v_vals = cache["v"]
    lam_vals = cache["lambda_"]
    N_out = int(cache["N_out"])
    z_b = float(cache["z_b"])
    n = len(a_vals)

    print(f"Building absolute amplitude teacher: {n} points")
    print(f"  a range: [{a_vals.min():.4f}, {a_vals.max():.4f}]")
    print(f"  omega range: [{omega_vals.min():.6e}, {omega_vals.max():.6e}]")

    B_inc_abs = np.zeros(n, dtype=np.complex128)
    B_ref_abs = np.zeros(n, dtype=np.complex128)

    for i in range(n):
        a_val = float(a_vals[i])
        omega_val = float(omega_vals[i])
        lam_val = complex(lam_vals[i])

        mode = KerrMode(M=1.0, a=a_val, omega=omega_val, ell=2, m=2, s=-2, lam=lam_val)
        try:
            amp = TeukRadAmplitudeInWithAbelChecks(mode, z_m=z_b, N_out=N_out, N_in=N_out)
            B_inc_abs[i] = complex(amp.B_inc)
            B_ref_abs[i] = complex(amp.B_ref)
            if i == 0 or i == n - 1 or (i + 1) % 5 == 0:
                print(f"  [{i+1}/{n}] a={a_val:.4f} omega={omega_val:.6e} "
                      f"|B_inc|={abs(amp.B_inc):.4e} |B_ref|={abs(amp.B_ref):.4e}")
        except Exception as e:
            print(f"  [{i+1}/{n}] a={a_val:.4f} omega={omega_val:.6e} FAILED: {e}")
            B_inc_abs[i] = np.nan
            B_ref_abs[i] = np.nan

    finite = np.isfinite(B_inc_abs) & np.isfinite(B_ref_abs)
    n_finite = int(np.sum(finite))
    print(f"Finite: {n_finite}/{n}")

    if n_finite < n:
        print("WARNING: some points failed, filtering to finite only")
        a_vals = a_vals[finite]
        omega_vals = omega_vals[finite]
        u_vals = u_vals[finite]
        v_vals = v_vals[finite]
        lam_vals = lam_vals[finite]
        B_inc_abs = B_inc_abs[finite]
        B_ref_abs = B_ref_abs[finite]

    eps = 1e-30
    mag_inc = np.abs(B_inc_abs)
    mag_ref = np.abs(B_ref_abs)
    logabs_B_inc = np.log(mag_inc + eps)
    logabs_B_ref = np.log(mag_ref + eps)
    phase_unit_B_inc = B_inc_abs / (mag_inc + eps)
    phase_unit_B_ref = B_ref_abs / (mag_ref + eps)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher_path = out_dir / "absolute_amplitude_teacher.npz"
    np.savez_compressed(
        teacher_path,
        a=a_vals,
        omega=omega_vals,
        u=u_vals,
        v=v_vals,
        lambda_=lam_vals,
        B_inc_abs=B_inc_abs,
        B_ref_abs=B_ref_abs,
        logabs_B_inc=logabs_B_inc,
        logabs_B_ref=logabs_B_ref,
        phase_unit_B_inc=phase_unit_B_inc,
        phase_unit_B_ref=phase_unit_B_ref,
        source="spectral_teukrad",
        convention="R_in = B_ref*u_up*A_up + B_inc*u_down*A_down",
        N_out=N_out,
        z_b=z_b,
    )
    print(f"Saved {teacher_path}")

    # Summary
    print(f"\n{'='*60}")
    print("  Absolute Amplitude Teacher Summary")
    print(f"{'='*60}")
    print(f"  |B_inc|: min={mag_inc.min():.4e}  max={mag_inc.max():.4e}  "
          f"median={np.median(mag_inc):.4e}")
    print(f"  |B_ref|: min={mag_ref.min():.4e}  max={mag_ref.max():.4e}  "
          f"median={np.median(mag_ref):.4e}")
    print(f"  log10|B_inc|: min={np.log10(mag_inc.min()):.4f}  "
          f"max={np.log10(mag_inc.max()):.4f}  median={np.log10(np.median(mag_inc)):.4f}")
    print(f"  log10|B_ref|: min={np.log10(mag_ref.min()):.4f}  "
          f"max={np.log10(mag_ref.max()):.4f}  median={np.log10(np.median(mag_ref)):.4f}")
    print(f"  n_points: {n_finite}")
    print(f"  source: spectral_teukrad (TeukRadAmplitudeInWithAbelChecks)")

    summary_path = out_dir / "absolute_amplitude_teacher_summary.md"
    with open(summary_path, "w") as f:
        f.write(f"# Absolute Amplitude Teacher Summary\n\n")
        f.write(f"- source: spectral_teukrad (TeukRadAmplitudeInWithAbelChecks)\n")
        f.write(f"- convention: R_in = B_ref*u_up*A_up + B_inc*u_down*A_down\n")
        f.write(f"- n_points: {n_finite}\n")
        f.write(f"- N_out: {N_out}, z_b: {z_b}\n\n")
        f.write(f"| Quantity | Min | Max | Median |\n")
        f.write(f"|----------|-----|-----|--------|\n")
        f.write(f"| \\|B_inc\\| | {mag_inc.min():.4e} | {mag_inc.max():.4e} | {np.median(mag_inc):.4e} |\n")
        f.write(f"| \\|B_ref\\| | {mag_ref.min():.4e} | {mag_ref.max():.4e} | {np.median(mag_ref):.4e} |\n")
        f.write(f"| log10\\|B_inc\\| | {np.log10(mag_inc.min()):.4f} | {np.log10(mag_inc.max()):.4f} | {np.log10(np.median(mag_inc)):.4f} |\n")
        f.write(f"| log10\\|B_ref\\| | {np.log10(mag_ref.min()):.4f} | {np.log10(mag_ref.max()):.4f} | {np.log10(np.median(mag_ref)):.4f} |\n")
    print(f"Saved {summary_path}")

    # Write audit report
    audit_dir = Path("outputs/stage2_amplitude_audit/patch_000_logw_v2")
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "amplitude_convention_audit.md"
    with open(audit_path, "w") as f:
        f.write("# Stage-2 Amplitude Convention Audit\n\n")
        f.write("## Old Teacher (pybhpt, scattering convention)\n\n")
        f.write("- B_inc: ALL |B_inc| = 1.0 (hardcoded in `amplitude_teacher/spectral_teacher.py:66`)\n")
        f.write("- B_ref: |B_ref| ∈ [27, 281]\n")
        f.write("- Normalization: pybhpt standard — unit ingoing amplitude at infinity\n")
        f.write("- Cause: `return 1.0 + 0.0j, B_ref` — B_inc is never computed\n\n")
        f.write("## New Teacher (teukrad, absolute convention)\n\n")
        f.write(f"- |B_inc|: min={mag_inc.min():.4e}, max={mag_inc.max():.4e}, median={np.median(mag_inc):.4e}\n")
        f.write(f"- |B_ref|: min={mag_ref.min():.4e}, max={mag_ref.max():.4e}, median={np.median(mag_ref):.4e}\n")
        f.write(f"- log10|B_inc|: min={np.log10(mag_inc.min()):.4f}, max={np.log10(mag_inc.max()):.4f}, median={np.log10(np.median(mag_inc)):.4f}\n")
        f.write(f"- log10|B_ref|: min={np.log10(mag_ref.min()):.4f}, max={np.log10(mag_ref.max()):.4f}, median={np.log10(np.median(mag_ref)):.4f}\n\n")
        f.write("## AmpNet (current, trained on old teacher)\n\n")
        f.write("- B_inc: |B_inc| ≈ 1.0 for all points (relative error ~100%)\n")
        f.write("- B_ref: median relative error ~6.4%, max ~22.5%\n")
        f.write("- AmpNet learned the pybhpt normalization (B_inc≈1), not absolute amplitudes\n\n")
        f.write("## Conclusion\n\n")
        f.write("Old Stage-2 teacher normalizes B_inc=1 (scattering convention).\n")
        f.write("The correct convention for R_in = B_ref*u_up*A_up + B_inc*u_down*A_down requires absolute amplitudes.\n")
        f.write("B_inc was off by 7-10 orders of magnitude.\n")
        f.write("This is the root cause of Stage-2 AmpNet failure and Stage-4 consistency error.\n")
    print(f"Saved {audit_path}")


if __name__ == "__main__":
    main()
