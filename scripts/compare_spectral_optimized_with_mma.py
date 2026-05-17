#!/usr/bin/env python3
"""Compare optimized spectral amplitudes against Mathematica.

Usage:
  python scripts/compare_spectral_optimized_with_mma.py
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from utils.mode import KerrMode
from utils.amplitude import TeukRadAmplitudeInWithAbelChecks


def rel_err(val, ref):
    denom = abs(ref)
    if denom < 1e-30:
        return abs(val - ref)
    return abs(val - ref) / denom


def main():
    pts_json = PROJECT_ROOT / "outputs/spectral_calibration/patch_000_logw_v2/patch0_calibration_points.json"
    with open(pts_json) as f:
        points = json.load(f)

    # Optimized parameters from scan
    optimal_params = {
        "center":     {"z_m": 0.08, "N_out": 120, "N_in": 200},
        "low_omega":  {"z_m": 0.07, "N_out": 250, "N_in": 300},
        "high_omega": {"z_m": 0.30, "N_out":  80, "N_in":  80},
        "low_a":      {"z_m": 0.08, "N_out": 160, "N_in": 200},
        "high_a":     {"z_m": 0.30, "N_out":  80, "N_in":  80},
    }
    # Baseline for comparison
    baseline_params = {"z_m": 0.30, "N_out": 80, "N_in": 80}

    print(f"{'point':12s} {'param':12s} {'detS':>10s} {'abel_out':>10s} {'B_inc_mag':>14s} {'B_ref_mag':>14s}")
    print("-" * 78)

    results = {}
    for pt in points:
        label = pt["label"]
        a_val = float(pt["a"])
        omega_val = float(pt["omega"])
        lam_val = complex(float(pt["lambda_real"]), float(pt.get("lambda_imag", 0)))

        mode = KerrMode(M=1.0, a=a_val, omega=omega_val, ell=2, m=2, s=-2, lam=lam_val)

        res = {"label": label, "a": a_val, "omega": omega_val}

        for param_label, params in [("baseline", baseline_params), ("optimized", optimal_params.get(label, baseline_params))]:
            amp = TeukRadAmplitudeInWithAbelChecks(mode, **params)
            detS = float(amp.detS_residual)
            abel_out = float(amp.outer_abel_residual)
            key = f"{param_label}_detS"
            res[key] = detS
            res[f"{param_label}_abel_out"] = abel_out
            res[f"{param_label}_B_inc"] = str(amp.B_inc)
            res[f"{param_label}_B_ref"] = str(amp.B_ref)

            marker = " <--" if param_label == "optimized" else ""
            print(f"{label:12s} {param_label:12s} {detS:10.3e} {abel_out:10.3e} "
                  f"{abs(amp.B_inc):14.6e} {abs(amp.B_ref):14.6e}{marker}")

        results[label] = res

    # MMA comparison with optimized params
    print(f"\n{'='*70}")
    print("  MMA Comparison (optimized params)")
    print(f"{'='*70}")

    try:
        from wolframclient.evaluation import WolframLanguageSession
        from wolframclient.language import wlexpr

        # Find kernel
        import subprocess
        kernel_result = subprocess.run(["which", "WolframKernel"], capture_output=True, text=True)
        kernel_path = kernel_result.stdout.strip() if kernel_result.returncode == 0 else None
        if not kernel_path:
            import os
            kernel_path = os.environ.get("WOLFRAM_KERNEL", None)
        if not kernel_path:
            print("  WolframKernel not found, skipping MMA")
            kernel_path = None

        if kernel_path:
            wl_path = PROJECT_ROOT / "mma" / "Radial_Function.wl"
            with open(wl_path) as f:
                wl_src = f.read()

            session = WolframLanguageSession(kernel_path)
            try:
                session.start()
                session.evaluate(wlexpr(wl_src))

                print(f"  {'point':12s} {'B_inc_err':>10s} {'B_ref_err':>10s} {'mma_B_inc_mag':>14s}")
                print(f"  {'-'*50}")

                for pt in points:
                    label = pt["label"]
                    a_val = float(pt["a"])
                    omega_val = float(pt["omega"])

                    result = session.evaluate(
                        wlexpr(f'ComputeAmplitudes[{a_val},{omega_val},2,2,-2,1.0]'))

                    if result is None:
                        print(f"  {label:12s} MMA returned None")
                        results[label]["mma_error"] = "None"
                        continue

                    def wl_to_c(w):
                        if hasattr(w, 'args') and len(w.args) == 2:
                            return complex(float(w.args[0]), float(w.args[1]))
                        return complex(w)

                    mma_B_inc = wl_to_c(result["Incidence"])
                    mma_B_ref = wl_to_c(result["Reflection"])

                    # Re-run spectral with optimal params
                    params = optimal_params.get(label, baseline_params)
                    mode = KerrMode(M=1.0, a=a_val, omega=omega_val, ell=2, m=2, s=-2,
                                    lam=complex(float(pt["lambda_real"]),
                                                float(pt.get("lambda_imag", 0))))
                    amp = TeukRadAmplitudeInWithAbelChecks(mode, **params)
                    spec_B_inc = amp.B_inc
                    spec_B_ref = amp.B_ref

                    err_inc = rel_err(spec_B_inc, mma_B_inc)
                    err_ref = rel_err(spec_B_ref, mma_B_ref)

                    results[label]["optimized_B_inc_err"] = float(err_inc)
                    results[label]["optimized_B_ref_err"] = float(err_ref)
                    results[label]["mma_B_inc"] = str(mma_B_inc)
                    results[label]["mma_B_ref"] = str(mma_B_ref)
                    results[label]["mma_B_trans"] = str(wl_to_c(result["Transmission"]))

                    print(f"  {label:12s} {err_inc:10.3e} {err_ref:10.3e} {abs(mma_B_inc):14.6e}")

            finally:
                session.terminate()
    except ImportError:
        print("  wolframclient unavailable, skipping MMA")

    # Save
    out_dir = PROJECT_ROOT / "outputs/spectral_calibration/patch_000_logw_v2"
    with open(out_dir / "spectral_optimized_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*70}")
    print("  Summary: Improvement over baseline (detS)")
    print(f"{'='*70}")
    for label, r in results.items():
        base = r.get("baseline_detS", float("nan"))
        opt = r.get("optimized_detS", float("nan"))
        if base > 0 and opt > 0:
            factor = base / opt
            inc_err = r.get("optimized_B_inc_err", float("nan"))
            ref_err = r.get("optimized_B_ref_err", float("nan"))
            print(f"  {label:12s}: {base:.2e} -> {opt:.2e} ({factor:.0f}x), "
                  f"B_inc_err={inc_err:.2e}, B_ref_err={ref_err:.2e}")

    print(f"\nSaved to {out_dir / 'spectral_optimized_comparison.json'}")


if __name__ == "__main__":
    main()
