#!/usr/bin/env python3
"""Scan N and z_m to find spectral parameters achieving <1e-8 MMA agreement.

For each calibration point, sweeps over N_out, N_in, z_m and measures:
  - Internal consistency (detS_residual, abel_residuals)
  - MMA comparison (B_inc, B_ref) when Mathematica is available

Usage:
  python scripts/scan_spectral_accuracy.py
  python scripts/scan_spectral_accuracy.py --fast  # fewer scan points
  python scripts/scan_spectral_accuracy.py --labels low_omega  # single point
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from utils.mode import KerrMode
from utils.amplitude import TeukRadAmplitudeInWithAbelChecks


def _try_import_wolfram():
    try:
        from wolframclient.evaluation import WolframLanguageSession
        from wolframclient.language import wlexpr
        return (WolframLanguageSession, wlexpr), None
    except ImportError as e:
        return None, str(e)


def run_mma_for_point(label, a, omega, lam, wl_path, kernel_path,
                       WolframLanguageSession, wlexpr):
    """Run Mathematica and return B_inc, B_ref."""
    path_wl = wl_path
    with open(path_wl) as f:
        wl_src = f.read()

    M_val = 1.0
    s_val = -2
    ell_val = 2
    m_val = 2

    expr = wlexpr(wl_src)
    session = WolframLanguageSession(kernel_path)
    try:
        session.start()
        session.evaluate(expr)
        result = session.evaluate(
            wlexpr(f'ComputeAmplitudes[{a},{omega},{ell_val},{m_val},{s_val},{M_val}]'))
        if result is None:
            return None, "ComputeAmplitudes returned None"

        # Extract complex values from WLFunction
        def wl_to_complex(wl_val):
            if hasattr(wl_val, 'args') and len(wl_val.args) == 2:
                return complex(float(wl_val.args[0]), float(wl_val.args[1]))
            return complex(wl_val)

        B_inc = wl_to_complex(result["Incidence"])
        B_ref = wl_to_complex(result["Reflection"])
        B_trans = wl_to_complex(result["Transmission"])
        return {"B_inc": B_inc, "B_ref": B_ref, "B_trans": B_trans}, None
    except Exception as e:
        return None, str(e)
    finally:
        session.terminate()


def rel_err(val, ref):
    denom = abs(ref)
    if denom < 1e-30:
        return abs(val - ref)
    return abs(val - ref) / denom


def main():
    parser = argparse.ArgumentParser(description="Scan spectral N/z_m for accuracy")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    args_cli = parser.parse_args()

    pts_json = PROJECT_ROOT / "outputs/spectral_calibration/patch_000_logw_v2/patch0_calibration_points.json"
    with open(pts_json) as f:
        points = json.load(f)

    if args_cli.labels:
        points = [p for p in points if p["label"] in args_cli.labels]

    # Scan ranges
    if args_cli.fast:
        z_m_vals = [0.3, 0.5, 0.7]
        N_vals = [(80, 80), (120, 120)]
    else:
        z_m_vals = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        N_vals = [(60, 60), (80, 80), (100, 100), (120, 120), (80, 120), (120, 80)]

    results = {}

    for pt in points:
        label = pt["label"]
        a_val = float(pt["a"])
        omega_val = float(pt["omega"])
        lam_val = complex(float(pt["lambda_real"]), float(pt.get("lambda_imag", 0)))

        mode = KerrMode(M=1.0, a=a_val, omega=omega_val, ell=2, m=2, s=-2,
                        lam=lam_val)

        print(f"\n{'='*70}")
        print(f"  {label}: a={a_val:.6f}, omega={omega_val:.6e}")
        print(f"{'='*70}")
        print(f"  {'zm':>6s} {'N_out':>6s} {'N_in':>6s}  {'detS_res':>10s}  {'abel_out':>10s}  {'abel_in':>10s}  {'B_inc_mag':>14s}")

        best = None
        best_detS = 1e100

        for z_m in z_m_vals:
            for n_out, n_in in N_vals:
                try:
                    amp = TeukRadAmplitudeInWithAbelChecks(
                        mode, N_in=n_in, N_out=n_out, z_m=z_m,
                    )
                except Exception as e:
                    print(f"  {z_m:6.2f} {n_out:6d} {n_in:6d}  ERROR: {e}")
                    continue

                detS = float(amp.detS_residual)
                abel_out = float(amp.outer_abel_residual)
                abel_in = float(amp.inner_abel_residual)

                marker = ""
                if detS < best_detS:
                    best_detS = detS
                    best = {
                        "z_m": z_m, "N_out": n_out, "N_in": n_in,
                        "detS": detS, "abel_out": abel_out, "abel_in": abel_in,
                        "B_inc": amp.B_inc, "B_ref": amp.B_ref, "B_trans": amp.B_trans,
                    }
                    marker = " <--"

                print(f"  {z_m:6.2f} {n_out:6d} {n_in:6d}  "
                      f"{detS:10.3e}  {abel_out:10.3e}  {abel_in:10.3e}  "
                      f"{abs(amp.B_inc):14.6e}{marker}")

        if best is None:
            print(f"  No successful computation for {label}")
            results[label] = {"error": "all failed"}
            continue

        # Now run MMA once for this point and compare all results
        results[label] = {
            "best_params": {
                "z_m": best["z_m"], "N_out": best["N_out"], "N_in": best["N_in"],
                "detS_residual": best["detS"],
                "abel_out_residual": best["abel_out"],
                "abel_in_residual": best["abel_in"],
            }
        }

        print(f"\n  Best: zm={best['z_m']}, N_out={best['N_out']}, N_in={best['N_in']}")
        print(f"         detS={best['detS']:.3e}, abel_out={best['abel_out']:.3e}, "
              f"abel_in={best['abel_in']:.3e}")

    # ---- MMA comparison for best params ----
    print(f"\n{'='*70}")
    print("  MMA comparison (best params vs Mathematica)")
    print(f"{'='*70}")

    WolframClient, mma_err = _try_import_wolfram()
    if WolframClient is not None:
        WolframLanguageSession, wlexpr = WolframClient
        from scripts.compare_spectral_with_mma_patch0 import _get_mma_kernel_path, _wl_complex_to_py

        from wolframclient.language import wlexpr as wl_expr_fn

        kernel_path = _get_mma_kernel_path()
        wl_path = str(PROJECT_ROOT / "mma" / "Radial_Function.wl")

        if kernel_path:
            session = WolframLanguageSession(kernel_path)
            try:
                session.start()
                with open(wl_path) as f:
                    session.evaluate(wl_expr_fn(f.read()))

                for pt in points:
                    label = pt["label"]
                    if label not in results or "error" in results[label]:
                        continue
                    if "best_params" not in results[label]:
                        continue

                    a_val = float(pt["a"])
                    omega_val = float(pt["omega"])

                    mma_result = session.evaluate(
                        wl_expr_fn(f'ComputeAmplitudes[{a_val},{omega_val},2,2,-2,1.0]'))

                    if mma_result is None:
                        print(f"  {label}: MMA returned None")
                        continue

                    def wl_to_c(w):
                        if hasattr(w, 'args') and len(w.args) == 2:
                            return complex(float(w.args[0]), float(w.args[1]))
                        return complex(w)

                    mma_B_inc = wl_to_c(mma_result["Incidence"])
                    mma_B_ref = wl_to_c(mma_result["Reflection"])

                    best = results[label]["best_params"]
                    spec_B_inc = results[label].get("best_B_inc")
                    spec_B_ref = results[label].get("best_B_ref")

                    # Re-run with best params to get B_inc/B_ref
                    mode = KerrMode(M=1.0, a=a_val, omega=omega_val, ell=2, m=2, s=-2,
                                    lam=complex(float(pt["lambda_real"]),
                                                float(pt.get("lambda_imag", 0))))
                    amp = TeukRadAmplitudeInWithAbelChecks(
                        mode, N_in=best["N_in"], N_out=best["N_out"], z_m=best["z_m"])
                    spec_B_inc = amp.B_inc
                    spec_B_ref = amp.B_ref

                    err_inc = rel_err(spec_B_inc, mma_B_inc)
                    err_ref = rel_err(spec_B_ref, mma_B_ref)

                    results[label]["best_B_inc"] = f"({spec_B_inc.real}+{spec_B_inc.imag}j)"
                    results[label]["best_B_ref"] = f"({spec_B_ref.real}+{spec_B_ref.imag}j)"
                    results[label]["mma_B_inc"] = f"({mma_B_inc.real}+{mma_B_inc.imag}j)"
                    results[label]["mma_B_ref"] = f"({mma_B_ref.real}+{mma_B_ref.imag}j)"
                    results[label]["rel_err_B_inc"] = float(err_inc)
                    results[label]["rel_err_B_ref"] = float(err_ref)

                    print(f"  {label:12s}: "
                          f"B_inc_err={err_inc:.3e}  B_ref_err={err_ref:.3e}  "
                          f"detS={best['detS']:.2e}")

            finally:
                session.terminate()
        else:
            print("  MMA kernel not found, skipping MMA comparison")
    else:
        print(f"  WolframClient unavailable: {mma_err}")

    # Save
    out_dir = PROJECT_ROOT / "outputs/spectral_calibration/patch_000_logw_v2"
    with open(out_dir / "spectral_accuracy_scan.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_dir / 'spectral_accuracy_scan.json'}")

    # Summary table
    print(f"\n{'='*70}")
    print("  Summary: Best parameters per point")
    print(f"{'='*70}")
    for label, r in results.items():
        if "best_params" in r:
            bp = r["best_params"]
            err_inc = r.get("rel_err_B_inc", float("nan"))
            err_ref = r.get("rel_err_B_ref", float("nan"))
            print(f"  {label:12s}: zm={bp['z_m']}, N=({bp['N_out']},{bp['N_in']}), "
                  f"detS={bp['detS']:.2e}, "
                  f"B_inc_err={err_inc:.2e}, B_ref_err={err_ref:.2e}")


if __name__ == "__main__":
    main()
