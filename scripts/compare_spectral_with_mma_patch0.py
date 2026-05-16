#!/usr/bin/env python3
"""Compare Python spectral amplitudes (B_inc, B_ref) with Mathematica reference.

For each patch 0 calibration point:
1. Call Mathematica ComputeAmplitudes to get B_inc, B_ref
2. Compute Python spectral amplitudes via TeukRadAmplitudeInWithAbelChecks
3. Compare and report differences

Usage:
  python scripts/compare_spectral_with_mma_patch0.py \
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


def _try_import_wolfram():
    """Try importing wolframclient. Returns (module, error_msg)."""
    try:
        from wolframclient.evaluation import WolframLanguageSession
        from wolframclient.language import wlexpr
        return (WolframLanguageSession, wlexpr), None
    except ImportError:
        return None, "wolframclient not installed (pip install wolframclient)"


def _get_mma_kernel_path():
    """Find Mathematica kernel path from env or common locations."""
    import os
    env_path = os.environ.get("WOLFRAM_KERNEL")
    if env_path and Path(env_path).exists():
        return env_path
    # Common paths
    candidates = [
        "/mnt/f/mma/WolframKernel.exe",
        "/usr/local/Wolfram/Mathematica/14.0/Executables/WolframKernel",
        "/usr/local/Wolfram/Mathematica/13.0/Executables/WolframKernel",
    ]
    for cand in candidates:
        if Path(cand).exists():
            return cand
    return None


def _py_list_to_mma_list(py_list):
    """Convert Python list to Mathematica list string {a, b, c, ...}."""
    return "{" + ", ".join(str(x) for x in py_list) + "}"


def _wl_complex_to_py(wl_val):
    """Convert wolframclient WLFunction Complex to Python complex."""
    if hasattr(wl_val, 'args') and len(wl_val.args) == 2:
        return complex(float(wl_val.args[0]), float(wl_val.args[1]))
    return complex(wl_val)


def compute_mma_amplitudes(session, wlexpr, s, l, m, a, omega):
    """Call Mathematica ComputeAmplitudes and return B_inc, B_ref, B_trans.

    The wolframclient returns an ImmutableDict whose values are WLFunction
    Complex objects. We extract real/imag via .args.
    """
    expr = f"ComputeAmplitudes[{s}, {l}, {m}, {a:.16g}, {omega:.16g}]"
    result = session.evaluate(wlexpr(expr))
    try:
        return {
            "B_inc": _wl_complex_to_py(result["Incidence"]),
            "B_ref": _wl_complex_to_py(result["Reflection"]),
            "B_trans": _wl_complex_to_py(result["Transmission"]),
        }
    except (TypeError, KeyError, IndexError) as e:
        return {"raw": str(result), "error": str(e)}


def compute_mma_rin_tail(session, wlexpr, s, l, m, a, omega, r_max, n_pts=512,
                          B_inc_check=None, B_ref_check=None):
    """Sample R_in on a large-r grid and optionally check spectral coefficients.

    Returns the sampled grid and, if B_inc_check/B_ref_check provided,
    the reconstruction residual ||R_in - (B_inc*A_down + B_ref*A_up)||.
    """
    from physical_ansatz.mapping import r_plus
    from physical_ansatz.u_basis import A_up as A_up_fn, A_down as A_down_fn
    import torch

    rp = r_plus(torch.tensor([[a]], dtype=torch.float64), 1.0).item()
    r_grid = np.geomspace(rp + 10, r_max, n_pts)
    r_list_str = _py_list_to_mma_list(r_grid)

    expr = f"SampleRinAtPoints[{s}, {l}, {m}, {a:.16g}, {omega:.16g}, {r_list_str}]"
    result = session.evaluate(wlexpr(expr))
    arr = np.array(result, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Unexpected Mathematica output shape: {arr.shape}")

    r_vals = arr[:, 0]
    Rin_vals = arr[:, 1] + 1j * arr[:, 2]

    result = {
        "r_range": [float(r_vals[0]), float(r_vals[-1])],
        "n_pts": n_pts,
        "Rin_at_rmax": complex(Rin_vals[-1]),
    }

    if B_inc_check is not None and B_ref_check is not None:
        r_t = torch.tensor(r_vals, dtype=torch.float64)
        a_t = torch.tensor([a], dtype=torch.float64)
        omega_t = torch.tensor([omega], dtype=torch.float64)
        A_d = A_down_fn(r_t, a_t, omega_t, M=1.0).numpy().ravel()
        A_u = A_up_fn(r_t, a_t, omega_t, M=1.0).numpy().ravel()
        Rin_recon = B_inc_check * A_d + B_ref_check * A_u
        recon_residual = float(np.linalg.norm(Rin_recon - Rin_vals))
        rel_recon = recon_residual / float(np.linalg.norm(Rin_vals))
        result["recon_residual"] = recon_residual
        result["recon_rel"] = rel_recon

    return result


def main():
    parser = argparse.ArgumentParser(description="Compare spectral amplitudes with Mathematica")
    parser.add_argument("--points-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--wl-path", type=str,
                        default=str(PROJECT_ROOT / "mma" / "Radial_Function.wl"))
    parser.add_argument("--kernel-path", type=str, default=None)
    parser.add_argument("--N-spectral", type=int, default=80)
    parser.add_argument("--zm", type=float, default=0.30)
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    args = parser.parse_args()

    with open(args.points_json) as f:
        points = json.load(f)
    if args.labels is not None:
        points = [p for p in points if p["label"] in args.labels]
    print(f"[mma-compare] {len(points)} points")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Python spectral ----
    print(f"\n[mma-compare] Python spectral: N={args.N_spectral} zm={args.zm}")
    spectral_results = {}
    for pt in points:
        label = pt["label"]
        a_val = pt["a"]
        omega_val = pt["omega"]
        lam_val = pt.get("lambda_real", pt.get("lam"))
        mode = KerrMode(M=1.0, a=a_val, omega=omega_val, ell=2, m=2, s=-2,
                        lam=complex(lam_val) if lam_val is not None else None)
        amp = TeukRadAmplitudeInWithAbelChecks(
            mode, N_in=args.N_spectral, N_out=args.N_spectral, z_m=args.zm,
        )
        spectral_results[label] = {
            "B_inc": amp.B_inc, "B_ref": amp.B_ref, "B_trans": amp.B_trans,
            "outer_abel_residual": amp.outer_abel_residual,
            "inner_abel": amp.inner_abel_residual,
            "detS_residual": amp.detS_residual,
        }
        print(f"  {label:12s}: B_inc=({amp.B_inc.real:.6e},{amp.B_inc.imag:.6e}) "
              f"B_ref=({amp.B_ref.real:.6e},{amp.B_ref.imag:.6e}) "
              f"Abel_out={amp.outer_abel_residual:.2e}")

    # ---- Mathematica ----
    mma_available = False
    mma_results = {}

    WolframClient, mma_err = _try_import_wolfram()
    if WolframClient is None:
        print(f"\n[mma-compare] Mathematica unavailable: {mma_err}")
    else:
        WolframLanguageSession, wlexpr = WolframClient
        kernel_path = args.kernel_path or _get_mma_kernel_path()
        if kernel_path is None:
            print("\n[mma-compare] Mathematica kernel not found. "
                  "Set WOLFRAM_KERNEL env var or use --kernel-path.")
        else:
            print(f"\n[mma-compare] Starting Mathematica kernel: {kernel_path}")
            session = None
            try:
                session = WolframLanguageSession(kernel=kernel_path)
                session.start()
                # Load the WL package
                wl_path = str(Path(args.wl_path).resolve())
                wl_load = f'Get["{wl_path}"]'
                session.evaluate(wlexpr(wl_load))
                print(f"[mma-compare] Loaded {wl_path}")

                for pt in points:
                    label = pt["label"]
                    a_val = pt["a"]
                    omega_val = pt["omega"]
                    s, l, m_val = -2, 2, 2
                    t0 = time.time()
                    try:
                        # Method 1: Direct ComputeAmplitudes
                        amps = compute_mma_amplitudes(session, wlexpr, s, l, m_val, a_val, omega_val)
                        # Method 2: R_in reconstruction check using spectral B_inc/B_ref
                        sp = spectral_results[label]
                        tail = compute_mma_rin_tail(session, wlexpr, s, l, m_val, a_val, omega_val,
                                                    r_max=2000.0,
                                                    B_inc_check=sp["B_inc"],
                                                    B_ref_check=sp["B_ref"])
                        elapsed = time.time() - t0
                        mma_results[label] = {
                            "B_inc_direct": amps.get("B_inc", None),
                            "B_ref_direct": amps.get("B_ref", None),
                            "B_trans_direct": amps.get("B_trans", None),
                            "recon_residual": tail.get("recon_residual", None),
                            "recon_rel": tail.get("recon_rel", None),
                        }
                        recon_str = f"recon={tail.get('recon_rel', 0):.2e}" if tail.get("recon_rel") is not None else ""
                        print(f"  {label:12s}: MMA B_inc={amps.get('B_inc')} "
                              f"B_ref={amps.get('B_ref')}  "
                              f"{recon_str}  ({elapsed:.1f}s)")
                    except Exception as pt_err:
                        print(f"  {label:12s}: MMA FAILED — {pt_err}")
                        mma_results[label] = {"error": str(pt_err)}

                mma_available = True
            except Exception as e:
                print(f"\n[mma-compare] Mathematica error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if session is not None:
                    try:
                        session.terminate()
                        print("[mma-compare] Mathematica kernel terminated.")
                    except Exception:
                        pass

    # ---- Comparison table ----
    print(f"\n{'='*80}")
    print(f"{'Label':12s} {'B_inc (spectral)':40s} {'B_ref (spectral)':40s} {'Abel_out':12s}")
    print(f"{'='*80}")
    comparisons = []
    for pt in points:
        label = pt["label"]
        sp = spectral_results[label]
        binc_s = f"({sp['B_inc'].real:.6e},{sp['B_inc'].imag:.6e})"
        bref_s = f"({sp['B_ref'].real:.6e},{sp['B_ref'].imag:.6e})"
        print(f"{label:12s} {binc_s:40s} {bref_s:40s} {sp['outer_abel_residual']:.2e}")

        comp = {
            "label": label, "a": pt["a"], "omega": pt["omega"],
            "spectral_B_inc": str(sp["B_inc"]),
            "spectral_B_ref": str(sp["B_ref"]),
            "spectral_B_trans": str(sp["B_trans"]),
            "spectral_abel": sp["outer_abel_residual"],
            "spectral_detS": sp["detS_residual"],
        }
        if label in mma_results:
            m = mma_results[label]
            if "error" not in m:
                comp["mma_B_inc"] = str(m["B_inc_direct"])
                comp["mma_B_ref"] = str(m["B_ref_direct"])
                comp["mma_B_trans"] = str(m["B_trans_direct"])
                comp["mma_recon_residual"] = m.get("recon_residual")
                comp["mma_recon_rel"] = m.get("recon_rel")
                if m["B_inc_direct"] is not None:
                    try:
                        binc_m = complex(m["B_inc_direct"])
                        bref_m = complex(m["B_ref_direct"])
                        binc_diff = abs(sp["B_inc"] - binc_m) / max(abs(binc_m), 1e-30)
                        bref_diff = abs(sp["B_ref"] - bref_m) / max(abs(bref_m), 1e-30)
                        comp["rel_diff_B_inc"] = float(binc_diff)
                        comp["rel_diff_B_ref"] = float(bref_diff)
                        print(f"  MMA rel_diff: B_inc={binc_diff:.4e} B_ref={bref_diff:.4e}")
                    except (TypeError, ValueError):
                        pass
            else:
                comp["mma_error"] = m["error"]
        comparisons.append(comp)

    with open(out_dir / "mma_comparison_summary.json", "w") as f:
        json.dump({
            "mma_available": mma_available,
            "spectral_N": args.N_spectral,
            "spectral_zm": args.zm,
            "comparisons": comparisons,
        }, f, indent=2, default=str)
    print(f"\n[mma-compare] Summary saved to {out_dir / 'mma_comparison_summary.json'}")


if __name__ == "__main__":
    main()
