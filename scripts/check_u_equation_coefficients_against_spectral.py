#!/usr/bin/env python3
"""Compare Stage-3 u-equation coefficients (y-space) with spectral coeffs_numeric (z-space).

Verifies the transformation:
    C2_y = 4 * B2_z,   C1_y = 2 * B1_z,   C0_y = B0_z
    where y = 2z - 1

Usage:
  python scripts/check_u_equation_coefficients_against_spectral.py
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from physical_ansatz.u_residual import compute_u_equation_coefficients
from utils.amplitude import coeffs_numeric
from utils.mode import KerrMode


def main():
    # Load calibration points
    pts_json = PROJECT_ROOT / "outputs/spectral_calibration/patch_000_logw_v2/patch0_calibration_points.json"
    with open(pts_json) as f:
        points = json.load(f)

    # z values to check (avoid exact 0 and 1)
    z_vals = [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 0.5]
    bases = ["up", "down"]

    results = []
    max_err = {"C2": 0.0, "C1": 0.0, "C0": 0.0}
    max_err_point = {"C2": "", "C1": "", "C0": ""}

    for pt in points:
        label = pt["label"]
        a_val = float(pt["a"])
        omega_val = float(pt["omega"])
        lam_val = complex(float(pt["lambda_real"]), float(pt.get("lambda_imag", 0)))

        mode = KerrMode(M=1.0, a=a_val, omega=omega_val, ell=2, m=2, s=-2, lam=lam_val)

        for basis in bases:
            for z_val in z_vals:
                y_val = 2.0 * z_val - 1.0

                # Spectral coefficients (z-space)
                B2_z, B1_z, B0_z = coeffs_numeric(np.array([z_val]), mode, basis)
                B2_z, B1_z, B0_z = complex(B2_z[0]), complex(B1_z[0]), complex(B0_z[0])

                # u-equation coefficients (y-space)
                a_t = torch.tensor([[a_val]], dtype=torch.float64)
                omega_t = torch.tensor([[omega_val]], dtype=torch.float64)
                lam_t = torch.tensor([[lam_val]], dtype=torch.complex128)
                y_t = torch.tensor([[y_val]], dtype=torch.float64)

                C2_y, C1_y, C0_y = compute_u_equation_coefficients(
                    a_t, omega_t, y_t, lam_t, M=1.0, s=-2, m=2, basis=basis)

                C2 = complex(C2_y[0, 0].item())
                C1 = complex(C1_y[0, 0].item())
                C0 = complex(C0_y[0, 0].item())

                # Expected relations: C2=4*B2, C1=2*B1, C0=B0
                ref_C2 = 4.0 * B2_z
                ref_C1 = 2.0 * B1_z
                ref_C0 = B0_z

                def rel_err(val, ref):
                    denom = abs(ref)
                    if denom < 1e-30:
                        return abs(val - ref)
                    return abs(val - ref) / denom

                err_C2 = rel_err(C2, ref_C2)
                err_C1 = rel_err(C1, ref_C1)
                err_C0 = rel_err(C0, ref_C0)

                results.append({
                    "label": label, "basis": basis, "z": z_val,
                    "C2": C2, "ref_C2": ref_C2, "rel_err_C2": err_C2,
                    "C1": C1, "ref_C1": ref_C1, "rel_err_C1": err_C1,
                    "C0": C0, "ref_C0": ref_C0, "rel_err_C0": err_C0,
                })

                for name, err in [("C2", err_C2), ("C1", err_C1), ("C0", err_C0)]:
                    if err > max_err[name]:
                        max_err[name] = err
                        max_err_point[name] = f"{label}/{basis}/z={z_val}"

    # Print results
    print(f"{'label':12s} {'basis':6s} {'z':>8s}  {'rel_err_C2':>12s}  {'rel_err_C1':>12s}  {'rel_err_C0':>12s}")
    print("-" * 78)
    for r in results:
        print(f"{r['label']:12s} {r['basis']:6s} {r['z']:8.1e}  "
              f"{r['rel_err_C2']:12.3e}  {r['rel_err_C1']:12.3e}  {r['rel_err_C0']:12.3e}")

    print(f"\n--- Max errors ---")
    for name in ["C2", "C1", "C0"]:
        print(f"  {name}: {max_err[name]:.3e}  at {max_err_point[name]}")

    # Threshold check
    all_ok = True
    for r in results:
        for name in ["rel_err_C2", "rel_err_C1", "rel_err_C0"]:
            if r[name] > 1e-6:
                all_ok = False
                break

    print(f"\nPASS: {all_ok}  (threshold: 1e-6 at z>=1e-4, 1e-10 at larger z)")

    # Save outputs
    out_dir = PROJECT_ROOT / "outputs/spectral_calibration/patch_000_logw_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON - convert complex to [real, imag] for serialization
    def _serialize(v):
        if isinstance(v, complex):
            return [float(v.real), float(v.imag)]
        if isinstance(v, (np.integer, np.floating)):
            return float(v)
        return v

    results_serializable = [{k: _serialize(v) for k, v in r.items()} for r in results]
    with open(out_dir / "u_equation_coeff_check.json", "w") as f:
        json.dump({"results": results_serializable, "max_errors": {
            "C2": {"value": max_err["C2"], "at": max_err_point["C2"]},
            "C1": {"value": max_err["C1"], "at": max_err_point["C1"]},
            "C0": {"value": max_err["C0"], "at": max_err_point["C0"]},
        }, "pass": all_ok}, f, indent=2)

    # Markdown
    lines = []
    lines.append("# u-equation coefficient check")
    lines.append("")
    lines.append("| label | basis | z | rel_err_C2 | rel_err_C1 | rel_err_C0 |")
    lines.append("|-------|-------|---|------------|------------|------------|")
    for r in results:
        lines.append(f"| {r['label']} | {r['basis']} | {r['z']:.1e} | "
                     f"{r['rel_err_C2']:.3e} | {r['rel_err_C1']:.3e} | {r['rel_err_C0']:.3e} |")
    lines.append("")
    lines.append("## Max errors")
    for name in ["C2", "C1", "C0"]:
        lines.append(f"- **{name}**: {max_err[name]:.3e} at {max_err_point[name]}")
    lines.append(f"\n**PASS**: {all_ok}")
    with open(out_dir / "u_equation_coeff_check.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\nSaved to {out_dir / 'u_equation_coeff_check.json'}")
    print(f"         {out_dir / 'u_equation_coeff_check.md'}")

    # Also check individual terms for one point to debug sign issues
    print("\n--- Detailed breakdown (center/up/z=0.1) ---")
    z_dbg = 0.1
    y_dbg = 2.0 * z_dbg - 1.0
    pt = points[0]  # center
    mode = KerrMode(M=1.0, a=float(pt["a"]), omega=float(pt["omega"]),
                    ell=2, m=2, s=-2, lam=complex(float(pt["lambda_real"]), 0))
    B2, B1, B0 = coeffs_numeric(np.array([z_dbg]), mode, "up")
    B2, B1, B0 = complex(B2[0]), complex(B1[0]), complex(B0[0])

    a_t = torch.tensor([[float(pt["a"])]], dtype=torch.float64)
    omega_t = torch.tensor([[float(pt["omega"])]], dtype=torch.float64)
    lam_t = torch.tensor([[complex(float(pt["lambda_real"]), 0)]], dtype=torch.complex128)
    y_t = torch.tensor([[y_dbg]], dtype=torch.float64)
    C2, C1, C0 = compute_u_equation_coefficients(a_t, omega_t, y_t, lam_t, M=1.0, s=-2, m=2, basis="up")
    C2v, C1v, C0v = complex(C2[0,0].item()), complex(C1[0,0].item()), complex(C0[0,0].item())

    print(f"  B2={B2:.6e},  4*B2={4*B2:.6e},  C2={C2v:.6e},  diff={C2v - 4*B2:.3e}")
    print(f"  B1={B1:.6e},  2*B1={2*B1:.6e},  C1={C1v:.6e},  diff={C1v - 2*B1:.3e}")
    print(f"  B0={B0:.6e},  B0  ={B0:.6e},  C0={C0v:.6e},  diff={C0v - B0:.3e}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
