#!/usr/bin/env python3
"""Diagnose infinity Robin constraint at y=-1 for the current Stage-1 checkpoint.

Computes S(-1), S_y(-1), c_inf, and Robin residual for a set of parameter
points from patch 0. Also computes near-infinity PDE residual for comparison.

Usage:
  python scripts/diagnose_infinity_robin_stage1.py \
    --checkpoint outputs/stage1_artifacts/patch_000_logw_v2/stage1_best_model.pt \
    --device cuda
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from physical_ansatz.infinity_robin import (
    analytic_c_inf,
    infinity_robin_residual,
    compute_S_and_Sy_at_infinity,
)
from physical_ansatz.transform_y import horizon_regularity_slope
from physical_ansatz.residual_pinn import compute_f_derivatives_autograd
from utils.mode import KerrMode
from utils.compute_lambda_usage import compute_lambda


def load_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    from model.autoencoder_pinn import AutoencoderTeukolskyPINN
    model = AutoencoderTeukolskyPINN()
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/stage1_artifacts/patch_000_logw_v2/stage1_best_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="outputs/infinity_robin_diagnostics")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        # Auto-find latest stage1 checkpoint
        candidates = sorted(Path("outputs").rglob("stage1_best_model.pt"))
        if candidates:
            checkpoint_path = candidates[-1]
            print(f"Auto-found checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"No checkpoint at {args.checkpoint} and no alternatives found")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_checkpoint(checkpoint_path, device)
    print(f"  Loaded. encoder_config: {model.encoder.local_coord_mode}")

    # Parameter points covering the patch 0 range
    # a ∈ [0.01, 0.99], omega ∈ [0.01, 1.5], M=1, s=-2, l=2, m=2
    param_points = [
        # (a, omega, label)
        (0.5127, 0.0198, "difficult_point"),
        (0.1, 0.1, "low_a"),
        (0.5, 0.1, "center"),
        (0.9, 0.1, "high_a"),
        (0.5, 0.02, "low_omega"),
        (0.5, 0.5, "mid_omega"),
        (0.7, 0.05, "moderate"),
        (0.3, 0.03, "moderate_low"),
    ]

    M, s, l, m = 1.0, -2, 2, 2
    y_inf = torch.tensor([-1.0], device=device, dtype=torch.float64)
    y_near_inf = torch.tensor([-0.999, -0.997, -0.995, -0.99], device=device, dtype=torch.float64)

    results = []
    print(f"\n{'='*80}")
    print(f"Infinity Robin Diagnostic")
    print(f"{'='*80}")

    for a_val, omega_val, label in param_points:
        lam_val = compute_lambda(a_val, omega_val, l, m, s=s)
        mode = KerrMode(M=M, a=a_val, omega=omega_val, ell=l, m=m, s=s, lam=lam_val)

        a_t = torch.tensor([a_val], device=device, dtype=torch.float64)
        omega_t = torch.tensor([omega_val], device=device, dtype=torch.float64)
        lam_t = torch.tensor(lam_val, device=device, dtype=torch.complex128)

        # Compute c_inf analytically
        c_inf = analytic_c_inf(a_t, omega_t, lam_t, m=m, M=M, s=s)[0]

        # Compute slope for S construction
        slope = horizon_regularity_slope(a=a_t, omega=omega_t, lambda_=lam_t, m=m, M=M, s=s)

        # Evaluate f, f_y at y=-1 precisely (must NOT be inside torch.no_grad())
        f_inf, fy_inf, _ = compute_f_derivatives_autograd(model, a_t, omega_t, y_inf)
        f_val = f_inf[0, 0].detach()
        fy_val = fy_inf[0, 0].detach()

        S_val, Sy_val = compute_S_and_Sy_at_infinity(f_val, fy_val, slope[0])

        robin_residual = infinity_robin_residual(S_val, Sy_val, c_inf)
        robin_abs = abs(robin_residual)
        robin_rel = robin_abs / (abs(Sy_val) + abs(c_inf * S_val) + 1e-12)

        # Near-infinity PDE residual (not at y=-1 directly to avoid singularity)
        f_near, fy_near, fyy_near = compute_f_derivatives_autograd(model, a_t, omega_t, y_near_inf)

        # Compute S and derivatives at near-infinity points
        from physical_ansatz.transform_y import compose_reduced_shape_from_f
        S_near_list = []
        Sy_near_list = []
        for i in range(len(y_near_inf)):
            yi = y_near_inf[i:i+1]
            fi = f_near[:, i:i+1]
            fyi = fy_near[:, i:i+1]
            Si = compose_reduced_shape_from_f(fi[0], yi[0], slope[0])
            S_near_list.append(Si)
            # Approximate S_y from f, f_y
            _, Syi = compute_S_and_Sy_at_infinity(fi[0, 0], fyi[0, 0], slope[0])
            Sy_near_list.append(Syi.unsqueeze(0))
        S_near = torch.stack(S_near_list)
        Sy_near = torch.stack(Sy_near_list)

        # Compute PDE residual D2*S_yy + D1*S_y + D0*S at near-infinity points
        # For simplicity, compute A coeffs and D coeffs
        from physical_ansatz.teukolsky_coeffs import coeffs_x
        from physical_ansatz.mapping import r_plus, r_from_x
        from physical_ansatz.transform_y import transform_coeffs_x_to_y_S

        rp = r_plus(a_t, M)
        x_near = (y_near_inf + 1.0) / 2.0
        r_near = rp / x_near

        A2_near, A1_near, A0_near = coeffs_x(
            x_near.unsqueeze(0).unsqueeze(-1),
            a_t.unsqueeze(0).unsqueeze(-1),
            omega_t.unsqueeze(0).unsqueeze(-1),
            m, lam_t.unsqueeze(0).unsqueeze(-1), s=s, M=M,
        )
        D2_near, D1_near, D0_near = transform_coeffs_x_to_y_S(
            A2_near, A1_near, A0_near,
            r_near.unsqueeze(0).unsqueeze(-1),
            a_t.unsqueeze(0), omega_t.unsqueeze(0), m=m, M=M, s=s,
        )

        # S_yy estimate via finite diff of S_y
        Sy_near_vals = Sy_near.squeeze()
        Syy_near_approx = torch.zeros_like(Sy_near_vals)
        for i in range(1, len(y_near_inf) - 1):
            dy = y_near_inf[i+1] - y_near_inf[i-1]
            Syy_near_approx[i] = (Sy_near_vals[i+1] - Sy_near_vals[i-1]) / dy
        dy0 = y_near_inf[1] - y_near_inf[0]
        Syy_near_approx[0] = (Sy_near_vals[1] - Sy_near_vals[0]) / dy0
        dy_last = y_near_inf[-1] - y_near_inf[-2]
        Syy_near_approx[-1] = (Sy_near_vals[-1] - Sy_near_vals[-2]) / dy_last

        D2v = D2_near.squeeze(-1).squeeze(-1)
        D1v = D1_near.squeeze(-1).squeeze(-1)
        D0v = D0_near.squeeze(-1).squeeze(-1)
        S_v = S_near.squeeze()
        pde_res = D2v * Syy_near_approx + D1v * Sy_near_vals + D0v * S_v
        pde_res_abs = torch.abs(pde_res).detach().cpu().numpy()

        point_result = {
            "label": label,
            "a": a_val,
            "omega": omega_val,
            "lambda": {"real": lam_val.real, "imag": lam_val.imag},
            "r_plus": mode.rp,
            "k_hor": mode.k_hor,
            "S_minus1": {"real": S_val.real.item(), "imag": S_val.imag.item()},
            "Sy_minus1": {"real": Sy_val.real.item(), "imag": Sy_val.imag.item()},
            "c_inf": {"real": c_inf.real.item(), "imag": c_inf.imag.item()},
            "robin_residual_abs": robin_abs.item(),
            "robin_residual_rel": robin_rel.item(),
            "pde_residual_near_inf": pde_res_abs.tolist(),
        }

        print(f"\n--- {label}: a={a_val:.6g}, omega={omega_val:.6g} ---")
        print(f"  r_+ = {mode.rp:.6f}, k_hor = {mode.k_hor:.6f}")
        print(f"  c_inf (analytic) = {c_inf.item():.6e}")
        print(f"  S(-1)  = {S_val.item():.6e}")
        print(f"  S_y(-1) = {Sy_val.item():.6e}")
        print(f"  Robin residual |B| = {robin_abs.item():.4e}")
        print(f"  Robin residual rel = {robin_rel.item():.4e}")
        print(f"  Near-inf PDE |res| (max) = {pde_res_abs.max():.4e}")
        print(f"  c vs eps: ", end="")
        results.append(point_result)

    # Summary statistics
    robin_abs_vals = [r["robin_residual_abs"] for r in results]
    robin_rel_vals = [r["robin_residual_rel"] for r in results]

    print(f"\n{'='*80}")
    print(f"Summary ({len(results)} points)")
    print(f"{'='*80}")
    print(f"  Robin |B|: median={np.median(robin_abs_vals):.4e}, max={np.max(robin_abs_vals):.4e}")
    print(f"  Robin rel: median={np.median(robin_rel_vals):.4e}, max={np.max(robin_rel_vals):.4e}")

    # Save
    diagnostics = {
        "timestamp": timestamp,
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "summary": {
            "robin_abs_median": float(np.median(robin_abs_vals)),
            "robin_abs_max": float(np.max(robin_abs_vals)),
            "robin_rel_median": float(np.median(robin_rel_vals)),
            "robin_rel_max": float(np.max(robin_rel_vals)),
            "n_points": len(results),
        },
        "points": results,
    }

    json_path = out_dir / "diagnostics.json"
    with open(json_path, "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)
    print(f"\nSaved {json_path}")

    # CSV table
    csv_path = out_dir / "robin_residual_table.csv"
    with open(csv_path, "w") as f:
        f.write("label,a,omega,c_inf_real,c_inf_imag,S_real,S_imag,"
                "Sy_real,Sy_imag,robin_abs,robin_rel\n")
        for r in results:
            f.write(f"{r['label']},{r['a']},{r['omega']},"
                    f"{r['c_inf']['real']},{r['c_inf']['imag']},"
                    f"{r['S_minus1']['real']},{r['S_minus1']['imag']},"
                    f"{r['Sy_minus1']['real']},{r['Sy_minus1']['imag']},"
                    f"{r['robin_residual_abs']},{r['robin_residual_rel']}\n")
    print(f"Saved {csv_path}")

    # Markdown summary
    md_path = out_dir / "diagnostics_summary.md"
    with open(md_path, "w") as f:
        f.write(f"# Infinity Robin Diagnostic\n\n")
        f.write(f"- **Timestamp**: {timestamp}\n")
        f.write(f"- **Checkpoint**: {checkpoint_path}\n")
        f.write(f"- **Device**: {device}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Robin \\|B\\| median | {np.median(robin_abs_vals):.4e} |\n")
        f.write(f"| Robin \\|B\\| max | {np.max(robin_abs_vals):.4e} |\n")
        f.write(f"| Robin rel median | {np.median(robin_rel_vals):.4e} |\n")
        f.write(f"| Robin rel max | {np.max(robin_rel_vals):.4e} |\n\n")
        f.write(f"## Per-Point Results\n\n")
        f.write(f"| Label | a | omega | c_inf | S(-1) | S_y(-1) | |B| | rel B |\n")
        f.write(f"|-------|---|-------|-------|-------|---------|-----|-------|\n")
        for r in results:
            f.write(f"| {r['label']} | {r['a']} | {r['omega']} | "
                    f"{complex(r['c_inf']['real'], r['c_inf']['imag']):.4e} | "
                    f"{complex(r['S_minus1']['real'], r['S_minus1']['imag']):.4e} | "
                    f"{complex(r['Sy_minus1']['real'], r['Sy_minus1']['imag']):.4e} | "
                    f"{r['robin_residual_abs']:.4e} | {r['robin_residual_rel']:.4e} |\n")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
