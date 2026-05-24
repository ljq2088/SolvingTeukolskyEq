#!/usr/bin/env python3
"""Load adaptive-trained model and compare vs pybhpt reference.

3x2 grid per (a,omega) point: Re(R), Im(R), |R| vs r  |  Re(S), Im(S), |S| vs y
Also computes relative amplitude error.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from dataset.sampling import sample_points_chebyshev_grid
from physical_ansatz.transform_y import (
    compose_reduced_shape_from_f, horizon_regularity_slope, h_factor,
)
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives
from physical_ansatz.mapping import r_plus
from domain.atlas_builder import load_atlas, map_to_chart
from utils.compute_lambda_usage import compute_lambda


def predict_R_S(model, a_val, omega_val, u_val, v_val, lam, r_grid, M=1.0, m=2, s=-2):
    device = next(model.parameters()).device
    dtype = torch.float64

    a_t = torch.tensor([a_val], device=device, dtype=dtype)
    omega_t = torch.tensor([omega_val], device=device, dtype=dtype)
    u_t = torch.tensor([u_val], device=device, dtype=dtype)
    v_t = torch.tensor([v_val], device=device, dtype=dtype)
    lam_t = torch.tensor([lam], device=device, dtype=torch.complex128)
    r_grid_t = torch.tensor(r_grid, device=device, dtype=dtype)

    rp = r_plus(a_t, M)
    x_grid = rp / r_grid_t
    y_grid = 2.0 * x_grid - 1.0

    with torch.no_grad():
        f_pred = model(a_t, omega_t, y_grid, u=u_t, v=v_t)
        slope = horizon_regularity_slope(a=a_t, omega=omega_t, lambda_=lam_t, m=m, M=M, s=s)
        shape = compose_reduced_shape_from_f(
            f=f_pred.squeeze(0), y=y_grid.squeeze(0), slope=slope.squeeze(0))
        h2 = h_factor(a_t, omega_t, m=m, M=M, s=s)
        rp_val = r_plus(a_t, M)
        _, rm, _, _, _ = build_prefactor_primitives(r_grid_t, a_t, M=M, need_rs=False)
        P, _, _ = Leaver_prefactors(r_grid_t, a_t, omega_t, m=m, M=M, s=s, rp=rp_val, rm=rm)
        R_pred = P.squeeze(0) * h2 * shape

    return (R_pred.detach().cpu().numpy(), shape.detach().cpu().numpy(),
            y_grid.detach().cpu().numpy())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="outputs/gaussian_collocation_exp/adaptive_long_rw1e5/model_final.pt")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/gaussian_collocation_exp/adaptive_long_rw1e5")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-r", type=int, default=300)
    parser.add_argument("--skip-pybhpt", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    M_val, ell, m_val, s_val = 1.0, 2, 2, -2

    # Load atlas
    atlas = load_atlas("outputs/domain/atlas_l2_m2_logw.json")
    comp = atlas.components[0]
    omega_chart_mode = atlas.meta.get("omega_chart_mode", "log10")

    # Load model
    print(f"Loading model: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device)
    model = AutoencoderTeukolskyPINN()
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    print(f"  Loaded {len(sd)} keys")

    # pybhpt uses fork, which conflicts with CUDA. Run on CPU, sync first.
    model.to("cpu")
    torch.cuda.empty_cache()
    device = torch.device("cpu")
    points = [
        (0.50, 0.020, "center-low"),
        (0.50, 0.050, "center-mid"),
        (0.35, 0.030, "off-center-1"),
        (0.65, 0.015, "off-center-2"),
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from pybhpt_usage.compute_solution import compute_pybhpt_solution

    fig, axes = plt.subplots(4, 6, figsize=(24, 22))

    all_errors = []

    for pi, (a_val, omega_val, label) in enumerate(points):
        # Compute (u,v) and lambda
        u_val, v_val = map_to_chart(comp, a_val, omega_val, omega_chart_mode=omega_chart_mode)
        lam = compute_lambda(a_val, omega_val, ell, m_val, s=s_val)

        rp = float(M_val + np.sqrt(M_val**2 - a_val**2))
        r_min = rp + 0.01
        r_max = 100.0

        # Left: uniform r-grid
        r_grid = np.linspace(r_min, r_max, args.n_r)
        # Right: Chebyshev y-grid
        y_min = 2.0 * rp / r_max - 1.0
        y_max = 2.0 * rp / r_min - 1.0
        y_cheb = sample_points_chebyshev_grid(args.n_r, y_min, y_max).numpy()
        r_from_y = rp / (0.5 * (y_cheb + 1.0))

        # Model prediction (uniform r-grid)
        R_pred, _, _ = predict_R_S(model, a_val, omega_val, u_val, v_val, lam, r_grid,
                                    M=M_val, m=m_val, s=s_val)
        # Model prediction (y-grid)
        R_pred_y, S_pred_y, _ = predict_R_S(model, a_val, omega_val, u_val, v_val, lam,
                                              r_from_y, M=M_val, m=m_val, s=s_val)

        # pybhpt reference
        ref_available = False
        if not args.skip_pybhpt:
            try:
                a_f = float(a_val)
                w_f = float(omega_val)
                _, R_ref = compute_pybhpt_solution(a_f, w_f, ell=ell, m=m_val,
                                                    r_grid=r_grid, timeout=30.0)
                R_ref = np.asarray(R_ref, dtype=np.complex128)
                _, R_ref_y = compute_pybhpt_solution(a_f, w_f, ell=ell, m=m_val,
                                                      r_grid=r_from_y, timeout=30.0)
                R_ref_y = np.asarray(R_ref_y, dtype=np.complex128)
                # Compute S_ref
                h2_val = float(h_factor(torch.tensor([a_val]), torch.tensor([omega_val]),
                                         m=m_val, M=M_val, s=s_val).item())
                rp_t = r_plus(torch.tensor([a_val]), M_val)
                _, rm_t, _, _, _ = build_prefactor_primitives(
                    torch.tensor(r_from_y), torch.tensor([a_val]), M=M_val, need_rs=False)
                P_y, _, _ = Leaver_prefactors(torch.tensor(r_from_y), torch.tensor([a_val]),
                                               torch.tensor([omega_val]), m=m_val, M=M_val,
                                               s=s_val, rp=rp_t, rm=rm_t)
                P_y_np = P_y.squeeze(0).numpy()
                S_ref = R_ref_y / (P_y_np * complex(h2_val))
                ref_available = True

                # Compute relative error
                abs_R_ref = np.abs(R_ref)
                mask = abs_R_ref > 1e-15
                rel_err = np.abs(np.abs(R_pred)[mask] - abs_R_ref[mask]) / abs_R_ref[mask]
                med_err = np.median(rel_err)
                max_err = np.max(rel_err)
                all_errors.append({"label": label, "a": a_val, "omega": omega_val,
                                   "med_err": med_err, "max_err": max_err})
                error_str = f"med err={med_err:.2e}, max err={max_err:.2e}"
            except Exception as e:
                print(f"  pybhpt failed for a={a_val:.3f}, w={omega_val:.4f}: {e}")
                error_str = "no ref"
        else:
            error_str = "no ref"

        print(f"  {label}: a={a_val:.2f}, w={omega_val:.4f}  {error_str}")

        prefix = f"a={a_val:.2f}, w={omega_val:.3f}"
        title = f"{prefix}  {error_str}"

        # Left column: R(r) vs r
        for col, (data_name, pred_data, ref_data_fn) in enumerate([
            ("Re(R)", R_pred.real, lambda r: r.real),
            ("Im(R)", R_pred.imag, lambda r: r.imag),
            ("|R|", np.abs(R_pred), lambda r: np.abs(r)),
        ]):
            ax = axes[pi, col]
            ax.plot(r_grid, pred_data, lw=1.6, label="Pred")
            if ref_available:
                ax.plot(r_grid, ref_data_fn(R_ref), "--", lw=1.0, label="ref")
            if col == 0:
                ax.set_ylabel(f"{prefix}\n{data_name}")
            else:
                ax.set_ylabel(data_name)
            if col == 2:
                ax.set_xlabel("r")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_title(title, fontsize=9)

        # Right column: S(y) vs y
        for col, (data_name, pred_data, ref_data_fn) in enumerate([
            ("Re(S)", S_pred_y.real, lambda r: r.real),
            ("Im(S)", S_pred_y.imag, lambda r: r.imag),
            ("|S|", np.abs(S_pred_y), lambda r: np.abs(r)),
        ]):
            ax = axes[pi, 3 + col]
            ax.plot(y_cheb, pred_data, lw=1.6, label="Pred")
            if ref_available:
                ax.plot(y_cheb, ref_data_fn(S_ref), "--", lw=1.0, label="ref")
            if col == 0:
                ax.set_ylabel(f"{prefix}\n{data_name}")
            else:
                ax.set_ylabel(data_name)
            if col == 2:
                ax.set_xlabel("y")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

    fig.suptitle("Adaptive Collocation (200ep, rw=1e5, resample/20ep) vs pybhpt Reference",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "ref_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_dir / 'ref_comparison.png'}")

    # Print error summary
    if all_errors:
        print(f"\n{'='*60}")
        print("Relative Amplitude Error Summary (|R| vs pybhpt):")
        print(f"{'='*60}")
        for e in all_errors:
            print(f"  {e['label']:15s} a={e['a']:.2f} w={e['omega']:.3f}  "
                  f"med={e['med_err']:.4e}  max={e['max_err']:.4e}")
        # Overall
        meds = [e['med_err'] for e in all_errors]
        maxs = [e['max_err'] for e in all_errors]
        print(f"  {'OVERALL':15s}  med={np.median(meds):.4e}  max={np.max(maxs):.4e}")


if __name__ == "__main__":
    main()
