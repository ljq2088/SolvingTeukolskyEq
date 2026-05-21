#!/usr/bin/env python3
"""Evaluate patch 0 autoencoder Stage-1 model: error stats on (a,ω) grid.

Computes relative amplitude and phase errors comparing model R(r) against
pybhpt benchmark on a grid of (a,ω) points covering patch 0.
"""
import json
import os
import sys
import time
from datetime import datetime
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
from physical_ansatz.transform_y import (
    compose_reduced_shape_from_f,
    horizon_regularity_slope,
    h_factor,
)
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives
from physical_ansatz.mapping import r_plus
from utils.compute_lambda_usage import compute_lambda
from domain.atlas_builder import load_atlas, map_to_chart


def predict_R_full(model, a, omega, u, v, lam, r_grid, M=1.0, m=2, s=-2):
    """Model prediction of R_in(r) for a single (a,omega) point."""
    device = next(model.parameters()).device
    dtype = torch.float64

    a_t = torch.tensor([a], device=device, dtype=dtype)
    omega_t = torch.tensor([omega], device=device, dtype=dtype)
    u_t = torch.tensor([u], device=device, dtype=dtype)
    v_t = torch.tensor([v], device=device, dtype=dtype)
    lam_t = torch.tensor([lam], device=device, dtype=torch.complex128)

    r_grid_t = torch.tensor(r_grid, device=device, dtype=dtype)
    rp = r_plus(a_t, M)
    x_grid = rp / r_grid_t
    y_grid = 2.0 * x_grid - 1.0

    with torch.no_grad():
        # Filter y to [-1, 1)
        valid = (y_grid >= -1) & (y_grid < 1)
        if not valid.all():
            y_grid = y_grid[valid]

        f_pred = model(a_t, omega_t, y_grid, u=u_t, v=v_t)
        slope = horizon_regularity_slope(
            a=a_t, omega=omega_t, lambda_=lam_t, m=m, M=M, s=s,
        )
        shape = compose_reduced_shape_from_f(f=f_pred.squeeze(0), y=y_grid.squeeze(0), slope=slope.squeeze(0))

        h2 = h_factor(a_t, omega_t, m=m, M=M, s=s)
        rp_val = r_plus(a_t, M)
        r_eff = r_grid_t[valid.squeeze(0) if valid.ndim > 1 else valid]

        _, rm, _, _, _ = build_prefactor_primitives(r_eff, a_t, M=M, need_rs=False)
        P, _, _ = Leaver_prefactors(r_eff, a_t, omega_t, m=m, M=M, s=s, rp=rp_val, rm=rm)

        R_pred = P.squeeze(0) * h2 * shape
        r_out = r_eff.numpy()
        R_out = R_pred.detach().cpu().numpy()

    return r_out, R_out


def sample_grid(comp, u_c=0.5, v_c=0.6026456460798848, h_u=0.3, h_v=0.12,
                n_a=6, n_w=6, omega_chart_mode="log10"):
    """Sample (a,omega) points within the patch's (u,v) support.

    Patch 0: u in [u_c-h_u, u_c+h_u], v in [v_c-h_v, v_c+h_v].
    """
    from domain.atlas_builder import map_from_chart

    u_min, u_max = u_c - h_u, u_c + h_u
    v_min, v_max = v_c - h_v, v_c + h_v

    # Map patch corners to (a,omega)
    a0, _ = map_from_chart(comp, u_min, v_c, omega_chart_mode=omega_chart_mode)
    a1, _ = map_from_chart(comp, u_max, v_c, omega_chart_mode=omega_chart_mode)
    _, w0 = map_from_chart(comp, u_c, v_min, omega_chart_mode=omega_chart_mode)
    _, w1 = map_from_chart(comp, u_c, v_max, omega_chart_mode=omega_chart_mode)

    # Add margin
    a_margin = (a1 - a0) * 0.05
    w_margin = (w1 - w0) * 0.05
    a_vals = np.linspace(a0 + a_margin, a1 - a_margin, n_a)
    w_vals = np.logspace(np.log10(w0 + w_margin), np.log10(w1 - w_margin), n_w)
    return a_vals, w_vals


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-a", type=int, default=6, help="Number of a points")
    parser.add_argument("--n-w", type=int, default=6, help="Number of omega points")
    parser.add_argument("--n-r", type=int, default=200, help="Number of r points")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-pybhpt", action="store_true", help="Skip pybhpt comparison, only compute model profiles")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64

    # Load model
    ckpt_path = "outputs/autoencoder_stage1_rin_train/20260518_194655_patch_000_comp_0_u_0.500_v_0.603/checkpoints/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = AutoencoderTeukolskyPINN()
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint with {len(sd)} keys")

    M, ell, m_mode, s_val = 1.0, 2, 2, -2

    # Load atlas component for (a,omega) -> (u,v) mapping
    atlas = load_atlas("outputs/domain/atlas_l2_m2_logw.json")
    comp = atlas.components[0]
    omega_chart_mode = atlas.meta.get("omega_chart_mode", "log10")
    print(f"Atlas component 0: a ∈ [{comp.a_support[0]:.3f}, {comp.a_support[-1]:.3f}], "
          f"omega ∈ [{np.min(comp.omega_lower):.2e}, {np.max(comp.omega_upper):.2e}], "
          f"mode={omega_chart_mode}")

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs/patch0_eval") / timestamp
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    # Sample points
    a_vals, w_vals = sample_grid(comp, n_a=args.n_a, n_w=args.n_w, omega_chart_mode=omega_chart_mode)
    print(f"\nGrid: {len(a_vals)} × {len(w_vals)} = {len(a_vals)*len(w_vals)} points")
    print(f"a ∈ [{a_vals[0]:.2f}, {a_vals[-1]:.2f}]")
    print(f"ω ∈ [{w_vals[0]:.2e}, {w_vals[-1]:.2e}]")

    # r-grid: log-spaced from horizon to r=100
    a_mid = 0.5
    rp_mid = float(M + np.sqrt(M ** 2 - a_mid ** 2))
    r_grid = np.logspace(np.log10(rp_mid + 0.001), 2.0, args.n_r)

    # Compute model predictions and pybhpt references
    results = []
    A, W = np.meshgrid(a_vals, w_vals)

    for i, (a_val, w_val) in enumerate([(A.flat[j], W.flat[j]) for j in range(A.size)]):
        print(f"  [{i+1}/{A.size}] a={a_val:.3f}, ω={w_val:.4f} ...", end=" ", flush=True)
        t0 = time.time()

        # Lambda
        lam = compute_lambda(float(a_val), float(w_val), ell, m_mode, s=s_val)
        # Compute correct (u,v) from atlas mapping
        try:
            u_val, v_val = map_to_chart(comp, float(a_val), float(w_val), omega_chart_mode=omega_chart_mode)
        except ValueError:
            print(f"outside component bounds, skipping")
            continue

        entry = {"a": float(a_val), "omega": float(w_val), "lambda": complex(lam),
                 "u": u_val, "v": v_val}

        # Model prediction
        try:
            r_mod, R_mod = predict_R_full(model, a_val, w_val, u_val, v_val, lam, r_grid, M=M, m=m_mode, s=s_val)
            entry["r_model"] = r_mod.tolist()
            entry["R_model_abs"] = np.abs(R_mod).tolist()
            entry["R_model_phase"] = np.angle(R_mod).tolist()
        except Exception as e:
            print(f"model failed: {e}")
            entry["model_error"] = str(e)
            results.append(entry)
            continue

        # pybhpt benchmark
        if not args.skip_pybhpt:
            try:
                from pybhpt_usage.compute_solution import compute_pybhpt_solution

                r_ref, R_ref = compute_pybhpt_solution(a_val, w_val, ell=ell, m=m_mode, r_grid=r_mod, timeout=20.0)
                R_ref = np.asarray(R_ref, dtype=np.complex128)

                # Interpolate model to pybhpt r-grid for comparison
                R_mod_interp = np.interp(r_ref, r_mod, np.abs(R_mod)) * np.exp(
                    1j * np.interp(r_ref, r_mod, np.angle(R_mod))
                )

                R_ref_abs = np.abs(R_ref)
                mask = R_ref_abs > 1e-15
                rel_err = np.abs(np.abs(R_mod_interp)[mask] - R_ref_abs[mask]) / R_ref_abs[mask]

                phase_ref = np.angle(R_ref[mask])
                phase_mod = np.angle(R_mod_interp[mask])
                phase_diff = np.mod(phase_mod - phase_ref + np.pi, 2 * np.pi) - np.pi

                entry["rel_err_mean"] = float(np.mean(rel_err))
                entry["rel_err_median"] = float(np.median(rel_err))
                entry["rel_err_max"] = float(np.max(rel_err))
                entry["phase_err_mean"] = float(np.mean(np.abs(phase_diff)))
                entry["R_ref_abs_max"] = float(R_ref_abs.max())

                print(f"rel_mean={entry['rel_err_mean']:.3e}, phase_mean={entry['phase_err_mean']:.3e} ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"pybhpt failed: {e}")
                entry["pybhpt_error"] = str(e)
        else:
            # Without pybhpt, compute PDE residual as error proxy
            # We'll compute the full S(y) PDE residual later
            print(f"done ({time.time()-t0:.1f}s)")

        results.append(entry)

    # Save raw results
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ========== Statistics ==========
    if not args.skip_pybhpt and any("rel_err_mean" in r for r in results):
        rel_means = [r["rel_err_mean"] for r in results if "rel_err_mean" in r]
        rel_medians = [r["rel_err_median"] for r in results if "rel_err_median" in r]
        phase_means = [r["phase_err_mean"] for r in results if "phase_err_mean" in r]

        summary = {
            "n_total": len(results),
            "n_success": len(rel_means),
            "rel_err_mean_of_means": float(np.mean(rel_means)),
            "rel_err_median_of_medians": float(np.median(rel_medians)),
            "rel_err_worst": float(np.max(rel_means)),
            "rel_err_best": float(np.min(rel_means)),
            "phase_err_mean": float(np.mean(phase_means)),
        }

        print(f"\n{'='*70}")
        print("Patch 0 Error Statistics")
        print(f"{'='*70}")
        print(f"Points evaluated: {summary['n_success']}/{summary['n_total']}")
        print(f"Relative amplitude error (mean of per-point means): {summary['rel_err_mean_of_means']:.4e}")
        print(f"Relative amplitude error (median of per-point medians): {summary['rel_err_median_of_medians']:.4e}")
        print(f"Best point:  {summary['rel_err_best']:.4e}")
        print(f"Worst point: {summary['rel_err_worst']:.4e}")
        print(f"Phase error (mean): {summary['phase_err_mean']:.4e} rad")

        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # ========== Heatmap: relative error ==========
        err_grid = np.full((args.n_w, args.n_a), np.nan)
        phase_grid = np.full((args.n_w, args.n_a), np.nan)
        for r in results:
            if "rel_err_mean" not in r:
                continue
            # Find index
            ia = np.argmin(np.abs(a_vals - r["a"]))
            iw = np.argmin(np.abs(w_vals - r["omega"]))
            err_grid[iw, ia] = r["rel_err_median"]
            phase_grid[iw, ia] = r["phase_err_mean"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        im0 = axes[0].pcolormesh(A, W, err_grid, shading="auto", cmap="YlOrRd")
        axes[0].set_xlabel("a")
        axes[0].set_ylabel("ω")
        axes[0].set_title("Median Relative Amplitude Error")
        axes[0].set_yscale("log")
        plt.colorbar(im0, ax=axes[0], format="%.2e")

        im1 = axes[1].pcolormesh(A, W, phase_grid, shading="auto", cmap="YlOrRd")
        axes[1].set_xlabel("a")
        axes[1].set_ylabel("ω")
        axes[1].set_title("Mean Phase Error [rad]")
        axes[1].set_yscale("log")
        plt.colorbar(im1, ax=axes[1], format="%.2e")

        fig.suptitle("Patch 0 — Autoencoder Stage-1 vs pybhpt", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / "error_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"\nHeatmap saved to {out_dir / 'plots' / 'error_heatmap.png'}")

        # ========== Detailed per-point bar chart ==========
        labels = [f"a={r['a']:.2f}\nω={r['omega']:.2f}" for r in results if "rel_err_mean" in r]
        rel_vals = [r["rel_err_median"] for r in results if "rel_err_mean" in r]
        sorted_idx = np.argsort(rel_vals)
        labels_sorted = [labels[i] for i in sorted_idx]
        rel_sorted = [rel_vals[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(max(12, len(labels)*0.8), 5))
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(rel_sorted)))
        ax.barh(range(len(rel_sorted)), rel_sorted, color=colors)
        ax.set_yticks(range(len(rel_sorted)))
        ax.set_yticklabels(labels_sorted, fontsize=7)
        ax.set_xlabel("Median Relative Amplitude Error")
        ax.set_xscale("log")
        ax.set_title("Patch 0 — Per-Point Median Relative Error (sorted)")
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / "error_bars.png", dpi=150)
        plt.close(fig)

    # ========== Sample profile plot ==========
    plot_idx = np.linspace(0, len(results) - 1, min(6, len(results)), dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flat
    for pi, idx in enumerate(plot_idx):
        r = results[idx]
        if "r_model" not in r:
            continue
        ax = axes[pi]
        ax.plot(r["r_model"], r["R_model_abs"], "b-", label="Model |R(r)|", lw=1.5)
        ax.set_xlabel("r")
        ax.set_ylabel("|R|")
        ax.set_title(f"a={r['a']:.3f}, ω={r['omega']:.4f}")
        if "rel_err_mean" in r:
            ax.text(0.95, 0.90, f"err={r['rel_err_mean']:.2e}",
                    transform=ax.transAxes, ha="right", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    for pi in range(len(plot_idx), len(axes)):
        axes[pi].set_visible(False)
    fig.suptitle("Patch 0 — Model |R(r)| Profiles", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "sample_profiles.png", dpi=150)
    plt.close(fig)

    print(f"\nAll outputs saved to {out_dir}")
    print(f"  {out_dir / 'results.json'}")
    print(f"  {out_dir / 'summary.json'}")
    print(f"  {out_dir / 'plots/'}")


if __name__ == "__main__":
    main()
