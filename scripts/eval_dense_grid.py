#!/usr/bin/env python3
"""Dense-grid a-omega-r relative error evaluation for Stage-1 autoencoder.

Evaluates model R(r) against pybhpt benchmark on a dense (a,omega) grid
covering the patch's parameter range. Saves heatmaps and raw data.

Usage:
  # After training:
  python scripts/eval_dense_grid.py \
    --checkpoint outputs/stage1_retrain/{run_dir}/checkpoints/best_model.pt \
    --config config/autoencoder_stage1_retrain.yaml \
    --patch-id 0 --device cuda

  # Manual on any checkpoint:
  python scripts/eval_dense_grid.py \
    --checkpoint path/to/checkpoint.pt \
    --config config/autoencoder_stage1_retrain.yaml \
    --patch-id 0 --device cpu --n-a 10 --n-w 10
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
        valid = (y_grid >= -1) & (y_grid < 1)
        if not valid.all():
            y_grid = y_grid[valid]

        f_pred = model(a_t, omega_t, y_grid, u=u_t, v=v_t)
        slope = horizon_regularity_slope(
            a=a_t, omega=omega_t, lambda_=lam_t, m=m, M=M, s=s,
        )
        shape = compose_reduced_shape_from_f(
            f=f_pred.squeeze(0), y=y_grid.squeeze(0), slope=slope.squeeze(0),
        )

        h2 = h_factor(a_t, omega_t, m=m, M=M, s=s)
        rp_val = r_plus(a_t, M)
        r_eff = r_grid_t[valid.squeeze(0) if valid.ndim > 1 else valid]

        _, rm, _, _, _ = build_prefactor_primitives(r_eff, a_t, M=M, need_rs=False)
        P, _, _ = Leaver_prefactors(r_eff, a_t, omega_t, m=m, M=M, s=s, rp=rp_val, rm=rm)

        R_pred = P.squeeze(0) * h2 * shape
        r_out = r_eff.cpu().numpy()
        R_out = R_pred.detach().cpu().numpy()

    return r_out, R_out


def build_grid_from_patch(patch_boundaries, n_a=15, n_w=15, omega_chart_mode="log10"):
    """Build (a,omega) evaluation grid within patch boundaries."""
    a_min, a_max = patch_boundaries["a"]
    w_min, w_max = patch_boundaries["omega"]

    a_margin = (a_max - a_min) * 0.02
    a_vals = np.linspace(a_min + a_margin, a_max - a_margin, n_a)

    if omega_chart_mode == "log10":
        w_vals = np.logspace(np.log10(w_min * 1.02), np.log10(w_max * 0.98), n_w)
    else:
        w_vals = np.linspace(w_min * 1.02, w_max * 0.98, n_w)

    return a_vals, w_vals


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str,
                        default="config/autoencoder_stage1_retrain.yaml")
    parser.add_argument("--patch-id", type=int, default=0)
    parser.add_argument("--atlas-json", type=str,
                        default="outputs/domain/atlas_l2_m2_logw.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-a", type=int, default=15)
    parser.add_argument("--n-w", type=int, default=15)
    parser.add_argument("--n-r", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-pybhpt", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    M, ell, m_mode, s_val = 1.0, 2, 2, -2

    # ---- Load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = AutoencoderTeukolskyPINN()
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    # ---- Extract patch info from checkpoint ----
    patch_boundaries = ckpt.get("patch_boundaries", None)
    if patch_boundaries is None:
        print("[warn] checkpoint missing patch_boundaries, using atlas defaults")
        atlas = load_atlas(args.atlas_json)
        comp = atlas.components[0]
        omega_chart_mode = atlas.meta.get("omega_chart_mode", "log10")
        # Fallback: use full component range — will be clipped by map_to_chart
        a_vals = np.linspace(0.01, 0.99, args.n_a)
        w_vals = np.logspace(-4, 0, args.n_w)
    else:
        omega_chart_mode = patch_boundaries.get("omega_chart_mode", "log10")
        a_vals, w_vals = build_grid_from_patch(
            patch_boundaries, n_a=args.n_a, n_w=args.n_w,
            omega_chart_mode=omega_chart_mode,
        )

    # ---- Load atlas for (a,omega)->(u,v) mapping ----
    atlas = load_atlas(args.atlas_json)
    comp = atlas.components[0]

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Patch boundaries: {patch_boundaries}")
    print(f"Grid: {len(a_vals)}a x {len(w_vals)}w = {len(a_vals)*len(w_vals)} points")
    print(f"a in [{a_vals[0]:.3f}, {a_vals[-1]:.3f}]")
    print(f"omega in [{w_vals[0]:.6f}, {w_vals[-1]:.4f}]")

    # ---- Output dir ----
    if args.output_dir is None:
        ckpt_dir = Path(args.checkpoint).parent.parent  # checkpoints/ -> run_dir
        out_dir = ckpt_dir / "benchmarks"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- r-grid ----
    a_mid = 0.5
    rp_mid = float(M + np.sqrt(M ** 2 - a_mid ** 2))
    r_grid = np.logspace(np.log10(rp_mid + 0.001), 2.0, args.n_r)

    # ---- Evaluate ----
    A, W = np.meshgrid(a_vals, w_vals)
    results = []
    n_total = A.size

    for i in range(n_total):
        a_val = float(A.flat[i])
        w_val = float(W.flat[i])
        print(f"  [{i+1}/{n_total}] a={a_val:.3f}, omega={w_val:.6f} ...", end=" ", flush=True)
        t0 = time.time()

        lam = compute_lambda(a_val, w_val, ell, m_mode, s=s_val)
        try:
            u_val, v_val = map_to_chart(comp, a_val, w_val, omega_chart_mode=omega_chart_mode)
        except ValueError:
            print("outside component")
            results.append({"a": a_val, "omega": w_val, "lambda": complex(lam),
                            "error": "outside_component"})
            continue

        entry = {"a": a_val, "omega": w_val, "lambda": complex(lam),
                 "u": u_val, "v": v_val}

        try:
            r_mod, R_mod = predict_R_full(model, a_val, w_val, u_val, v_val, lam, r_grid,
                                          M=M, m=m_mode, s=s_val)
            entry["r_model"] = r_mod.tolist()
            entry["R_model_abs"] = np.abs(R_mod).tolist()
        except Exception as e:
            print(f"model failed: {e}")
            entry["model_error"] = str(e)
            results.append(entry)
            continue

        if not args.skip_pybhpt:
            try:
                from pybhpt_usage.compute_solution import compute_pybhpt_solution
                r_ref, R_ref = compute_pybhpt_solution(a_val, w_val, ell=ell, m=m_mode,
                                                       r_grid=r_mod, timeout=30.0)
                R_ref = np.asarray(R_ref, dtype=np.complex128)

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
                entry["r_ref"] = r_ref.tolist()
                entry["R_ref_abs"] = np.abs(R_ref).tolist()

                print(f"median_rel={entry['rel_err_median']:.4e} ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"pybhpt failed: {e}")
                entry["pybhpt_error"] = str(e)
        else:
            print(f"done ({time.time()-t0:.1f}s)")

        results.append(entry)

    # ---- Save raw data ----
    with open(out_dir / "error_grid.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ---- Statistics ----
    success = [r for r in results if "rel_err_mean" in r]
    if not success:
        print("No successful evaluations. Exiting.")
        return

    rel_medians = [r["rel_err_median"] for r in success]
    rel_means = [r["rel_err_mean"] for r in success]
    rel_maxs = [r["rel_err_max"] for r in success]
    phase_means = [r["phase_err_mean"] for r in success]

    summary = {
        "checkpoint": args.checkpoint,
        "n_total": len(results),
        "n_success": len(success),
        "n_failed": len(results) - len(success),
        "grid_n_a": args.n_a,
        "grid_n_w": args.n_w,
        "a_range": [float(a_vals[0]), float(a_vals[-1])],
        "omega_range": [float(w_vals[0]), float(w_vals[-1])],
        "rel_err_median_of_medians": float(np.median(rel_medians)),
        "rel_err_mean_of_means": float(np.mean(rel_means)),
        "rel_err_worst_median": float(np.max(rel_medians)),
        "rel_err_best_median": float(np.min(rel_medians)),
        "rel_err_max_overall": float(np.max(rel_maxs)),
        "phase_err_mean": float(np.mean(phase_means)),
        "worst_point": {
            "a": float(success[np.argmax(rel_medians)]["a"]),
            "omega": float(success[np.argmax(rel_medians)]["omega"]),
        },
        "best_point": {
            "a": float(success[np.argmin(rel_medians)]["a"]),
            "omega": float(success[np.argmin(rel_medians)]["omega"]),
        },
    }
    with open(out_dir / "error_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Dense Grid Error Summary")
    print(f"{'='*60}")
    print(f"Evaluated: {summary['n_success']}/{summary['n_total']} points")
    print(f"Median-of-medians rel err: {summary['rel_err_median_of_medians']:.4%}")
    print(f"Mean-of-means rel err:    {summary['rel_err_mean_of_means']:.4%}")
    print(f"Best point:  {summary['best_point']} -> {summary['rel_err_best_median']:.4%}")
    print(f"Worst point: {summary['worst_point']} -> {summary['rel_err_worst_median']:.4%}")
    print(f"Max overall: {summary['rel_err_max_overall']:.4%}")

    # ---- Heatmap ----
    err_grid = np.full((args.n_w, args.n_a), np.nan)
    phase_grid = np.full((args.n_w, args.n_a), np.nan)
    for r in success:
        ia = np.argmin(np.abs(a_vals - r["a"]))
        iw = np.argmin(np.abs(w_vals - r["omega"]))
        err_grid[iw, ia] = r["rel_err_median"]
        phase_grid[iw, ia] = r["phase_err_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    im0 = axes[0].pcolormesh(A, W, err_grid, shading="auto", cmap="YlOrRd")
    axes[0].set_xlabel("a")
    axes[0].set_ylabel("omega")
    axes[0].set_title("Median Relative Amplitude Error |R_in|")
    axes[0].set_yscale("log")
    plt.colorbar(im0, ax=axes[0], format="%.2e")

    im1 = axes[1].pcolormesh(A, W, phase_grid, shading="auto", cmap="YlOrRd")
    axes[1].set_xlabel("a")
    axes[1].set_ylabel("omega")
    axes[1].set_title("Mean Phase Error [rad]")
    axes[1].set_yscale("log")
    plt.colorbar(im1, ax=axes[1], format="%.2e")

    fig.suptitle("Stage-1 Autoencoder vs pybhpt — Patch 0 Dense Grid", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "error_heatmap_a_omega.png", dpi=150)
    plt.close(fig)
    print(f"Heatmap saved to {out_dir / 'error_heatmap_a_omega.png'}")

    # ---- r-error curves for selected omega slices ----
    n_slices = min(4, args.n_w)
    slice_indices = np.linspace(0, args.n_w - 1, n_slices, dtype=int)
    fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 4))
    if n_slices == 1:
        axes = [axes]

    for si, iw in enumerate(slice_indices):
        ax = axes[si]
        w_slice = w_vals[iw]
        slice_results = [r for r in success if abs(r["omega"] - w_slice) < 1e-10]
        if not slice_results:
            ax.set_title(f"omega={w_slice:.4f} (no data)")
            continue
        for r_entry in slice_results:
            if "r_model" in r_entry and "r_ref" in r_entry:
                R_mod_abs = np.array(r_entry["R_model_abs"])
                r_mod = np.array(r_entry["r_model"])
                r_ref = np.array(r_entry["r_ref"])
                R_ref_abs = np.array(r_entry["R_ref_abs"])
                # Compute per-r relative error
                R_mod_interp = np.interp(r_ref, r_mod, R_mod_abs)
                mask = R_ref_abs > 1e-15
                r_err = np.abs(R_mod_interp[mask] - R_ref_abs[mask]) / R_ref_abs[mask]
                ax.semilogy(r_ref[mask], r_err, alpha=0.7,
                            label=f"a={r_entry['a']:.2f}")
        ax.set_xlabel("r / M")
        ax.set_ylabel("Relative Error |R|")
        ax.set_title(f"omega = {w_slice:.4f}")
        ax.grid(True, alpha=0.3)
        if len(slice_results) <= 6:
            ax.legend(fontsize=7)

    fig.suptitle("Per-r Relative Error — Selected omega Slices", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "error_r_slices.png", dpi=150)
    plt.close(fig)
    print(f"R-error slices saved to {out_dir / 'error_r_slices.png'}")

    print(f"\nAll outputs: {out_dir}/")


if __name__ == "__main__":
    main()
