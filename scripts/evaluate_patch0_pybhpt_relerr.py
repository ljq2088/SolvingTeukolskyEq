#!/usr/bin/env python3
"""Evaluate a Stage-1 model against pybhpt on a patch-0 (a, omega, r) grid."""
import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import load_pinn_full_config
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives, r_plus
from physical_ansatz.residual import AuxCache, get_lambda_from_cfg
from physical_ansatz.transform_y import (
    compose_reduced_shape_from_f,
    h_factor,
    horizon_regularity_slope,
)
from pybhpt_usage.compute_solution import compute_pybhpt_solution


def build_model(model_cfg):
    return AutoencoderTeukolskyPINN(
        hidden_dims=model_cfg.get("hidden_dims", [128, 128, 128, 128]),
        activation=model_cfg.get("activation", "silu"),
        fourier_num_freqs=model_cfg.get("fourier_num_freqs", 2),
        fourier_base_scale=model_cfg.get("fourier_base_scale", 1.0),
        param_embed_dim=model_cfg.get("param_embed_dim", 64),
        use_film=model_cfg.get("use_film", True),
        use_residual=model_cfg.get("use_residual", True),
    )


def predict_R(model, physics_cfg, a_value, omega_value, r_grid, device, dtype):
    problem_cfg = physics_cfg["problem"]
    M = float(problem_cfg.get("M", 1.0))
    m = int(problem_cfg.get("m", 2))
    s = int(problem_cfg.get("s", -2))

    a_t = torch.tensor(float(a_value), device=device, dtype=dtype)
    omega_t = torch.tensor(float(omega_value), device=device, dtype=dtype)
    u_t = torch.tensor((float(a_value) - 0.206) / (0.794 - 0.206), device=device, dtype=dtype)
    v_t = torch.tensor((float(omega_value) - 0.008523) / (0.0777 - 0.008523), device=device, dtype=dtype)
    r_t = torch.tensor(r_grid, device=device, dtype=dtype)
    rp = r_plus(a_t, M)
    y_t = 2.0 * rp / r_t - 1.0

    cache = AuxCache()
    lam = get_lambda_from_cfg(physics_cfg, cache, a_t, omega_t)
    lam = lam.detach().clone() if torch.is_tensor(lam) else torch.tensor(lam)
    lam = lam.to(device=device, dtype=torch.complex128)

    with torch.no_grad():
        f = model(
            a_t.unsqueeze(0),
            omega_t.unsqueeze(0),
            y_t,
            u=u_t.unsqueeze(0),
            v=v_t.unsqueeze(0),
        ).squeeze(0)
        slope = horizon_regularity_slope(
            a=a_t.unsqueeze(0),
            omega=omega_t.unsqueeze(0),
            lambda_=lam.unsqueeze(0),
            m=m,
            M=M,
            s=s,
        ).squeeze(0)
        shape = compose_reduced_shape_from_f(f=f, y=y_t, slope=slope)
        rp_r, rm_r, _, _, _ = build_prefactor_primitives(r_t, a_t, M=M, need_rs=False)
        P, _, _ = Leaver_prefactors(r_t, a_t, omega_t, m=m, M=M, s=s, rp=rp_r, rm=rm_r)
        return (P * h_factor(a_t, omega_t, m=m, M=M, s=s) * shape).detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config/autoencoder_stage1_pinn_random_v3.yaml")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-a", type=int, default=7)
    parser.add_argument("--n-omega", type=int, default=7)
    parser.add_argument("--n-r", type=int, default=128)
    parser.add_argument("--r-min", type=float, default=2.0)
    parser.add_argument("--r-max", type=float, default=1000.0)
    parser.add_argument("--r-grid", choices=["linear", "log"], default="linear")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    full_cfg = load_pinn_full_config(args.config)
    physics_cfg = full_cfg["physics"]
    model_cfg = full_cfg["train"].get("model", {})

    model = build_model(model_cfg)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()

    if args.output_root is None:
        ckpt_path = Path(args.checkpoint)
        run_dir = ckpt_path.parents[1] if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent
        output_root = run_dir / "evaluations"
    else:
        output_root = Path(args.output_root)
    out_dir = output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_patch0_pybhpt_relerr"
    out_dir.mkdir(parents=True, exist_ok=True)

    a_values = np.linspace(0.206, 0.794, args.n_a, dtype=np.float64)
    omega_values = np.linspace(0.008523, 0.0777, args.n_omega, dtype=np.float64)
    if args.r_grid == "log":
        r_grid = np.geomspace(args.r_min, args.r_max, args.n_r, dtype=np.float64)
    else:
        r_grid = np.linspace(args.r_min, args.r_max, args.n_r, dtype=np.float64)

    rel_err = np.full((args.n_a, args.n_omega, args.n_r), np.nan, dtype=np.float64)
    abs_err = np.full_like(rel_err, np.nan)
    pred_abs = np.full_like(rel_err, np.nan)
    ref_abs = np.full_like(rel_err, np.nan)
    failures = []

    print(f"Loaded checkpoint: {args.checkpoint} step={ckpt.get('step')} best={ckpt.get('best_val_mean')}")
    print(f"Output: {out_dir}")
    print(f"Grid: a={args.n_a}, omega={args.n_omega}, r={args.n_r}, r=[{r_grid[0]}, {r_grid[-1]}]")

    rows = []
    for i, a_value in enumerate(a_values):
        for j, omega_value in enumerate(omega_values):
            print(f"[{i+1}/{args.n_a}, {j+1}/{args.n_omega}] a={a_value:.6f} omega={omega_value:.6f}", flush=True)
            try:
                R_pred = predict_R(model, physics_cfg, a_value, omega_value, r_grid, device, dtype)
                _, R_ref = compute_pybhpt_solution(
                    float(a_value),
                    float(omega_value),
                    ell=2,
                    m=2,
                    r_grid=r_grid,
                    timeout=args.timeout,
                )
                err = np.abs(R_pred - R_ref)
                denom = np.abs(R_ref) + 1.0e-14
                rel = err / denom
                rel_err[i, j, :] = rel
                abs_err[i, j, :] = err
                pred_abs[i, j, :] = np.abs(R_pred)
                ref_abs[i, j, :] = np.abs(R_ref)
                rows.append({
                    "a": float(a_value),
                    "omega": float(omega_value),
                    "median_rel": float(np.nanmedian(rel)),
                    "mean_rel": float(np.nanmean(rel)),
                    "p90_rel": float(np.nanpercentile(rel, 90)),
                    "max_rel": float(np.nanmax(rel)),
                    "r_at_max": float(r_grid[int(np.nanargmax(rel))]),
                })
            except Exception as exc:
                failures.append({"a": float(a_value), "omega": float(omega_value), "error": str(exc)})
                print(f"  FAILED: {exc}", flush=True)

    np.savez_compressed(
        out_dir / "patch0_aw_r_relerr.npz",
        a_values=a_values,
        omega_values=omega_values,
        r_grid=r_grid,
        rel_err=rel_err,
        abs_err=abs_err,
        pred_abs=pred_abs,
        ref_abs=ref_abs,
    )

    with open(out_dir / "case_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["a", "omega", "median_rel", "mean_rel", "p90_rel", "max_rel", "r_at_max"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": ckpt.get("step"),
        "checkpoint_best_val_mean": ckpt.get("best_val_mean"),
        "grid": {
            "n_a": args.n_a,
            "n_omega": args.n_omega,
            "n_r": args.n_r,
            "r_min": float(r_grid[0]),
            "r_max": float(r_grid[-1]),
            "r_grid": args.r_grid,
        },
        "overall": {
            "median_rel": float(np.nanmedian(rel_err)),
            "mean_rel": float(np.nanmean(rel_err)),
            "p90_rel": float(np.nanpercentile(rel_err, 90)),
            "p99_rel": float(np.nanpercentile(rel_err, 99)),
            "max_rel": float(np.nanmax(rel_err)),
        },
        "failures": failures,
    }
    if rows:
        worst = max(rows, key=lambda row: row["median_rel"])
        best = min(rows, key=lambda row: row["median_rel"])
        summary["worst_case_by_median"] = worst
        summary["best_case_by_median"] = best

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(full_cfg, f)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        median_aw = np.nanmedian(rel_err, axis=2)
        max_aw = np.nanmax(rel_err, axis=2)
        median_r = np.nanmedian(rel_err, axis=(0, 1))
        p90_r = np.nanpercentile(rel_err, 90, axis=(0, 1))

        for data, name, title in [
            (median_aw, "median_rel_heatmap.png", "median relative error over r"),
            (max_aw, "max_rel_heatmap.png", "max relative error over r"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(
                data.T,
                origin="lower",
                aspect="auto",
                extent=[a_values[0], a_values[-1], omega_values[0], omega_values[-1]],
            )
            ax.set_xlabel("a")
            ax.set_ylabel("omega")
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(out_dir / name, dpi=180)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(r_grid, median_r, label="median over (a,omega)")
        ax.plot(r_grid, p90_r, label="p90 over (a,omega)")
        ax.set_xlabel("r")
        ax.set_ylabel("relative error")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "relerr_vs_r.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        summary["plot_error"] = str(exc)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
