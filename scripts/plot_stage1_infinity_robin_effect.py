#!/usr/bin/env python3
"""Plot before/after comparison for Stage-1 infinity Robin refinement.

Compares original checkpoint, Adam best, and L-BFGS best:
  Re(S), Im(S), |S|, arg(S), Robin residual, Sy(-1) vs c_inf*S(-1),
  near-infinity PDE residual.
"""
import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.infinity_robin import analytic_c_inf, infinity_robin_residual
from physical_ansatz.stage1_endpoint import compute_stage1_S_and_Sy_at_infinity
from physical_ansatz.transform_y import compose_reduced_shape_from_f, horizon_regularity_slope
from physical_ansatz.residual_pinn import compute_f_derivatives_autograd
from physical_ansatz.teukolsky_coeffs import coeffs_x
from physical_ansatz.transform_y import transform_coeffs_x_to_y
from utils.compute_lambda_usage import compute_lambda


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = AutoencoderTeukolskyPINN()
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def compute_S_profile(model, a, omega, u, v, lambda_, y_grid, M=1.0, m=2, s=-2):
    """Compute S(y) and Sy(y) for a single (a, omega) point."""
    device = a.device
    B = a.shape[0]
    N = y_grid.shape[1]

    f, fy, fyy = compute_f_derivatives_autograd(
        model, a, omega, y_grid, u_batch=u, v_batch=v,
    )
    slope = horizon_regularity_slope(a=a, omega=omega, lambda_=lambda_, m=m, M=M, s=s)

    S_vals = []
    for i in range(B):
        S_i = compose_reduced_shape_from_f(f[i], y_grid[i], slope[i])
        S_vals.append(S_i.unsqueeze(0))
    S_all = torch.cat(S_vals, dim=0)

    # PDE residual
    pde_res = []
    for i in range(B):
        x_i = (y_grid[i:i+1] + 1) / 2
        A2_i, A1_i, A0_i = coeffs_x(x=x_i, a=a[i], omega=omega[i], m=m, lambda_=lambda_[i], s=s, M=M)
        B2, B1, B0, rhs = transform_coeffs_x_to_y(A2_i, A1_i, A0_i, y_grid[i:i+1], slope[i:i+1])
        B2_c = B2.to(dtype=torch.complex128)
        B1_c = B1.to(dtype=torch.complex128)
        B0_c = B0.to(dtype=torch.complex128)
        rhs_c = rhs.to(dtype=torch.complex128) if rhs is not None else 0.0
        res = B2_c * fyy[i:i+1].to(dtype=torch.complex128) + \
              B1_c * fy[i:i+1].to(dtype=torch.complex128) + \
              B0_c * f[i:i+1].to(dtype=torch.complex128) - rhs_c
        pde_res.append((res.real**2 + res.imag**2).sqrt().detach().cpu().numpy().flatten())

    return S_all.detach().cpu().numpy(), np.array(pde_res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=str, required=True, help="Original checkpoint path")
    parser.add_argument("--adam-best", type=str, default=None, help="Adam best checkpoint")
    parser.add_argument("--lbfgs-best", type=str, default=None, help="L-BFGS best checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-points", type=int, default=4, help="Number of (a,omega) points to plot")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    cdtype = torch.complex128

    M_val = 1.0
    ell, m_mode, s_val = 2, 2, -2
    n_y = 200

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs/stage1_infinity_robin_compare") / timestamp
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    # Load all checkpoints
    print(f"Loading original: {args.original}")
    models = {"original": load_model(args.original, device)}

    if args.adam_best:
        print(f"Loading Adam best: {args.adam_best}")
        models["adam"] = load_model(args.adam_best, device)
    if args.lbfgs_best:
        print(f"Loading L-BFGS best: {args.lbfgs_best}")
        models["lbfgs"] = load_model(args.lbfgs_best, device)

    # Sample parameter points
    a_points = [0.1, 0.3, 0.5, 0.7, 0.9][:args.n_points]
    omega_points = [0.2, 0.35, 0.51, 0.65, 0.8][:args.n_points]

    # Compute lambda for each
    lam_list = []
    for a_v, w_v in zip(a_points, omega_points):
        lam = compute_lambda(a_v, w_v, ell, m_mode, s=s_val)
        lam_list.append(lam)

    a_t = torch.tensor(a_points, device=device, dtype=dtype)
    omega_t = torch.tensor(omega_points, device=device, dtype=dtype)
    lam_t = torch.tensor(np.array(lam_list), device=device, dtype=cdtype)
    u_t = torch.zeros(args.n_points, device=device, dtype=dtype)
    v_t = torch.ones(args.n_points, device=device, dtype=dtype) * 0.6

    # y grid for profiles
    y_grid = torch.linspace(-0.999, 0.999, n_y, device=device, dtype=dtype).unsqueeze(0).expand(args.n_points, -1)

    # y grid for near-infinity
    y_near = torch.linspace(-0.999, -0.95, 96, device=device, dtype=dtype).unsqueeze(0).expand(args.n_points, -1)

    # Compute profiles
    profiles = {}
    for name, mdl in models.items():
        print(f"Computing {name} profiles...")
        S_prof, pde_prof = compute_S_profile(mdl, a_t, omega_t, u_t, v_t, lam_t, y_grid, M=M_val, m=m_mode, s=s_val)
        _, pde_near = compute_S_profile(mdl, a_t, omega_t, u_t, v_t, lam_t, y_near, M=M_val, m=m_mode, s=s_val)
        ep = compute_stage1_S_and_Sy_at_infinity(
            mdl, a_t.squeeze(-1), omega_t.squeeze(-1), u=u_t, v=v_t, lambda_=lam_t, m=m_mode, M=M_val, s=s_val,
        )
        profiles[name] = {
            "S": S_prof,
            "pde_residual": pde_prof,
            "pde_near": pde_near,
            "S_inf": ep["S_inf"].detach().cpu().numpy(),
            "Sy_inf": ep["Sy_inf"].detach().cpu().numpy(),
            "c_inf": ep["c_inf"].detach().cpu().numpy(),
            "robin_res": ep["robin_residual"].detach().cpu().numpy(),
            "robin_rel": ep["robin_rel"].detach().cpu().numpy(),
        }

    y_np = y_grid[0].detach().cpu().numpy()
    y_near_np = y_near[0].detach().cpu().numpy()

    # ==== Figure 1: Re(S) and Im(S) ====
    fig, axes = plt.subplots(args.n_points, 2, figsize=(14, 3 * args.n_points))
    if args.n_points == 1:
        axes = axes.reshape(1, -1)
    colors = {"original": "blue", "adam": "green", "lbfgs": "red"}
    styles = {"original": "--", "adam": "-", "lbfgs": "-."}

    for i in range(args.n_points):
        for name in models:
            S = profiles[name]["S"][i]
            axes[i, 0].plot(y_np, S.real, color=colors[name], ls=styles[name], label=name, alpha=0.8)
            axes[i, 1].plot(y_np, S.imag, color=colors[name], ls=styles[name], label=name, alpha=0.8)
        axes[i, 0].set_ylabel(f"a={a_points[i]:.2f}, w={omega_points[i]:.2f}\nRe(S)")
        axes[i, 1].set_ylabel("Im(S)")
        axes[i, 0].legend(fontsize=7)
        axes[i, 1].legend(fontsize=7)
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("y")
    axes[-1, 1].set_xlabel("y")
    fig.suptitle("Re(S) and Im(S) Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "re_im_S_comparison.png", dpi=150)
    plt.close(fig)

    # ==== Figure 2: |S| and arg(S) ====
    fig, axes = plt.subplots(args.n_points, 2, figsize=(14, 3 * args.n_points))
    if args.n_points == 1:
        axes = axes.reshape(1, -1)
    for i in range(args.n_points):
        for name in models:
            S = profiles[name]["S"][i]
            axes[i, 0].plot(y_np, np.abs(S), color=colors[name], ls=styles[name], label=name, alpha=0.8)
            axes[i, 1].plot(y_np, np.angle(S), color=colors[name], ls=styles[name], label=name, alpha=0.8)
        axes[i, 0].set_ylabel(f"a={a_points[i]:.2f}\n|S|")
        axes[i, 1].set_ylabel("arg(S)")
        axes[i, 0].legend(fontsize=7)
        axes[i, 1].legend(fontsize=7)
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("y")
    axes[-1, 1].set_xlabel("y")
    fig.suptitle("|S| and arg(S) Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "abs_arg_S_comparison.png", dpi=150)
    plt.close(fig)

    # ==== Figure 3: Robin residual per point ====
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(args.n_points)
    width = 0.25
    for j, name in enumerate(models):
        robin_rel = profiles[name]["robin_rel"]
        ax.bar(x_pos + j * width, robin_rel, width, label=name, color=colors[name], alpha=0.8)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([f"a={a:.2f}\nw={w:.2f}" for a, w in zip(a_points, omega_points)])
    ax.set_ylabel("Robin relative residual")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Robin Residual at y=-1")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "robin_residual_bar.png", dpi=150)
    plt.close(fig)

    # ==== Figure 4: Sy(-1) vs c_inf * S(-1) scatter ====
    fig, ax = plt.subplots(figsize=(8, 8))
    for name in models:
        Sy = profiles[name]["Sy_inf"]
        cS = profiles[name]["c_inf"] * profiles[name]["S_inf"]
        ax.scatter(cS.real, cS.imag, marker="o", label=f"{name} c_inf*S(-1)", color=colors[name], s=80)
        ax.scatter(Sy.real, Sy.imag, marker="x", label=f"{name} Sy(-1)", color=colors[name], s=80)
    ax.axhline(0, color="gray", alpha=0.3)
    ax.axvline(0, color="gray", alpha=0.3)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title("Sy(-1) vs c_inf*S(-1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "Sy_vs_cS_scatter.png", dpi=150)
    plt.close(fig)

    # ==== Figure 5: Near-infinity PDE residual ====
    fig, axes = plt.subplots(args.n_points, 1, figsize=(12, 3 * args.n_points))
    if args.n_points == 1:
        axes = [axes]
    for i in range(args.n_points):
        for name in models:
            axes[i].plot(y_near_np, profiles[name]["pde_near"][i], color=colors[name], ls=styles[name], label=name, alpha=0.8)
        axes[i].set_ylabel(f"a={a_points[i]:.2f}, w={omega_points[i]:.2f}\n|PDE res|")
        axes[i].set_yscale("log")
        axes[i].legend(fontsize=7)
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("y")
    fig.suptitle("Near-infinity PDE Residual", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "near_pde_residual.png", dpi=150)
    plt.close(fig)

    # ==== Figure 6: Full y-range PDE residual ====
    fig, axes = plt.subplots(args.n_points, 1, figsize=(12, 3 * args.n_points))
    if args.n_points == 1:
        axes = [axes]
    for i in range(args.n_points):
        for name in models:
            axes[i].plot(y_np, profiles[name]["pde_residual"][i], color=colors[name], ls=styles[name], label=name, alpha=0.8)
        axes[i].set_ylabel(f"a={a_points[i]:.2f}\n|PDE res|")
        axes[i].set_yscale("log")
        axes[i].legend(fontsize=7)
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("y")
    fig.suptitle("Full-range PDE Residual", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "full_pde_residual.png", dpi=150)
    plt.close(fig)

    # ==== Save metrics JSON ====
    metrics = {}
    for name in models:
        metrics[name] = {
            "robin_rel_median": float(np.median(profiles[name]["robin_rel"])),
            "robin_rel_max": float(np.max(profiles[name]["robin_rel"])),
            "robin_rel_per_point": [float(x) for x in profiles[name]["robin_rel"]],
            "S_inf_magnitude": float(np.mean(np.abs(profiles[name]["S_inf"]))),
            "Sy_inf_magnitude": float(np.mean(np.abs(profiles[name]["Sy_inf"]))),
            "c_inf_range": [float(np.min(np.abs(profiles[name]["c_inf"]))), float(np.max(np.abs(profiles[name]["c_inf"])))],
        }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ==== Write summary ====
    lines = ["# Stage-1 Infinity Robin Refinement — Comparison Summary\n"]
    lines.append(f"Original: `{args.original}`")
    if args.adam_best:
        lines.append(f"Adam best: `{args.adam_best}`")
    if args.lbfgs_best:
        lines.append(f"L-BFGS best: `{args.lbfgs_best}`")
    lines.append("")
    lines.append("## Robin Residual (relative)")
    lines.append("| Model | Median | Max |")
    lines.append("|-------|--------|-----|")
    for name in models:
        m = metrics[name]
        lines.append(f"| {name} | {m['robin_rel_median']:.4e} | {m['robin_rel_max']:.4e} |")
    lines.append("")
    lines.append("## Per-point Robin relative residual")
    for name in models:
        pts = ", ".join(f"{x:.4e}" for x in metrics[name]["robin_rel_per_point"])
        lines.append(f"- {name}: [{pts}]")
    lines.append("")
    lines.append("## S(-1) and Sy(-1) magnitudes (mean |·|)")
    for name in models:
        lines.append(f"- {name}: |S(-1)|={metrics[name]['S_inf_magnitude']:.4e}, |Sy(-1)|={metrics[name]['Sy_inf_magnitude']:.4e}")

    summary = "\n".join(lines)
    with open(out_dir / "comparison_summary.md", "w") as f:
        f.write(summary)
    print(summary)
    print(f"\nPlots saved to {out_dir / 'plots'}")
    print(f"Metrics saved to {out_dir / 'metrics.json'}")
    print(f"Summary saved to {out_dir / 'comparison_summary.md'}")


if __name__ == "__main__":
    main()
