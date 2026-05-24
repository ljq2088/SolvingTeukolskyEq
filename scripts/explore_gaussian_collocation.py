#!/usr/bin/env python3
"""Explore truncated-normal y-sampling strategies for Stage-1 PINN training.

Compares three collocation strategies:
  baseline  — Chebyshev-Lobatto grid (current default)
  gaussian  — Truncated normal N(-1, sigma) concentrated near infinity
  mixed     — 70% normal + 30% uniform (domain coverage + infinity focus)

Each strategy adds explicit y=-1 Robin BC constraint with configurable weight.
Short runs (30 epochs) to quickly reveal loss trends.
"""
import argparse
import json
import os
import sys
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
from dataset.sampling import (
    sample_points_chebyshev_grid,
    sample_points_random_uniform,
    sample_points_normal_truncated,
    sample_points_residual_weighted,
)
from physical_ansatz.residual_pinn import compute_f_derivatives_autograd
from physical_ansatz.teukolsky_coeffs import coeffs_x
from physical_ansatz.transform_y import (
    transform_coeffs_x_to_y,
    horizon_regularity_slope,
)
from physical_ansatz.infinity_robin import (
    analytic_c_inf,
    infinity_robin_loss,
    compute_S_and_Sy_at_infinity,
)
from domain.atlas_builder import load_atlas, map_to_chart, interp_lower_upper
from utils.compute_lambda_usage import compute_lambda


# ============================================================
# helpers
# ============================================================

def sample_param_batch(comp, batch_size, device, dtype, omega_chart_mode="log10"):
    """Sample random (a, omega) within component envelope, compute (u,v)."""
    a0, a1 = comp.a_support[0], comp.a_support[-1]
    a = a0 + torch.rand(batch_size, device=device, dtype=dtype) * (a1 - a0)
    omega = torch.empty(batch_size, device=device, dtype=dtype)
    u = torch.empty(batch_size, device=device, dtype=dtype)
    v = torch.empty(batch_size, device=device, dtype=dtype)
    for i in range(batch_size):
        ai = float(a[i])
        low, up = interp_lower_upper(comp, ai)
        omega[i] = low + torch.rand(1, device=device, dtype=dtype).item() * (up - low)
        ui, vi = map_to_chart(comp, ai, float(omega[i]), omega_chart_mode=omega_chart_mode)
        u[i] = ui
        v[i] = vi
    return a, omega, u, v


_lambda_cache = {}

def compute_lambda_cached(a, omega, ell, m, s):
    key = (round(a, 10), round(omega, 10))
    if key not in _lambda_cache:
        _lambda_cache[key] = compute_lambda(a, omega, ell, m, s=s)
    return _lambda_cache[key]


# ============================================================
# residual evaluation for adaptive sampling
# ============================================================

def evaluate_residual_profile(model, a_batch, omega_batch, u_batch, v_batch,
                              lambda_batch, y_eval, M=1.0, m_val=2, s_val=-2):
    """Evaluate PDE |residual| profile on dense y-grid (no grad accum).

    Returns (N_eval,) tensor of mean |residual| across batch.
    """
    B = a_batch.shape[0]
    N = y_eval.shape[0]

    y_in = y_eval.unsqueeze(0).expand(B, -1).contiguous()  # (B, N)

    f_int, f_y_int, f_yy_int = compute_f_derivatives_autograd(
        model, a_batch, omega_batch, y_in,
        u_batch=u_batch, v_batch=v_batch,
    )

    x_int = (y_eval + 1.0) / 2.0  # (N,)
    A2_l, A1_l, A0_l = [], [], []
    for i in range(B):
        A2, A1, A0 = coeffs_x(
            x=x_int, a=a_batch[i], omega=omega_batch[i],
            m=m_val, lambda_=lambda_batch[i], s=s_val, M=M,
        )
        A2_l.append(A2); A1_l.append(A1); A0_l.append(A0)
    A2_int = torch.stack(A2_l, dim=0)  # (B, N)
    A1_int = torch.stack(A1_l, dim=0)
    A0_int = torch.stack(A0_l, dim=0)

    slope_int = horizon_regularity_slope(
        a=a_batch, omega=omega_batch, lambda_=lambda_batch,
        m=m_val, M=M, s=s_val,
    )

    y_ode = y_eval.unsqueeze(0)  # (1, N)
    B2, B1, B0, rhs = transform_coeffs_x_to_y(
        A2_int, A1_int, A0_int, y_ode, slope=slope_int,
    )

    residual = B2 * f_yy_int + B1 * f_y_int + B0 * f_int - rhs  # (B, N)
    abs_residual = torch.abs(residual)  # (B, N)

    # Average across batch, detach to free autograd graph
    profile = abs_residual.mean(dim=0).detach()  # (N,)

    return profile


# ============================================================
# loss functions
# ============================================================

def compute_pde_loss(model, a_batch, omega_batch, u_batch, v_batch,
                     lambda_batch, y_interior, M=1.0, m_val=2, s_val=-2,
                     y_mode="1d"):
    """PDE residual loss for f(y) formulation.

    y_mode='2d': y_interior is 1D (N,) — expanded to (B,N) per sample.
    y_mode='1d': y_interior is 1D (B,) — one y-point per batch element (matches
                 original AtlasPatchTrainer strategy, more stable).
    """
    B = a_batch.shape[0]
    if y_mode == "2d":
        y_in = y_interior.unsqueeze(0).expand(B, -1).contiguous()
        y_ode = y_interior  # (N,) for transforming ODE coefficients
    else:
        y_in = y_interior  # (B,) — one point per sample
        y_ode = y_interior.unsqueeze(0)  # (1, B) for transform_coeffs_x_to_y

    f_int, f_y_int, f_yy_int = compute_f_derivatives_autograd(
        model, a_batch, omega_batch, y_in,
        u_batch=u_batch, v_batch=v_batch,
    )
    # f_int: (B, N) for 2d, (B, 1) for 1d

    if y_mode == "1d":
        # Per-sample y: need per-sample ODE coefficients
        x_pts = (y_interior + 1.0) / 2.0  # (B,)
        A2_l, A1_l, A0_l = [], [], []
        for i in range(B):
            A2, A1, A0 = coeffs_x(
                x=x_pts[i:i+1], a=a_batch[i], omega=omega_batch[i],
                m=m_val, lambda_=lambda_batch[i], s=s_val, M=M,
            )
            A2_l.append(A2.squeeze()); A1_l.append(A1.squeeze()); A0_l.append(A0.squeeze())
        A2_int = torch.stack(A2_l, dim=0)  # (B,)
        A1_int = torch.stack(A1_l, dim=0)
        A0_int = torch.stack(A0_l, dim=0)
    else:
        x_int = (y_interior + 1.0) / 2.0  # (N,)
        A2_l, A1_l, A0_l = [], [], []
        for i in range(B):
            A2, A1, A0 = coeffs_x(
                x=x_int, a=a_batch[i], omega=omega_batch[i],
                m=m_val, lambda_=lambda_batch[i], s=s_val, M=M,
            )
            A2_l.append(A2); A1_l.append(A1); A0_l.append(A0)
        A2_int = torch.stack(A2_l, dim=0)  # (B, N)
        A1_int = torch.stack(A1_l, dim=0)
        A0_int = torch.stack(A0_l, dim=0)

    slope_int = horizon_regularity_slope(
        a=a_batch, omega=omega_batch, lambda_=lambda_batch,
        m=m_val, M=M, s=s_val,
    )

    B2, B1, B0, rhs = transform_coeffs_x_to_y(
        A2_int, A1_int, A0_int, y_ode, slope=slope_int,
    )
    if y_mode == "1d":
        B2 = B2.squeeze(-1); B1 = B1.squeeze(-1); B0 = B0.squeeze(-1)
        rhs = rhs.squeeze(-1) if rhs is not None else 0.0
        f_s = f_int.squeeze(-1); fy_s = f_y_int.squeeze(-1); fyy_s = f_yy_int.squeeze(-1)
    else:
        f_s, fy_s, fyy_s = f_int, f_y_int, f_yy_int

    residual = B2 * fyy_s + B1 * fy_s + B0 * f_s - rhs
    return torch.mean(torch.abs(residual) ** 2)


def compute_robin_loss(model, a_batch, omega_batch, u_batch, v_batch,
                       lambda_batch, M=1.0, m_val=2, s_val=-2):
    """Infinity Robin BC loss at y=-1."""
    B = a_batch.shape[0]
    device = a_batch.device
    dtype = a_batch.dtype
    cdtype = torch.complex128

    y_inf = torch.full((B, 1), -1.0, device=device, dtype=dtype)

    f_inf, fy_inf, _ = compute_f_derivatives_autograd(
        model, a_batch, omega_batch, y_inf,
        u_batch=u_batch, v_batch=v_batch,
    )
    f_inf = f_inf.squeeze(-1)
    fy_inf = fy_inf.squeeze(-1)
    if fy_inf.ndim > 1 and fy_inf.shape[0] == fy_inf.shape[1]:
        fy_inf = fy_inf.diagonal()

    slope_inf = horizon_regularity_slope(
        a=a_batch, omega=omega_batch, lambda_=lambda_batch,
        m=m_val, M=M, s=s_val,
    )
    S_inf, Sy_inf = compute_S_and_Sy_at_infinity(f_inf, fy_inf, slope_inf)

    if not torch.is_complex(S_inf):
        S_inf = S_inf.to(dtype=cdtype)
    if not torch.is_complex(Sy_inf):
        Sy_inf = Sy_inf.to(dtype=cdtype)
    lambda_c = lambda_batch.to(dtype=cdtype) if not torch.is_complex(lambda_batch) else lambda_batch

    c_inf = analytic_c_inf(a_batch, omega_batch, lambda_c, m=m_val, M=M, s=s_val)
    return infinity_robin_loss(S_inf, Sy_inf, c_inf).mean()


# ============================================================
# training loop
# ============================================================

def run_experiment(model, comp, strategy, sigma, robin_weight, epochs,
                   batch_size, n_interior, lr, device, dtype, omega_chart_mode,
                   y_mode="1d",
                   ell=2, m_val=2, s_val=-2, M_val=1.0,
                   resample_every=5, n_eval=256,
                   uniform_frac_init=1.0, uniform_frac_final=0.3,
                   temperature=1.0):
    """Train for N epochs with given strategy, return loss history.

    y_mode='1d': one y-point per batch element (original trainer style, stable).
    y_mode='2d': full (B,N) y-grid (higher variance, for distribution testing).

    strategy='adaptive': cosine-annealing from uniform to residual-based sampling.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"epoch": [], "loss_pde": [], "loss_robin": [],
               "loss_total": [], "grad_norm": [], "uniform_frac": []}

    # Adaptive strategy: build dense evaluation grid (fixed, for residual profiling)
    y_eval_grid = None
    if strategy == "adaptive":
        y_eval_grid = sample_points_chebyshev_grid(
            n_eval, y_min=-0.999, y_max=0.999, device=device, dtype=dtype)
        # Initial collocation points from Chebyshev grid (warm-up)
        if y_mode == "1d":
            current_y = sample_points_chebyshev_grid(
                batch_size, y_min=-0.999, y_max=0.999, device=device, dtype=dtype)
        else:
            current_y = sample_points_chebyshev_grid(
                n_interior, y_min=-0.999, y_max=0.999, device=device, dtype=dtype)

    for epoch in range(epochs):
        current_uniform_frac = float("nan")  # only used by adaptive
        # Sample parameters
        a_b, omega_b, u_b, v_b = sample_param_batch(
            comp, batch_size, device, dtype, omega_chart_mode)
        lam_b = torch.tensor(
            [compute_lambda_cached(float(a_b[i]), float(omega_b[i]),
                                   ell, m_val, s_val) for i in range(batch_size)],
            device=device, dtype=torch.complex128)

        # Adaptive resampling
        if strategy == "adaptive" and epoch % resample_every == 0:
            # Cosine-annealing progress
            warmup = max(1, epochs // 10)
            if epoch < warmup:
                alpha = 0.0
            else:
                progress = min(1.0, (epoch - warmup) / max(1, epochs - warmup))
                import math
                alpha = 0.5 * (1.0 - math.cos(math.pi * progress))
            uniform_frac = uniform_frac_init + alpha * (uniform_frac_final - uniform_frac_init)
            current_uniform_frac = uniform_frac

            # Evaluate residual profile
            profile = evaluate_residual_profile(
                model, a_b, omega_b, u_b, v_b, lam_b, y_eval_grid,
                M=M_val, m_val=m_val, s_val=s_val,
            )

            # Resample collocation points
            n_select = batch_size if y_mode == "1d" else n_interior
            current_y = sample_points_residual_weighted(
                y_eval_grid, profile, n_select,
                uniform_frac=uniform_frac, temperature=temperature,
                device=device, dtype=dtype,
            )

            if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"  [adaptive] epoch {epoch+1}: alpha={alpha:.2f}, "
                      f"uniform_frac={uniform_frac:.2f}, "
                      f"residual max={profile.max().item():.1e}, mean={profile.mean().item():.1e}")

        # Sample y-points
        if strategy == "adaptive":
            y_int = current_y
        elif strategy == "baseline":
            if y_mode == "1d":
                y_int = sample_points_chebyshev_grid(
                    batch_size, y_min=-0.999, y_max=0.999, device=device, dtype=dtype)
            else:
                y_int = sample_points_chebyshev_grid(
                    n_interior, y_min=-0.999, y_max=0.999, device=device, dtype=dtype)
        elif strategy == "gaussian":
            if y_mode == "1d":
                y_int = sample_points_normal_truncated(
                    batch_size, sigma=sigma, y_min=-0.999, y_max=0.999,
                    device=device, dtype=dtype)
            else:
                y_int = sample_points_normal_truncated(
                    n_interior, sigma=sigma, y_min=-0.999, y_max=0.999,
                    device=device, dtype=dtype)
        elif strategy == "mixed":
            if y_mode == "1d":
                n_gauss = int(0.7 * batch_size)
                n_unif = batch_size - n_gauss
            else:
                n_gauss = int(0.7 * n_interior)
                n_unif = n_interior - n_gauss
            y_g = sample_points_normal_truncated(
                n_gauss, sigma=sigma, y_min=-0.999, y_max=0.999,
                device=device, dtype=dtype)
            y_u = sample_points_random_uniform(
                n_unif, y_min=-0.999, y_max=0.999, device=device, dtype=dtype)
            y_int = torch.sort(torch.cat([y_g, y_u])).values
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Losses
        loss_pde = compute_pde_loss(model, a_b, omega_b, u_b, v_b, lam_b,
                                    y_int, M=M_val, m_val=m_val, s_val=s_val,
                                    y_mode=y_mode)
        loss_robin = compute_robin_loss(model, a_b, omega_b, u_b, v_b, lam_b,
                                        M=M_val, m_val=m_val, s_val=s_val)
        loss_total = loss_pde + robin_weight * loss_robin

        # Step
        optimizer.zero_grad()
        loss_total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        history["epoch"].append(epoch)
        history["loss_pde"].append(float(loss_pde.detach()))
        history["loss_robin"].append(float(loss_robin.detach()))
        history["loss_total"].append(float(loss_total.detach()))
        history["grad_norm"].append(float(grad_norm))
        history["uniform_frac"].append(
            current_uniform_frac if strategy == "adaptive" else float("nan"))

        if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  [{strategy}] epoch {epoch+1:3d}/{epochs}  "
                  f"pde={loss_pde.detach().item():.3e}  robin={loss_robin.detach().item():.3e}  "
                  f"total={loss_total.detach().item():.3e}  |g|={grad_norm.item():.1f}")

    return history


# ============================================================
# plotting
# ============================================================

def plot_comparison(histories, out_dir, sigma, robin_weight):
    """Plot loss curves for multiple strategies."""
    colors = {"baseline": "gray", "gaussian": "C0", "mixed": "C2", "adaptive": "C3"}
    styles = {"baseline": "--", "gaussian": "-", "mixed": "-.", "adaptive": "-"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for key in ["loss_pde", "loss_robin"]:
        ax = axes[0][0] if key == "loss_pde" else axes[0][1]
        for name, hist in histories.items():
            ax.plot(hist["epoch"], hist[key], color=colors[name], ls=styles[name],
                    label=name, lw=1.5, alpha=0.85)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key.replace("_", " ").title())
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Total loss
    ax = axes[1][0]
    for name, hist in histories.items():
        ax.plot(hist["epoch"], hist["loss_total"],
                color=colors[name], ls=styles[name], label=name, lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Gradient norm
    ax = axes[1][1]
    for name, hist in histories.items():
        ax.plot(hist["epoch"], hist["grad_norm"],
                color=colors[name], ls=styles[name], label=name, lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Collocation Strategy Comparison  ($\\sigma$={sigma}, Robin weight={robin_weight})",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_loss.png", dpi=150)
    plt.close(fig)

    # Also plot y-point distribution for each strategy
    fig, ax = plt.subplots(figsize=(10, 4))
    for name in histories:
        if name == "baseline":
            y_pts = sample_points_chebyshev_grid(128, -0.999, 0.999).numpy()
        elif name == "gaussian":
            y_pts = sample_points_normal_truncated(128, sigma=sigma).numpy()
        elif name == "mixed":
            y_g = sample_points_normal_truncated(int(128*0.7), sigma=sigma).numpy()
            y_u = sample_points_random_uniform(int(128*0.3), -0.999, 0.999).numpy()
            y_pts = np.sort(np.concatenate([y_g, y_u]))
        ax.hist(y_pts, bins=40, alpha=0.5, label=name, color=colors[name])
    ax.axvline(-1.0, color='red', ls='--', lw=1, label='y=-1 (infinity)')
    ax.set_xlabel("y")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.set_title(f"Collocation Point Distribution  ($\\sigma$={sigma})")
    fig.tight_layout()
    fig.savefig(out_dir / "point_distribution.png", dpi=150)
    plt.close(fig)


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Explore truncated-normal y-sampling for Stage-1 PINN")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/autoencoder_stage1_rin_train/"
                                "20260518_194655_patch_000_comp_0_u_0.500_v_0.603/"
                                "checkpoints/best_model.pt")
    parser.add_argument("--atlas-json", type=str,
                        default="outputs/domain/atlas_l2_m2_logw.json")
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["baseline", "gaussian", "mixed", "adaptive", "all"])
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--robin-weight", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-interior", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--y-mode", type=str, default="1d",
                        choices=["1d", "2d"],
                        help="1d: one y-point/batch (stable), 2d: full grid (high var)")
    parser.add_argument("--resample-every", type=int, default=5,
                        help="Adaptive: resample collocation points every N epochs")
    parser.add_argument("--n-eval", type=int, default=256,
                        help="Adaptive: number of evaluation grid points")
    parser.add_argument("--uniform-frac-init", type=float, default=1.0,
                        help="Adaptive: initial uniform fraction (warm-up)")
    parser.add_argument("--uniform-frac-final", type=float, default=0.3,
                        help="Adaptive: final uniform fraction (after annealing)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Adaptive: softmax temperature for residual-based sampling")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64
    cdtype = torch.complex128
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    M_val, ell, m_val, s_val = 1.0, 2, 2, -2

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs/gaussian_collocation_exp") / timestamp
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load atlas
    atlas = load_atlas(args.atlas_json)
    comp = atlas.components[0]
    omega_chart_mode = atlas.meta.get("omega_chart_mode", "log10")

    # Load model base
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_base = AutoencoderTeukolskyPINN()
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model_base.load_state_dict(sd)
    print(f"  Loaded {len(sd)} keys")

    # Save config
    with open(out_dir / "experiment_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    strategies = [args.strategy] if args.strategy != "all" else ["baseline", "gaussian", "mixed", "adaptive"]
    all_histories = {}

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}  sigma={args.sigma}  robin_weight={args.robin_weight}  epochs={args.epochs}")
        print(f"{'='*60}")

        # Fresh model copy
        model = AutoencoderTeukolskyPINN()
        model.load_state_dict(sd)
        model.to(device)

        hist = run_experiment(
            model=model, comp=comp, strategy=strategy,
            sigma=args.sigma, robin_weight=args.robin_weight,
            epochs=args.epochs, batch_size=args.batch_size,
            n_interior=args.n_interior, lr=args.lr,
            device=device, dtype=dtype,
            omega_chart_mode=omega_chart_mode,
            y_mode=args.y_mode,
            ell=ell, m_val=m_val, s_val=s_val, M_val=M_val,
            resample_every=args.resample_every, n_eval=args.n_eval,
            uniform_frac_init=args.uniform_frac_init,
            uniform_frac_final=args.uniform_frac_final,
            temperature=args.temperature,
        )
        all_histories[strategy] = hist

        # Save per-strategy history
        np.savez(out_dir / f"history_{strategy}_s{args.sigma}.npz", **hist)

    if len(all_histories) > 1:
        plot_comparison(all_histories, out_dir, args.sigma, args.robin_weight)
        print(f"\nComparison plots saved to {out_dir}")

    # Print summary
    print(f"\n{'='*60}")
    print("Final loss summary:")
    print(f"{'='*60}")
    for name, hist in all_histories.items():
        pde_first, pde_last = hist["loss_pde"][0], hist["loss_pde"][-1]
        robin_first, robin_last = hist["loss_robin"][0], hist["loss_robin"][-1]
        print(f"  {name:10s}:  PDE {pde_first:.3e} -> {pde_last:.3e}  "
              f"Robin {robin_first:.3e} -> {robin_last:.3e}")

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
