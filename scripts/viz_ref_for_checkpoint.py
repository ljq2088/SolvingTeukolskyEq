#!/usr/bin/env python3
"""Generate reference visualization for a checkpoint (L-BFGS refined or training).

Usage:
  python scripts/viz_ref_for_checkpoint.py \
    --checkpoint outputs/stage1_lbfgs_refine/.../checkpoints/best_model.pt \
    --config config/autoencoder_stage1_pinn_random_v3.yaml \
    --a 0.5 --omega 0.04 --u 0.5 --v 0.6 \
    --device cuda
"""
import argparse, sys, os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.residual import get_lambda_from_cfg, AuxCache
from physical_ansatz.residual_pinn import compose_reduced_shape_from_f, horizon_regularity_slope
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives, r_plus
from physical_ansatz.transform_y import h_factor
from dataset.sampling import sample_points_chebyshev_grid
from config.config_loader import load_pinn_full_config

# Try spectral benchmark
try:
    from teukrad import KerrMode, TeukRadAmplitudeInWithInterpolant
    HAS_SPECTRAL = True
except ImportError:
    HAS_SPECTRAL = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--omega", type=float, default=0.04)
    parser.add_argument("--u", type=float, default=0.5)
    parser.add_argument("--v", type=float, default=0.6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--viz-n-points", type=int, default=400)
    parser.add_argument("--r-min", type=float, default=2.0)
    parser.add_argument("--r-max", type=float, default=1000.0)
    parser.add_argument("--spectral-N", type=int, default=80)
    parser.add_argument("--spectral-z-m", type=float, default=0.15)
    parser.add_argument("--output", default=None, help="Output image path")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Load config
    full_cfg = load_pinn_full_config(args.config)
    physics_cfg = full_cfg["physics"]
    problem_cfg = physics_cfg["problem"]
    M = float(problem_cfg.get("M", 1.0))
    l = int(problem_cfg.get("l", 2))
    m = int(problem_cfg.get("m", 2))
    s = int(problem_cfg.get("s", -2))
    cfg = full_cfg["train"]
    mc = cfg.get("model", {})

    # Build model
    model = AutoencoderTeukolskyPINN(
        hidden_dims=mc.get("hidden_dims", [128, 128, 128, 128]),
        activation=mc.get("activation", "silu"),
        fourier_num_freqs=mc.get("fourier_num_freqs", 2),
        fourier_base_scale=mc.get("fourier_base_scale", 1.0),
        param_embed_dim=mc.get("param_embed_dim", 64),
        use_film=mc.get("use_film", True),
        use_residual=mc.get("use_residual", True),
    )
    ckpt = torch.load(str(ckpt_path), map_location=device)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    step = ckpt.get("step", "?")
    print(f"Loaded: {ckpt_path} (step={step})")

    # Reference params
    a_t = torch.tensor(args.a, device=device, dtype=dtype)
    omega_t = torch.tensor(args.omega, device=device, dtype=dtype)
    u_t = torch.tensor(args.u, device=device, dtype=dtype)
    v_t = torch.tensor(args.v, device=device, dtype=dtype)

    cache = AuxCache()
    lam = get_lambda_from_cfg(physics_cfg, cache, a_t, omega_t)
    lam_complex = torch.tensor(lam, device=device, dtype=torch.complex128)
    h2 = h_factor(a_t, omega_t, m=m, M=M, s=s)

    rp = r_plus(a_t, M)
    r_min = max(args.r_min, float(rp.detach().cpu().item()) + 1e-4)
    r_max = args.r_max

    # Left: uniform r grid
    r_grid = torch.linspace(r_min, r_max, args.viz_n_points, device=device, dtype=dtype)
    x_grid_from_r = rp / r_grid
    y_grid_from_r = 2.0 * x_grid_from_r - 1.0

    # Right: Chebyshev y grid
    y_min = 2.0 * float(rp.detach().cpu().item()) / r_max - 1.0
    y_max = 2.0 * float(rp.detach().cpu().item()) / r_min - 1.0
    y_grid_cheb = sample_points_chebyshev_grid(
        n_points=args.viz_n_points, y_min=y_min, y_max=y_max,
        device=device, dtype=dtype,
    )
    x_grid_from_y = 0.5 * (y_grid_cheb + 1.0)
    r_grid_from_y = rp / x_grid_from_y

    # Predict
    with torch.no_grad():
        # On r-grid
        f_pred_r = model(a_t.unsqueeze(0), omega_t.unsqueeze(0), y_grid_from_r,
                         u=u_t.unsqueeze(0), v=v_t.unsqueeze(0)).squeeze(0)
        slope_r = horizon_regularity_slope(
            a=a_t.unsqueeze(0), omega=omega_t.unsqueeze(0),
            lambda_=lam_complex.unsqueeze(0), m=m, M=M, s=s,
        ).squeeze(0)
        shape_pred_r = compose_reduced_shape_from_f(f=f_pred_r, y=y_grid_from_r, slope=slope_r)

        rp_r, rm_r, _, _, _ = build_prefactor_primitives(r_grid, a_t, M=M, need_rs=False)
        P_r, _, _ = Leaver_prefactors(r_grid, a_t, omega_t, m=m, M=M, s=s, rp=rp_r, rm=rm_r)
        R_pred_r = P_r * h2 * shape_pred_r

        # On y-grid
        f_pred_y = model(a_t.unsqueeze(0), omega_t.unsqueeze(0), y_grid_cheb,
                         u=u_t.unsqueeze(0), v=v_t.unsqueeze(0)).squeeze(0)
        slope_y = horizon_regularity_slope(
            a=a_t.unsqueeze(0), omega=omega_t.unsqueeze(0),
            lambda_=lam_complex.unsqueeze(0), m=m, M=M, s=s,
        ).squeeze(0)
        shape_pred_y = compose_reduced_shape_from_f(f=f_pred_y, y=y_grid_cheb, slope=slope_y)

        rp_y, rm_y, _, _, _ = build_prefactor_primitives(r_grid_from_y, a_t, M=M, need_rs=False)
        P_y, _, _ = Leaver_prefactors(r_grid_from_y, a_t, omega_t, m=m, M=M, s=s, rp=rp_y, rm=rm_y)
        R_pred_y = P_y * h2 * shape_pred_y

    # Convert to numpy
    r_np = r_grid.detach().cpu().numpy()
    y_np = y_grid_cheb.detach().cpu().numpy()
    R_pred_r_np = R_pred_r.detach().cpu().numpy()
    shape_pred_y_np = shape_pred_y.detach().cpu().numpy()

    # Benchmark
    a_scalar = args.a
    omega_scalar = args.omega
    benchmark_available = False
    benchmark_status = "benchmark=off"
    R_ref_r_np = None
    shape_ref_y_np = None

    if HAS_SPECTRAL:
        try:
            mode = KerrMode(M=M, a=a_scalar, omega=omega_scalar, ell=l, m=m,
                          lam=complex(lam), s=s)
            spectral = TeukRadAmplitudeInWithInterpolant(
                mode=mode, N_in=args.spectral_N, N_out=args.spectral_N, z_m=args.spectral_z_m,
            )
            profile = spectral.profile
            R_ref_r_np = np.asarray(profile.R_of_r(r_np), dtype=np.complex128)
            R_ref_y_np = np.asarray(profile.R_of_r(r_grid_from_y.detach().cpu().numpy()), dtype=np.complex128)
            shape_ref_y_np = R_ref_y_np / (P_y.detach().cpu().numpy() * complex(h2.detach().cpu().item()))
            benchmark_available = True
            benchmark_status = f"benchmark=spectral(N={args.spectral_N})"
        except Exception as e:
            benchmark_status = f"benchmark=spectral-failed: {e}"

    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=False)

    axes[0, 0].plot(r_np, np.real(R_pred_r_np), label="Pred Re(R)", lw=1.6)
    if benchmark_available:
        axes[0, 0].plot(r_np, np.real(R_ref_r_np), "--", label="ref Re(R)", lw=1.0)
    axes[0, 0].set_ylabel("Re(R)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    axes[1, 0].plot(r_np, np.imag(R_pred_r_np), label="Pred Im(R)", lw=1.6)
    if benchmark_available:
        axes[1, 0].plot(r_np, np.imag(R_ref_r_np), "--", label="ref Im(R)", lw=1.0)
    axes[1, 0].set_ylabel("Im(R)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    axes[2, 0].plot(r_np, np.abs(R_pred_r_np), label="Pred |R|", lw=1.6)
    if benchmark_available:
        axes[2, 0].plot(r_np, np.abs(R_ref_r_np), "--", label="ref |R|", lw=1.0)
    axes[2, 0].set_ylabel("|R|")
    axes[2, 0].set_xlabel("r")
    axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(alpha=0.3)

    axes[0, 1].plot(y_np, np.real(shape_pred_y_np), label="Pred Re(S)", lw=1.6)
    if benchmark_available:
        axes[0, 1].plot(y_np, np.real(shape_ref_y_np), "--", label="ref Re(S)", lw=1.0)
    axes[0, 1].set_ylabel("Re(S)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 1].plot(y_np, np.imag(shape_pred_y_np), label="Pred Im(S)", lw=1.6)
    if benchmark_available:
        axes[1, 1].plot(y_np, np.imag(shape_ref_y_np), "--", label="ref Im(S)", lw=1.0)
    axes[1, 1].set_ylabel("Im(S)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    axes[2, 1].plot(y_np, np.abs(shape_pred_y_np), label="Pred |S|", lw=1.6)
    if benchmark_available:
        axes[2, 1].plot(y_np, np.abs(shape_ref_y_np), "--", label="ref |S|", lw=1.0)
    axes[2, 1].set_ylabel("|S|")
    axes[2, 1].set_xlabel("y")
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(alpha=0.3)

    rel_err_str = ""
    if benchmark_available and R_ref_r_np is not None:
        rel_err = np.abs(R_pred_r_np - R_ref_r_np) / (np.abs(R_ref_r_np) + 1e-14)
        rel_err_str = f" medRelErr={np.median(rel_err):.2e} maxRelErr={np.max(rel_err):.2e}"

    fig.suptitle(
        f"step={step}, a={args.a:.6f}, omega={args.omega:.6f}, "
        f"u={args.u:.3f}, v={args.v:.3f}\n{benchmark_status}{rel_err_str}",
        fontsize=11
    )
    fig.tight_layout()

    # Output path
    if args.output:
        save_path = Path(args.output)
    else:
        ckpt_dir = ckpt_path.parent
        ckpt_name = ckpt_path.stem
        save_path = ckpt_dir / f"{ckpt_name}_ref.png"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
