#!/usr/bin/env python3
"""Stage-1 near-infinity PDE + spectral consistency refinement.

Only trains RinDecoder. Encoder, AmplitudeNet, Up/Down decoders frozen.

Loss:
  L = w_pde * L_pde + w_cons * L_cons + w_drift * L_drift

  L_pde:   Stage-1 Teukolsky PDE residual on near-infinity y-grid
  L_cons:  |R_model/P - R_combo/P|^2  (spectral consistency)
  L_drift: |R_model/P - R_frozen/P|^2  (prevent catastrophic forgetting)

Usage:
  python scripts/train_stage1_nearinf_consistency_refine.py \
    --config config/autoencoder_stage1_nearinf_consistency_refine.yaml \
    --device cuda --epochs 50 --verbose
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from config.config_loader import load_pinn_full_config
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.mapping import r_plus, r_minus
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives
from physical_ansatz.u_basis import A_up, A_down
from physical_ansatz.transform_y import (
    compose_reduced_shape_from_f, horizon_regularity_slope, h_factor,
    transform_coeffs_x_to_y,
)
from physical_ansatz.teukolsky_coeffs import coeffs_x
from physical_ansatz.stage1_reconstruction import compute_R_over_P_from_model


def Leaver_P(r, a, omega, m=2, M=1.0, s=-2):
    rp = r_plus(a, M)
    rm = r_minus(a, M)
    sigma_p = (2.0 * omega * rp - m * a) / (rp - rm)
    pp = -s - 1j * sigma_p
    pm = -1 - s + 2j * omega + 1j * sigma_p
    return ((r - rp) ** pp) * ((r - rm) ** pm) * torch.exp(1j * omega * r)


def _get_dtype(dtype_name):
    return torch.float32 if dtype_name == "float32" else torch.float64


def compute_f_derivatives_autograd(model, a, omega, y, u=None, v=None):
    """Compute f, f_y, f_yy via autograd."""
    y.requires_grad_(True)
    f = model.forward(a, omega, y, u=u, v=v)
    fy = torch.autograd.grad(f, y, grad_outputs=torch.ones_like(f),
                             create_graph=True, retain_graph=True)[0]
    fyy = torch.autograd.grad(fy, y, grad_outputs=torch.ones_like(fy),
                              create_graph=True, retain_graph=True)[0]
    return f, fy, fyy


def compute_pde_residual(model, a_b, omega_b, y_b, lambda_b, cfg, u_b=None, v_b=None):
    """Compute pointwise PDE residual |R(y)|^2 using the Stage-1 f(y) formulation."""
    M = float(cfg.get("problem", {}).get("M", 1.0))
    s = int(cfg.get("problem", {}).get("s", -2))
    m = int(cfg.get("problem", {}).get("m", 2))

    B = a_b.shape[0]
    N = y_b.shape[1]

    f_out, fy_out, fyy_out = compute_f_derivatives_autograd(
        model, a_b, omega_b, y_b, u=u_b, v=v_b,
    )

    slope = horizon_regularity_slope(
        a=a_b, omega=omega_b, lambda_=lambda_b, m=m, M=M, s=s,
    )

    x_int = (y_b + 1.0) / 2.0
    A2_list, A1_list, A0_list = [], [], []
    for i in range(B):
        A2, A1, A0 = coeffs_x(
            x=x_int, a=a_b[i], omega=omega_b[i], m=m,
            lambda_=lambda_b[i], s=s, M=M,
        )
        A2_list.append(A2)
        A1_list.append(A1)
        A0_list.append(A0)
    A2_int = torch.stack(A2_list, dim=0)
    A1_int = torch.stack(A1_list, dim=0)
    A0_int = torch.stack(A0_list, dim=0)

    B2_int, B1_int, B0_int, rhs = transform_coeffs_x_to_y(
        A2_int, A1_int, A0_int, y_b, slope=slope,
    )
    residual_f = B2_int * fyy_out + B1_int * fy_out + B0_int * f_out - rhs
    pointwise = torch.abs(residual_f) ** 2

    # Normalize
    scale = (1.0 + torch.abs(B2_int.detach()) ** 2 + torch.abs(B1_int.detach()) ** 2 +
             torch.abs(B0_int.detach()) ** 2 + torch.abs(rhs.detach()) ** 2)
    pointwise = pointwise / scale.clamp_min(1e-12)

    return pointwise


def compute_R_combo_over_P(model, a_b, omega_b, y_b, u_b, v_b,
                            u_up_spec, u_down_spec, m=2, M=1.0, s=-2):
    """Compute (B_ref_net * u_up_spec * A_up + B_inc_net * u_down_spec * A_down) / P."""
    with torch.no_grad():
        B_inc, B_ref, _ = model.predict_amplitudes(a_b, omega_b, u_b, v_b)
        rp = r_plus(a_b, M)
        x = (y_b + 1.0) / 2.0
        r = rp / x
        A_u = A_up(r, a_b, omega_b, M)
        A_d = A_down(r, a_b, omega_b, M)
        P = Leaver_P(r, a_b, omega_b, m=m, M=M, s=s)
        # Keep everything complex — A_up, A_down are real, u_up/u_down are complex
        A_u_c = A_u.to(dtype=torch.complex128)
        A_d_c = A_d.to(dtype=torch.complex128)
        P_c = P.to(dtype=torch.complex128)
        B_ref_c = B_ref.unsqueeze(-1).to(dtype=torch.complex128)
        B_inc_c = B_inc.unsqueeze(-1).to(dtype=torch.complex128)
        u_up_c = u_up_spec.to(dtype=torch.complex128, device=r.device)
        u_down_c = u_down_spec.to(dtype=torch.complex128, device=r.device)
        R_combo = (B_ref_c * A_u_c * u_up_c + B_inc_c * A_d_c * u_down_c) / P_c
    return R_combo


def main():
    parser = argparse.ArgumentParser(description="Stage-1 near-infinity consistency refine")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    full_cfg = load_pinn_full_config(args.config)
    runtime_cfg = full_cfg.get("runtime", full_cfg.get("train", {}).get("runtime", {}))
    dtype_name = runtime_cfg.get("dtype", "float64")
    dtype = _get_dtype(dtype_name)
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    physics_cfg = full_cfg.get("physics", {})

    ref_cfg = full_cfg.get("train", {}).get("stage1_refine", {})
    loss_cfg = ref_cfg.get("loss", {})
    opt_cfg = ref_cfg.get("optimizer", {})
    samp_cfg = ref_cfg.get("sampling", {})

    # ---- Checkpoints ----
    stage1_ckpt = Path(ref_cfg["stage1_checkpoint"])
    stage2_ckpt = Path(ref_cfg["stage2_checkpoint"])
    spectral_cache_path = Path(ref_cfg["spectral_basis_cache"])

    for p, name in [(stage1_ckpt, "Stage-1"), (stage2_ckpt, "Stage-2"),
                     (spectral_cache_path, "Spectral cache")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    # ---- Load spectral cache ----
    cache = np.load(spectral_cache_path)
    a_cache = cache["a"]
    omega_cache = cache["omega"]
    u_cache = cache["u"]
    v_cache = cache["v"]
    lam_cache = cache["lambda_"]
    u_up_cache = cache["u_up"]
    u_down_cache = cache["u_down"]
    y_cache_np = cache["y"]
    n_cache = len(a_cache)
    cache_n_y = len(y_cache_np)
    z_b = float(cache["z_b"])
    N_out = int(cache["N_out"])
    print(f"[refine] Spectral cache: {n_cache} params, y=[{y_cache_np[0]:.4f}, {y_cache_np[-1]:.4f}]")

    # ---- Build y-grid: use spectral cache's grid (already near-infinity) ----
    y_grid = torch.tensor(y_cache_np, device=device, dtype=dtype)  # (N_y,)
    n_y = len(y_grid)
    y_t = y_grid.unsqueeze(0)  # (1, N_y)
    # Also add transition points for PDE sampling (interleave)
    n_near = len(y_grid)
    y_trans = torch.linspace(-0.95, 0.0, int(samp_cfg.get("transition", {}).get("n_y", 32)),
                             device=device, dtype=dtype)
    # For PDE: use near-inf grid (where consistency is computed) + transition
    # But for consistency loss, only use near-inf portion
    nearinf_mask = torch.ones(n_y, dtype=torch.bool, device=device)
    print(f"[refine] y-grid: {n_y} cache points + {len(y_trans)} transition, "
          f"cache y=[{y_grid[0]:.4f}, {y_grid[-1]:.4f}]")

    # ---- Build model ----
    model_cfg = full_cfg.get("train", {}).get("model", full_cfg.get("model", {}))
    model = AutoencoderTeukolskyPINN(
        hidden_dims=model_cfg.get("hidden_dims", [128, 128, 128, 128]),
        activation=str(model_cfg.get("activation", "silu")),
        param_embed_dim=int(model_cfg.get("param_embed_dim", 64)),
        fourier_num_freqs=int(model_cfg.get("fourier_num_freqs", 2)),
        fourier_base_scale=float(model_cfg.get("fourier_base_scale", 1.0)),
        use_film=bool(model_cfg.get("use_film", True)),
        use_residual=bool(model_cfg.get("use_residual", True)),
        amp_hidden_dim=int(model_cfg.get("amp_hidden_dim", 128)),
        amp_n_blocks=int(model_cfg.get("amp_n_blocks", 3)),
        decoder_hidden_dim=int(model_cfg.get("decoder_hidden_dim", 128)),
        decoder_n_hidden=int(model_cfg.get("decoder_n_hidden", 0)),
    )

    # Load Stage-1 weights
    ckpt1 = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
    sd1 = ckpt1.get("model_state_dict", ckpt1)
    sd1_f = {k: v for k, v in sd1.items() if not k.startswith("amplitude_net.")}
    missing, unexpected = model.load_state_dict(sd1_f, strict=False)
    print(f"[refine] Stage-1: {len(missing)} missing, {len(unexpected)} unexpected")

    # Load Stage-2 AmplitudeNet weights
    ckpt2 = torch.load(stage2_ckpt, map_location="cpu", weights_only=False)
    sd2 = ckpt2.get("model_state_dict", ckpt2)
    amp_keys = {k: v for k, v in sd2.items() if k.startswith("amplitude_net.")}
    model.load_state_dict(amp_keys, strict=False)
    print(f"[refine] Stage-2 AmplitudeNet: {len(amp_keys)} keys")

    model.to(device=device, dtype=dtype)

    # ---- Freeze policy ----
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.rin_decoder.named_parameters():
        p.requires_grad = True

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[refine] Params: {n_trainable}/{n_total} trainable (RinDecoder only)")

    # ---- Pre-compute frozen R_over_P (for drift loss) ----
    print(f"[refine] Computing frozen R_over_P for {n_cache} points ...")
    frozen_R_over_P = []
    with torch.no_grad():
        for i in range(n_cache):
            a_t = torch.tensor([[float(a_cache[i])]], device=device, dtype=dtype)
            omega_t = torch.tensor([[float(omega_cache[i])]], device=device, dtype=dtype)
            lam_t = torch.tensor([[complex(lam_cache[i])]], device=device, dtype=cdtype)
            u_t = torch.tensor([[float(u_cache[i])]], device=device, dtype=dtype)
            v_t = torch.tensor([[float(v_cache[i])]], device=device, dtype=dtype)
            R_op = compute_R_over_P_from_model(
                model, a_t, omega_t, y_t, lam_t, u_t, v_t,
                m=int(physics_cfg.get("problem", {}).get("m", 2)),
                M=float(physics_cfg.get("problem", {}).get("M", 1.0)),
                s=int(physics_cfg.get("problem", {}).get("s", -2)),
            )
            frozen_R_over_P.append(R_op.squeeze(0))
    print(f"[refine] Frozen R_over_P computed.")

    # ---- Pre-convert cache points to tensors ----
    cache_a = [torch.tensor([[float(a_cache[i])]], device=device, dtype=dtype) for i in range(n_cache)]
    cache_omega = [torch.tensor([[float(omega_cache[i])]], device=device, dtype=dtype) for i in range(n_cache)]
    cache_lam = [torch.tensor([[complex(lam_cache[i])]], device=device, dtype=cdtype) for i in range(n_cache)]
    cache_u = [torch.tensor([[float(u_cache[i])]], device=device, dtype=dtype) for i in range(n_cache)]
    cache_v = [torch.tensor([[float(v_cache[i])]], device=device, dtype=dtype) for i in range(n_cache)]
    cache_up = [torch.tensor(u_up_cache[i], device=device, dtype=cdtype).unsqueeze(0) for i in range(n_cache)]
    cache_down = [torch.tensor(u_down_cache[i], device=device, dtype=cdtype).unsqueeze(0) for i in range(n_cache)]

    # ---- Optimizer ----
    lr = args.lr if args.lr is not None else float(opt_cfg.get("lr_rin_decoder", 1e-6))
    optimizer = torch.optim.Adam(model.rin_decoder.parameters(), lr=lr,
                                  weight_decay=float(opt_cfg.get("weight_decay", 0.0)))
    grad_clip = float(ref_cfg.get("grad_clip", 0.1))

    # ---- Loss config ----
    w_pde = float(loss_cfg.get("weight_pde", 1.0))
    w_cons = float(loss_cfg.get("weight_consistency", 0.1))
    w_drift = float(loss_cfg.get("weight_rin_drift", 10.0))
    eps_loss = float(loss_cfg.get("eps", 1e-12))

    # ---- Output ----
    output_root = ref_cfg.get("output_root", "outputs/autoencoder_stage1_nearinf_consistency_refine")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{timestamp}_stage1_nearinf_refine"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    epochs_max = args.epochs if args.epochs is not None else int(ref_cfg.get("epochs", 500))
    val_every = int(ref_cfg.get("val_every_epochs", 10))
    save_every = int(ref_cfg.get("save_every_epochs", 50))
    train_seed = int(ref_cfg.get("train_seed", 1234))
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)

    # Resume if specified
    start_epoch = 0
    if args.resume_checkpoint:
        resume_ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        start_epoch = resume_ckpt.get("epoch", 0)
        print(f"[refine] Resumed from epoch {start_epoch}")

    model.eval()
    model.rin_decoder.train()

    history = []
    best_loss = float("inf")
    best_epoch = 0

    print(f"\n[refine] Training: {epochs_max} epochs, lr={lr}, batch_size={n_cache}")
    print(f"         w_pde={w_pde}, w_cons={w_cons}, w_drift={w_drift}")
    print(f"         run_dir: {run_dir}")
    print(f"{'='*60}")

    for epoch in range(start_epoch, epochs_max):
        epoch_pde = 0.0
        epoch_cons = 0.0
        epoch_drift = 0.0

        # Process all cache points (full-batch on small dataset)
        for i in range(n_cache):
            a_b = cache_a[i]
            omega_b = cache_omega[i]
            lam_b = cache_lam[i]
            u_b = cache_u[i]
            v_b = cache_v[i]
            u_up_b = cache_up[i]
            u_down_b = cache_down[i]
            frozen_R = frozen_R_over_P[i].unsqueeze(0)

            # 1. PDE residual
            pde_res = compute_pde_residual(
                model, a_b, omega_b, y_t, lam_b, physics_cfg, u_b=u_b, v_b=v_b,
            )
            L_pde = pde_res.mean()

            # 2. R_model/P (current)
            R_model = compute_R_over_P_from_model(
                model, a_b, omega_b, y_t, lam_b, u_b, v_b,
                m=int(physics_cfg.get("problem", {}).get("m", 2)),
                M=float(physics_cfg.get("problem", {}).get("M", 1.0)),
                s=int(physics_cfg.get("problem", {}).get("s", -2)),
            )

            # 3. R_combo/P (spectral consistency target)
            R_combo = compute_R_combo_over_P(
                model, a_b, omega_b, y_t, u_b, v_b, u_up_b, u_down_b,
                m=int(physics_cfg.get("problem", {}).get("m", 2)),
                M=float(physics_cfg.get("problem", {}).get("M", 1.0)),
                s=int(physics_cfg.get("problem", {}).get("s", -2)),
            )

            # Consistency loss
            L_cons = torch.mean(torch.abs(R_model - R_combo) ** 2 /
                               (torch.abs(R_model) ** 2 + eps_loss))

            # Drift loss
            L_drift = torch.mean(torch.abs(R_model - frozen_R) ** 2 /
                                (torch.abs(frozen_R) ** 2 + eps_loss))

            loss = w_pde * L_pde + w_cons * L_cons + w_drift * L_drift

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.rin_decoder.parameters(), grad_clip)
            optimizer.step()

            epoch_pde += L_pde.detach().item()
            epoch_cons += L_cons.detach().item()
            epoch_drift += L_drift.detach().item()

        avg_pde = epoch_pde / n_cache
        avg_cons = epoch_cons / n_cache
        avg_drift = epoch_drift / n_cache
        total_loss = w_pde * avg_pde + w_cons * avg_cons + w_drift * avg_drift

        if (epoch + 1) % val_every == 0 or epoch == 0:
            history.append({
                "epoch": epoch + 1,
                "L_pde": avg_pde,
                "L_cons": avg_cons,
                "L_drift": avg_drift,
                "total_loss": total_loss,
                "lr": optimizer.param_groups[0]["lr"],
            })

            if args.verbose or (epoch + 1) % val_every == 0:
                marker = ""
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_epoch = epoch + 1
                    marker = " [BEST]"
                print(f"  epoch {epoch+1:5d}/{epochs_max} | "
                      f"pde={avg_pde:.6e} cons={avg_cons:.6e} drift={avg_drift:.6e} "
                      f"total={total_loss:.6e}{marker}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == epochs_max - 1:
            ckpt_data = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "config": str(Path(args.config).resolve()),
                "stage1_checkpoint": str(stage1_ckpt.resolve()),
                "stage2_checkpoint": str(stage2_ckpt.resolve()),
            }
            torch.save(ckpt_data, ckpt_dir / "latest_model.pt")

    # ---- Final ----
    print(f"\n{'='*60}")
    print(f"[refine] Training complete. run_dir: {run_dir}")
    if history:
        last = history[-1]
        print(f"         final: pde={last['L_pde']:.6e} cons={last['L_cons']:.6e} "
              f"drift={last['L_drift']:.6e}")
        print(f"         best_epoch={best_epoch} best_total={best_loss:.6e}")

    summary = {
        "run_dir": str(run_dir),
        "stage1_checkpoint": str(stage1_ckpt.resolve()),
        "stage2_checkpoint": str(stage2_ckpt.resolve()),
        "n_cache": int(n_cache),
        "n_y": int(n_y),
        "epochs_total": len(history),
        "history": history,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
