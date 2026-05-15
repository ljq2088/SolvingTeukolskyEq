#!/usr/bin/env python3
"""
Stage-3 autoencoder training: u_up / u_down decoder training via PDE residual.

Loads a Stage-2 best checkpoint, freezes encoder + rin_decoder + amplitude_net,
and trains only up_decoder and down_decoder to satisfy the Teukolsky ODE.

Loss:
    L_up   = mean |TeukResidual[R_up]|^2
    L_down = mean |TeukResidual[R_down]|^2
    where R_up = A_up * u_up, R_down = A_down * u_down

Usage:
  python scripts/train_autoencoder_stage3_ubasis.py \\
    --config config/autoencoder_stage3_ubasis.yaml \\
    --checkpoint outputs/autoencoder_stage2_amplitude_train/.../best_model.pt \\
    --device cuda \\
    --epochs 500 \\
    --verbose
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from config.config_loader import load_pinn_full_config
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.u_residual import (
    compute_stage3_loss_u_equation,
    compute_stage3_loss,
)


def _get_dtype(dtype_name):
    return torch.float32 if dtype_name == "float32" else torch.float64


def _build_y_grid(n_interior, n_near_inf, y_inf_min, y_inf_max, device, dtype,
                  y_eps=1e-3, y_min=-1.0, y_max=1.0, y_strategy="chebyshev"):
    """Build y-grid for PDE collocation.

    y_strategy:
        "chebyshev":       full-domain Chebyshev-Gauss-Lobatto grid, clipped to [y_min+y_eps, y_max-y_eps]
        "near_infinity_only": train only in [y_min, y_max], typically [-1, 0]
    """
    if y_strategy == "near_infinity_only":
        y_all = torch.linspace(y_min + y_eps, y_max - y_eps, n_interior,
                               device=device, dtype=dtype)
    else:
        k = torch.arange(n_interior, device=device, dtype=dtype)
        y_cheb = -torch.cos(torch.pi * k / (n_interior - 1))
        y_cheb = y_cheb.clamp(-1.0 + y_eps, 1.0 - y_eps)
        y_all = y_cheb

    if n_near_inf > 0:
        y_extra = torch.linspace(y_inf_min, y_inf_max, n_near_inf,
                                  device=device, dtype=dtype)
        y_extra = y_extra.clamp(-1.0 + y_eps, 1.0 - y_eps)
        y_all = torch.cat([y_all, y_extra])

    return y_all  # (N_total,)


def _load_artifact(artifact_dir):
    """Load parameter pool from rpred_cache."""
    data = np.load(Path(artifact_dir) / "rpred_cache.npz")
    valid = np.isfinite(data["a"]) & np.isfinite(data["omega"]) & np.isfinite(data["lambda_"])
    return {
        "a": data["a"][valid],
        "omega": data["omega"][valid],
        "u": data["u"][valid],
        "v": data["v"][valid],
        "lambda_": data["lambda_"][valid],
    }


def _grad_norm(module):
    """Total L2 gradient norm over all parameters of a module."""
    sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            sq += p.grad.detach().norm().item() ** 2
    return sq ** 0.5


def _sample_batch(pool, batch_size, device, dtype, cdtype):
    """Sample random batch from parameter pool."""
    n = len(pool["a"])
    idx = np.random.choice(n, min(batch_size, n), replace=False)
    a = torch.tensor(pool["a"][idx], device=device, dtype=dtype)
    omega = torch.tensor(pool["omega"][idx], device=device, dtype=dtype)
    u = torch.tensor(pool["u"][idx], device=device, dtype=dtype)
    v = torch.tensor(pool["v"][idx], device=device, dtype=dtype)
    lam = torch.tensor(pool["lambda_"][idx], device=device, dtype=cdtype)
    return a, omega, u, v, lam


def main():
    parser = argparse.ArgumentParser(description="Stage-3 u-basis decoder training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Stage-2 best checkpoint path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Load config ----
    full_cfg = load_pinn_full_config(args.config)
    runtime_cfg = full_cfg.get("runtime", {})
    dtype_name = runtime_cfg.get("dtype", "float64")
    dtype = _get_dtype(dtype_name)
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    stage3_cfg = full_cfg.get("train", {}).get("stage3", full_cfg.get("stage3", {}))
    loss_cfg = stage3_cfg.get("loss", {})
    opt_cfg = stage3_cfg.get("optimizer", {})
    sched_cfg = stage3_cfg.get("scheduler", {})
    samp_cfg = stage3_cfg.get("sampling", {})
    batch_size = int(stage3_cfg.get("batch_size", 8))
    n_interior = int(stage3_cfg.get("n_interior", 128))
    epochs_max = args.epochs if args.epochs is not None else int(stage3_cfg.get("epochs", 5000))
    val_every = int(stage3_cfg.get("val_every_epochs", 50))
    output_root = stage3_cfg.get("output_root", "outputs/autoencoder_stage3_ubasis_train")
    save_every = int(stage3_cfg.get("save_every_epochs", 500))
    grad_clip = float(stage3_cfg.get("grad_clip", 1.0))
    artifact_dir = stage3_cfg.get("artifact_dir",
                                    "outputs/stage1_artifacts/patch_000_logw_v2")

    # Physics
    physics = full_cfg.get("physics", full_cfg)
    prob = physics.get("problem", {})
    M = float(prob.get("M", 1.0))
    s = int(prob.get("s", -2))
    l = int(prob.get("l", 2))
    m = int(prob.get("m", 2))

    # ---- Load checkpoint ----
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # ---- Build model ----
    full_cfg_raw = ckpt.get("full_cfg", full_cfg)
    model_cfg = full_cfg_raw.get("model", full_cfg.get("model", {}))
    ckpt_epoch = ckpt.get("epoch", None)

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
        **model_cfg.get("encoder_kwargs", {}),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device=device, dtype=dtype)

    # ---- Stage-3 freeze: only up_decoder + down_decoder trainable ----
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.up_decoder.named_parameters():
        p.requires_grad = True
    for _, p in model.down_decoder.named_parameters():
        p.requires_grad = True

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_up = sum(p.numel() for p in model.up_decoder.parameters())
    n_down = sum(p.numel() for p in model.down_decoder.parameters())
    print(f"[stage3] Model loaded from {ckpt_path}")
    print(f"         Source: {'Stage-3 continuation' if ckpt.get('history') else 'Stage-2 init'}, epoch: {ckpt_epoch}")
    print(f"         Params: {n_trainable}/{n_total} trainable")
    print(f"         up_decoder: {n_up}, down_decoder: {n_down}")

    # ---- Load parameter pool ----
    pool = _load_artifact(artifact_dir)
    n_pool = len(pool["a"])
    print(f"[stage3] Parameter pool: {n_pool} samples")
    print(f"         a: [{pool['a'].min():.4f}, {pool['a'].max():.4f}]")
    print(f"         omega: [{pool['omega'].min():.6e}, {pool['omega'].max():.6e}]")

    # ---- Build y grid ----
    n_near_inf = int(samp_cfg.get("n_near_infinity", 0))
    y_inf_min = float(samp_cfg.get("y_inf_min", -1.0))
    y_inf_max = float(samp_cfg.get("y_inf_max", -0.95))
    y_eps = float(samp_cfg.get("y_eps", 1e-3))
    y_grid_min = float(samp_cfg.get("y_min", -1.0))
    y_grid_max = float(samp_cfg.get("y_max", 1.0))
    y_strategy = str(samp_cfg.get("y_strategy", "chebyshev"))
    y_grid = _build_y_grid(n_interior, n_near_inf, y_inf_min, y_inf_max,
                            device, dtype, y_eps=y_eps,
                            y_min=y_grid_min, y_max=y_grid_max,
                            y_strategy=y_strategy)  # (N,)
    n_y = len(y_grid)
    print(f"[stage3] y-grid: {n_y} points ({n_interior} Cheb + {n_near_inf} near-inf), y_eps={y_eps}")

    # ---- Optimizer ----
    lr = args.lr if args.lr is not None else float(opt_cfg.get("lr", 1e-4))
    wd = float(opt_cfg.get("weight_decay", 0.0))
    opt_params = list(model.up_decoder.parameters()) + list(model.down_decoder.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=lr, weight_decay=wd)

    # ---- Scheduler ----
    scheduler = None
    if sched_cfg.get("enabled", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 300)),
            min_lr=float(sched_cfg.get("min_lr", 1e-6)),
        )

    # ---- Output ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{timestamp}_stage3_ubasis"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # ---- Training ----
    residual_mode = str(stage3_cfg.get("residual_mode", "u_equation"))
    weight_up = float(loss_cfg.get("weight_up", 1.0))
    weight_down = float(loss_cfg.get("weight_down", 1.0))
    normalize_res = bool(loss_cfg.get("normalize_residual", False))
    eps = float(loss_cfg.get("eps", 1e-12))

    if residual_mode == "u_equation":
        loss_fn = compute_stage3_loss_u_equation
        loss_kwargs = dict(weight_up=weight_up, weight_down=weight_down)
    else:
        loss_fn = compute_stage3_loss
        loss_kwargs = dict(weight_up=weight_up, weight_down=weight_down,
                           normalize_residual=normalize_res, eps=eps)

    best_val = float("inf")
    best_epoch = 0
    history = []

    model.eval()  # encoder/rin_decoder frozen, up/down in eval for BN but we need train mode
    model.up_decoder.train()
    model.down_decoder.train()

    print(f"\n[stage3] Training: {epochs_max} epochs, batch_size={batch_size}, lr={lr}")
    print(f"         residual_mode: {residual_mode}, y_eps={y_eps}, grad_clip={grad_clip}")
    print(f"         run_dir: {run_dir}")
    print(f"{'='*60}")

    for epoch in range(epochs_max):
        # Sample batch and expand y grid
        a_b, omega_b, u_b, v_b, lam_b = _sample_batch(
            pool, batch_size, device, dtype, cdtype)
        B_actual = a_b.shape[0]
        y_b = y_grid.unsqueeze(0).expand(B_actual, -1)  # (B, N_y)

        # Compute loss
        total_loss, info = loss_fn(
            model, a_b, omega_b, y_b, lam_b, u_b, v_b,
            M=M, s=s, m=m, **loss_kwargs,
        )

        optimizer.zero_grad()
        total_loss.backward()

        # ---- Gradient norms before clipping ----
        gn_up = _grad_norm(model.up_decoder)
        gn_down = _grad_norm(model.down_decoder)
        gn_total = (gn_up ** 2 + gn_down ** 2) ** 0.5

        # ---- Gradient clipping ----
        torch.nn.utils.clip_grad_norm_(opt_params, grad_clip)

        optimizer.step()

        train_loss = total_loss.detach().item()

        # ---- NaN/Inf detection on loss components ----
        if not (np.isfinite(train_loss) and np.isfinite(info.get("loss_up", 0))
                and np.isfinite(info.get("loss_down", 0))):
            print(f"[stage3] NaN/Inf at epoch {epoch+1}: "
                  f"train={train_loss}, up={info.get('loss_up')}, down={info.get('loss_down')}")
            # Save debug checkpoint
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "error": "NaN/Inf detected",
                "info": info,
            }, ckpt_dir / "nan_debug.pt")
            break

        # ---- Validation (on same grid, different param samples) ----
        if epoch == 0 or (epoch + 1) % val_every == 0:
            a_v, omega_v, u_v, v_v, lam_v = _sample_batch(
                pool, batch_size, device, dtype, cdtype)
            Bv = a_v.shape[0]
            y_v = y_grid.unsqueeze(0).expand(Bv, -1)
            val_loss, val_info = loss_fn(
                model, a_v, omega_v, y_v, lam_v, u_v, v_v,
                M=M, s=s, m=m, **loss_kwargs,
            )

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_info["total_loss"],
                "val_loss_up": val_info["loss_up"],
                "val_loss_down": val_info["loss_down"],
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm_up": float(gn_up),
                "grad_norm_down": float(gn_down),
                "grad_norm_total": float(gn_total),
                "y_eps": y_eps,
                "residual_mode": residual_mode,
            })

            val_float = history[-1]["val_loss"]
            is_best = val_float < best_val
            status = ""
            if is_best:
                best_val = val_float
                best_epoch = epoch + 1
                status = " [BEST]"

            if args.verbose or (epoch + 1) % val_every == 0:
                print(f"  epoch {epoch+1:5d}/{epochs_max} | "
                      f"train={train_loss:.6e} val={val_float:.6e} | "
                      f"up={val_info['loss_up']:.4e} down={val_info['loss_down']:.4e} | "
                      f"gn_up={gn_up:.1f} gn_down={gn_down:.1f}{status}")

            if scheduler is not None:
                scheduler.step(val_float)

        # ---- Save checkpoints ----
        if (epoch + 1) % save_every == 0 or (epoch == epochs_max - 1):
            ckpt_data = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "history": history,
                "config": str(Path(args.config).resolve()),
                "stage2_checkpoint": str(ckpt_path.resolve()),
            }
            torch.save(ckpt_data, ckpt_dir / "latest_model.pt")

        if is_best:
            ckpt_data = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "history": history,
                "config": str(Path(args.config).resolve()),
                "stage2_checkpoint": str(ckpt_path.resolve()),
            }
            torch.save(ckpt_data, ckpt_dir / "best_model.pt")

    # ---- Summary ----
    final_val = history[-1]["val_loss"] if history else float("nan")
    print(f"\n{'='*60}")
    print(f"[stage3] Training complete.")
    print(f"         run_dir: {run_dir}")
    print(f"         best_epoch: {best_epoch}, best_val_loss: {best_val:.6e}")
    print(f"         final_val_loss: {final_val:.6e}")
    if history:
        last = history[-1]
        print(f"         final loss_up: {last['val_loss_up']:.4e}")
        print(f"         final loss_down: {last['val_loss_down']:.4e}")

    summary = {
        "run_dir": str(run_dir),
        "stage2_checkpoint": str(ckpt_path.resolve()),
        "n_pool": int(n_pool),
        "batch_size": int(batch_size),
        "n_interior": int(n_interior),
        "epochs_total": len(history),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "final_val_loss": float(final_val),
        "y_eps": y_eps,
        "residual_mode": residual_mode,
        "grad_clip": grad_clip,
        "history": history,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
