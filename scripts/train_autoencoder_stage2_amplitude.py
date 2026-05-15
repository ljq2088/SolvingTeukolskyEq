#!/usr/bin/env python3
"""
Stage-2 autoencoder training: AmplitudeNet regression against teacher.

Loads a Stage-1 checkpoint, freezes encoder + all decoders, and trains
only AmplitudeNet to match B_inc, B_ref from the amplitude teacher.

Loss (方案B):
  loss = log-mag MSE + normalised-phase MSE
       = MSE(log|B_pred|, log|B_true|)
       + MSE(Re(B_pred/|B_pred|), Re(B_true/|B_true|))
       + MSE(Im(B_pred/|B_pred|), Im(B_true/|B_true|))

Usage:
  python scripts/train_autoencoder_stage2_amplitude.py \\
    --config config/autoencoder_stage2_amplitude.yaml \\
    --checkpoint outputs/stage1_artifacts/patch_000_logw_v2/stage1_best_model.pt \\
    --device cuda \\
    --epochs 100
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
import yaml

from config.config_loader import load_pinn_full_config
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from amplitude_teacher import AmplitudeTeacher


# ============================================================
# 方案B loss
# ============================================================
def amplitude_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8,
                   weight_logmag: float = 1.0, weight_phase: float = 1.0):
    """方案B amplitude loss — log-magnitude + normalised phase.

    Args:
        pred: (B,) complex tensor
        target: (B,) complex tensor
        eps: small value to avoid log(0)
        weight_logmag: weight for log-magnitude MSE
        weight_phase: weight for normalised-phase MSE

    Returns:
        scalar loss
    """
    logmag_pred = torch.log(torch.abs(pred) + eps)
    logmag_target = torch.log(torch.abs(target) + eps)
    loss_logmag = torch.nn.functional.mse_loss(logmag_pred, logmag_target)

    norm_pred = pred / (torch.abs(pred) + eps)
    norm_target = target / (torch.abs(target) + eps)
    loss_phase_real = torch.nn.functional.mse_loss(norm_pred.real, norm_target.real)
    loss_phase_imag = torch.nn.functional.mse_loss(norm_pred.imag, norm_target.imag)

    return weight_logmag * loss_logmag + weight_phase * (loss_phase_real + loss_phase_imag)


def _get_dtype(dtype_name):
    return torch.float32 if dtype_name == "float32" else torch.float64


def main():
    parser = argparse.ArgumentParser(description="Stage-2 AmplitudeNet training")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Stage-1 checkpoint path (stage1_best_model.pt)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Load config ----
    full_cfg = load_pinn_full_config(args.config)
    runtime_cfg = full_cfg.get("runtime", {})
    dtype_name = runtime_cfg.get("dtype", "float64")
    dtype = _get_dtype(dtype_name)
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    stage2_cfg = full_cfg.get("stage2", {})
    teacher_dir = Path(stage2_cfg.get("teacher_dir", "outputs/stage1_artifacts/patch_000_logw_v2"))
    loss_cfg = stage2_cfg.get("loss", {})
    opt_cfg = stage2_cfg.get("optimizer", {})
    sched_cfg = stage2_cfg.get("scheduler", {})
    train_frac = float(stage2_cfg.get("train_fraction", 0.8))
    batch_size = int(stage2_cfg.get("batch_size", 32))
    epochs_max = args.epochs if args.epochs is not None else int(stage2_cfg.get("epochs", 5000))
    val_every = int(stage2_cfg.get("val_every_epochs", 50))
    es_cfg = stage2_cfg.get("early_stopping", {})
    output_root = stage2_cfg.get("output_root", "outputs/autoencoder_stage2_amplitude_train")
    save_every = int(stage2_cfg.get("save_every_epochs", 500))
    train_seed = int(stage2_cfg.get("train_seed", 1234))

    torch.manual_seed(train_seed)
    np.random.seed(train_seed)

    # ---- Load checkpoint ----
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # ---- Build model ----
    full_cfg_raw = ckpt.get("full_cfg", full_cfg)
    model_cfg = full_cfg_raw.get("model", full_cfg.get("model", {}))
    ckpt_step = ckpt.get("step", None)

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

    # ---- Stage-2 freeze: only AmplitudeNet trainable ----
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.amplitude_net.named_parameters():
        p.requires_grad = True

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[stage2] Model loaded from {ckpt_path}")
    print(f"         Stage-1 step: {ckpt_step}")
    print(f"         Params: {n_trainable}/{n_total} trainable (amplitude_net only)")

    # ---- Load teacher ----
    teacher_path = teacher_dir / "amplitude_teacher.npz"
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher not found: {teacher_path}")
    teacher = AmplitudeTeacher.load(teacher_path)
    n_teacher = len(teacher.a_vals)

    # Filter out NaN
    valid = np.isfinite(teacher.B_inc) & np.isfinite(teacher.B_ref)
    a_vals = teacher.a_vals[valid]
    omega_vals = teacher.omega_vals[valid]
    B_inc_true = teacher.B_inc[valid]
    B_ref_true = teacher.B_ref[valid]
    n_valid = len(a_vals)
    print(f"[stage2] Teacher loaded: {n_valid}/{n_teacher} valid samples")
    print(f"         B_inc mag: [{np.min(np.abs(B_inc_true)):.4e}, {np.max(np.abs(B_inc_true)):.4e}]")
    print(f"         B_ref mag: [{np.min(np.abs(B_ref_true)):.4e}, {np.max(np.abs(B_ref_true)):.4e}]")

    # ---- Train/val split ----
    idx = np.random.permutation(n_valid)
    n_train = int(n_valid * train_frac)
    idx_train = idx[:n_train]
    idx_val = idx[n_train:]

    a_train = torch.tensor(a_vals[idx_train], device=device, dtype=dtype)
    omega_train = torch.tensor(omega_vals[idx_train], device=device, dtype=dtype)
    B_inc_train = torch.tensor(B_inc_true[idx_train], device=device, dtype=cdtype)
    B_ref_train = torch.tensor(B_ref_true[idx_train], device=device, dtype=cdtype)

    if len(idx_val) > 0:
        a_val = torch.tensor(a_vals[idx_val], device=device, dtype=dtype)
        omega_val = torch.tensor(omega_vals[idx_val], device=device, dtype=dtype)
        B_inc_val = torch.tensor(B_inc_true[idx_val], device=device, dtype=cdtype)
        B_ref_val = torch.tensor(B_ref_true[idx_val], device=device, dtype=cdtype)
    else:
        a_val = omega_val = B_inc_val = B_ref_val = None

    # ---- Optimizer ----
    lr = args.lr if args.lr is not None else float(opt_cfg.get("lr", 1e-4))
    wd = float(opt_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.Adam(model.amplitude_net.parameters(), lr=lr, weight_decay=wd)

    # ---- Scheduler ----
    scheduler = None
    if sched_cfg.get("enabled", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_cfg.get("mode", "min"),
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 200)),
            min_lr=float(sched_cfg.get("min_lr", 1e-6)),
        )

    # ---- Output ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{timestamp}_stage2_amplitude"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # ---- Training ----
    loss_cfg_dict = {
        "weight_logmag": float(loss_cfg.get("weight_logmag", 1.0)),
        "weight_phase": float(loss_cfg.get("weight_phase", 1.0)),
        "eps": float(loss_cfg.get("eps", 1e-8)),
    }

    es_patience = int(es_cfg.get("patience", 500))
    es_min_delta = float(es_cfg.get("min_delta", 1e-7))

    best_val = float("inf")
    best_epoch = 0
    steps_since_best = 0
    is_best = False
    history = []

    model.eval()  # encoders/decoders are frozen anyway
    model.amplitude_net.train()

    print(f"\n[stage2] Training: {epochs_max} epochs, batch_size={batch_size}, lr={lr}")
    print(f"         run_dir: {run_dir}")
    print(f"{'='*60}")

    for epoch in range(epochs_max):
        # Shuffle
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for b_start in range(0, n_train, batch_size):
            b_idx = perm[b_start:b_start + batch_size]
            a_b = a_train[b_idx].unsqueeze(-1)
            omega_b = omega_train[b_idx].unsqueeze(-1)

            B_inc_pred, B_ref_pred, _ = model.predict_amplitudes(a_b, omega_b)
            B_inc_t = B_inc_train[b_idx]
            B_ref_t = B_ref_train[b_idx]

            loss_inc = amplitude_loss(B_inc_pred, B_inc_t, **loss_cfg_dict)
            loss_ref = amplitude_loss(B_ref_pred, B_ref_t, **loss_cfg_dict)
            loss = loss_inc + loss_ref

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # ---- Validation ----
        if a_val is not None and (epoch == 0 or (epoch + 1) % val_every == 0):
            with torch.no_grad():
                B_inc_pred_v, B_ref_pred_v, _ = model.predict_amplitudes(
                    a_val.unsqueeze(-1), omega_val.unsqueeze(-1),
                )
                val_loss_inc = amplitude_loss(B_inc_pred_v, B_inc_val, **loss_cfg_dict)
                val_loss_ref = amplitude_loss(B_ref_pred_v, B_ref_val, **loss_cfg_dict)
                val_loss = (val_loss_inc + val_loss_ref).item()

                # Also compute relative errors on magnitudes
                rel_err_inc = torch.abs(torch.abs(B_inc_pred_v) - torch.abs(B_inc_val)) / (
                    torch.abs(B_inc_val) + 1e-8)
                rel_err_ref = torch.abs(torch.abs(B_ref_pred_v) - torch.abs(B_ref_val)) / (
                    torch.abs(B_ref_val) + 1e-8)

            history.append({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "val_median_rel_err_inc": float(torch.median(rel_err_inc).item()),
                "val_median_rel_err_ref": float(torch.median(rel_err_ref).item()),
                "lr": optimizer.param_groups[0]["lr"],
            })

            is_best = val_loss < best_val - es_min_delta
            status = ""
            if is_best:
                best_val = val_loss
                best_epoch = epoch + 1
                steps_since_best = 0
                status = " [BEST]"
            else:
                steps_since_best += 1

            if args.verbose or (epoch + 1) % val_every == 0:
                print(f"  epoch {epoch+1:5d}/{epochs_max} | train_loss={avg_loss:.6e} "
                      f"val_loss={val_loss:.6e} | "
                      f"rel_inc={torch.median(rel_err_inc).item():.4e} "
                      f"rel_ref={torch.median(rel_err_ref).item():.4e}{status}")

            if scheduler is not None:
                scheduler.step(val_loss)

            # Early stop
            if steps_since_best >= es_patience:
                print(f"[stage2] Early stopping at epoch {epoch+1} (best={best_epoch}, val={best_val:.6e})")
                break

        # ---- Save checkpoints ----
        if (epoch + 1) % save_every == 0 or (epoch == epochs_max - 1) or is_best:
            ckpt_data = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "history": history,
                "config": str(Path(args.config).resolve()),
                "stage1_checkpoint": str(ckpt_path.resolve()),
                "loss_cfg": loss_cfg_dict,
            }
            torch.save(ckpt_data, ckpt_dir / "latest_model.pt")

        if is_best:
            torch.save(ckpt_data, ckpt_dir / "best_model.pt")

    # ---- Summary ----
    final_val = history[-1]["val_loss"] if history else float("nan")
    print(f"\n{'='*60}")
    print(f"[stage2] Training complete.")
    print(f"         run_dir: {run_dir}")
    print(f"         best_epoch: {best_epoch}, best_val_loss: {best_val:.6e}")
    print(f"         final_val_loss: {final_val:.6e}")
    if history:
        last = history[-1]
        print(f"         final rel_err_inc: {last['val_median_rel_err_inc']:.4e}")
        print(f"         final rel_err_ref: {last['val_median_rel_err_ref']:.4e}")

    # Save summary
    summary = {
        "run_dir": str(run_dir),
        "stage1_checkpoint": str(ckpt_path.resolve()),
        "stage1_step": int(ckpt_step) if ckpt_step is not None else None,
        "n_train": int(n_train),
        "n_val": len(idx_val),
        "epochs_total": len(history),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "final_val_loss": float(final_val),
        "history": history,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
