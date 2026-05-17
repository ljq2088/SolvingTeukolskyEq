#!/usr/bin/env python3
"""Stage-4: Spectral-basis consistency polishing with drift regularization.

Freezes encoder and up/down decoders. Only RinDecoder and AmplitudeNet
are fine-tuned at very low LR. Uses FIXED spectral u_up/u_down basis
(not learned Stage-3 decoders).

Usage:
  # 20-epoch smoke test
  python scripts/train_autoencoder_stage4_spectral_consistency.py \
    --config config/autoencoder_stage4_spectral_consistency.yaml \
    --device cuda --epochs 20 --verbose

  # 200-epoch continuation
  python scripts/train_autoencoder_stage4_spectral_consistency.py \
    --config config/autoencoder_stage4_spectral_consistency.yaml \
    --device cuda --epochs 200 \
    --resume-checkpoint <best_checkpoint.pt> --verbose
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from config.config_loader import load_pinn_full_config
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.mapping import r_plus, r_minus
from physical_ansatz.u_basis import A_up, A_down
from physical_ansatz.stage1_reconstruction import compute_R_over_P_from_model


def _get_dtype(dtype_name):
    return torch.float32 if dtype_name == "float32" else torch.float64


def Leaver_P(r, a, omega, m=2, M=1.0, s=-2):
    rp = r_plus(a, M)
    rm = r_minus(a, M)
    sigma_p = (2.0 * omega * rp - m * a) / (rp - rm)
    pp = -s - 1j * sigma_p
    pm = -1 - s + 2j * omega + 1j * sigma_p
    return ((r - rp) ** pp) * ((r - rm) ** pm) * torch.exp(1j * omega * r)


def compute_stage4_loss(model, a, omega, y, u, v, lambda_, stage1_model,
                        weight_rin_drift=50.0, weight_amp_drift=50.0, eps=1e-12,
                        M=1.0, s=-2, m=2, u_up_spec=None, u_down_spec=None):
    """Compute Stage-4 v2 consistency + drift losses.

    v2 fix: R_over_P = h2 * S(y) via compute_R_over_P_from_model, NOT f_R.
    Parameters sampled from spectral cache (exact u_up/u_down match).
    """
    B = a.shape[0]
    N = y.shape[1]
    device = a.device
    dtype = a.dtype
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    # ---- R_model / P from RinDecoder (CORRECTED) ----
    R_over_P = compute_R_over_P_from_model(
        model, a, omega, y, lambda_, u, v, m=m, M=M, s=s,
    )

    # ---- B_inc, B_ref from AmplitudeNet ----
    B_inc, B_ref, _ = model.predict_amplitudes(a, omega, u, v)

    # ---- r from y for A_up, A_down, P ----
    rp = r_plus(a, M)
    x = (y + 1.0) / 2.0
    r = rp / x

    A_u = A_up(r, a, omega, M)
    A_d = A_down(r, a, omega, M)
    P = Leaver_P(r, a, omega, m=m, M=M, s=s)

    # ---- R_combo / P ----
    B_ref_exp = B_ref.unsqueeze(-1)
    B_inc_exp = B_inc.unsqueeze(-1)
    R_combo_over_P = (B_ref_exp * A_u * u_up_spec + B_inc_exp * A_d * u_down_spec) / P

    # ---- L_cons ----
    diff = R_over_P - R_combo_over_P
    L_cons = torch.mean(torch.abs(diff) ** 2) / (torch.mean(torch.abs(R_over_P) ** 2) + eps)

    # ---- L_rin_drift: CORRECTED R_over_P vs frozen Stage-1 ----
    with torch.no_grad():
        R_stage1_over_P = compute_R_over_P_from_model(
            stage1_model, a, omega, y, lambda_, u, v, m=m, M=M, s=s,
        )
    L_rin_drift = torch.mean(torch.abs(R_over_P - R_stage1_over_P) ** 2) / \
                  (torch.mean(torch.abs(R_stage1_over_P) ** 2) + eps)

    # ---- L_amp_drift: B_inc/B_ref vs frozen Stage-2 ----
    with torch.no_grad():
        B_inc_s2, B_ref_s2, _ = stage1_model.predict_amplitudes(a, omega, u, v)
    L_amp_drift_inc = torch.mean(torch.abs(B_inc - B_inc_s2) ** 2) / \
                      (torch.mean(torch.abs(B_inc_s2) ** 2) + eps)
    L_amp_drift_ref = torch.mean(torch.abs(B_ref - B_ref_s2) ** 2) / \
                      (torch.mean(torch.abs(B_ref_s2) ** 2) + eps)
    L_amp_drift = L_amp_drift_inc + L_amp_drift_ref

    # ---- Total ----
    total_loss = L_cons + weight_rin_drift * L_rin_drift + weight_amp_drift * L_amp_drift

    # ---- Metrics ----
    rel_err = torch.abs(R_over_P - R_combo_over_P) / (torch.abs(R_over_P) + eps)
    med_rel_err = torch.median(rel_err)
    max_rel_err = torch.max(rel_err)

    info = {
        "L_cons": float(L_cons.detach()),
        "L_rin_drift": float(L_rin_drift.detach()),
        "L_amp_drift": float(L_amp_drift.detach()),
        "total_loss": float(total_loss.detach()),
        "median_rel_err": float(med_rel_err.detach()),
        "max_rel_err": float(max_rel_err.detach()),
        "B_inc_drift": float(torch.mean(torch.abs(B_inc - B_inc_s2)).detach()),
        "B_ref_drift": float(torch.mean(torch.abs(B_ref - B_ref_s2)).detach()),
    }
    return total_loss, info


def main():
    parser = argparse.ArgumentParser(description="Stage-4 spectral consistency training")
    parser.add_argument("--config", type=str, default="config/autoencoder_stage4_spectral_consistency.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--rollback-to-pre-stage4", action="store_true",
                        help="Print rollback path and exit")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[stage4-train] Device: {device}")

    # ---- Load config ----
    full_cfg = load_pinn_full_config(args.config)
    train_cfg = full_cfg.get("train", {})
    runtime_cfg = train_cfg.get("runtime", {})
    dtype_name = runtime_cfg.get("dtype", "float64")
    dtype = _get_dtype(dtype_name)
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    stage4_cfg = train_cfg.get("stage4", full_cfg.get("stage4", {}))
    loss_cfg = stage4_cfg.get("loss", {})
    opt_cfg = stage4_cfg.get("optimizer", {})
    y_cfg = stage4_cfg.get("y_region", {})
    M_phys = float(full_cfg.get("physics", {}).get("problem", {}).get("M", 1.0))
    s_phys = int(full_cfg.get("physics", {}).get("problem", {}).get("s", -2))
    m_phys = int(full_cfg.get("physics", {}).get("problem", {}).get("m", 2))

    # ---- Rollback mode ----
    bundle_dir = Path(stage4_cfg["pre_stage4_bundle"])
    if args.rollback_to_pre_stage4:
        print(f"[stage4-train] Rollback bundle: {bundle_dir}")
        print(f"  stage1_best_model.pt, stage2_best_model.pt")
        sys.exit(0)

    # ---- Load spectral basis cache ----
    cache_path = Path(stage4_cfg["spectral_basis_cache"])
    cache = np.load(cache_path)
    a_cache = cache["a"]
    omega_cache = cache["omega"]
    u_cache = cache["u"]
    v_cache = cache["v"]
    lam_cache = cache["lambda_"]
    u_up_cache = cache["u_up"]
    u_down_cache = cache["u_down"]
    y_grid_np = cache["y"]
    n_cache = len(a_cache)
    n_y = len(y_grid_np)
    print(f"[stage4-train] Spectral cache: {n_cache} params, {n_y} y-points (direct index sampling)")

    # ---- Build model ----
    model_cfg = train_cfg.get("model", {})
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
        decoder_n_hidden=int(model_cfg.get("decoder_n_hidden", 2)),
    )

    # Load Stage-1 weights
    stage1_ckpt = torch.load(bundle_dir / "stage1_best_model.pt", map_location="cpu", weights_only=False)
    sd1 = stage1_ckpt.get("model_state_dict", stage1_ckpt)
    missing, unexpected = model.load_state_dict(sd1, strict=False)
    print(f"[stage4-train] Stage-1: {len(missing)} missing, {len(unexpected)} unexpected keys")

    # Load Stage-2 AmplitudeNet weights
    stage2_ckpt_path = Path(stage4_cfg["stage2_checkpoint"])
    ckpt2 = torch.load(stage2_ckpt_path, map_location="cpu", weights_only=False)
    sd2 = ckpt2.get("model_state_dict", ckpt2)
    amp_keys = {k: v for k, v in sd2.items() if k.startswith("amplitude_net.")}
    model.load_state_dict(amp_keys, strict=False)
    print(f"[stage4-train] Stage-2 AmplitudeNet: {len(amp_keys)} keys loaded")

    model.to(device=device, dtype=dtype)

    # ---- Freeze policy ----
    train_encoder = bool(stage4_cfg.get("train_encoder", False))
    train_rin = bool(stage4_cfg.get("train_rin_decoder", True))
    train_amp = bool(stage4_cfg.get("train_amplitude_net", True))
    train_up = bool(stage4_cfg.get("train_up_decoder", False))
    train_down = bool(stage4_cfg.get("train_down_decoder", False))

    for name, p in model.named_parameters():
        p.requires_grad = False
    if train_rin:
        for p in model.rin_decoder.parameters():
            p.requires_grad = True
    if train_amp:
        for p in model.amplitude_net.parameters():
            p.requires_grad = True
    if train_up:
        for p in model.up_decoder.parameters():
            p.requires_grad = True
    if train_down:
        for p in model.down_decoder.parameters():
            p.requires_grad = True
    if train_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[stage4-train] Trainable: {trainable:,} / {total:,} params")
    print(f"  encoder: {'train' if train_encoder else 'frozen'}")
    print(f"  rin_decoder: {'train' if train_rin else 'frozen'}")
    print(f"  amplitude_net: {'train' if train_amp else 'frozen'}")
    print(f"  up_decoder: {'train' if train_up else 'frozen'}")
    print(f"  down_decoder: {'train' if train_down else 'frozen'}")

    # ---- Build frozen reference model (Stage-1 + Stage-2 AmplitudeNet) ----
    ref_model = AutoencoderTeukolskyPINN(
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
        decoder_n_hidden=int(model_cfg.get("decoder_n_hidden", 2)),
    )
    ref_model.load_state_dict(sd1, strict=False)
    ref_model.load_state_dict(amp_keys, strict=False)
    ref_model.to(device=device, dtype=dtype)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # ---- Optimizer ----
    lr_rin = float(opt_cfg.get("lr_rin_decoder", 1e-7))
    lr_amp = float(opt_cfg.get("lr_amplitude_net", 5e-7))
    wd = float(opt_cfg.get("weight_decay", 0.0))
    param_groups = []
    if train_rin:
        param_groups.append({"params": model.rin_decoder.parameters(), "lr": lr_rin, "name": "rin_decoder"})
    if train_amp:
        param_groups.append({"params": model.amplitude_net.parameters(), "lr": lr_amp, "name": "amplitude_net"})
    if train_up:
        param_groups.append({"params": model.up_decoder.parameters(), "lr": lr_rin, "name": "up_decoder"})
    if train_down:
        param_groups.append({"params": model.down_decoder.parameters(), "lr": lr_rin, "name": "down_decoder"})
    if train_encoder:
        lr_enc = float(opt_cfg.get("lr_encoder", 0.0))
        param_groups.append({"params": model.encoder.parameters(), "lr": lr_enc, "name": "encoder"})
    optimizer = torch.optim.Adam(param_groups, weight_decay=wd)
    print(f"[stage4-train] Optimizer: lr_rin={lr_rin}, lr_amp={lr_amp}, wd={wd}")

    # ---- Resume checkpoint ----
    start_epoch = 0
    best_val = float("inf")
    best_epoch = -1
    history = []
    if args.resume_checkpoint:
        resume = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model_state_dict"], strict=False)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = resume.get("epoch", 0)
        best_val = resume.get("best_val_loss", float("inf"))
        best_epoch = resume.get("best_epoch", -1)
        history = resume.get("history", [])
        print(f"[stage4-train] Resumed from epoch {start_epoch}, best_val={best_val:.6e}")

    # ---- Output dir ----
    output_root = Path(stage4_cfg.get("output_root", "outputs/autoencoder_stage4_spectral_consistency"))
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_stage4_spectral")
    run_dir = output_root / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    print(f"[stage4-train] Output: {run_dir}")

    # ---- Training hyperparams ----
    epochs = args.epochs
    batch_size = int(stage4_cfg.get("batch_size", 8))
    grad_clip = float(stage4_cfg.get("grad_clip", 0.1))
    weight_consistency = float(loss_cfg.get("weight_consistency", 1.0))
    weight_rin_drift = float(loss_cfg.get("weight_rin_drift", 50.0))
    weight_amp_drift = float(loss_cfg.get("weight_amp_drift", 50.0))
    eps = float(loss_cfg.get("eps", 1e-12))
    save_every = int(stage4_cfg.get("save_every_epochs", 100))
    val_every = int(stage4_cfg.get("val_every_epochs", 20))

    # ---- Training points: sampled directly from spectral cache ----
    print(f"[stage4-train] Parameter source: spectral cache ({n_cache} points, exact u_up/u_down match)")

    y_t = torch.tensor(y_grid_np, device=device, dtype=dtype).unsqueeze(0)  # (1, N_y)
    rng = np.random.default_rng(42)

    def _make_cache_batch(indices):
        """Build batch tensors from cache indices."""
        B = len(indices)
        a_b = torch.tensor(a_cache[indices], device=device, dtype=dtype).reshape(-1, 1)
        omega_b = torch.tensor(omega_cache[indices], device=device, dtype=dtype).reshape(-1, 1)
        u_b = torch.tensor(u_cache[indices], device=device, dtype=dtype).reshape(-1, 1)
        v_b = torch.tensor(v_cache[indices], device=device, dtype=dtype).reshape(-1, 1)
        lam_b = torch.tensor(lam_cache[indices], device=device, dtype=cdtype).reshape(-1, 1)
        u_up_b = torch.tensor(u_up_cache[indices], device=device, dtype=cdtype)
        u_down_b = torch.tensor(u_down_cache[indices], device=device, dtype=cdtype)
        return a_b, omega_b, u_b, v_b, lam_b, u_up_b, u_down_b

    # ---- Initial metrics ----
    print(f"\n[stage4-train] === Initial metrics ===")
    model.eval()
    idx_init = rng.choice(n_cache, min(16, n_cache), replace=False)
    init_infos = []
    for i in idx_init:
        a_t, omega_t, u_t, v_t, lam_t, u_up_t, u_down_t = _make_cache_batch([i])
        _, info = compute_stage4_loss(
            model, a_t, omega_t, y_t, u_t, v_t, lam_t,
            stage1_model=ref_model,
            weight_rin_drift=weight_rin_drift, weight_amp_drift=weight_amp_drift, eps=eps,
            M=M_phys, s=s_phys, m=m_phys,
            u_up_spec=u_up_t, u_down_spec=u_down_t,
        )
        init_infos.append(info)

    def _avg_infos(infos, key):
        return float(np.mean([x[key] for x in infos]))

    print(f"  L_cons:        {_avg_infos(init_infos, 'L_cons'):.6e}")
    print(f"  L_rin_drift:   {_avg_infos(init_infos, 'L_rin_drift'):.6e}")
    print(f"  L_amp_drift:   {_avg_infos(init_infos, 'L_amp_drift'):.6e}")
    print(f"  median_rel_err: {_avg_infos(init_infos, 'median_rel_err'):.6e}")
    print(f"  max_rel_err:    {_avg_infos(init_infos, 'max_rel_err'):.6e}")
    print(f"  B_inc_drift:    {_avg_infos(init_infos, 'B_inc_drift'):.6e}")
    print(f"  B_ref_drift:    {_avg_infos(init_infos, 'B_ref_drift'):.6e}")

    # ---- Training loop ----
    print(f"\n[stage4-train] === Training {epochs} epochs ===")
    t_start = time.time()
    nan_streak = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        # Sample batch from cache indices
        idx = rng.choice(n_cache, min(batch_size, n_cache), replace=False)
        a_b, omega_b, u_b, v_b, lam_b, u_up_b, u_down_b = _make_cache_batch(idx)
        y_b = y_t.expand(len(idx), -1)

        optimizer.zero_grad(set_to_none=True)
        loss, loss_info = compute_stage4_loss(
            model, a_b, omega_b, y_b, u_b, v_b, lam_b,
            stage1_model=ref_model,
            weight_rin_drift=weight_rin_drift, weight_amp_drift=weight_amp_drift, eps=eps,
            M=M_phys, s=s_phys, m=m_phys,
            u_up_spec=u_up_b, u_down_spec=u_down_b,
        )

        if not torch.isfinite(loss):
            nan_streak += 1
            print(f"  [Epoch {epoch+1}] NaN detected (streak={nan_streak}), skipping step")
            if nan_streak >= 5:
                print(f"[stage4-train] ABORT: 5 consecutive NaN losses")
                break
            continue
        nan_streak = 0

        loss.backward()
        if grad_clip > 0:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()

        # ---- Logging ----
        if args.verbose and (epoch + 1) % max(1, min(10, epochs // 20)) == 0:
            print(f"  [Epoch {epoch+1}/{start_epoch+epochs}] "
                  f"L_cons={loss_info['L_cons']:.4e} "
                  f"L_rin={loss_info['L_rin_drift']:.4e} "
                  f"L_amp={loss_info['L_amp_drift']:.4e} "
                  f"med={loss_info['median_rel_err']:.4e}")

        # ---- Validation ----
        if (epoch + 1) % val_every == 0:
            model.eval()
            val_idx = rng.choice(n_cache, min(16, n_cache), replace=False)
            val_infos = []
            with torch.no_grad():
                for vi in val_idx:
                    a_t, omega_t, u_t, v_t, lam_t, u_up_t, u_down_t = _make_cache_batch([vi])
                    _, info = compute_stage4_loss(
                        model, a_t, omega_t, y_t, u_t, v_t, lam_t,
                        stage1_model=ref_model,
                        weight_rin_drift=weight_rin_drift, weight_amp_drift=weight_amp_drift, eps=eps,
                        M=M_phys, s=s_phys, m=m_phys,
                        u_up_spec=u_up_t, u_down_spec=u_down_t,
                    )
                    val_infos.append(info)
            val_loss = float(np.mean([x["total_loss"] for x in val_infos]))
            val_med = float(np.mean([x["median_rel_err"] for x in val_infos]))
            val_max = float(np.mean([x["max_rel_err"] for x in val_infos]))
            print(f"  [Val {epoch+1}] loss={val_loss:.4e} med_err={val_med:.4e} max_err={val_max:.4e}")

            history.append({
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_median_rel_err": val_med,
                "val_max_rel_err": val_max,
                "L_cons": float(np.mean([x["L_cons"] for x in val_infos])),
                "L_rin_drift": float(np.mean([x["L_rin_drift"] for x in val_infos])),
                "L_amp_drift": float(np.mean([x["L_amp_drift"] for x in val_infos])),
            })

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch + 1
                ckpt = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                    "history": history,
                    "config": str(Path(args.config).resolve()),
                    "pre_stage4_bundle": str(bundle_dir.resolve()),
                    "stage2_checkpoint": str(stage2_ckpt_path.resolve()),
                    "spectral_basis_cache": str(cache_path.resolve()),
                }
                torch.save(ckpt, ckpt_dir / "best_model.pt")
                print(f"  [Best {epoch+1}] val_loss={best_val:.4e}")

            model.train()

        # ---- Save latest ----
        if (epoch + 1) % save_every == 0:
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "history": history,
                "config": str(Path(args.config).resolve()),
            }
            torch.save(ckpt, ckpt_dir / f"epoch_{epoch+1}.pt")
            torch.save(ckpt, ckpt_dir / "latest_model.pt")
            print(f"  [Save {epoch+1}] Checkpoint saved")

    # ---- Final ----
    train_time = time.time() - t_start
    print(f"\n[stage4-train] === Training complete ({train_time:.0f}s) ===")
    print(f"  Best epoch: {best_epoch}, best_val_loss: {best_val:.4e}")

    # Save final checkpoint
    final_ckpt = {
        "epoch": start_epoch + epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "history": history,
        "config": str(Path(args.config).resolve()),
        "pre_stage4_bundle": str(bundle_dir.resolve()),
        "spectral_basis_cache": str(cache_path.resolve()),
    }
    torch.save(final_ckpt, ckpt_dir / "latest_model.pt")

    # Save summary
    summary = {
        "run_dir": str(run_dir.resolve()),
        "pre_stage4_bundle": str(bundle_dir.resolve()),
        "stage2_checkpoint": str(stage2_ckpt_path.resolve()),
        "spectral_basis_cache": str(cache_path.resolve()),
        "epochs": start_epoch + epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "history": history,
        "rollback_path": str(bundle_dir.resolve()),
        "rollback_files": ["stage1_best_model.pt", "stage2_best_model.pt"],
    }
    with open(run_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Final metrics
    model.eval()
    final_idx = rng.choice(n_cache, min(16, n_cache), replace=False)
    final_infos = []
    with torch.no_grad():
        for fi in final_idx:
            a_t, omega_t, u_t, v_t, lam_t, u_up_t, u_down_t = _make_cache_batch([fi])
            _, info = compute_stage4_loss(
                model, a_t, omega_t, y_t, u_t, v_t, lam_t,
                stage1_model=ref_model,
                weight_rin_drift=weight_rin_drift, weight_amp_drift=weight_amp_drift, eps=eps,
                M=M_phys, s=s_phys, m=m_phys,
                u_up_spec=u_up_t, u_down_spec=u_down_t,
            )
            final_infos.append(info)

    print(f"\n[stage4-train] === Final metrics ===")
    print(f"  L_cons:        {_avg_infos(final_infos, 'L_cons'):.6e}")
    print(f"  L_rin_drift:   {_avg_infos(final_infos, 'L_rin_drift'):.6e}")
    print(f"  L_amp_drift:   {_avg_infos(final_infos, 'L_amp_drift'):.6e}")
    print(f"  median_rel_err: {_avg_infos(final_infos, 'median_rel_err'):.6e}")
    print(f"  max_rel_err:    {_avg_infos(final_infos, 'max_rel_err'):.6e}")
    print(f"  B_inc_drift:    {_avg_infos(final_infos, 'B_inc_drift'):.6e}")
    print(f"  B_ref_drift:    {_avg_infos(final_infos, 'B_ref_drift'):.6e}")

    drift_ok = _avg_infos(final_infos, 'L_rin_drift') < 1e-2 and _avg_infos(final_infos, 'L_amp_drift') < 1e-2
    has_nan = bool(np.any([not np.isfinite(x['L_cons']) for x in final_infos]))
    print(f"\n  Drift OK: {drift_ok}")
    print(f"  NaN/Inf:  {has_nan}")
    print(f"  Best checkpoint: {ckpt_dir / 'best_model.pt'}")
    print(f"  Rollback path:   {bundle_dir}")

    # Warn if drift too large
    if not drift_ok:
        print(f"\n  *** WARNING: Drift exceeded 1e-2 threshold ***")
        print(f"  *** Consider increasing drift weight or reducing LR ***")
        print(f"  *** Roll back with: --rollback-to-pre-stage4 ***")


if __name__ == "__main__":
    main()
