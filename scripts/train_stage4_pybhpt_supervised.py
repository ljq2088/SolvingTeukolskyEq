#!/usr/bin/env python3
"""Stage-4 pybhpt-supervised training: R_in/P match + PDE residual.

Only RinDecoder trainable. Encoder and AmplitudeNet frozen.
Loss = w_sup * |R_model/P - R_pybhpt/P|^2 + w_pde * |PDE_residual|^2.
w_sup decays on schedule so PDE residual takes over later epochs.

Usage:
  # Build pybhpt references first:
  python scripts/build_pybhpt_reference_cache.py \
    --rpred-cache outputs/pre_stage4_bundle/patch_000_logw_v2/rpred_cache.npz \
    --n-points 16 --output-dir outputs/pybhpt_reference_cache/patch_000_logw_v2

  # Train:
  python scripts/train_stage4_pybhpt_supervised.py \
    --config config/autoencoder_stage4_pybhpt_supervised.yaml \
    --device cuda --epochs 500 --verbose
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
from physical_ansatz.prefactor import delta, delta_r, V_of_r, Leaver_prefactors
from physical_ansatz.stage1_reconstruction import compute_R_over_P_from_model


def _get_dtype(dtype_name):
    return torch.float32 if dtype_name == "float32" else torch.float64


def build_y_grid(cfg_sampling):
    """Build near-infinity-biased y-grid from config sampling settings."""
    near = cfg_sampling.get("near_core", {})
    y_near = np.linspace(near.get("y_min", -0.999), near.get("y_max", -0.95),
                         near.get("n_y", 96))
    trans = cfg_sampling.get("transition", {})
    # Start transition just after near_core end to avoid duplicate
    y_trans = np.linspace(trans.get("y_min", -0.95), trans.get("y_max", 0.0),
                          trans.get("n_y", 32))
    return np.concatenate([y_near, y_trans])


def compute_pde_residual(model, a, omega, y, lambda_, u=None, v=None,
                         m=2, M=1.0, s=-2):
    """Compute normalized Teukolsky PDE residual on R_in (not R_in/P).

    R_in = P(r) * (R_in/P)_model, then compute
    |Delta * R_in_rr + (s+1) * Delta_r * R_in_r + V * R_in|^2 / |R_in|^2.
    """
    B, N = y.shape
    if not y.requires_grad:
        y = y.clone().requires_grad_(True)

    device = y.device
    dtype = y.dtype
    rp = r_plus(a, M)

    # R_over_P = h2 * S(y) from RinDecoder
    R_over_P = compute_R_over_P_from_model(model, a, omega, y, lambda_, u, v, m=m, M=M, s=s)

    # R_over_P derivatives w.r.t y via autograd
    ROP_y = torch.complex(
        torch.autograd.grad(R_over_P.real.sum(), y, create_graph=True, retain_graph=True)[0],
        torch.autograd.grad(R_over_P.imag.sum(), y, create_graph=True, retain_graph=True)[0],
    )
    ROP_yy = torch.complex(
        torch.autograd.grad(ROP_y.real.sum(), y, create_graph=True, retain_graph=True)[0],
        torch.autograd.grad(ROP_y.imag.sum(), y, create_graph=True, retain_graph=True)[0],
    )

    # y → r conversion
    y1 = y + 1.0
    dy_dr = -y1 ** 2 / (2.0 * rp)
    d2y_dr2 = y1 ** 3 / (2.0 * rp ** 2)

    ROP_r = ROP_y * dy_dr
    ROP_rr = ROP_yy * dy_dr ** 2 + ROP_y * d2y_dr2

    # r coordinate
    x = (y + 1.0) / 2.0
    r = rp / x

    # Leaver prefactor P(r) and its analytic r-derivatives
    P, P_r, P_rr = Leaver_prefactors(r, a, omega, m=m, M=M, s=s)

    # R_in = P * R_over_P, with product-rule derivatives
    R_in = P * R_over_P
    R_in_r = P_r * R_over_P + P * ROP_r
    R_in_rr = P_rr * R_over_P + 2.0 * P_r * ROP_r + P * ROP_rr

    # Teukolsky coefficients
    Delta = delta(r, a, M)
    Delta_r_v = delta_r(r, M)
    V = V_of_r(r, a, omega, m, s, lambda_, M)

    residual = Delta * R_in_rr + (s + 1.0) * Delta_r_v * R_in_r + V * R_in
    pw = torch.abs(residual) ** 2 / (torch.abs(R_in.detach()) ** 2 + 1e-12)
    return R_over_P, pw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="config/autoencoder_stage4_pybhpt_supervised.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    full_cfg = load_pinn_full_config(args.config)
    train_cfg = full_cfg.get("train", {})
    s4_cfg = train_cfg["stage4_pybhpt"]
    runtime_cfg = train_cfg.get("runtime", {})
    model_cfg = train_cfg.get("model", {})
    physics_cfg = full_cfg.get("physics", {})

    dtype = _get_dtype(runtime_cfg.get("dtype", "float64"))
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    M_phys = float(physics_cfg.get("problem", {}).get("M", 1.0))
    s_phys = int(physics_cfg.get("problem", {}).get("s", -2))
    m_phys = int(physics_cfg.get("problem", {}).get("m", 2))

    epochs = args.epochs if args.epochs is not None else s4_cfg.get("epochs", 500)
    batch_size = s4_cfg.get("batch_size", 4)

    # ---- Output dir ----
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_stage4_pybhpt")
    out_root = Path(s4_cfg.get("output_root", "outputs/autoencoder_stage4_pybhpt_supervised"))
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # ---- Build y-grid ----
    y_grid_np = build_y_grid(s4_cfg.get("sampling", {}))
    n_y = len(y_grid_np)
    print(f"[init] y-grid: [{y_grid_np[0]:.4f}, {y_grid_np[-1]:.4f}], {n_y} points")

    # ---- Load pybhpt references ----
    ref_path = Path(s4_cfg["pybhpt_references"])
    refs = np.load(ref_path)
    a_refs = refs["a"]
    omega_refs = refs["omega"]
    u_refs = refs["u"]
    v_refs = refs["v"]
    lam_refs = refs["lam"]
    y_ref = refs["y"]
    R_over_P_refs = refs["R_over_P"]

    # Verify y-grid matches
    if not np.allclose(y_ref, y_grid_np, rtol=1e-10):
        # Re-interpolate references to current y-grid
        print(f"[init] Interpolating pybhpt refs from y[{y_ref[0]:.4f},{y_ref[-1]:.4f}] "
              f"to training y[{y_grid_np[0]:.4f},{y_grid_np[-1]:.4f}]")
        R_new = np.zeros((len(a_refs), n_y), dtype=np.complex128)
        for i in range(len(a_refs)):
            R_new[i] = (np.interp(y_grid_np, y_ref, R_over_P_refs[i].real)
                        + 1j * np.interp(y_grid_np, y_ref, R_over_P_refs[i].imag))
        R_over_P_refs = R_new
        y_ref = y_grid_np

    n_refs = len(a_refs)
    print(f"[init] pybhpt references: {n_refs} points")

    # ---- Load model ----
    stage1_ckpt = Path(s4_cfg["stage1_checkpoint"])
    model = AutoencoderTeukolskyPINN(
        hidden_dims=list(model_cfg.get("hidden_dims", [128, 128, 128, 128])),
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
    ckpt = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.to(device=device, dtype=dtype)
    model.eval()

    # ---- Freeze policy: only RinDecoder trainable ----
    freeze = s4_cfg.get("freeze", {})
    for name, param in model.named_parameters():
        if "rin_decoder" in name:
            param.requires_grad = freeze.get("train_rin_decoder", True)
        elif "amplitude_net" in name:
            param.requires_grad = freeze.get("train_amplitude_net", False)
        elif "up_decoder" in name:
            param.requires_grad = freeze.get("train_up_decoder", False)
        elif "down_decoder" in name:
            param.requires_grad = freeze.get("train_down_decoder", False)
        elif "encoder" in name:
            param.requires_grad = freeze.get("train_encoder", False)
        else:
            param.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[init] Trainable: {n_trainable}/{n_total} ({100*n_trainable/n_total:.1f}%)")

    # ---- Optimizer ----
    opt_cfg = s4_cfg.get("optimizer", {})
    lr = float(opt_cfg.get("lr", 5e-6))
    wd = float(opt_cfg.get("weight_decay", 0.0))
    grad_clip = float(opt_cfg.get("grad_clip", 0.1))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=wd)

    start_epoch = 0
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        start_epoch = resume_ckpt.get("epoch", 0)
        print(f"[init] Resumed from epoch {start_epoch}")

    # ---- Loss weights ----
    loss_cfg = s4_cfg.get("loss", {})
    w_sup_init = float(loss_cfg.get("weight_supervision", 1.0))
    w_pde_val = float(loss_cfg.get("weight_pde", 0.1))
    w_sup_decay = float(loss_cfg.get("supervision_decay", 0.98))

    # ---- Pre-load all pybhpt refs to GPU ----
    y_t = torch.tensor(y_grid_np, device=device, dtype=dtype).unsqueeze(0)
    refs_gpu = {}
    for i in range(n_refs):
        refs_gpu[i] = {
            "a": torch.tensor([[float(a_refs[i])]], device=device, dtype=dtype),
            "omega": torch.tensor([[float(omega_refs[i])]], device=device, dtype=dtype),
            "u": torch.tensor([[float(u_refs[i])]], device=device, dtype=dtype),
            "v": torch.tensor([[float(v_refs[i])]], device=device, dtype=dtype),
            "lam": torch.tensor([[complex(lam_refs[i])]], device=device, dtype=cdtype),
            "R_over_P": torch.tensor(R_over_P_refs[i].reshape(1, -1), device=device, dtype=cdtype),
        }

    # ---- Training ----
    n_batches_per_epoch = max(1, n_refs // batch_size)
    print(f"\n[training] {epochs} epochs, lr={lr}, batch_size={batch_size}, "
          f"w_sup_init={w_sup_init}, w_pde={w_pde_val}, w_sup_decay={w_sup_decay}")
    print(f"           grad_clip={grad_clip}, {n_batches_per_epoch} batches/epoch")

    best_loss = float("inf")
    nan_streak = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        epoch_loss = 0.0
        epoch_sup = 0.0
        epoch_pde = 0.0

        # w_sup decay
        w_sup = w_sup_init * (w_sup_decay ** epoch)
        w_sup = max(w_sup, loss_cfg.get("supervision_min", 0.01))

        # Shuffle indices
        perm = torch.randperm(n_refs).tolist()
        for b_start in range(0, n_refs, batch_size):
            b_idx = perm[b_start:b_start + batch_size]
            if len(b_idx) == 0:
                continue

            # Stack batch
            a_b = torch.cat([refs_gpu[i]["a"] for i in b_idx], dim=0)
            omega_b = torch.cat([refs_gpu[i]["omega"] for i in b_idx], dim=0)
            u_b = torch.cat([refs_gpu[i]["u"] for i in b_idx], dim=0)
            v_b = torch.cat([refs_gpu[i]["v"] for i in b_idx], dim=0)
            lam_b = torch.cat([refs_gpu[i]["lam"] for i in b_idx], dim=0)
            R_ref_b = torch.cat([refs_gpu[i]["R_over_P"] for i in b_idx], dim=0)

            y_b = y_t.expand(len(b_idx), -1).clone().requires_grad_(True)

            # Forward
            R_model, pde_pw = compute_pde_residual(
                model, a_b, omega_b, y_b, lam_b, u_b, v_b,
                m=m_phys, M=M_phys, s=s_phys,
            )

            # L_sup: |R_model - R_ref|^2
            sup_pw = torch.abs(R_model - R_ref_b) ** 2
            L_sup = torch.mean(sup_pw)
            L_pde = torch.mean(pde_pw)

            total = w_sup * L_sup + w_pde_val * L_pde

            optimizer.zero_grad()
            total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()

            epoch_loss += float(total.detach())
            epoch_sup += float(L_sup.detach())
            epoch_pde += float(L_pde.detach())

        n_b = max(1, n_batches_per_epoch)
        avg_loss = epoch_loss / n_b
        avg_sup = epoch_sup / n_b
        avg_pde = epoch_pde / n_b

        # NaN guard
        if np.isnan(avg_loss):
            nan_streak += 1
            print(f"  epoch {epoch+1:5d} | NaN loss (streak {nan_streak})")
            if nan_streak >= 5:
                print("[abort] 5 consecutive NaN losses")
                break
            continue
        nan_streak = 0

        best_tag = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_tag = " [BEST]"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "w_sup": w_sup,
            }, ckpt_dir / "best_model.pt")

        if args.verbose or (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:5d}/{start_epoch+epochs} | "
                  f"sup={avg_sup:.6e} pde={avg_pde:.6e} "
                  f"total={avg_loss:.6e} w_sup={w_sup:.4f}{best_tag}")

    # ---- Final save ----
    torch.save({
        "epoch": start_epoch + epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_dir / "last_model.pt")

    print(f"\n[done] run_dir: {run_dir}")
    print(f"        best_loss: {best_loss:.6e}")


if __name__ == "__main__":
    main()
