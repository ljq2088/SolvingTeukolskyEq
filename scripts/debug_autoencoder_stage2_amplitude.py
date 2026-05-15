#!/usr/bin/env python3
"""Smoke test for Stage-2 AmplitudeNet training.

Verifies:
  1. Model loads from Stage-1 checkpoint
  2. Stage-2 freeze: only amplitude_net trainable
  3. AmplitudeNet input/output shapes
  4. Teacher data loads and matches model expectations
  5. 方案B loss computes correctly
  6. One forward/backward pass succeeds
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from config.config_loader import load_pinn_full_config
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from amplitude_teacher import AmplitudeTeacher


def _get_dtype(dtype_name):
    return torch.float32 if dtype_name == "float32" else torch.float64


def amplitude_loss(pred, target, eps=1e-8, weight_logmag=1.0, weight_phase=1.0):
    logmag_pred = torch.log(torch.abs(pred) + eps)
    logmag_target = torch.log(torch.abs(target) + eps)
    loss_logmag = torch.nn.functional.mse_loss(logmag_pred, logmag_target)

    norm_pred = pred / (torch.abs(pred) + eps)
    norm_target = target / (torch.abs(target) + eps)
    loss_phase = (torch.nn.functional.mse_loss(norm_pred.real, norm_target.real)
                  + torch.nn.functional.mse_loss(norm_pred.imag, norm_target.imag))
    return weight_logmag * loss_logmag + weight_phase * loss_phase


def main():
    config_path = "config/autoencoder_stage2_amplitude.yaml"
    ckpt_path = "outputs/stage1_artifacts/patch_000_logw_v2/stage1_best_model.pt"
    teacher_path = "outputs/stage1_artifacts/patch_000_logw_v2/amplitude_teacher.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    passed = 0
    failed = 0
    errors = []

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            msg = f"  FAIL: {name}"
            if detail:
                msg += f" -- {detail}"
            print(msg)
            errors.append(name)

    # ============================================================
    # 1. Model load
    # ============================================================
    print("=" * 60)
    print("1. Model loading")
    full_cfg = load_pinn_full_config(config_path)
    runtime_cfg = full_cfg.get("runtime", {})
    dtype_name = runtime_cfg.get("dtype", "float64")
    dtype = _get_dtype(dtype_name)
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg_raw = ckpt.get("full_cfg", full_cfg)
    model_cfg = model_cfg_raw.get("model", full_cfg.get("model", {}))

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
    model.eval()
    print(f"  Model loaded, device={device}, dtype={dtype}")
    check("model.load_state_dict", True)

    # ============================================================
    # 2. Stage-2 freeze
    # ============================================================
    print("\n" + "=" * 60)
    print("2. Stage-2 freeze policy")

    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.amplitude_net.named_parameters():
        p.requires_grad = True

    def _module_trainable(module):
        return all(p.requires_grad for p in module.parameters())

    def _module_frozen(mod):
        return all(not p.requires_grad for p in mod.parameters())

    check("encoder frozen", _module_frozen(model.encoder))
    check("rin_decoder frozen", _module_frozen(model.rin_decoder))
    check("up_decoder frozen", _module_frozen(model.up_decoder))
    check("down_decoder frozen", _module_frozen(model.down_decoder))
    check("amplitude_net trainable", _module_trainable(model.amplitude_net))

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {n_trainable}/{n_total}")
    check("only amplitude_net trainable",
          n_trainable == sum(p.numel() for p in model.amplitude_net.parameters()))

    # ============================================================
    # 3. AmplitudeNet shapes
    # ============================================================
    print("\n" + "=" * 60)
    print("3. AmplitudeNet input/output shapes")

    B = 4
    a_t = torch.rand(B, 1, device=device, dtype=dtype)
    omega_t = torch.rand(B, 1, device=device, dtype=dtype) * 0.1
    u_t = torch.rand(B, 1, device=device, dtype=dtype)
    v_t = torch.rand(B, 1, device=device, dtype=dtype)

    with torch.no_grad():
        param_feats, alpha, xi = model.encoder._build_param_features(
            a=a_t, omega=omega_t, u=u_t, v=v_t,
        )
        B_inc, B_ref, raw = model.amplitude_net(param_feats)

    check("param_feats shape", param_feats.shape == (B, model.encoder.param_in_dim),
          f"got {param_feats.shape}")
    check("B_inc shape", B_inc.shape == (B,), f"got {B_inc.shape}")
    check("B_ref shape", B_ref.shape == (B,), f"got {B_ref.shape}")
    check("B_inc is complex", torch.is_complex(B_inc))
    check("B_ref is complex", torch.is_complex(B_ref))
    check("raw is dict", isinstance(raw, dict))
    check("raw has rho/phi keys", all(k in raw for k in ["rho_inc", "phi_inc", "rho_ref", "phi_ref"]))

    print(f"  B_inc: {B_inc.detach().cpu().numpy()}")
    print(f"  B_ref: {B_ref.detach().cpu().numpy()}")

    # ============================================================
    # 4. Teacher data
    # ============================================================
    print("\n" + "=" * 60)
    print("4. Teacher data")

    teacher = AmplitudeTeacher.load(teacher_path)
    n_t = len(teacher.a_vals)
    valid = np.isfinite(teacher.B_inc) & np.isfinite(teacher.B_ref)
    n_valid = int(valid.sum())

    check("teacher has a_vals", len(teacher.a_vals) > 0)
    check("teacher has omega_vals", len(teacher.omega_vals) > 0)
    check("teacher has B_inc", len(teacher.B_inc) > 0)
    check("teacher has B_ref", len(teacher.B_ref) > 0)
    check("a and omega same length", len(teacher.a_vals) == len(teacher.omega_vals))
    check("B_inc and B_ref same length", len(teacher.B_inc) == len(teacher.B_ref))
    check("teacher finite samples", n_valid > 0, f"{n_valid}/{n_t} valid")
    check("B_inc ~ 1", np.allclose(np.abs(teacher.B_inc[valid]), 1.0, rtol=1e-6),
          f"|B_inc| = {np.abs(teacher.B_inc[valid])}")
    check("B_ref non-zero", np.all(np.abs(teacher.B_ref[valid]) > 0))

    print(f"  Samples: {n_t}, Valid: {n_valid}")
    print(f"  B_inc mag: [{np.min(np.abs(teacher.B_inc[valid])):.4e}, "
          f"{np.max(np.abs(teacher.B_inc[valid])):.4e}]")
    print(f"  B_ref mag: [{np.min(np.abs(teacher.B_ref[valid])):.4e}, "
          f"{np.max(np.abs(teacher.B_ref[valid])):.4e}]")

    # ============================================================
    # 5. 方案B loss
    # ============================================================
    print("\n" + "=" * 60)
    print("5. 方案B loss")

    B_inc_pred = torch.tensor([1.2 + 0.1j, 0.9 - 0.05j], device=device, dtype=cdtype)
    B_inc_true = torch.tensor([1.0 + 0.0j, 1.0 + 0.0j], device=device, dtype=cdtype)
    loss_inc = amplitude_loss(B_inc_pred, B_inc_true)
    check("loss is finite", torch.isfinite(loss_inc), f"loss={loss_inc.item():.6e}")
    check("loss > 0", loss_inc.item() > 0, f"loss={loss_inc.item():.6e}")

    # Perfect match should give ~0 loss
    loss_perfect = amplitude_loss(B_inc_true, B_inc_true)
    check("perfect match loss ~ 0", loss_perfect.item() < 1e-5,
          f"loss={loss_perfect.item():.6e}")

    B_ref_pred = torch.tensor([100.0 + 50.0j, -30.0 + 80.0j], device=device, dtype=cdtype)
    B_ref_true = torch.tensor([95.0 + 55.0j, -28.0 + 75.0j], device=device, dtype=cdtype)
    loss_ref = amplitude_loss(B_ref_pred, B_ref_true)
    check("B_ref loss finite", torch.isfinite(loss_ref), f"loss={loss_ref.item():.6e}")

    print(f"  loss_inc (mismatch): {loss_inc.item():.6e}")
    print(f"  loss_inc (perfect):  {loss_perfect.item():.6e}")
    print(f"  loss_ref (mismatch): {loss_ref.item():.6e}")

    # ============================================================
    # 6. Forward/backward pass
    # ============================================================
    print("\n" + "=" * 60)
    print("6. Forward/backward pass")

    model.amplitude_net.train()
    idx = np.random.choice(n_valid, min(8, n_valid), replace=False)
    a_b = torch.tensor(teacher.a_vals[idx], device=device, dtype=dtype).unsqueeze(-1)
    omega_b = torch.tensor(teacher.omega_vals[idx], device=device, dtype=dtype).unsqueeze(-1)
    B_inc_t = torch.tensor(teacher.B_inc[idx], device=device, dtype=cdtype)
    B_ref_t = torch.tensor(teacher.B_ref[idx], device=device, dtype=cdtype)

    B_inc_p, B_ref_p, _ = model.predict_amplitudes(a_b, omega_b)
    loss = amplitude_loss(B_inc_p, B_inc_t) + amplitude_loss(B_ref_p, B_ref_t)
    loss.backward()

    check("forward B_inc shape matches", B_inc_p.shape == B_inc_t.shape)
    check("forward B_ref shape matches", B_ref_p.shape == B_ref_t.shape)
    check("loss backward succeeds", True)
    check("amplitude_net grads exist",
          all(p.grad is not None for p in model.amplitude_net.parameters()))
    check("encoder grads are None",
          all(p.grad is None for p in model.encoder.parameters()))
    check("rin_decoder grads are None",
          all(p.grad is None for p in model.rin_decoder.parameters()))

    # Check no NaN in grads
    grad_nan = any(
        torch.isnan(p.grad).any()
        for p in model.amplitude_net.parameters() if p.grad is not None
    )
    check("grads are finite", not grad_nan)

    print(f"  loss: {loss.item():.6e}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print(f"TOTAL: {passed + failed} checks, {passed} PASS, {failed} FAIL")
    if errors:
        print("FAILURES:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
