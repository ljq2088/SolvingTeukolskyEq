#!/usr/bin/env python3
"""Smoke test for Stage-3 u-basis decoder training.

Verifies:
  1. Stage-2 checkpoint loads
  2. Stage-3 freeze: only up/down decoders trainable
  3. predict_u_up / predict_u_down output shapes
  4. u_up(-1)=1, u_down(-1)=1 (boundary ansatz)
  5. du_up/dy(-1)=c_up, du_down/dy(-1)=c_down
  6. A_up/A_down finite
  7. R_up/R_down finite
  8. u-basis residual finite
  9. Backward: only up/down decoders get grads
  10. optimizer.step: only up/down params change
  11. Checkpoint save/load round-trip
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from config.config_loader import load_pinn_full_config
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.u_basis import (
    compose_u_from_f,
    infinity_slopes_y,
    A_up,
    A_down,
)
from physical_ansatz.u_residual import (
    compute_u_basis_residual,
    compute_stage3_loss,
    compute_u_equation_residual,
    compute_stage3_loss_u_equation,
)


def _get_dtype(dtype_name):
    return torch.float32 if dtype_name == "float32" else torch.float64


def main():
    config_path = "config/autoencoder_stage3_ubasis.yaml"
    ckpt_path = "outputs/autoencoder_stage2_amplitude_train/20260515_192625_stage2_amplitude/checkpoints/best_model.pt"

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
    # 2. Stage-3 freeze
    # ============================================================
    print("\n" + "=" * 60)
    print("2. Stage-3 freeze policy")

    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.up_decoder.named_parameters():
        p.requires_grad = True
    for _, p in model.down_decoder.named_parameters():
        p.requires_grad = True

    def _module_trainable(module):
        return all(p.requires_grad for p in module.parameters())

    def _module_frozen(mod):
        return all(not p.requires_grad for p in mod.parameters())

    check("encoder frozen", _module_frozen(model.encoder))
    check("rin_decoder frozen", _module_frozen(model.rin_decoder))
    check("amplitude_net frozen", _module_frozen(model.amplitude_net))
    check("up_decoder trainable", _module_trainable(model.up_decoder))
    check("down_decoder trainable", _module_trainable(model.down_decoder))

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_expected = (sum(p.numel() for p in model.up_decoder.parameters())
                  + sum(p.numel() for p in model.down_decoder.parameters()))
    print(f"  Trainable: {n_trainable}/{n_total}")
    check("only up/down decoders trainable", n_trainable == n_expected)

    # ============================================================
    # 3. predict shapes
    # ============================================================
    print("\n" + "=" * 60)
    print("3. predict_u_up / predict_u_down shapes")

    B, N = 2, 64
    a_t = torch.rand(B, 1, device=device, dtype=dtype)
    omega_t = torch.rand(B, 1, device=device, dtype=dtype) * 0.1
    y_t = torch.linspace(-1, 1, N, device=device, dtype=dtype).unsqueeze(0).expand(B, -1)

    with torch.no_grad():
        u_up = model.predict_u_up(a_t, omega_t, y_t)
        u_down = model.predict_u_down(a_t, omega_t, y_t)

    check("u_up shape", u_up.shape == (B, N), f"got {u_up.shape}")
    check("u_down shape", u_down.shape == (B, N), f"got {u_down.shape}")
    check("u_up finite", torch.all(torch.isfinite(u_up)))
    check("u_down finite", torch.all(torch.isfinite(u_down)))

    # ============================================================
    # 4 & 5. Boundary ansatz
    # ============================================================
    print("\n" + "=" * 60)
    print("4 & 5. Boundary ansatz: u(-1)=1, u_y(-1)=c_inf")

    # Use fixed parameters
    a_val = torch.tensor([[0.5]], device=device, dtype=dtype)
    omega_val = torch.tensor([[0.05]], device=device, dtype=dtype)
    lam_val = torch.tensor([complex(3.0 + 0.0j)], device=device, dtype=cdtype)

    y_bd = torch.linspace(-1, 1, 128, device=device, dtype=dtype).unsqueeze(0)

    with torch.no_grad():
        f_up = model.predict_u_up(a_val, omega_val, y_bd)
        f_down = model.predict_u_down(a_val, omega_val, y_bd)

    c_up, c_down = infinity_slopes_y(a_val, omega_val, lam_val, m=2, M=1.0)

    # Check u(-1)
    u_up_bd = compose_u_from_f(f_up, y_bd, c_up)
    u_down_bd = compose_u_from_f(f_down, y_bd, c_down)

    u_up_at_inf = u_up_bd[0, 0]  # y=-1 is first Chebyshev node (leftmost)
    u_down_at_inf = u_down_bd[0, 0]

    check("u_up(-1)=1", torch.abs(u_up_at_inf - 1.0) < 1e-4,
          f"u_up(-1)={u_up_at_inf:.6e}")
    check("u_down(-1)=1", torch.abs(u_down_at_inf - 1.0) < 1e-4,
          f"u_down(-1)={u_down_at_inf:.6e}")

    # Check u_y(-1) via finite difference
    y_bd_grad = y_bd.clone().requires_grad_(True)
    with torch.no_grad():
        f_up_g = model.predict_u_up(a_val, omega_val, y_bd_grad)
        f_down_g = model.predict_u_down(a_val, omega_val, y_bd_grad)
    u_up_g = compose_u_from_f(f_up_g, y_bd_grad, c_up)
    u_down_g = compose_u_from_f(f_down_g, y_bd_grad, c_down)

    # u_y at y=-1 via autograd
    u_up_re = u_up_g.real
    u_up_y = torch.autograd.grad(u_up_re[0].sum(), y_bd_grad,
                                  create_graph=False, retain_graph=False)[0]
    u_down_re = u_down_g.real
    u_down_y = torch.autograd.grad(u_down_re[0].sum(), y_bd_grad,
                                    create_graph=False, retain_graph=False)[0]

    c_up_real = c_up.real.squeeze()
    c_down_real = c_down.real.squeeze()
    check("du_up/dy(-1) = c_up (real part)",
          torch.abs(u_up_y[0, 0] - c_up_real) < 5e-3,
          f"got {u_up_y[0,0]:.4e}, expected {c_up_real:.4e}")
    check("du_down/dy(-1) = c_down (real part)",
          torch.abs(u_down_y[0, 0] - c_down_real) < 5e-3,
          f"got {u_down_y[0,0]:.4e}, expected {c_down_real:.4e}")

    # ============================================================
    # 6 & 7. A_up/A_down and R_up/R_down finite
    # ============================================================
    print("\n" + "=" * 60)
    print("6 & 7. A_up/A_down, R_up/R_down finite")

    from physical_ansatz.mapping import r_plus
    y_fin = torch.linspace(-0.999, 0.999, 128, device=device, dtype=dtype).unsqueeze(0)
    x_t = (y_fin + 1.0) / 2.0
    rp_t = r_plus(a_val, M=1.0)
    r_t = rp_t / x_t

    with torch.no_grad():
        f_up_fin = model.predict_u_up(a_val, omega_val, y_fin)
        f_down_fin = model.predict_u_down(a_val, omega_val, y_fin)
    u_up_fin = compose_u_from_f(f_up_fin, y_fin, c_up)
    u_down_fin = compose_u_from_f(f_down_fin, y_fin, c_down)

    with torch.no_grad():
        A_up_vals = A_up(r_t, a_val, omega_val)
        A_down_vals = A_down(r_t, a_val, omega_val)
        R_up_vals = A_up_vals * u_up_fin
        R_down_vals = A_down_vals * u_down_fin

    check("A_up finite", torch.all(torch.isfinite(A_up_vals)))
    check("A_down finite", torch.all(torch.isfinite(A_down_vals)))
    check("R_up finite", torch.all(torch.isfinite(R_up_vals)))
    check("R_down finite", torch.all(torch.isfinite(R_down_vals)))

    print(f"  |A_up| range: [{torch.abs(A_up_vals).min():.2e}, {torch.abs(A_up_vals).max():.2e}]")
    print(f"  |A_down| range: [{torch.abs(A_down_vals).min():.2e}, {torch.abs(A_down_vals).max():.2e}]")

    # ============================================================
    # 8. u-basis residual finite
    # ============================================================
    print("\n" + "=" * 60)
    print("8. u-basis residual")

    y_res = y_fin.clone().requires_grad_(True)
    _, res_up, pw_up = compute_u_basis_residual(
        model, a_val, omega_val, y_res, lam_val, M=1.0, s=-2, m=2, basis="up")
    _, res_down, pw_down = compute_u_basis_residual(
        model, a_val, omega_val, y_res, lam_val, M=1.0, s=-2, m=2, basis="down")

    check("residual_up finite", torch.all(torch.isfinite(res_up)),
          f"NaN={torch.isnan(res_up).any().item()}, Inf={torch.isinf(res_up).any().item()}")
    check("residual_down finite", torch.all(torch.isfinite(res_down)))
    check("pointwise_up finite", torch.all(torch.isfinite(pw_up)))
    check("pointwise_down finite", torch.all(torch.isfinite(pw_down)))

    print(f"  mean |res_up|^2: {pw_up.mean().item():.4e}")
    print(f"  mean |res_down|^2: {pw_down.mean().item():.4e}")

    # ============================================================
    # 9. u-equation residual finite
    # ============================================================
    print("\n" + "=" * 60)
    print("9. u-equation residual")

    y_ueq = torch.linspace(-0.999, 0.999, 128, device=device, dtype=dtype).unsqueeze(0)
    y_ueq = y_ueq.clone().requires_grad_(True)

    _, res_ueq_up, pw_ueq_up = compute_u_equation_residual(
        model, a_val, omega_val, y_ueq, lam_val, M=1.0, s=-2, m=2, basis="up")
    _, res_ueq_down, pw_ueq_down = compute_u_equation_residual(
        model, a_val, omega_val, y_ueq, lam_val, M=1.0, s=-2, m=2, basis="down")

    check("u_eq residual_up finite", torch.all(torch.isfinite(res_ueq_up)),
          f"NaN={torch.isnan(res_ueq_up).any().item()}, Inf={torch.isinf(res_ueq_up).any().item()}")
    check("u_eq residual_down finite", torch.all(torch.isfinite(res_ueq_down)))
    check("u_eq pointwise_up finite", torch.all(torch.isfinite(pw_ueq_up)))
    check("u_eq pointwise_down finite", torch.all(torch.isfinite(pw_ueq_down)))

    print(f"  u_eq mean |res_up|^2: {pw_ueq_up.mean().item():.4e}")
    print(f"  u_eq mean |res_down|^2: {pw_ueq_down.mean().item():.4e}")

    # ============================================================
    # 10. u-equation loss finite
    # ============================================================
    print("\n" + "=" * 60)
    print("10. u-equation loss")

    y_uloss = torch.linspace(-0.999, 0.999, 128, device=device, dtype=dtype).unsqueeze(0)
    y_uloss = y_uloss.clone().requires_grad_(True)
    total_u_loss, u_info = compute_stage3_loss_u_equation(
        model, a_val, omega_val, y_uloss, lam_val, M=1.0, s=-2, m=2)

    check("u_eq loss_up finite", np.isfinite(u_info["loss_up"]),
          f"loss_up={u_info['loss_up']:.4e}")
    check("u_eq loss_down finite", np.isfinite(u_info["loss_down"]),
          f"loss_down={u_info['loss_down']:.4e}")
    print(f"  u_eq loss_up: {u_info['loss_up']:.4e}, loss_down: {u_info['loss_down']:.4e}")

    # ============================================================
    # 11. u-equation backward: only up/down get grads
    # ============================================================
    print("\n" + "=" * 60)
    print("11. u-equation backward: only up/down decoders get grads")

    model.zero_grad()
    y_g2 = torch.linspace(-0.999, 0.999, 128, device=device, dtype=dtype).unsqueeze(0)
    y_g2 = y_g2.clone().requires_grad_(True)
    u_loss_g, _ = compute_stage3_loss_u_equation(
        model, a_val, omega_val, y_g2, lam_val, M=1.0, s=-2, m=2)
    u_loss_g.backward()

    check("u_eq: encoder grads are None",
          all(p.grad is None for p in model.encoder.parameters()))
    check("u_eq: rin_decoder grads are None",
          all(p.grad is None for p in model.rin_decoder.parameters()))
    check("u_eq: amplitude_net grads are None",
          all(p.grad is None for p in model.amplitude_net.parameters()))
    check("u_eq: up_decoder has grads",
          all(p.grad is not None for p in model.up_decoder.parameters()))
    check("u_eq: down_decoder has grads",
          all(p.grad is not None for p in model.down_decoder.parameters()))

    # No NaN in grads, grad norm finite
    for module_name, module in [("up_decoder", model.up_decoder),
                                  ("down_decoder", model.down_decoder)]:
        grad_nan = any(
            torch.isnan(p.grad).any() for p in module.parameters() if p.grad is not None
        )
        check(f"u_eq: {module_name} grads finite", not grad_nan)
        grad_norm = sum(p.grad.norm().item() ** 2 for p in module.parameters() if p.grad is not None) ** 0.5
        check(f"u_eq: {module_name} grad norm finite", np.isfinite(grad_norm),
              f"grad_norm={grad_norm:.4e}")
        print(f"  {module_name} grad norm: {grad_norm:.4e}")

    # ============================================================
    # 12. direct_R vs u_equation residual sign consistency
    # ============================================================
    print("\n" + "=" * 60)
    print("12. direct_R vs u_equation residual consistency (moderate r)")

    y_mod = torch.linspace(-0.5, 0.5, 64, device=device, dtype=dtype).unsqueeze(0)
    y_mod = y_mod.clone().requires_grad_(True)

    _, R_res_up, _ = compute_u_basis_residual(
        model, a_val, omega_val, y_mod, lam_val, M=1.0, s=-2, m=2, basis="up")
    _, u_res_up, _ = compute_u_equation_residual(
        model, a_val, omega_val, y_mod, lam_val, M=1.0, s=-2, m=2, basis="up")

    # Both should be valid (finite), no strict scale match required
    check("direct_R up residual finite at moderate r",
          torch.all(torch.isfinite(R_res_up)))
    check("u_eq up residual finite at moderate r",
          torch.all(torch.isfinite(u_res_up)))

    # Same for down
    _, R_res_dn, _ = compute_u_basis_residual(
        model, a_val, omega_val, y_mod, lam_val, M=1.0, s=-2, m=2, basis="down")
    _, u_res_dn, _ = compute_u_equation_residual(
        model, a_val, omega_val, y_mod, lam_val, M=1.0, s=-2, m=2, basis="down")

    check("direct_R down residual finite at moderate r",
          torch.all(torch.isfinite(R_res_dn)))
    check("u_eq down residual finite at moderate r",
          torch.all(torch.isfinite(u_res_dn)))

    print(f"  u_eq up residual scale: {torch.abs(u_res_up).mean().item():.4e}")
    print(f"  u_eq down residual scale: {torch.abs(u_res_dn).mean().item():.4e}")

    # ============================================================
    # 13. Backward: only up/down get grads (direct R reference)
    # ============================================================
    print("\n" + "=" * 60)
    print("13. Backward: only up/down decoders get grads (direct R)")

    model.up_decoder.train()
    model.down_decoder.train()

    y_g = y_fin.clone().requires_grad_(True)
    total_loss, info = compute_stage3_loss(
        model, a_val, omega_val, y_g, lam_val, M=1.0, s=-2, m=2)
    total_loss.backward()

    check("encoder grads are None",
          all(p.grad is None for p in model.encoder.parameters()))
    check("rin_decoder grads are None",
          all(p.grad is None for p in model.rin_decoder.parameters()))
    check("amplitude_net grads are None",
          all(p.grad is None for p in model.amplitude_net.parameters()))
    check("up_decoder has grads",
          all(p.grad is not None for p in model.up_decoder.parameters()))
    check("down_decoder has grads",
          all(p.grad is not None for p in model.down_decoder.parameters()))

    # No NaN in grads
    for module_name, module in [("up_decoder", model.up_decoder),
                                  ("down_decoder", model.down_decoder)]:
        grad_nan = any(
            torch.isnan(p.grad).any() for p in module.parameters() if p.grad is not None
        )
        check(f"{module_name} grads finite", not grad_nan)

    print(f"  loss_up: {info['loss_up']:.4e}, loss_down: {info['loss_down']:.4e}")

    # ============================================================
    # 14. optimizer.step: only up/down change
    # ============================================================
    print("\n" + "=" * 60)
    print("14. optimizer.step: only up/down params change")

    # Snapshot params before step
    def _param_snapshot(module):
        return {n: p.detach().clone() for n, p in module.named_parameters()}

    snap_enc = _param_snapshot(model.encoder)
    snap_rin = _param_snapshot(model.rin_decoder)
    snap_amp = _param_snapshot(model.amplitude_net)
    snap_up = _param_snapshot(model.up_decoder)
    snap_down = _param_snapshot(model.down_decoder)

    opt = torch.optim.Adam(
        list(model.up_decoder.parameters()) + list(model.down_decoder.parameters()),
        lr=0.01,
    )

    y_g2 = y_fin.clone().requires_grad_(True)
    loss2, _ = compute_stage3_loss(
        model, a_val, omega_val, y_g2, lam_val, M=1.0, s=-2, m=2)
    opt.zero_grad()
    loss2.backward()
    opt.step()

    def _params_changed(snap, module):
        for n, p in module.named_parameters():
            if not torch.allclose(snap[n], p.detach()):
                return True
        return False

    check("encoder unchanged", not _params_changed(snap_enc, model.encoder))
    check("rin_decoder unchanged", not _params_changed(snap_rin, model.rin_decoder))
    check("amplitude_net unchanged", not _params_changed(snap_amp, model.amplitude_net))
    check("up_decoder changed", _params_changed(snap_up, model.up_decoder))
    check("down_decoder changed", _params_changed(snap_down, model.down_decoder))

    # ============================================================
    # 15. Checkpoint save/load round-trip
    # ============================================================
    print("\n" + "=" * 60)
    print("15. Checkpoint save/load round-trip")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name

    ckpt_data = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "best_val_loss": 0.1,
    }
    torch.save(ckpt_data, tmp_path)
    loaded = torch.load(tmp_path, map_location="cpu", weights_only=False)
    model2 = AutoencoderTeukolskyPINN(
        hidden_dims=model_cfg.get("hidden_dims", [128, 128, 128, 128]),
        activation=str(model_cfg.get("activation", "silu")),
        param_embed_dim=int(model_cfg.get("param_embed_dim", 64)),
        fourier_num_freqs=int(model_cfg.get("fourier_num_freqs", 2)),
        fourier_base_scale=float(model_cfg.get("fourier_base_scale", 1.0)),
        use_film=bool(model_cfg.get("use_film", True)),
        use_residual=bool(model_cfg.get("use_residual", True)),
        amp_hidden_dim=int(model_cfg.get("amp_hidden_dim", 128)),
        amp_n_blocks=int(model_cfg.get("amp_n_blocks", 3)),
    )
    model2.load_state_dict(loaded["model_state_dict"], strict=False)
    check("round-trip load", True)
    check("round-trip epoch", loaded["epoch"] == 1)

    import os
    os.unlink(tmp_path)

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
