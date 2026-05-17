#!/usr/bin/env python3
"""15-point debug checklist for Stage-4 spectral consistency training.

Usage:
  python scripts/debug_stage4_spectral_consistency.py \
    --config config/autoencoder_stage4_spectral_consistency.yaml \
    --device cuda
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from config.config_loader import load_pinn_full_config
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.mapping import r_plus, r_minus
from physical_ansatz.u_basis import A_up, A_down


def _get_dtype(dtype_name):
    return torch.float32 if dtype_name == "float32" else torch.float64


def Leaver_P(r, a, omega, m=2, M=1.0, s=-2):
    rp = r_plus(a, M)
    rm = r_minus(a, M)
    sigma_p = (2.0 * omega * rp - m * a) / (rp - rm)
    pp = -s - 1j * sigma_p
    pm = -1 - s + 2j * omega + 1j * sigma_p
    return ((r - rp) ** pp) * ((r - rm) ** pm) * torch.exp(1j * omega * r)


def main():
    parser = argparse.ArgumentParser(description="Stage-4 debug checklist")
    parser.add_argument("--config", type=str, default="config/autoencoder_stage4_spectral_consistency.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    passed = 0
    failed = 0
    results = []

    def check(name, condition, detail=""):
        nonlocal passed, failed
        ok = bool(condition)
        status = "PASS" if ok else "FAIL"
        msg = f"  [{status}] {name}"
        if detail and not ok:
            msg += f" — {detail}"
        print(msg)
        results.append({"name": name, "status": status, "detail": detail if not ok else ""})
        if ok:
            passed += 1
        else:
            failed += 1
        return ok

    # ---- Load config ----
    full_cfg = load_pinn_full_config(args.config)
    train_cfg = full_cfg.get("train", {})
    runtime_cfg = train_cfg.get("runtime", {})
    dtype_name = runtime_cfg.get("dtype", "float64")
    dtype = _get_dtype(dtype_name)
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    stage4_cfg = full_cfg.get("train", {}).get("stage4", full_cfg.get("stage4", {}))
    M_phys = float(full_cfg.get("physics", {}).get("problem", {}).get("M", 1.0))
    s_phys = int(full_cfg.get("physics", {}).get("problem", {}).get("s", -2))
    m_phys = int(full_cfg.get("physics", {}).get("problem", {}).get("m", 2))

    # ---- 1. Pre-stage4 bundle loadable ----
    bundle_dir = Path(stage4_cfg["pre_stage4_bundle"])
    stage1_pt = bundle_dir / "stage1_best_model.pt"
    check("1. pre-stage4 bundle exists", bundle_dir.exists(), str(bundle_dir))
    check("1a. stage1_best_model.pt", stage1_pt.exists(), str(stage1_pt))

    # ---- 2. Spectral basis cache loadable ----
    cache_path = Path(stage4_cfg["spectral_basis_cache"])
    check("2a. spectral_basis_cache.npz exists", cache_path.exists(), str(cache_path))
    cache = np.load(cache_path)
    required_keys = ["a", "omega", "u", "v", "y", "u_up", "u_down"]
    for k in required_keys:
        check(f"2b. cache has '{k}'", k in cache, f"keys: {list(cache.keys())}")
    n_cache = len(cache["a"])
    n_y = len(cache["y"])
    check(f"2c. cache n_param={n_cache} > 0", n_cache > 0)
    check(f"2d. cache n_y={n_y} > 0", n_y > 0)

    # ---- 3. Stage-1 / Stage-2 checkpoints loadable ----
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

    ckpt1 = torch.load(stage1_pt, map_location="cpu", weights_only=False)
    sd1 = ckpt1.get("model_state_dict", ckpt1)
    missing, unexpected = model.load_state_dict(sd1, strict=False)
    check("3a. Stage-1 checkpoint loaded", len(unexpected) == 0,
          f"{len(missing)} missing, {len(unexpected)} unexpected")

    stage2_ckpt_path = Path(stage4_cfg["stage2_checkpoint"])
    check("3b. Stage-2 checkpoint exists", stage2_ckpt_path.exists(), str(stage2_ckpt_path))
    ckpt2 = torch.load(stage2_ckpt_path, map_location="cpu", weights_only=False)
    sd2 = ckpt2.get("model_state_dict", ckpt2)
    amp_keys = {k: v for k, v in sd2.items() if k.startswith("amplitude_net.")}
    model.load_state_dict(amp_keys, strict=False)
    check("3c. Stage-2 AmplitudeNet loaded", len(amp_keys) > 0, f"{len(amp_keys)} keys")

    model.to(device=device, dtype=dtype)
    model.eval()

    # ---- 4. R_model_over_P shape ----
    y_t = torch.tensor(cache["y"][:16], device=device, dtype=dtype).unsqueeze(0)
    a_t = torch.tensor([[0.5]], device=device, dtype=dtype)
    omega_t = torch.tensor([[0.5]], device=device, dtype=dtype)
    u_t = torch.tensor([[0.5]], device=device, dtype=dtype)
    v_t = torch.tensor([[0.5]], device=device, dtype=dtype)
    with torch.no_grad():
        R_over_P = model.predict_Rin(a_t, omega_t, y_t, u_t, v_t)
    check("4. R_model_over_P shape (1, N)", R_over_P.shape == (1, len(y_t[0])),
          f"got {R_over_P.shape}")
    check("4a. R_model_over_P finite", torch.all(torch.isfinite(torch.abs(R_over_P))).item())

    # ---- 5. B_inc/B_ref shape ----
    with torch.no_grad():
        B_inc, B_ref, _ = model.predict_amplitudes(a_t, omega_t, u_t, v_t)
    check("5. B_inc shape () or (1,)", B_inc.numel() == 1, f"got {B_inc.shape}")
    check("5a. B_ref shape () or (1,)", B_ref.numel() == 1, f"got {B_ref.shape}")
    check("5b. B_inc finite", torch.isfinite(torch.abs(B_inc)).item())
    check("5c. B_ref finite", torch.isfinite(torch.abs(B_ref)).item())

    # ---- 6. u_up/u_down spectral cache shape ----
    check("6a. u_up shape (n_param, n_y)", cache["u_up"].shape == (n_cache, n_y),
          f"got {cache['u_up'].shape}, expected ({n_cache}, {n_y})")
    check("6b. u_down shape (n_param, n_y)", cache["u_down"].shape == (n_cache, n_y),
          f"got {cache['u_down'].shape}, expected ({n_cache}, {n_y})")

    # ---- 7. A_up/A_down/P shape ----
    rp = r_plus(a_t, M_phys)
    x = (y_t + 1.0) / 2.0
    r = rp / x
    A_u = A_up(r, a_t, omega_t, M_phys)
    A_d = A_down(r, a_t, omega_t, M_phys)
    P = Leaver_P(r, a_t, omega_t, m=m_phys, M=M_phys, s=s_phys)
    check("7a. A_up shape (1, N)", A_u.shape == y_t.shape, f"got {A_u.shape}")
    check("7b. A_down shape (1, N)", A_d.shape == y_t.shape, f"got {A_d.shape}")
    check("7c. P shape (1, N)", P.shape == y_t.shape, f"got {P.shape}")
    check("7d. A_up finite", torch.all(torch.isfinite(A_u)).item())
    check("7e. A_down finite", torch.all(torch.isfinite(A_d)).item())
    check("7f. P finite", torch.all(torch.isfinite(torch.abs(P))).item())

    # ---- 8. R_combo_over_P finite ----
    u_up_spec = torch.tensor(cache["u_up"][0:1, :16], device=device, dtype=cdtype)
    u_down_spec = torch.tensor(cache["u_down"][0:1, :16], device=device, dtype=cdtype)
    R_combo = (B_ref.unsqueeze(-1) * A_u * u_up_spec + B_inc.unsqueeze(-1) * A_d * u_down_spec) / P
    check("8. R_combo_over_P finite", torch.all(torch.isfinite(torch.abs(R_combo))).item())

    # ---- 9. L_cons finite ----
    eps = float(stage4_cfg.get("loss", {}).get("eps", 1e-12))
    diff = R_over_P[:, :16] - R_combo
    L_cons = torch.mean(torch.abs(diff) ** 2) / (torch.mean(torch.abs(R_over_P[:, :16]) ** 2) + eps)
    check("9. L_cons finite", torch.isfinite(L_cons).item(), f"L_cons={float(L_cons):.4e}")

    # ---- 10. Drift loss finite ----
    # Build a frozen reference
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

    with torch.no_grad():
        R_s1 = ref_model.predict_Rin(a_t, omega_t, y_t, u_t, v_t)[:, :16]
        B_inc_s2, B_ref_s2, _ = ref_model.predict_amplitudes(a_t, omega_t, u_t, v_t)
    L_rin_drift = torch.mean(torch.abs(R_over_P[:, :16] - R_s1) ** 2) / \
                  (torch.mean(torch.abs(R_s1) ** 2) + eps)
    L_amp_inc = torch.mean(torch.abs(B_inc - B_inc_s2) ** 2) / \
                (torch.mean(torch.abs(B_inc_s2) ** 2) + eps)
    L_amp_ref = torch.mean(torch.abs(B_ref - B_ref_s2) ** 2) / \
                (torch.mean(torch.abs(B_ref_s2) ** 2) + eps)
    check("10a. L_rin_drift finite", torch.isfinite(L_rin_drift).item(),
          f"L_rin_drift={float(L_rin_drift):.4e}")
    check("10b. L_amp_drift finite", torch.isfinite(L_amp_inc + L_amp_ref).item(),
          f"L_amp_drift={float(L_amp_inc + L_amp_ref):.4e}")

    # ---- 11. Only RinDecoder / AmplitudeNet have gradients ----
    # Simulate freeze policy
    train_rin = bool(stage4_cfg.get("train_rin_decoder", True))
    train_amp = bool(stage4_cfg.get("train_amplitude_net", True))
    train_up = bool(stage4_cfg.get("train_up_decoder", False))
    train_down = bool(stage4_cfg.get("train_down_decoder", False))
    train_enc = bool(stage4_cfg.get("train_encoder", False))

    for p in model.parameters():
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
    if train_enc:
        for p in model.encoder.parameters():
            p.requires_grad = True

    # Check gradients after a backward pass
    y_full = torch.tensor(cache["y"], device=device, dtype=dtype).unsqueeze(0)
    R_over_P_full = model.predict_Rin(a_t, omega_t, y_full, u_t, v_t)
    B_inc_full, B_ref_full, _ = model.predict_amplitudes(a_t, omega_t, u_t, v_t)

    # Compute a simple loss and backward
    A_u_full = A_up(rp / ((y_full + 1.0) / 2.0), a_t, omega_t, M_phys)
    A_d_full = A_down(rp / ((y_full + 1.0) / 2.0), a_t, omega_t, M_phys)
    P_full = Leaver_P(rp / ((y_full + 1.0) / 2.0), a_t, omega_t, m=m_phys, M=M_phys, s=s_phys)
    u_up_f = torch.tensor(cache["u_up"][0:1], device=device, dtype=cdtype)
    u_down_f = torch.tensor(cache["u_down"][0:1], device=device, dtype=cdtype)
    R_combo_f = (B_ref_full.unsqueeze(-1) * A_u_full * u_up_f +
                 B_inc_full.unsqueeze(-1) * A_d_full * u_down_f) / P_full
    test_loss = torch.mean(torch.abs(R_over_P_full - R_combo_f) ** 2) / \
                (torch.mean(torch.abs(R_over_P_full) ** 2) + eps)
    test_loss.backward()

    rin_has_grad = any(p.grad is not None for p in model.rin_decoder.parameters())
    amp_has_grad = any(p.grad is not None for p in model.amplitude_net.parameters())
    encoder_has_grad = any(p.grad is not None and p.requires_grad
                           for p in model.encoder.parameters())
    up_has_grad = any(p.grad is not None for p in model.up_decoder.parameters())
    down_has_grad = any(p.grad is not None for p in model.down_decoder.parameters())

    check("11. RinDecoder has gradients", rin_has_grad == train_rin,
          f"expected {train_rin}, got {rin_has_grad}")
    check("11a. AmplitudeNet has gradients", amp_has_grad == train_amp,
          f"expected {train_amp}, got {amp_has_grad}")

    # ---- 12. Encoder has no gradients ----
    check("12. Encoder frozen (no grad)", not encoder_has_grad,
          f"expected False, got {encoder_has_grad}")

    # ---- 13. Up/down decoder have no gradients ----
    check("13a. Up decoder frozen (no grad)", not up_has_grad,
          f"expected False, got {up_has_grad}")
    check("13b. Down decoder frozen (no grad)", not down_has_grad,
          f"expected False, got {down_has_grad}")

    # ---- 14. Optimizer.step changes only RinDecoder / AmplitudeNet ----
    # Save pre-step weights
    pre_rin = {n: p.clone() for n, p in model.rin_decoder.named_parameters()}
    pre_amp = {n: p.clone() for n, p in model.amplitude_net.named_parameters()}
    pre_enc = {n: p.clone() for n, p in model.encoder.named_parameters()}
    pre_up = {n: p.clone() for n, p in model.up_decoder.named_parameters()}
    pre_down = {n: p.clone() for n, p in model.down_decoder.named_parameters()}

    opt_groups = []
    if train_rin:
        opt_groups.append({"params": model.rin_decoder.parameters(), "lr": 1e-7})
    if train_amp:
        opt_groups.append({"params": model.amplitude_net.parameters(), "lr": 5e-7})
    optimizer = torch.optim.Adam(opt_groups)

    model.zero_grad()
    # Re-forward
    R_over_P2 = model.predict_Rin(a_t, omega_t, y_full, u_t, v_t)
    B_inc2, B_ref2, _ = model.predict_amplitudes(a_t, omega_t, u_t, v_t)
    R_combo2 = (B_ref2.unsqueeze(-1) * A_u_full * u_up_f +
                B_inc2.unsqueeze(-1) * A_d_full * u_down_f) / P_full
    test_loss2 = torch.mean(torch.abs(R_over_P2 - R_combo2) ** 2) / \
                 (torch.mean(torch.abs(R_over_P2) ** 2) + eps)
    test_loss2.backward()
    optimizer.step()

    rin_changed = any(not torch.allclose(pre_rin[n], p) for n, p in model.rin_decoder.named_parameters())
    amp_changed = any(not torch.allclose(pre_amp[n], p) for n, p in model.amplitude_net.named_parameters())
    enc_changed = any(not torch.allclose(pre_enc[n], p) for n, p in model.encoder.named_parameters())
    up_changed = any(not torch.allclose(pre_up[n], p) for n, p in model.up_decoder.named_parameters())
    down_changed = any(not torch.allclose(pre_down[n], p) for n, p in model.down_decoder.named_parameters())

    check("14a. RinDecoder params changed", rin_changed == train_rin,
          f"expected {train_rin}, got {rin_changed}")
    check("14b. AmplitudeNet params changed", amp_changed == train_amp,
          f"expected {train_amp}, got {amp_changed}")
    check("14c. Encoder unchanged", not enc_changed,
          f"expected False, got {enc_changed}")
    check("14d. Up decoder unchanged", not up_changed,
          f"expected False, got {up_changed}")
    check("14e. Down decoder unchanged", not down_changed,
          f"expected False, got {down_changed}")

    # ---- 15. Rollback checkpoint exists ----
    check("15a. stage1_best_model.pt exists", (bundle_dir / "stage1_best_model.pt").exists())
    check("15b. stage2_best_model.pt exists", (bundle_dir / "stage2_best_model.pt").exists())
    check("15c. spectral_basis_cache.npz exists", cache_path.exists())

    # ---- Summary ----
    metadata_path = bundle_dir / "metadata.json"
    if metadata_path.exists():
        import json
        with open(metadata_path) as f:
            meta = json.load(f)
        check("15d. bundle metadata valid", "stage1_best_model.pt" in str(meta.get("rollback_files", [])))

    print(f"\n{'='*50}")
    print(f"  Results: {passed} PASS, {failed} FAIL")
    if failed > 0:
        print(f"  *** {failed} checks FAILED — review above ***")
        sys.exit(1)
    else:
        print(f"  All {passed} checks passed. Ready for Stage-4 training.")
        sys.exit(0)


if __name__ == "__main__":
    main()
