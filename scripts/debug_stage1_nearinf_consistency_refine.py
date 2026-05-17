#!/usr/bin/env python3
"""Debug checks for Stage-1 near-infinity consistency refinement.

Verifies:
  1. Stage-1 checkpoint loads correctly
  2. Stage-2 AmplitudeNet checkpoint loads correctly
  3. Spectral cache loads correctly
  4. R_model/P = h2 * S_model computes correctly
  5. PDE residual is finite
  6. Consistency loss is finite
  7. Drift loss is finite
  8. Only RinDecoder has requires_grad=True
  9. Encoder has no gradients
  10. AmplitudeNet has no gradients
  11. Up/Down decoders have no gradients
  12. optimizer.step() changes only RinDecoder
  13. No NaN/Inf in any computation

Usage:
  python scripts/debug_stage1_nearinf_consistency_refine.py \
    --config config/autoencoder_stage1_nearinf_consistency_refine.yaml \
    --device cuda
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import load_pinn_full_config
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.mapping import r_plus, r_minus
from physical_ansatz.prefactor import Leaver_prefactors
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


def check(name, condition, fatal=False):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    if not condition and fatal:
        raise RuntimeError(f"FATAL: {name}")
    return condition


def main():
    parser = argparse.ArgumentParser(description="Debug Stage-1 near-inf consistency refine")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
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
    samp_cfg = ref_cfg.get("sampling", {})

    M_phys = float(physics_cfg.get("problem", {}).get("M", 1.0))
    s_phys = int(physics_cfg.get("problem", {}).get("s", -2))
    m_phys = int(physics_cfg.get("problem", {}).get("m", 2))

    stage1_ckpt = Path(ref_cfg["stage1_checkpoint"])
    stage2_ckpt = Path(ref_cfg["stage2_checkpoint"])
    spec_cache_path = Path(ref_cfg["spectral_basis_cache"])

    print("=" * 60)
    print("Stage-1 Near-Inf Consistency Refine — Debug Checks")
    print("=" * 60)

    # ---- Check 1: Stage-1 checkpoint loads ----
    print("\n1. Stage-1 checkpoint:")
    check("file exists", stage1_ckpt.exists(), fatal=True)
    ckpt1 = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
    sd1 = ckpt1.get("model_state_dict", ckpt1)
    check(f"state_dict has {len(sd1)} keys", len(sd1) > 0)

    # ---- Check 2: Stage-2 checkpoint loads ----
    print("\n2. Stage-2 AmplitudeNet checkpoint:")
    check("file exists", stage2_ckpt.exists(), fatal=True)
    ckpt2 = torch.load(stage2_ckpt, map_location="cpu", weights_only=False)
    sd2 = ckpt2.get("model_state_dict", ckpt2)
    amp_keys = [k for k in sd2.keys() if k.startswith("amplitude_net.")]
    check(f"amplitude_net keys: {len(amp_keys)}", len(amp_keys) > 0)
    # Check B values are not ~1
    if "amplitude_net.inc_token" in sd2:
        print(f"    inc_token norm: {sd2['amplitude_net.inc_token'].norm().item():.4f}")

    # ---- Check 3: Spectral cache loads ----
    print("\n3. Spectral cache:")
    check("file exists", spec_cache_path.exists(), fatal=True)
    cache = np.load(spec_cache_path)
    for key in ["a", "omega", "u", "v", "lambda_", "u_up", "u_down", "y"]:
        check(f"has '{key}'", key in cache)
    n_cache = len(cache["a"])
    check(f"n_points={n_cache}", n_cache > 0)
    for i in range(min(3, n_cache)):
        check(f"  point {i}: u_up finite", np.all(np.isfinite(cache["u_up"][i])))
        check(f"  point {i}: u_down finite", np.all(np.isfinite(cache["u_down"][i])))

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

    sd1_f = {k: v for k, v in sd1.items() if not k.startswith("amplitude_net.")}
    model.load_state_dict(sd1_f, strict=False)
    amp_kv = {k: v for k, v in sd2.items() if k.startswith("amplitude_net.")}
    model.load_state_dict(amp_kv, strict=False)
    model.to(device=device, dtype=dtype)

    # ---- Check 4: R_model/P computes ----
    print("\n4. R_model/P computation:")
    y_t = torch.tensor(cache["y"], device=device, dtype=dtype).unsqueeze(0)
    n_y = y_t.shape[1]
    a_t = torch.tensor([[float(cache["a"][0])]], device=device, dtype=dtype)
    omega_t = torch.tensor([[float(cache["omega"][0])]], device=device, dtype=dtype)
    lam_t = torch.tensor([[complex(cache["lambda_"][0])]], device=device, dtype=cdtype)
    u_t = torch.tensor([[float(cache["u"][0])]], device=device, dtype=dtype)
    v_t = torch.tensor([[float(cache["v"][0])]], device=device, dtype=dtype)

    with torch.no_grad():
        R_op = compute_R_over_P_from_model(model, a_t, omega_t, y_t, lam_t, u_t, v_t,
                                           m=m_phys, M=M_phys, s=s_phys)
    check(f"R_model/P shape correct ({R_op.shape})", R_op.shape[1] == n_y)
    check("R_model/P finite", torch.all(torch.isfinite(R_op)).item())
    check("R_model/P not all zeros", torch.max(torch.abs(R_op)) > 0)

    # ---- Check 5: PDE residual finite ----
    print("\n5. PDE residual:")
    from scripts.train_stage1_nearinf_consistency_refine import compute_pde_residual
    pde_res = compute_pde_residual(model, a_t, omega_t, y_t, lam_t, physics_cfg,
                                   u_b=u_t, v_b=v_t)
    check("PDE residual finite", torch.all(torch.isfinite(pde_res)).item())
    L_pde = pde_res.mean()
    check(f"L_pde = {L_pde.item():.6e}", L_pde.item() > 0 and np.isfinite(L_pde.item()))

    # ---- Check 6: Consistency loss ----
    print("\n6. Consistency loss:")
    u_up_b = torch.tensor(cache["u_up"][0], device=device, dtype=cdtype).unsqueeze(0)
    u_down_b = torch.tensor(cache["u_down"][0], device=device, dtype=cdtype).unsqueeze(0)

    with torch.no_grad():
        B_inc, B_ref, _ = model.predict_amplitudes(a_t, omega_t, u_t, v_t)
        rp = r_plus(a_t, M_phys)
        x = (y_t + 1.0) / 2.0
        r = rp / x
        A_u = A_up(r, a_t, omega_t, M_phys)
        A_d = A_down(r, a_t, omega_t, M_phys)
        P = Leaver_P(r, a_t, omega_t, m=m_phys, M=M_phys, s=s_phys)
        R_combo = (B_ref.unsqueeze(-1) * A_u * u_up_b.to(dtype=r.dtype) +
                   B_inc.unsqueeze(-1) * A_d * u_down_b.to(dtype=r.dtype)) / P
    check("R_combo/P finite", torch.all(torch.isfinite(R_combo)).item())

    eps = float(loss_cfg.get("eps", 1e-12))
    L_cons = torch.mean(torch.abs(R_op - R_combo) ** 2 / (torch.abs(R_op) ** 2 + eps))
    check(f"L_cons = {L_cons.item():.6e}", L_cons.item() > 0 and np.isfinite(L_cons.item()))

    # ---- Check 7: Drift loss ----
    print("\n7. Drift loss:")
    frozen_R = R_op.clone()
    L_drift = torch.mean(torch.abs(R_op - frozen_R) ** 2 / (torch.abs(frozen_R) ** 2 + eps))
    check(f"L_drift (vs self) = {L_drift.item():.6e}", L_drift.item() < 1e-10)

    # ---- Checks 8-11: Freeze policy ----
    print("\n8-11. Freeze policy:")
    for name, p in model.named_parameters():
        prefix = name.split(".")[0]
        if prefix == "rin_decoder":
            check(f"  {name}: requires_grad=True", p.requires_grad, fatal=True)
        else:
            check(f"  {name}: requires_grad=False", not p.requires_grad)

    # ---- Check 12: optimizer.step() only changes RinDecoder ----
    print("\n12. Optimizer step isolation:")
    model.eval()
    model.rin_decoder.train()
    optimizer = torch.optim.Adam(model.rin_decoder.parameters(), lr=1e-7)

    # Capture pre-step params
    pre_params = {}
    for name, p in model.named_parameters():
        pre_params[name] = p.clone()

    # Forward/backward on a tiny test
    with torch.enable_grad():
        y_test = torch.tensor([[-0.999, -0.99, -0.98, -0.97]], device=device, dtype=dtype)
        f_test = model.forward(a_t, omega_t, y_test, u=u_t, v=v_t)
        loss_test = torch.mean(torch.abs(f_test) ** 2)
        loss_test.backward()
        optimizer.step()
        optimizer.zero_grad()

    for name, p in model.named_parameters():
        changed = not torch.allclose(pre_params[name], p, atol=1e-30)
        prefix = name.split(".")[0]
        if prefix == "rin_decoder":
            check(f"  {name}: changed (expected)", changed)
        else:
            check(f"  {name}: unchanged (expected)", not changed)

    # ---- Check 13: No NaN/Inf ----
    print("\n13. NaN/Inf check:")
    has_nan = False
    for name, p in model.named_parameters():
        if torch.any(~torch.isfinite(p)):
            print(f"  [FAIL] {name} has NaN/Inf")
            has_nan = True
    if not has_nan:
        check("All parameters finite", True)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("Debug checks complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
