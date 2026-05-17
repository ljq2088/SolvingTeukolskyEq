#!/usr/bin/env python3
"""Preflight diagnostic for Stage-4: compute consistency error WITHOUT training.

Checks whether R_model (Rin decoder) is consistent with:
  R_combo = B_ref * A_up * u_up_spec + B_inc * A_down * u_down_spec

Reports consistency metrics but does NOT update any model parameters.

Usage:
  python scripts/diagnose_stage4_spectral_consistency.py \
    --pre-stage4-bundle outputs/pre_stage4_bundle/patch_000_logw_v2 \
    --stage2-checkpoint outputs/autoencoder_stage2_amplitude_train/20260515_192625_stage2_amplitude/checkpoints/best_model.pt \
    --spectral-basis-cache outputs/stage4_spectral_basis_cache/patch_000_logw_v2/spectral_basis_cache.npz \
    --config config/autoencoder_stage3_ubasis_refine.yaml \
    --device cuda \
    --output-dir outputs/stage4_spectral_preflight/patch_000_logw_v2
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from config.config_loader import load_pinn_full_config
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.mapping import r_plus, r_minus
from physical_ansatz.prefactor import r_star
from physical_ansatz.u_basis import A_up, A_down


def _get_dtype(dtype_name):
    return torch.float32 if dtype_name == "float32" else torch.float64


def Leaver_P(r, a, omega, m=2, M=1.0, s=-2):
    """Leaver prefactor P(r) for Teukolsky equation."""
    rp = r_plus(a, M)
    rm = r_minus(a, M)
    sigma_p = (2.0 * omega * rp - m * a) / (rp - rm)
    pp = -s - 1j * sigma_p
    pm = -1 - s + 2j * omega + 1j * sigma_p
    return ((r - rp) ** pp) * ((r - rm) ** pm) * torch.exp(1j * omega * r)


def main():
    parser = argparse.ArgumentParser(description="Stage-4 preflight diagnostic")
    parser.add_argument("--pre-stage4-bundle", type=str, required=True)
    parser.add_argument("--stage2-checkpoint", type=str, required=True)
    parser.add_argument("--spectral-basis-cache", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/autoencoder_stage3_ubasis_refine.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-param-eval", type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load config ----
    full_cfg = load_pinn_full_config(args.config)
    train_cfg = full_cfg.get("train", {})
    runtime_cfg = train_cfg.get("runtime", {})
    dtype_name = runtime_cfg.get("dtype", "float64")
    dtype = _get_dtype(dtype_name)
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    physics_cfg = full_cfg.get("physics", {})
    M_phys = float(physics_cfg.get("problem", {}).get("M", 1.0))
    s_phys = int(physics_cfg.get("problem", {}).get("s", -2))
    m_phys = int(physics_cfg.get("problem", {}).get("m", 2))

    # ---- Load spectral cache ----
    cache = np.load(args.spectral_basis_cache)
    y_grid_np = cache["y"]
    z_grid_np = cache["z"]
    u_up_cache = cache["u_up"]
    u_down_cache = cache["u_down"]
    a_cache = cache["a"]
    omega_cache = cache["omega"]
    u_cache = cache["u"]
    v_cache = cache["v"]
    lam_cache = cache["lambda_"]
    n_cache = len(a_cache)
    n_y = len(y_grid_np)
    print(f"[preflight] Spectral cache: {n_cache} params, {n_y} y-points, "
          f"y=[{y_grid_np[0]:.4f}, {y_grid_np[-1]:.4f}]")

    # ---- Load Stage-1 model from pre-stage4 bundle ----
    bundle_dir = Path(args.pre_stage4_bundle)
    stage1_ckpt_path = bundle_dir / "stage1_best_model.pt"
    if not stage1_ckpt_path.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {stage1_ckpt_path}")

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
        decoder_n_hidden=int(model_cfg.get("decoder_n_hidden", 0)),
    )

    # Load stage-1 weights
    ckpt1 = torch.load(stage1_ckpt_path, map_location="cpu", weights_only=False)
    sd1 = ckpt1.get("model_state_dict", ckpt1)
    missing, unexpected = model.load_state_dict(sd1, strict=False)
    print(f"[preflight] Stage-1 model: {len(missing)} missing, {len(unexpected)} unexpected keys")

    # Load stage-2 amplitude weights
    stage2_ckpt_path = Path(args.stage2_checkpoint)
    ckpt2 = torch.load(stage2_ckpt_path, map_location="cpu", weights_only=False)
    sd2 = ckpt2.get("model_state_dict", ckpt2)
    amp_keys = {k: v for k, v in sd2.items() if k.startswith("amplitude_net.")}
    model.load_state_dict(amp_keys, strict=False)
    print(f"[preflight] Stage-2 AmplitudeNet: loaded {len(amp_keys)} keys")

    model.to(device=device, dtype=dtype)
    model.eval()

    # ---- Evaluate on sample of cache points ----
    n_eval = min(args.n_param_eval, n_cache)
    idx_eval = np.linspace(0, n_cache - 1, n_eval, dtype=int)

    y_t = torch.tensor(y_grid_np, device=device, dtype=dtype).unsqueeze(0)  # (1, N_y)

    results = []
    all_rel_errs = []

    for i in idx_eval:
        a_val = float(a_cache[i])
        omega_val = float(omega_cache[i])
        u_val = float(u_cache[i])
        v_val = float(v_cache[i])

        a_t = torch.tensor([[a_val]], device=device, dtype=dtype)
        omega_t = torch.tensor([[omega_val]], device=device, dtype=dtype)
        u_t = torch.tensor([[u_val]], device=device, dtype=dtype)
        v_t = torch.tensor([[v_val]], device=device, dtype=dtype)

        with torch.no_grad():
            # R_model / P = f_R(y) from Rin decoder
            R_over_P = model.predict_Rin(a_t, omega_t, y_t, u_t, v_t)  # (1, N_y) complex

            # B_inc, B_ref from AmplitudeNet
            B_inc, B_ref, _ = model.predict_amplitudes(a_t, omega_t, u_t, v_t)

            # Compute r from y for A_up, A_down, P
            rp = r_plus(a_t, M_phys)
            x = (y_t + 1.0) / 2.0
            r = rp / x

            A_u = A_up(r, a_t, omega_t, M_phys)  # (1, N_y)
            A_d = A_down(r, a_t, omega_t, M_phys)  # (1, N_y)
            P = Leaver_P(r, a_t, omega_t, m=m_phys, M=M_phys, s=s_phys)

            # Spectral u
            u_up_spec = torch.tensor(u_up_cache[i], device=device, dtype=cdtype).unsqueeze(0)
            u_down_spec = torch.tensor(u_down_cache[i], device=device, dtype=cdtype).unsqueeze(0)

            # R_combo / P
            B_ref_exp = B_ref.unsqueeze(-1)
            B_inc_exp = B_inc.unsqueeze(-1)
            R_combo_over_P = (B_ref_exp * A_u * u_up_spec + B_inc_exp * A_d * u_down_spec) / P

            # Relative error
            rel_err = torch.abs(R_over_P - R_combo_over_P) / (torch.abs(R_over_P) + 1e-12)
            rel_err_np = rel_err.cpu().numpy().ravel()

        med_err = float(np.median(rel_err_np))
        max_err = float(np.max(rel_err_np))
        all_rel_errs.extend(rel_err_np.tolist())

        results.append({
            "idx": int(i),
            "a": a_val,
            "omega": omega_val,
            "u": u_val,
            "v": v_val,
            "median_rel_err": med_err,
            "max_rel_err": max_err,
        })

    # ---- Summary ----
    all_rel = np.array(all_rel_errs)
    median_all = float(np.median(all_rel))
    max_all = float(np.max(all_rel))
    has_nan = bool(np.any(~np.isfinite(all_rel)))

    print(f"\n[preflight] Results ({n_eval} points):")
    print(f"  median_rel_err: {median_all:.4e}")
    print(f"  max_rel_err:    {max_all:.4e}")
    print(f"  NaN/Inf:        {has_nan}")

    # Per-point details
    print(f"\n  {'idx':>4s} {'a':>8s} {'omega':>12s} {'med_err':>10s} {'max_err':>10s}")
    for r in results:
        print(f"  {r['idx']:4d} {r['a']:8.4f} {r['omega']:12.6e} {r['median_rel_err']:10.4e} {r['max_rel_err']:10.4e}")

    # ---- Save ----
    summary = {
        "pre_stage4_bundle": str(bundle_dir.resolve()),
        "stage2_checkpoint": str(stage2_ckpt_path.resolve()),
        "spectral_basis_cache": str(Path(args.spectral_basis_cache).resolve()),
        "n_param_eval": int(n_eval),
        "n_y": int(n_y),
        "y_range": [float(y_grid_np[0]), float(y_grid_np[-1])],
        "median_rel_err": median_all,
        "max_rel_err": max_all,
        "has_nan_inf": has_nan,
        "per_point": results,
        "verdict": "GOOD" if (median_all < 1.0 and not has_nan) else "NEEDS_IMPROVEMENT",
    }
    with open(out_dir / "preflight_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[preflight] Saved to {out_dir / 'preflight_summary.json'}")
    print(f"[preflight] Verdict: {summary['verdict']}")


if __name__ == "__main__":
    main()
