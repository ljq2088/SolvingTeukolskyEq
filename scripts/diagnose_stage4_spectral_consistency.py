#!/usr/bin/env python3
"""Stage-4 v2 preflight: compute consistency error with CORRECT R_over_P.

Key fix: R_over_P = h2 * S(y), NOT f_R(y).
S(y) = compose_reduced_shape_from_f(f_R, y, c_H)

Also adds error decomposition:
  1. spectral self-consistency
  2. amplitude-net consistency
  3. Stage-1 consistency
  4. actual Stage-4 consistency

Usage:
  python scripts/diagnose_stage4_spectral_consistency.py \
    --pre-stage4-bundle outputs/pre_stage4_bundle/patch_000_logw_v2 \
    --stage2-checkpoint outputs/autoencoder_stage2_amplitude_train/20260515_192625_stage2_amplitude/checkpoints/best_model.pt \
    --spectral-basis-cache outputs/stage4_spectral_basis_cache/patch_000_logw_v2/spectral_basis_cache.npz \
    --config config/autoencoder_stage4_spectral_consistency.yaml \
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
from physical_ansatz.stage1_reconstruction import compute_R_over_P_from_model
from utils.mode import KerrMode
from utils.amplitude import solve_basis_domain


def _get_dtype(dtype_name):
    return torch.float32 if dtype_name == "float32" else torch.float64


def Leaver_P(r, a, omega, m=2, M=1.0, s=-2):
    rp = r_plus(a, M)
    rm = r_minus(a, M)
    sigma_p = (2.0 * omega * rp - m * a) / (rp - rm)
    pp = -s - 1j * sigma_p
    pm = -1 - s + 2j * omega + 1j * sigma_p
    return ((r - rp) ** pp) * ((r - rm) ** pm) * torch.exp(1j * omega * r)


def compute_spectral_Rin_over_P(a, omega, y, lambda_, M=1.0, s=-2, m=2, N_out=80, z_b=0.05):
    """Compute R_in/P directly from spectral method for error decomposition.

    Solves Teukolsky ODE for Rin and returns R_in/P at grid points.
    """
    a_val = float(a.squeeze().cpu().numpy())
    omega_val = float(omega.squeeze().cpu().numpy())
    lam_val = complex(float(lambda_.squeeze().cpu().real.numpy()),
                      float(lambda_.squeeze().cpu().imag.numpy()))

    mode = KerrMode(M=M, a=a_val, omega=omega_val, ell=2, m=m, s=s, lam=lam_val)
    sol = solve_basis_domain(mode, "in", N_out, 0.0, z_b, "right")

    z_grid_np = ((y.squeeze().cpu().numpy() + 1.0) / 2.0)

    # Interpolate spectral Rin to y-grid (mask z=0 to avoid 1/z divergence)
    z_mask = z_grid_np > 1e-12
    u_interp = np.zeros(len(z_grid_np), dtype=np.complex128)
    u_interp[z_mask] = np.interp(z_grid_np[z_mask], sol["z"][::-1], sol["u"][::-1].real) + \
                       1j * np.interp(z_grid_np[z_mask], sol["z"][::-1], sol["u"][::-1].imag)

    return torch.tensor(u_interp, device=y.device, dtype=torch.complex128).unsqueeze(0)


def rel_error(val, ref, eps=1e-12):
    return np.abs(val - ref) / (np.abs(ref) + eps)


def main():
    parser = argparse.ArgumentParser(description="Stage-4 v2 preflight diagnostic")
    parser.add_argument("--pre-stage4-bundle", type=str, required=True)
    parser.add_argument("--stage2-checkpoint", type=str, required=True)
    parser.add_argument("--spectral-basis-cache", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/autoencoder_stage4_spectral_consistency.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, required=True)
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
    a_cache = cache["a"]
    omega_cache = cache["omega"]
    u_cache = cache["u"]
    v_cache = cache["v"]
    lam_cache = cache["lambda_"]
    u_up_cache = cache["u_up"]
    u_down_cache = cache["u_down"]
    y_grid_np = cache["y"]
    z_grid_np = cache["z"]
    n_cache = len(a_cache)
    n_y = len(y_grid_np)
    z_b = float(cache["z_b"])
    N_out = int(cache["N_out"])
    print(f"[preflight-v2] Spectral cache: {n_cache} params, {n_y} y-points, "
          f"y=[{y_grid_np[0]:.4f}, {y_grid_np[-1]:.4f}]")

    # ---- Load Stage-1 model ----
    bundle_dir = Path(args.pre_stage4_bundle)
    stage1_ckpt_path = bundle_dir / "stage1_best_model.pt"

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

    ckpt1 = torch.load(stage1_ckpt_path, map_location="cpu", weights_only=False)
    sd1 = ckpt1.get("model_state_dict", ckpt1)
    missing, unexpected = model.load_state_dict(sd1, strict=False)
    print(f"[preflight-v2] Stage-1: {len(missing)} missing, {len(unexpected)} unexpected keys")

    stage2_ckpt_path = Path(args.stage2_checkpoint)
    ckpt2 = torch.load(stage2_ckpt_path, map_location="cpu", weights_only=False)
    sd2 = ckpt2.get("model_state_dict", ckpt2)
    amp_keys = {k: v for k, v in sd2.items() if k.startswith("amplitude_net.")}
    model.load_state_dict(amp_keys, strict=False)
    print(f"[preflight-v2] Stage-2 AmplitudeNet: {len(amp_keys)} keys")

    model.to(device=device, dtype=dtype)
    model.eval()

    # ---- Evaluate ALL cache points ----
    y_t = torch.tensor(y_grid_np, device=device, dtype=dtype).unsqueeze(0)  # (1, N_y)

    # For spectral error decomposition, compute spectral Rin for a few points
    n_decomp = min(5, n_cache)
    decomp_idx = np.linspace(0, n_cache - 1, n_decomp, dtype=int)

    results = []
    decomp_results = []

    for i in range(n_cache):
        a_val = float(a_cache[i])
        omega_val = float(omega_cache[i])
        u_val = float(u_cache[i])
        v_val = float(v_cache[i])
        lam_val = complex(lam_cache[i])

        a_t = torch.tensor([[a_val]], device=device, dtype=dtype)
        omega_t = torch.tensor([[omega_val]], device=device, dtype=dtype)
        u_t = torch.tensor([[u_val]], device=device, dtype=dtype)
        v_t = torch.tensor([[v_val]], device=device, dtype=dtype)
        lam_t = torch.tensor([[lam_val]], device=device, dtype=cdtype)

        with torch.no_grad():
            # CORRECTED: R_model/P = h2 * S(y)
            R_over_P = compute_R_over_P_from_model(
                model, a_t, omega_t, y_t, lam_t, u_t, v_t,
                m=m_phys, M=M_phys, s=s_phys,
            )  # (1, N_y) complex

            # B_inc, B_ref from AmplitudeNet
            B_inc, B_ref, _ = model.predict_amplitudes(a_t, omega_t, u_t, v_t)

            # r from y
            rp = r_plus(a_t, M_phys)
            x = (y_t + 1.0) / 2.0
            r = rp / x

            A_u = A_up(r, a_t, omega_t, M_phys)
            A_d = A_down(r, a_t, omega_t, M_phys)
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

        results.append({
            "idx": int(i),
            "a": a_val,
            "omega": omega_val,
            "u": u_val,
            "v": v_val,
            "median_rel_err": med_err,
            "max_rel_err": max_err,
        })

        # ---- Error decomposition for subset ----
        if i in decomp_idx:
            with torch.no_grad():
                # 1. Spectral self-consistency: spectral R_in/P vs spectral combo
                R_spec_over_P = compute_spectral_Rin_over_P(
                    a_t, omega_t, y_t, lam_t, M=M_phys, s=s_phys, m=m_phys,
                    N_out=N_out, z_b=z_b,
                )
                # Get spectral B_inc, B_ref
                from utils.amplitude import TeukRadAmplitudeInWithAbelChecks
                mode = KerrMode(M=M_phys, a=a_val, omega=omega_val, ell=2, m=m_phys, s=s_phys, lam=lam_val)
                amp = TeukRadAmplitudeInWithAbelChecks(mode, z_m=z_b, N_out=N_out, N_in=N_out)
                B_inc_spec = amp.B_inc
                B_ref_spec = amp.B_ref

                B_inc_s = torch.tensor([[B_inc_spec]], device=device, dtype=cdtype)
                B_ref_s = torch.tensor([[B_ref_spec]], device=device, dtype=cdtype)
                R_spec_combo = (B_ref_s.unsqueeze(-1) * A_u * u_up_spec +
                               B_inc_s.unsqueeze(-1) * A_d * u_down_spec) / P

                spec_self = rel_error(R_spec_over_P.cpu().numpy().ravel(),
                                      R_spec_combo.cpu().numpy().ravel())

                # 2. Amplitude-net: B_net + spectral u vs spectral R_in/P
                R_amp_combo = (B_ref.unsqueeze(-1) * A_u * u_up_spec +
                              B_inc.unsqueeze(-1) * A_d * u_down_spec) / P
                amp_cons = rel_error(R_amp_combo.cpu().numpy().ravel(),
                                    R_spec_over_P.cpu().numpy().ravel())

                # 3. Stage-1: R_model/P vs spectral R_in/P
                s1_cons = rel_error(R_over_P.cpu().numpy().ravel(),
                                   R_spec_over_P.cpu().numpy().ravel())

                # 4. Actual Stage-4: R_model/P vs B_net*spectral_u/P
                s4_cons = rel_error(R_over_P.cpu().numpy().ravel(),
                                   R_amp_combo.cpu().numpy().ravel())

                decomp_results.append({
                    "idx": int(i),
                    "a": a_val,
                    "omega": omega_val,
                    "spectral_self_consistency_med": float(np.median(spec_self)),
                    "amplitude_net_consistency_med": float(np.median(amp_cons)),
                    "stage1_consistency_med": float(np.median(s1_cons)),
                    "stage4_consistency_med": float(np.median(s4_cons)),
                })

    # ---- Summary ----
    rel_errs_all = np.array([r["median_rel_err"] for r in results])
    max_errs_all = np.array([r["max_rel_err"] for r in results])
    median_all = float(np.median(rel_errs_all))
    max_all = float(np.max(max_errs_all))
    has_nan = bool(np.any(~np.isfinite(rel_errs_all)))

    print(f"\n[preflight-v2] Results ({n_cache} points):")
    print(f"  median_rel_err: {median_all:.4e}")
    print(f"  max_rel_err:    {max_all:.4e}")
    print(f"  NaN/Inf:        {has_nan}")

    # Per-point details
    print(f"\n  {'idx':>4s} {'a':>8s} {'omega':>12s} {'med_err':>10s} {'max_err':>10s}")
    for r in results:
        print(f"  {r['idx']:4d} {r['a']:8.4f} {r['omega']:12.6e} {r['median_rel_err']:10.4e} {r['max_rel_err']:10.4e}")

    # ---- Error decomposition ----
    if decomp_results:
        print(f"\n{'='*80}")
        print("  Error Decomposition (5 sample points)")
        print(f"{'='*80}")
        print(f"  {'idx':>4s} {'a':>8s} {'omega':>12s} {'spec_self':>10s} {'amp_net':>10s} {'stage1':>10s} {'stage4':>10s}")
        for d in decomp_results:
            print(f"  {d['idx']:4d} {d['a']:8.4f} {d['omega']:12.6e} "
                  f"{d['spectral_self_consistency_med']:10.4e} {d['amplitude_net_consistency_med']:10.4e} "
                  f"{d['stage1_consistency_med']:10.4e} {d['stage4_consistency_med']:10.4e}")

    # ---- Save ----
    summary = {
        "version": "v2",
        "note": "CORRECTED: R_over_P = h2 * S(y), NOT f_R(y)",
        "pre_stage4_bundle": str(bundle_dir.resolve()),
        "stage2_checkpoint": str(stage2_ckpt_path.resolve()),
        "spectral_basis_cache": str(Path(args.spectral_basis_cache).resolve()),
        "n_points": int(n_cache),
        "n_y": int(n_y),
        "y_range": [float(y_grid_np[0]), float(y_grid_np[-1])],
        "median_rel_err": median_all,
        "max_rel_err": max_all,
        "has_nan_inf": has_nan,
        "per_point": results,
        "error_decomposition": decomp_results,
        "verdict": "GOOD" if (median_all < 1.0 and not has_nan) else "NEEDS_IMPROVEMENT",
    }
    with open(out_dir / "preflight_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[preflight-v2] Saved to {out_dir / 'preflight_summary.json'}")
    print(f"[preflight-v2] Verdict: {summary['verdict']}")

    if decomp_results:
        s1_med = float(np.median([d["stage1_consistency_med"] for d in decomp_results]))
        amp_med = float(np.median([d["amplitude_net_consistency_med"] for d in decomp_results]))
        spec_med = float(np.median([d["spectral_self_consistency_med"] for d in decomp_results]))
        print(f"\n[preflight-v2] Bottleneck analysis:")
        print(f"  spectral self-consistency: {spec_med:.4e}")
        print(f"  amplitude-net consistency: {amp_med:.4e}")
        print(f"  Stage-1 PINN consistency:  {s1_med:.4e}")
        if s1_med > amp_med and s1_med > spec_med:
            print(f"  => Primary bottleneck: Stage-1 Rin PINN")
        elif amp_med > s1_med:
            print(f"  => Primary bottleneck: AmplitudeNet")
        else:
            print(f"  => Bottleneck: spectral self-consistency (check cache)")


if __name__ == "__main__":
    main()
