#!/usr/bin/env python3
"""Oracle-B bottleneck diagnosis with pybhpt benchmark and visualization.

Three-way comparison at near-infinity (y close to -1):
  A. Stage-1 network:  R_model/P = h2 * S_RinDecoder
  B. pybhpt benchmark: R_pybhpt/P (numerical ODE reference)
  C. AmpNet + spectral: (B_ref_net*u_up_spec*A_up + B_inc_net*u_down_spec*A_down)/P

Key question: is Stage-1 RinDecoder the bottleneck, or is AmplitudeNet still limiting?

Usage:
  python scripts/diagnose_stage1_nearinf_bottleneck.py \
    --pre-stage4-bundle outputs/pre_stage4_bundle/patch_000_logw_v2 \
    --stage2-checkpoint outputs/autoencoder_stage2_absolute_amplitude_train/20260517_182923_stage2_absolute_amp/checkpoints/best_model.pt \
    --spectral-cache outputs/stage4_spectral_basis_cache/patch_000_logw_v2/spectral_basis_cache.npz \
    --config config/autoencoder_stage4_spectral_consistency.yaml \
    --device cuda \
    --output-dir outputs/stage1_nearinf_bottleneck_diagnosis/patch_000_logw_v2
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import load_pinn_full_config
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.mapping import r_plus, r_minus
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives
from physical_ansatz.u_basis import A_up, A_down
from physical_ansatz.stage1_reconstruction import compute_R_over_P_from_model
from utils.amplitude import TeukRadAmplitudeInWithAbelChecks
from utils.mode import KerrMode
from pybhpt_usage.compute_solution import compute_pybhpt_solution


def _get_dtype(dtype_name):
    return torch.float32 if dtype_name == "float32" else torch.float64


def Leaver_P(r, a, omega, m=2, M=1.0, s=-2):
    rp = r_plus(a, M)
    rm = r_minus(a, M)
    sigma_p = (2.0 * omega * rp - m * a) / (rp - rm)
    pp = -s - 1j * sigma_p
    pm = -1 - s + 2j * omega + 1j * sigma_p
    return ((r - rp) ** pp) * ((r - rm) ** pm) * torch.exp(1j * omega * r)


def rel_error(val, ref, eps=1e-12):
    return np.abs(val - ref) / (np.abs(ref) + eps)


def compute_R_over_P_pybhpt(a_val, omega_val, y_np, m=2, M=1.0, s=-2, timeout=15.0):
    """Compute R_in/P from pybhpt on the same y-grid."""
    rp = 1.0 + np.sqrt(1.0 - a_val * a_val)
    x_np = (y_np + 1.0) / 2.0
    r_np = rp / np.clip(x_np, 1e-12, None)
    r_sorted = np.sort(r_np)
    sort_idx = np.argsort(r_np)

    try:
        r_vals, R_in = compute_pybhpt_solution(a_val, omega_val, ell=2, m=m,
                                                r_grid=r_sorted, timeout=timeout)
    except Exception as e:
        return None

    R_in_interp = np.interp(r_np, r_vals, R_in.real) + 1j * np.interp(r_np, r_vals, R_in.imag)

    # Compute P on the same grid
    a_t = torch.tensor([a_val], dtype=torch.float64)
    omega_t = torch.tensor([omega_val], dtype=torch.float64)
    r_t = torch.tensor(r_np, dtype=torch.float64)
    rp_t = r_plus(a_t, M)
    rm_t = r_minus(a_t, M)
    P_np, _, _ = Leaver_prefactors(r_t, a_t, omega_t, m=m, M=M, s=s, rp=rp_t, rm=rm_t)
    P_np = P_np.numpy()

    R_over_P = R_in_interp / P_np
    return R_over_P


def compute_R_over_P_ampnet_spectral(model, a_t, omega_t, u_t, v_t, y_t, u_up_spec, u_down_spec,
                                      m=2, M=1.0, s=-2):
    """Compute AmpNet+spectral combo: (B_ref*u_up*A_up + B_inc*u_down*A_down)/P."""
    with torch.no_grad():
        B_inc, B_ref, _ = model.predict_amplitudes(a_t, omega_t, u_t, v_t)
        rp = r_plus(a_t, M)
        x = (y_t + 1.0) / 2.0
        r = rp / x
        A_u = A_up(r, a_t, omega_t, M)
        A_d = A_down(r, a_t, omega_t, M)
        P = Leaver_P(r, a_t, omega_t, m=m, M=M, s=s)
        R_combo = (B_ref.unsqueeze(-1) * A_u * u_up_spec +
                   B_inc.unsqueeze(-1) * A_d * u_down_spec) / P
    return R_combo.squeeze(0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Oracle-B bottleneck diagnosis with pybhpt")
    parser.add_argument("--pre-stage4-bundle", type=str, required=True)
    parser.add_argument("--stage2-checkpoint", type=str, required=True)
    parser.add_argument("--spectral-cache", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/autoencoder_stage4_spectral_consistency.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--pybhpt-max-points", type=int, default=10,
                        help="Max points for pybhpt evaluation (slow 10-15s per point)")
    parser.add_argument("--pybhpt-timeout", type=float, default=20.0)
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
    cache = np.load(args.spectral_cache)
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
    print(f"[diagnose] Spectral cache: {n_cache} params, {n_y} y-points, "
          f"y=[{y_grid_np[0]:.4f}, {y_grid_np[-1]:.4f}]")

    # Define near-infinity region
    nearinf_mask = y_grid_np >= -0.999
    nearinf_mask &= y_grid_np <= -0.95
    n_nearinf = int(np.sum(nearinf_mask))
    print(f"[diagnose] Near-infinity region: y=[-0.999, -0.95], {n_nearinf} points")

    # ---- Load Stage-1 model ----
    bundle_dir = Path(args.pre_stage4_bundle)
    stage1_ckpt_path = bundle_dir / "stage1_best_model.pt"
    if not stage1_ckpt_path.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {stage1_ckpt_path}")

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
    sd1_filtered = {k: v for k, v in sd1.items() if not k.startswith("amplitude_net.")}
    missing, unexpected = model.load_state_dict(sd1_filtered, strict=False)
    print(f"[diagnose] Stage-1: {len(missing)} missing, {len(unexpected)} unexpected")

    # Load Stage-2 AmplitudeNet
    stage2_ckpt_path = Path(args.stage2_checkpoint)
    if not stage2_ckpt_path.exists():
        raise FileNotFoundError(f"Stage-2 checkpoint not found: {stage2_ckpt_path}")
    ckpt2 = torch.load(stage2_ckpt_path, map_location="cpu", weights_only=False)
    sd2 = ckpt2.get("model_state_dict", ckpt2)
    amp_keys = {k: v for k, v in sd2.items() if k.startswith("amplitude_net.")}
    n_amp = len(amp_keys)
    model.load_state_dict(amp_keys, strict=False)
    print(f"[diagnose] Stage-2 AmplitudeNet: {n_amp} keys loaded")

    model.to(device=device, dtype=dtype)
    model.eval()

    # ---- Evaluate all points for Stage-1 vs AmpNet+spectral ----
    y_t = torch.tensor(y_grid_np, device=device, dtype=dtype).unsqueeze(0)

    results_all = []
    # Select pybhpt evaluation points (spread across parameter range)
    n_pybhpt = min(args.pybhpt_max_points, n_cache)
    pybhpt_idx = np.linspace(0, n_cache - 1, n_pybhpt, dtype=int)

    print(f"\n[diagnose] Evaluating {n_cache} points (Stage-1 + AmpNet) ...")
    print(f"[diagnose] pybhpt on {n_pybhpt} selected points ...")

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

        # Get spectral u_up/u_down
        u_up_spec = torch.tensor(u_up_cache[i], device=device, dtype=cdtype).unsqueeze(0)
        u_down_spec = torch.tensor(u_down_cache[i], device=device, dtype=cdtype).unsqueeze(0)

        with torch.no_grad():
            # A. Stage-1: R_model/P
            R_over_P_model = compute_R_over_P_from_model(
                model, a_t, omega_t, y_t, lam_t, u_t, v_t,
                m=m_phys, M=M_phys, s=s_phys,
            )  # (1, N_y) complex
            R_model_np = R_over_P_model.squeeze(0).cpu().numpy()

        # C. AmpNet + spectral u
        R_ampnet_np = compute_R_over_P_ampnet_spectral(
            model, a_t, omega_t, u_t, v_t, y_t,
            u_up_spec, u_down_spec,
            m=m_phys, M=M_phys, s=s_phys,
        )

        # Error metrics: Stage-1 vs AmpNet+spectral
        err_model_vs_ampnet = rel_error(R_model_np, R_ampnet_np)
        err_model_vs_ampnet_nearinf = err_model_vs_ampnet[nearinf_mask]

        # Error metrics for AmpNet vs spectral B
        with torch.no_grad():
            B_inc, B_ref, _ = model.predict_amplitudes(a_t, omega_t, u_t, v_t)
            B_inc_np = B_inc.squeeze().cpu().numpy()
            B_ref_np = B_ref.squeeze().cpu().numpy()
        mode = KerrMode(M=M_phys, a=a_val, omega=omega_val, ell=2, m=m_phys, s=s_phys, lam=lam_val)
        try:
            amp = TeukRadAmplitudeInWithAbelChecks(mode, z_m=float(cache["z_b"]),
                                                    N_out=int(cache["N_out"]), N_in=int(cache["N_out"]))
            B_inc_spec = complex(amp.B_inc)
            B_ref_spec = complex(amp.B_ref)
        except Exception:
            B_inc_spec = B_ref_spec = np.nan

        entry = {
            "idx": int(i),
            "a": a_val,
            "omega": omega_val,
            "median_rel_err_all": float(np.median(err_model_vs_ampnet)),
            "median_rel_err_nearinf": float(np.median(err_model_vs_ampnet_nearinf)),
            "max_rel_err_nearinf": float(np.max(err_model_vs_ampnet_nearinf)),
            "B_inc_net": complex(B_inc_np),
            "B_ref_net": complex(B_ref_np),
            "B_inc_spec": B_inc_spec,
            "B_ref_spec": B_ref_spec,
        }

        # B. pybhpt for selected points
        if i in pybhpt_idx:
            print(f"  [{i+1}/{n_cache}] pybhpt: a={a_val:.4f} omega={omega_val:.6e} ...", end=" ")
            R_pybhpt_np = compute_R_over_P_pybhpt(
                a_val, omega_val, y_grid_np,
                m=m_phys, M=M_phys, s=s_phys, timeout=args.pybhpt_timeout,
            )
            if R_pybhpt_np is not None:
                err_model_vs_pybhpt = rel_error(R_model_np, R_pybhpt_np)
                err_ampnet_vs_pybhpt = rel_error(R_ampnet_np, R_pybhpt_np)
                entry["pybhpt_median_err_model"] = float(np.median(err_model_vs_pybhpt))
                entry["pybhpt_median_err_ampnet"] = float(np.median(err_ampnet_vs_pybhpt))
                entry["pybhpt_median_err_model_nearinf"] = float(np.median(err_model_vs_pybhpt[nearinf_mask]))
                entry["pybhpt_median_err_ampnet_nearinf"] = float(np.median(err_ampnet_vs_pybhpt[nearinf_mask]))
                entry["pybhpt_R_over_P"] = R_pybhpt_np.tolist()  # save for plotting later
                entry["R_model_np"] = R_model_np.tolist()
                entry["R_ampnet_np"] = R_ampnet_np.tolist()
                print(f"model_err={entry['pybhpt_median_err_model']:.4e} "
                      f"ampnet_err={entry['pybhpt_median_err_ampnet']:.4e}")
            else:
                print("FAILED")
                entry["pybhpt_median_err_model"] = None
                entry["pybhpt_median_err_ampnet"] = None

        results_all.append(entry)

    # ---- Summary statistics ----
    nearinf_errs = [r["median_rel_err_nearinf"] for r in results_all]
    nearinf_med = float(np.median(nearinf_errs))

    pybhpt_model_errs = [r["pybhpt_median_err_model"] for r in results_all
                         if r.get("pybhpt_median_err_model") is not None]
    pybhpt_ampnet_errs = [r["pybhpt_median_err_ampnet"] for r in results_all
                          if r.get("pybhpt_median_err_ampnet") is not None]

    print(f"\n[diagnose] {'='*60}")
    print(f"[diagnose] Summary ({n_cache} points)")
    print(f"[diagnose] Stage-1 vs AmpNet+spectral (near-inf): median={nearinf_med:.4e}")

    if pybhpt_model_errs:
        print(f"[diagnose] Stage-1 vs pybhpt:             median={np.median(pybhpt_model_errs):.4e}")
    if pybhpt_ampnet_errs:
        print(f"[diagnose] AmpNet+spectral vs pybhpt:     median={np.median(pybhpt_ampnet_errs):.4e}")

    # Determine bottleneck
    print(f"\n[diagnose] Bottleneck analysis:")
    if pybhpt_model_errs:
        s1_vs_pybhpt = np.median(pybhpt_model_errs)
        amp_vs_pybhpt = np.median(pybhpt_ampnet_errs)
        print(f"  Stage-1 vs pybhpt:        {s1_vs_pybhpt:.4e}")
        print(f"  AmpNet+spectral vs pybhpt: {amp_vs_pybhpt:.4e}")
        if s1_vs_pybhpt > 5 * amp_vs_pybhpt:
            print(f"  => VERDICT: Stage-1 RinDecoder is the PRIMARY bottleneck")
        elif amp_vs_pybhpt > 5 * s1_vs_pybhpt:
            print(f"  => VERDICT: AmplitudeNet is the PRIMARY bottleneck")
        else:
            print(f"  => VERDICT: Both contribute comparably")

    # ---- Visualization ----
    print(f"\n[diagnose] Generating plots ...")
    n_plots = min(6, n_pybhpt)
    plot_idx = [pybhpt_idx[j] for j in range(n_plots)]

    fig, axes = plt.subplots(2, n_plots, figsize=(4 * n_plots, 8))
    if n_plots == 1:
        axes = axes.reshape(-1, 1)

    for j, pi in enumerate(plot_idx):
        r = results_all[pi]
        ax_abs = axes[0, j]
        ax_phase = axes[1, j]

        y_plot = y_grid_np
        R_model = np.array(r.get("R_model_np", np.zeros(len(y_plot))))
        R_ampnet = np.array(r.get("R_ampnet_np", np.zeros(len(y_plot))))
        R_pybhpt = np.array(r.get("pybhpt_R_over_P", np.zeros(len(y_plot))))

        # Focus on near-inf
        y_mask = y_plot >= -0.999

        ax_abs.semilogy(y_plot[y_mask], np.abs(R_model[y_mask]), label="Stage-1", linewidth=1)
        ax_abs.semilogy(y_plot[y_mask], np.abs(R_ampnet[y_mask]), label="AmpNet+Spectral", linewidth=1)
        if r.get("pybhpt_R_over_P") is not None:
            ax_abs.semilogy(y_plot[y_mask], np.abs(R_pybhpt[y_mask]), 'k--', label="pybhpt", linewidth=1)
        ax_abs.axvline(x=-0.95, color='gray', linestyle=':', alpha=0.5)
        ax_abs.set_title(f"a={r['a']:.3f} ω={r['omega']:.2e}")
        ax_abs.set_ylabel("|R_in/P|")
        ax_abs.legend(fontsize=7)
        ax_abs.grid(True, alpha=0.3)

        phase_model = np.angle(R_model[y_mask])
        phase_ampnet = np.angle(R_ampnet[y_mask])
        ax_phase.plot(y_plot[y_mask], phase_model, linewidth=1)
        ax_phase.plot(y_plot[y_mask], phase_ampnet, linewidth=1)
        if r.get("pybhpt_R_over_P") is not None:
            ax_phase.plot(y_plot[y_mask], np.angle(R_pybhpt[y_mask]), 'k--', linewidth=1)
        ax_phase.axvline(x=-0.95, color='gray', linestyle=':', alpha=0.5)
        ax_phase.set_xlabel("y")
        ax_phase.set_ylabel("Phase(R_in/P)")
        ax_phase.grid(True, alpha=0.3)

    fig.suptitle("Near-Infinity R_in/P Comparison: Stage-1 vs AmpNet+Spectral vs pybhpt",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    plot_path = out_dir / "nearinf_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[diagnose] Saved {plot_path}")

    # ---- Per-point error plot ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    ax0 = axes2[0]
    omegas = np.array([r["omega"] for r in results_all])
    nearinf_errs_all = np.array(nearinf_errs)
    scatter0 = ax0.scatter(omegas, nearinf_errs_all, c=np.log10(omegas), cmap='viridis', s=30)
    ax0.set_xscale('log')
    ax0.set_xlabel("omega")
    ax0.set_ylabel("median rel err (near-inf)")
    ax0.set_title("Stage-1 vs AmpNet+spectral (near-inf)")
    ax0.grid(True, alpha=0.3)
    ax0.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='err=0.5')
    ax0.legend()
    plt.colorbar(scatter0, ax=ax0, label="log10(omega)")

    ax1 = axes2[1]
    B_inc_errs = []
    B_ref_errs = []
    for r in results_all:
        B_inc_s = r.get("B_inc_spec", np.nan)
        B_ref_s = r.get("B_ref_spec", np.nan)
        if np.isfinite(abs(B_inc_s)):
            B_inc_errs.append(rel_error(abs(r["B_inc_net"]), abs(B_inc_s)))
            B_ref_errs.append(rel_error(abs(r["B_ref_net"]), abs(B_ref_s)))
    ax1.scatter(omegas[:len(B_inc_errs)], B_inc_errs, s=20, alpha=0.7, label="B_inc")
    ax1.scatter(omegas[:len(B_ref_errs)], B_ref_errs, s=20, alpha=0.7, label="B_ref")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel("omega")
    ax1.set_ylabel("relative error")
    ax1.set_title("AmplitudeNet B vs spectral B")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig2.tight_layout()
    plot_path2 = out_dir / "error_analysis.png"
    fig2.savefig(plot_path2, dpi=150)
    plt.close(fig2)
    print(f"[diagnose] Saved {plot_path2}")

    # ---- Save results ----
    summary = {
        "pre_stage4_bundle": str(bundle_dir.resolve()),
        "stage2_checkpoint": str(stage2_ckpt_path.resolve()),
        "spectral_cache": str(Path(args.spectral_cache).resolve()),
        "n_points": int(n_cache),
        "n_y": int(n_y),
        "n_nearinf": int(n_nearinf),
        "y_range": [float(y_grid_np[0]), float(y_grid_np[-1])],
        "nearinf_y_range": [-0.999, -0.95],
        "median_rel_err_nearinf": nearinf_med,
        "pybhpt_median_model_err": float(np.median(pybhpt_model_errs)) if pybhpt_model_errs else None,
        "pybhpt_median_ampnet_err": float(np.median(pybhpt_ampnet_errs)) if pybhpt_ampnet_errs else None,
        "per_point": [{k: v for k, v in r.items()
                       if k not in ("pybhpt_R_over_P", "R_model_np", "R_ampnet_np")}
                      for r in results_all],
    }
    with open(out_dir / "diagnosis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print per-point table
    print(f"\n[diagnose] Per-point results:")
    header = f"  {'idx':>4s} {'a':>8s} {'omega':>12s} {'nearinf_err':>12s} {'all_err':>12s}"
    if pybhpt_model_errs:
        header += f" {'pybhpt_m':>10s} {'pybhpt_a':>10s}"
    print(header)
    for r in results_all:
        line = f"  {r['idx']:4d} {r['a']:8.4f} {r['omega']:12.6e} {r['median_rel_err_nearinf']:12.4e} {r['median_rel_err_all']:12.4e}"
        if r.get("pybhpt_median_err_model") is not None:
            line += f" {r['pybhpt_median_err_model']:10.4e} {r['pybhpt_median_err_ampnet']:10.4e}"
        print(line)

    print(f"\n[diagnose] Saved {out_dir / 'diagnosis_summary.json'}")


if __name__ == "__main__":
    main()
