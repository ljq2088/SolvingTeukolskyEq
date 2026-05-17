#!/usr/bin/env python3
"""Regenerate near-infinity comparison plot with Mathematica B_inc/B_ref amplitudes.

Replaces AmpNet B values with Mathematica-computed Teukolsky amplitudes.

Usage:
  python scripts/redraw_nearinf_comparison_mma.py \
    --pre-stage4-bundle outputs/pre_stage4_bundle/patch_000_logw_v2 \
    --spectral-cache outputs/stage4_spectral_basis_cache/patch_000_logw_v2/spectral_basis_cache.npz \
    --config config/autoencoder_stage4_spectral_consistency.yaml \
    --device cuda \
    --output-dir outputs/stage1_nearinf_bottleneck_diagnosis/patch_000_logw_v2
"""
import argparse
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
from physical_ansatz.prefactor import Leaver_prefactors
from physical_ansatz.u_basis import A_up, A_down
from physical_ansatz.stage1_reconstruction import compute_R_over_P_from_model
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr
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
    rp = 1.0 + np.sqrt(1.0 - a_val * a_val)
    x_np = (y_np + 1.0) / 2.0
    r_np = rp / np.clip(x_np, 1e-12, None)
    r_sorted = np.sort(r_np)
    try:
        r_vals, R_in = compute_pybhpt_solution(a_val, omega_val, ell=2, m=m,
                                                r_grid=r_sorted, timeout=timeout)
    except Exception:
        return None
    R_in_interp = np.interp(r_np, r_vals, R_in.real) + 1j * np.interp(r_np, r_vals, R_in.imag)
    a_t = torch.tensor([a_val], dtype=torch.float64)
    omega_t = torch.tensor([omega_val], dtype=torch.float64)
    r_t = torch.tensor(r_np, dtype=torch.float64)
    rp_t = r_plus(a_t, M)
    rm_t = r_minus(a_t, M)
    P_np, _, _ = Leaver_prefactors(r_t, a_t, omega_t, m=m, M=M, s=s, rp=rp_t, rm=rm_t)
    return R_in_interp / P_np.numpy()


def _wl_complex_to_py(wl_val):
    """Convert wolframclient WLFunction Complex to Python complex."""
    if hasattr(wl_val, 'args') and len(wl_val.args) == 2:
        return complex(float(wl_val.args[0]), float(wl_val.args[1]))
    return complex(wl_val)


def compute_mma_amplitudes(session, wlexpr_func, s, l, m, a, omega):
    """Call Mathematica to compute B_inc, B_ref for given (a, omega)."""
    expr = f"ComputeAmplitudes[{s}, {l}, {m}, {a:.16g}, {omega:.16g}]"
    result = session.evaluate(wlexpr_func(expr))
    try:
        B_inc = _wl_complex_to_py(result["Incidence"])
        B_ref = _wl_complex_to_py(result["Reflection"])
        return B_inc, B_ref
    except (TypeError, KeyError, IndexError) as e:
        raise RuntimeError(f"MMA result parse error: {e}, raw={str(result)[:200]}")


def compute_R_over_P_mma_spectral(B_inc_mma, B_ref_mma, a_t, omega_t, y_t,
                                    u_up_spec, u_down_spec, M=1.0, s=-2):
    """Compute R/P using Mathematica B values + spectral u basis."""
    with torch.no_grad():
        rp = r_plus(a_t, M)
        x = (y_t + 1.0) / 2.0
        r = rp / x
        A_u = A_up(r, a_t, omega_t, M).to(dtype=torch.complex128)
        A_d = A_down(r, a_t, omega_t, M).to(dtype=torch.complex128)
        P = Leaver_P(r, a_t, omega_t, M=M, s=s).to(dtype=torch.complex128)
        B_ref_c = torch.tensor([[B_ref_mma]], dtype=torch.complex128, device=y_t.device)
        B_inc_c = torch.tensor([[B_inc_mma]], dtype=torch.complex128, device=y_t.device)
        R_combo = (B_ref_c * A_u * u_up_spec + B_inc_c * A_d * u_down_spec) / P
    return R_combo.squeeze(0).cpu().numpy()


def _get_mma_kernel_path():
    """Find the Mathematica kernel path."""
    candidates = [
        "/mnt/f/mma/WolframKernel.exe",
        "/usr/local/Wolfram/Mathematica/14.0/Executables/WolframKernel",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-stage4-bundle", type=str, required=True)
    parser.add_argument("--spectral-cache", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/autoencoder_stage4_spectral_consistency.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--pybhpt-max-points", type=int, default=10)
    parser.add_argument("--pybhpt-timeout", type=float, default=20.0)
    parser.add_argument("--mma-wl", type=str, default="mma/Radial_Function.wl")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load config ----
    full_cfg = load_pinn_full_config(args.config)
    train_cfg = full_cfg.get("train", {})
    runtime_cfg = train_cfg.get("runtime", {})
    dtype = _get_dtype(runtime_cfg.get("dtype", "float64"))
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
    print(f"Cache: {n_cache} params, y=[{y_grid_np[0]:.4f}, {y_grid_np[-1]:.4f}]")

    nearinf_mask = y_grid_np >= -0.999
    nearinf_mask &= y_grid_np <= -0.95

    # ---- Load Stage-1 model ----
    bundle_dir = Path(args.pre_stage4_bundle)
    stage1_ckpt_path = bundle_dir / "stage1_best_model.pt"
    if not stage1_ckpt_path.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint: {stage1_ckpt_path}")

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
    model.load_state_dict(sd1_filtered, strict=False)
    model.to(device=device, dtype=dtype)
    model.eval()

    y_t = torch.tensor(y_grid_np, device=device, dtype=dtype).unsqueeze(0)

    # ---- Start Mathematica ----
    kernel_path = _get_mma_kernel_path()
    print(f"Mathematica kernel: {kernel_path}")
    session = WolframLanguageSession(kernel=kernel_path)
    session.start()
    wl_path = Path(args.mma_wl).resolve()
    session.evaluate(wlexpr(f'Get["{wl_path}"]'))
    print(f"Loaded {wl_path}")

    wlexpr_func = wlexpr

    # ---- Compute MMA B values for all points ----
    print(f"\nComputing MMA amplitudes for {n_cache} points ...")
    mma_B_all = []
    for i in range(n_cache):
        a_val = float(a_cache[i])
        omega_val = float(omega_cache[i])
        try:
            B_inc, B_ref = compute_mma_amplitudes(session, wlexpr_func,
                                                   s_phys, 2, m_phys, a_val, omega_val)
            mma_B_all.append((B_inc, B_ref))
            print(f"  [{i+1}/{n_cache}] a={a_val:.4f} ω={omega_val:.6e} "
                  f"B_inc={B_inc:.4e} B_ref={B_ref:.4e}")
        except Exception as e:
            print(f"  [{i+1}/{n_cache}] a={a_val:.4f} ω={omega_val:.6e} FAILED: {e}")
            mma_B_all.append((None, None))

    session.stop()

    # ---- Evaluate all points ----
    n_pybhpt = min(args.pybhpt_max_points, n_cache)
    pybhpt_idx = list(np.linspace(0, n_cache - 1, n_pybhpt, dtype=int))
    results_all = []

    print(f"\nEvaluating {n_cache} points (Stage-1 + MMA+Spectral) ...")
    print(f"pybhpt on {n_pybhpt} selected points ...")

    for i in range(n_cache):
        a_val = float(a_cache[i])
        omega_val = float(omega_cache[i])
        u_val = float(u_cache[i])
        v_val = float(v_cache[i])
        lam_val = complex(lam_cache[i])
        B_inc_mma, B_ref_mma = mma_B_all[i]

        a_t = torch.tensor([[a_val]], device=device, dtype=dtype)
        omega_t = torch.tensor([[omega_val]], device=device, dtype=dtype)
        u_t = torch.tensor([[u_val]], device=device, dtype=dtype)
        v_t = torch.tensor([[v_val]], device=device, dtype=dtype)
        lam_t = torch.tensor([[lam_val]], device=device, dtype=cdtype)
        u_up_spec = torch.tensor(u_up_cache[i], device=device, dtype=cdtype).unsqueeze(0)
        u_down_spec = torch.tensor(u_down_cache[i], device=device, dtype=cdtype).unsqueeze(0)

        with torch.no_grad():
            R_over_P_model = compute_R_over_P_from_model(
                model, a_t, omega_t, y_t, lam_t, u_t, v_t,
                m=m_phys, M=M_phys, s=s_phys,
            )
            R_model_np = R_over_P_model.squeeze(0).cpu().numpy()

        # MMA + spectral u
        if B_ref_mma is not None:
            R_mma_np = compute_R_over_P_mma_spectral(
                B_inc_mma, B_ref_mma, a_t, omega_t, y_t,
                u_up_spec, u_down_spec, M=M_phys, s=s_phys,
            )
        else:
            R_mma_np = np.zeros_like(R_model_np)

        err_model_vs_mma = rel_error(R_model_np, R_mma_np)

        entry = {
            "idx": int(i),
            "a": a_val,
            "omega": omega_val,
            "B_inc_mma": B_inc_mma,
            "B_ref_mma": B_ref_mma,
            "median_rel_err_all": float(np.median(err_model_vs_mma)),
            "median_rel_err_nearinf": float(np.median(err_model_vs_mma[nearinf_mask])),
            "max_rel_err_nearinf": float(np.max(err_model_vs_mma[nearinf_mask])),
        }

        # pybhpt for selected points
        if i in pybhpt_idx:
            print(f"  [{i+1}/{n_cache}] pybhpt: a={a_val:.4f} ω={omega_val:.6e} ...", end=" ")
            R_pybhpt_np = compute_R_over_P_pybhpt(
                a_val, omega_val, y_grid_np,
                m=m_phys, M=M_phys, s=s_phys, timeout=args.pybhpt_timeout,
            )
            if R_pybhpt_np is not None:
                err_model_vs_pybhpt = rel_error(R_model_np, R_pybhpt_np)
                err_mma_vs_pybhpt = rel_error(R_mma_np, R_pybhpt_np) if B_ref_mma is not None else np.ones_like(R_model_np)
                entry["pybhpt_median_err_model"] = float(np.median(err_model_vs_pybhpt))
                entry["pybhpt_median_err_mma"] = float(np.median(err_mma_vs_pybhpt))
                entry["pybhpt_median_err_model_nearinf"] = float(np.median(err_model_vs_pybhpt[nearinf_mask]))
                entry["pybhpt_median_err_mma_nearinf"] = float(np.median(err_mma_vs_pybhpt[nearinf_mask]))
                entry["pybhpt_R_over_P"] = R_pybhpt_np.tolist()
                entry["R_model_np"] = R_model_np.tolist()
                entry["R_mma_np"] = R_mma_np.tolist()
                print(f"model_err={entry['pybhpt_median_err_model']:.4e} "
                      f"mma_err={entry['pybhpt_median_err_mma']:.4e}")
            else:
                print("FAILED")

        results_all.append(entry)

    # ---- Summary ----
    nearinf_errs = [r["median_rel_err_nearinf"] for r in results_all
                    if np.isfinite(r["median_rel_err_nearinf"])]
    pybhpt_model_errs = [r["pybhpt_median_err_model"] for r in results_all
                         if r.get("pybhpt_median_err_model") is not None]
    pybhpt_mma_errs = [r["pybhpt_median_err_mma"] for r in results_all
                       if r.get("pybhpt_median_err_mma") is not None]

    print(f"\nSummary:")
    print(f"  Stage-1 vs MMA+Spectral (near-inf): median={np.median(nearinf_errs):.4e}")
    if pybhpt_model_errs:
        print(f"  Stage-1 vs pybhpt:                  median={np.median(pybhpt_model_errs):.4e}")
    if pybhpt_mma_errs:
        print(f"  MMA+Spectral vs pybhpt:             median={np.median(pybhpt_mma_errs):.4e}")

    # ---- Plot: nearinf_comparison_mma.png ----
    print(f"\nGenerating plots ...")
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
        R_model = np.array(r.get("R_model_np", [0]))
        R_mma = np.array(r.get("R_mma_np", [0]))
        R_pybhpt = np.array(r.get("pybhpt_R_over_P", [0]))

        if len(R_model) != len(y_plot):
            R_model = np.zeros(len(y_plot))
            R_mma = np.zeros(len(y_plot))
            R_pybhpt = np.zeros(len(y_plot))

        y_mask = y_plot >= -0.999

        ax_abs.semilogy(y_plot[y_mask], np.abs(R_model[y_mask]), label="Stage-1 (PINN)", linewidth=1)
        ax_abs.semilogy(y_plot[y_mask], np.abs(R_mma[y_mask]), label="MMA+Spectral", linewidth=1)
        if r.get("pybhpt_R_over_P") is not None:
            ax_abs.semilogy(y_plot[y_mask], np.abs(R_pybhpt[y_mask]), 'k--', label="pybhpt (ODE)", linewidth=1)
        ax_abs.axvline(x=-0.95, color='gray', linestyle=':', alpha=0.5)
        ax_abs.set_title(f"a={r['a']:.3f}  $\omega$={r['omega']:.2e}")
        ax_abs.set_ylabel("|R_in/P|")
        ax_abs.legend(fontsize=7)
        ax_abs.grid(True, alpha=0.3)

        phase_model = np.unwrap(np.angle(R_model[y_mask]))
        phase_mma = np.unwrap(np.angle(R_mma[y_mask]))
        ax_phase.plot(y_plot[y_mask], phase_model, label="Stage-1 (PINN)", linewidth=1)
        ax_phase.plot(y_plot[y_mask], phase_mma, label="MMA+Spectral", linewidth=1)
        if r.get("pybhpt_R_over_P") is not None:
            phase_pybhpt = np.unwrap(np.angle(R_pybhpt[y_mask]))
            ax_phase.plot(y_plot[y_mask], phase_pybhpt, 'k--', label="pybhpt (ODE)", linewidth=1)
        ax_phase.axvline(x=-0.95, color='gray', linestyle=':', alpha=0.5)
        ax_phase.set_xlabel("y")
        ax_phase.set_ylabel("Phase(R_in/P)")
        ax_phase.legend(fontsize=7)
        ax_phase.grid(True, alpha=0.3)

    fig.suptitle("Near-Infinity R_in/P: Stage-1 (PINN) vs MMA+Spectral vs pybhpt (ODE)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    plot_path = out_dir / "nearinf_comparison_mma.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved {plot_path}")

    # ---- Per-point error plot ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    ax0 = axes2[0]
    omegas = np.array([r["omega"] for r in results_all])
    nearinf_errs_all = np.array([r["median_rel_err_nearinf"] for r in results_all])
    ax0.scatter(omegas, nearinf_errs_all, c=np.log10(omegas), cmap='viridis', s=30)
    ax0.set_xscale('log')
    ax0.set_xlabel("omega")
    ax0.set_ylabel("median rel err (near-inf)")
    ax0.set_title("Stage-1 (PINN) vs MMA+Spectral (near-inf)")
    ax0.grid(True, alpha=0.3)
    ax0.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='err=0.5')
    ax0.legend()

    ax1 = axes2[1]
    pybhpt_omega = [r["omega"] for r in results_all if r.get("pybhpt_median_err_model") is not None]
    p_model = [r["pybhpt_median_err_model"] for r in results_all if r.get("pybhpt_median_err_model") is not None]
    p_mma = [r["pybhpt_median_err_mma"] for r in results_all if r.get("pybhpt_median_err_mma") is not None]
    ax1.scatter(pybhpt_omega, p_model, s=20, alpha=0.7, label="Stage-1 vs pybhpt")
    ax1.scatter(pybhpt_omega, p_mma, s=20, alpha=0.7, label="MMA+Spectral vs pybhpt", marker='s')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel("omega")
    ax1.set_ylabel("median relative error")
    ax1.set_title("Error vs pybhpt reference")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig2.tight_layout()
    plot_path2 = out_dir / "error_analysis_mma.png"
    fig2.savefig(plot_path2, dpi=150)
    plt.close(fig2)
    print(f"Saved {plot_path2}")

    # ---- Summary JSON ----
    summary = {
        "spectral_cache": str(Path(args.spectral_cache).resolve()),
        "n_points": int(n_cache),
        "n_y": int(len(y_grid_np)),
        "median_rel_err_nearinf": float(np.median(nearinf_errs)),
        "pybhpt_median_model_err": float(np.median(pybhpt_model_errs)) if pybhpt_model_errs else None,
        "pybhpt_median_mma_err": float(np.median(pybhpt_mma_errs)) if pybhpt_mma_errs else None,
        "per_point": [{k: v for k, v in r.items()
                       if k not in ("pybhpt_R_over_P", "R_model_np", "R_mma_np")}
                      for r in results_all],
    }
    with open(out_dir / "mma_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nPer-point results:")
    for r in results_all:
        line = f"  {r['idx']:4d} {r['a']:8.4f} {r['omega']:12.6e} {r['median_rel_err_nearinf']:12.4e}"
        if r.get("pybhpt_median_err_model") is not None:
            line += f"  model_vs_pybhpt={r['pybhpt_median_err_model']:.4e}"
        if r.get("pybhpt_median_err_mma") is not None:
            line += f"  mma_vs_pybhpt={r['pybhpt_median_err_mma']:.4e}"
        print(line)

    print(f"\nSaved {out_dir / 'mma_comparison_summary.json'}")


if __name__ == "__main__":
    import json
    main()
