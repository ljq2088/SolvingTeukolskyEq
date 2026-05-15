#!/usr/bin/env python3
"""Diagnose Stage-3 u-basis solution quality.

Checks boundary conditions, u_up/u_down smoothness, and residual profiles.

Usage:
  python scripts/diagnose_stage3_ubasis_solution.py \\
    --checkpoint <stage3_best_model.pt> \\
    --config config/autoencoder_stage3_ubasis_refine.yaml \\
    --device cuda
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from config.config_loader import load_pinn_full_config
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.u_basis import compose_u_from_f, infinity_slopes_y, A_up, A_down
from physical_ansatz.u_residual import compute_u_equation_residual
from physical_ansatz.mapping import r_plus


def _get_dtype(dtype_name):
    return torch.float32 if dtype_name == "float32" else torch.float64


def _ensure_2d(t):
    if t is None:
        return None
    if t.ndim == 1:
        return t.unsqueeze(-1)
    return t


def main():
    parser = argparse.ArgumentParser(description="Diagnose Stage-3 u-basis solutions")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/autoencoder_stage3_ubasis_refine.yaml")
    parser.add_argument("--artifact-dir", type=str,
                        default="outputs/stage1_artifacts/patch_000_logw_v2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--n-y", type=int, default=256)
    parser.add_argument("--y-eps", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load config
    full_cfg = load_pinn_full_config(args.config)
    runtime_cfg = full_cfg.get("runtime", {})
    dtype_name = runtime_cfg.get("dtype", "float64")
    dtype = _get_dtype(dtype_name)
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    # Physics
    physics = full_cfg.get("physics", full_cfg)
    prob = physics.get("problem", {})
    M = float(prob.get("M", 1.0))
    s = int(prob.get("s", -2))
    m = int(prob.get("m", 2))

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    full_cfg_raw = ckpt.get("full_cfg", full_cfg)
    model_cfg = full_cfg_raw.get("model", full_cfg.get("model", {}))

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

    ckpt_epoch = ckpt.get("epoch", "?")
    is_stage3 = ckpt.get("history") is not None
    print(f"Model loaded: epoch={ckpt_epoch}, source={'Stage-3' if is_stage3 else 'Stage-2'}")

    # Load parameter pool
    pool_data = np.load(Path(args.artifact_dir) / "rpred_cache.npz")
    valid = np.isfinite(pool_data["a"]) & np.isfinite(pool_data["omega"]) & np.isfinite(pool_data["lambda_"])
    a_pool = pool_data["a"][valid]
    omega_pool = pool_data["omega"][valid]
    lam_pool = pool_data["lambda_"][valid]

    # Sample parameters
    n_total = len(a_pool)
    idx = np.linspace(0, n_total - 1, args.n_samples, dtype=int)
    print(f"Sampling {args.n_samples} parameter sets from {n_total} total")

    # y grid
    y = torch.linspace(-1 + args.y_eps, 1 - args.y_eps, args.n_y,
                       device=device, dtype=dtype).unsqueeze(0)  # (1, N_y)

    # Output dir
    run_dir = Path(args.checkpoint).parent.parent
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    md_lines = []
    md_lines.append("# Stage-3 u-basis Diagnosis\n")
    md_lines.append(f"- **checkpoint**: `{args.checkpoint}`\n")
    md_lines.append(f"- **epoch**: {ckpt_epoch}\n")
    md_lines.append(f"- **n_samples**: {args.n_samples}\n")
    md_lines.append(f"- **y_eps**: {args.y_eps}\n\n")

    # ---- Collect statistics ----
    all_u_up_max = []
    all_u_down_max = []
    all_res_up_median = []
    all_res_down_median = []
    all_res_up_global = []
    all_res_down_global = []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False
        print("matplotlib not available, skipping plots")

    for i_sample, i in enumerate(idx):
        a_val = torch.tensor([[a_pool[i]]], device=device, dtype=dtype)
        omega_val = torch.tensor([[omega_pool[i]]], device=device, dtype=dtype)
        lam_val = torch.tensor([complex(lam_pool[i].real, lam_pool[i].imag)],
                               device=device, dtype=cdtype)

        a_str = f"{a_pool[i]:.4f}"
        omega_str = f"{omega_pool[i]:.6e}"
        print(f"\nSample {i_sample+1}/{args.n_samples}: a={a_str}, omega={omega_str}")

        # ---- Forward pass ----
        y_g = y.clone().requires_grad_(True)
        with torch.no_grad():
            f_up = model.predict_u_up(a_val, omega_val, y_g)
            f_down = model.predict_u_down(a_val, omega_val, y_g)

        c_up, c_down = infinity_slopes_y(a_val, omega_val, lam_val, m=m, M=M)
        u_up = compose_u_from_f(f_up, y_g, c_up)
        u_down = compose_u_from_f(f_down, y_g, c_down)

        # ---- Boundary checks at EXACT y=-1 ----
        # Use a dedicated boundary point, NOT the clipped collocation grid
        y0 = torch.tensor([[-1.0]], device=device, dtype=dtype, requires_grad=True)
        with torch.no_grad():
            f_up_0 = model.predict_u_up(a_val, omega_val, y0)
            f_down_0 = model.predict_u_down(a_val, omega_val, y0)
        u_up_0 = compose_u_from_f(f_up_0, y0, c_up)
        u_down_0 = compose_u_from_f(f_down_0, y0, c_down)

        # u(-1) = 1
        u_up_at_inf = u_up_0[0, 0].detach().item()
        u_down_at_inf = u_down_0[0, 0].detach().item()

        # Analytic check: u_y(-1) = c_inf (by construction of ansatz)
        # This always holds mathematically. Report as analytic pass.
        c_up_val = c_up.detach().squeeze()
        c_down_val = c_down.detach().squeeze()

        ok_up_boundary_val = abs(u_up_at_inf - 1.0) < 1e-12
        ok_down_boundary_val = abs(u_down_at_inf - 1.0) < 1e-12

        print(f"  [Boundary] u_up(-1)={u_up_at_inf:.12e} (expect 1.0) "
              f"{'OK' if ok_up_boundary_val else 'FAIL'}")
        print(f"  [Boundary] u_down(-1)={u_down_at_inf:.12e} (expect 1.0) "
              f"{'OK' if ok_down_boundary_val else 'FAIL'}")
        print(f"  [Boundary] u_y(-1) analytic: du_up/dy={c_up_val:.6e}, du_down/dy={c_down_val:.6e} (PASS by construction)")

        # Autograd verification of u_y(-1) — check both real and imaginary parts
        y0_g = y0.clone().detach().requires_grad_(True)
        f_up_g0 = model.predict_u_up(a_val, omega_val, y0_g)
        f_down_g0 = model.predict_u_down(a_val, omega_val, y0_g)
        u_up_g0 = compose_u_from_f(f_up_g0, y0_g, c_up)
        u_down_g0 = compose_u_from_f(f_down_g0, y0_g, c_down)

        u_up_y_real = torch.autograd.grad(u_up_g0.real.sum(), y0_g,
                                          create_graph=False, retain_graph=True)[0]
        u_down_y_real = torch.autograd.grad(u_down_g0.real.sum(), y0_g,
                                            create_graph=False, retain_graph=True)[0]
        u_up_y_imag = torch.autograd.grad(u_up_g0.imag.sum(), y0_g,
                                          create_graph=False, retain_graph=True)[0]
        u_down_y_imag = torch.autograd.grad(u_down_g0.imag.sum(), y0_g,
                                            create_graph=False, retain_graph=False)[0]

        uy_up_val = complex(u_up_y_real[0, 0].item(), u_up_y_imag[0, 0].item())
        uy_down_val = complex(u_down_y_real[0, 0].item(), u_down_y_imag[0, 0].item())

        c_up_complex = complex(c_up_val.real.item(), c_up_val.imag.item())
        c_down_complex = complex(c_down_val.real.item(), c_down_val.imag.item())

        ok_up_deriv = abs(uy_up_val.real - c_up_complex.real) < 1e-9 and abs(uy_up_val.imag - c_up_complex.imag) < 1e-9
        ok_down_deriv = abs(uy_down_val.real - c_down_complex.real) < 1e-9 and abs(uy_down_val.imag - c_down_complex.imag) < 1e-9

        print(f"  [Boundary] u_y(-1) autograd: du_up/dy={uy_up_val:.6e} (expect {c_up_complex:.6e}) "
              f"{'OK' if ok_up_deriv else 'FAIL'}")
        print(f"  [Boundary] u_y(-1) autograd: du_down/dy={uy_down_val:.6e} (expect {c_down_complex:.6e}) "
              f"{'OK' if ok_down_deriv else 'FAIL'}")

        ok_up_boundary = ok_up_boundary_val and ok_up_deriv
        ok_down_boundary = ok_down_boundary_val and ok_down_deriv

        # ---- Residual ----
        _, res_up, _ = compute_u_equation_residual(
            model, a_val, omega_val, y_g, lam_val, M=M, s=s, m=m, basis="up")
        _, res_down, _ = compute_u_equation_residual(
            model, a_val, omega_val, y_g, lam_val, M=M, s=s, m=m, basis="down")

        res_up_abs = torch.abs(res_up).detach().cpu().numpy().ravel()
        res_down_abs = torch.abs(res_down).detach().cpu().numpy().ravel()

        # Split domain: near-infinity (y < -0.5) and whole-domain
        y_np = y.cpu().numpy().ravel()
        near_inf_mask = y_np < -0.5

        res_up_near_inf = np.median(res_up_abs[near_inf_mask]) if near_inf_mask.any() else np.nan
        res_down_near_inf = np.median(res_down_abs[near_inf_mask]) if near_inf_mask.any() else np.nan
        res_up_med = np.median(res_up_abs)
        res_down_med = np.median(res_down_abs)

        u_up_max = np.max(np.abs(u_up.detach().cpu().numpy()))
        u_down_max = np.max(np.abs(u_down.detach().cpu().numpy()))

        all_u_up_max.append(u_up_max)
        all_u_down_max.append(u_down_max)
        all_res_up_median.append(res_up_near_inf)
        all_res_down_median.append(res_down_near_inf)
        all_res_up_global.append(res_up_med)
        all_res_down_global.append(res_down_med)

        print(f"  max|u_up|={u_up_max:.4f}  max|u_down|={u_down_max:.4f}")
        print(f"  residual_up: median={res_up_med:.4e} near-inf={res_up_near_inf:.4e}")
        print(f"  residual_down: median={res_down_med:.4e} near-inf={res_down_near_inf:.4e}")

        # ---- Plot ----
        if HAS_MPL:
            u_up_np = u_up.detach().cpu().numpy().ravel()
            u_down_np = u_down.detach().cpu().numpy().ravel()

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f"Stage-3 u-basis — a={a_str}, $\\omega$={omega_str}")

            ax = axes[0, 0]
            ax.plot(y_np, np.abs(u_up_np), "b-", lw=1)
            ax.set_xlabel("y"); ax.set_ylabel("|u_up|")
            ax.set_title("|u_up(y)|"); ax.grid(True, alpha=0.3)

            ax = axes[0, 1]
            ax.plot(y_np, u_up_np.real, "b-", lw=1, label="Re")
            ax.plot(y_np, u_up_np.imag, "r--", lw=1, label="Im")
            ax.set_xlabel("y"); ax.set_ylabel("u_up")
            ax.set_title("Re/Im u_up(y)"); ax.legend(); ax.grid(True, alpha=0.3)

            ax = axes[0, 2]
            ax.semilogy(y_np, res_up_abs, "b-", lw=1)
            ax.set_xlabel("y"); ax.set_ylabel("|res_up|")
            ax.set_title("u-eq residual_up(y)"); ax.grid(True, alpha=0.3)

            ax = axes[1, 0]
            ax.plot(y_np, np.abs(u_down_np), "r-", lw=1)
            ax.set_xlabel("y"); ax.set_ylabel("|u_down|")
            ax.set_title("|u_down(y)|"); ax.grid(True, alpha=0.3)

            ax = axes[1, 1]
            ax.plot(y_np, u_down_np.real, "b-", lw=1, label="Re")
            ax.plot(y_np, u_down_np.imag, "r--", lw=1, label="Im")
            ax.set_xlabel("y"); ax.set_ylabel("u_down")
            ax.set_title("Re/Im u_down(y)"); ax.legend(); ax.grid(True, alpha=0.3)

            ax = axes[1, 2]
            ax.semilogy(y_np, res_down_abs, "r-", lw=1)
            ax.set_xlabel("y"); ax.set_ylabel("|res_down|")
            ax.set_title("u-eq residual_down(y)"); ax.grid(True, alpha=0.3)

            plt.tight_layout()
            fig_path = diag_dir / f"sample_{i_sample:02d}_a{a_str}_omega{omega_str}.png"
            fig.savefig(fig_path, dpi=100)
            plt.close(fig)
            print(f"  Saved {fig_path}")

    # ---- Summary statistics ----
    md_lines.append("## Boundary Conditions (EXACT y=-1)\n\n")
    md_lines.append(f"- u_up(-1)=1 : {'PASS' if ok_up_boundary_val else 'FAIL'}\n")
    md_lines.append(f"- u_down(-1)=1 : {'PASS' if ok_down_boundary_val else 'FAIL'}\n")
    md_lines.append(f"- du_up/dy(-1)=c_up (analytic) : PASS (by ansatz construction)\n")
    md_lines.append(f"- du_down/dy(-1)=c_down (analytic) : PASS (by ansatz construction)\n")
    md_lines.append(f"- du_up/dy(-1)=c_up (autograd, Re+Im) : {'PASS' if ok_up_deriv else 'FAIL'}\n")
    md_lines.append(f"- du_down/dy(-1)=c_down (autograd, Re+Im) : {'PASS' if ok_down_deriv else 'FAIL'}\n")
    md_lines.append(f"\nNote: boundary checks use exact y=-1, NOT the clipped collocation grid.\n")
    md_lines.append(f"Collocation grid: y >= -1 + y_eps ({args.y_eps}), for PDE residual only.\n\n")

    md_lines.append("## Statistics\n\n")
    md_lines.append("| Metric | min | median | max |\n")
    md_lines.append("|--------|-----|--------|-----|\n")
    md_lines.append(f"| max |u_up| | {np.min(all_u_up_max):.4f} | {np.median(all_u_up_max):.4f} | {np.max(all_u_up_max):.4f} |\n")
    md_lines.append(f"| max |u_down| | {np.min(all_u_down_max):.4f} | {np.median(all_u_down_max):.4f} | {np.max(all_u_down_max):.4f} |\n")
    md_lines.append(f"| residual_up near-inf median | {np.min(all_res_up_median):.4e} | {np.median(all_res_up_median):.4e} | {np.max(all_res_up_median):.4e} |\n")
    md_lines.append(f"| residual_down near-inf median | {np.min(all_res_down_median):.4e} | {np.median(all_res_down_median):.4e} | {np.max(all_res_down_median):.4e} |\n")
    md_lines.append(f"| residual_up global median | {np.min(all_res_up_global):.4e} | {np.median(all_res_up_global):.4e} | {np.max(all_res_up_global):.4e} |\n")
    md_lines.append(f"| residual_down global median | {np.min(all_res_down_global):.4e} | {np.median(all_res_down_global):.4e} | {np.max(all_res_down_global):.4e} |\n\n")

    # Overall assessment
    md_lines.append("## Assessment\n\n")
    issues = []
    if not ok_up_boundary:
        issues.append("- u_up boundary conditions FAIL")
    if not ok_down_boundary:
        issues.append("- u_down boundary conditions FAIL")
    if np.max(all_u_up_max) > 100:
        issues.append(f"- u_up has large amplitude (max={np.max(all_u_up_max):.1f})")
    if np.max(all_u_down_max) > 100:
        issues.append(f"- u_down has large amplitude (max={np.max(all_u_down_max):.1f})")
    if not issues:
        md_lines.append("No issues detected. Solutions appear smooth and boundary conditions satisfied.\n")
    else:
        md_lines.append("Issues:\n")
        for issue in issues:
            md_lines.append(issue + "\n")

    md_path = diag_dir / "stage3_ubasis_diagnosis.md"
    with open(md_path, "w") as f:
        f.writelines(md_lines)

    print(f"\nDiagnosis saved to {md_path}")


if __name__ == "__main__":
    main()
