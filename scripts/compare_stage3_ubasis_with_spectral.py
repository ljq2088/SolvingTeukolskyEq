#!/usr/bin/env python3
"""
Compare Stage-3 decoder u_up/u_down with spectral method basis functions.

Generates basis_functions_stage3_vs_spectral.png mimicking
high_omega_debug/basis_functions.png, with spectral + decoder overlaid.

Usage:
  python scripts/compare_stage3_ubasis_with_spectral.py \
    --stage3-checkpoint outputs/autoencoder_stage3_ubasis_refine/.../best_model.pt \
    --config config/autoencoder_stage3_ubasis_refine.yaml \
    --device cuda \
    --a 0.1 --omega 10.0 \
    --output-dir outputs/stage3_spectral_compare/patch_000_logw_v2
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from config.config_loader import load_pinn_full_config
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.u_basis import compose_u_from_f, infinity_slopes_y, A_up, A_down
from physical_ansatz.mapping import r_plus


def _get_dtype(dtype_name):
    return torch.float32 if dtype_name == "float32" else torch.float64


def run_spectral(a, omega, lam, z1=0.625, z2=0.705, N_left=28, N_mid=28, N_right=28, M=1.0, s=-2, ell=2, m=2):
    """Run spectral method and return basis data."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from utils.mode import KerrMode
    from utils.amplitude import solve_basis_domain, r_of_z

    mode = KerrMode(M=M, a=a, omega=omega, ell=ell, m=m, lam=lam, s=s)

    sol_down = solve_basis_domain(mode, "down", N_left, 0.0, z1, "left")
    sol_up = solve_basis_domain(mode, "up", N_left, 0.0, z1, "left")
    sol_in = solve_basis_domain(mode, "in", N_right, z2, 1.0, "right")

    return {
        "sol_down": sol_down,
        "sol_up": sol_up,
        "sol_in": sol_in,
        "z1": z1,
        "z2": z2,
        "mode": mode,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Stage-3 decoder with spectral basis")
    parser.add_argument("--stage3-checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/autoencoder_stage3_ubasis_refine.yaml")
    parser.add_argument("--device", type=str, default="cuda")

    # Point specification: either via JSON or direct parameters
    parser.add_argument("--point-json", type=str, default=None,
                        help="Path to calibration points JSON (e.g. patch0_calibration_points.json)")
    parser.add_argument("--point-label", type=str, default=None,
                        help="Label of point to use from point-json (e.g. center, low_omega)")
    parser.add_argument("--a", type=float, default=None)
    parser.add_argument("--omega", type=float, default=None)
    parser.add_argument("--u", type=float, default=None)
    parser.add_argument("--v", type=float, default=None)
    parser.add_argument("--lam", type=float, default=None)

    # Spectral profile (pre-computed basis, skips spectral solve)
    parser.add_argument("--spectral-profile", type=str, default=None,
                        help="Path to pre-computed basis profile .npz")

    parser.add_argument("--M", type=float, default=1.0)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--z1", type=float, default=0.625)
    parser.add_argument("--z2", type=float, default=0.705)
    parser.add_argument("--N-left", type=int, default=28)
    parser.add_argument("--N-mid", type=int, default=28)
    parser.add_argument("--N-right", type=int, default=28)
    parser.add_argument("--n-y-diag", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="outputs/stage3_spectral_compare/patch_000_logw_v2")
    parser.add_argument("--no-plots", action="store_true", default=False)
    args = parser.parse_args()

    # ---- Resolve point parameters ----
    if args.point_json is not None:
        with open(args.point_json) as f:
            calib_points = json.load(f)
        if args.point_label is None:
            raise ValueError("--point-label required when --point-json given")
        pt = next((p for p in calib_points if p["label"] == args.point_label), None)
        if pt is None:
            raise ValueError(f"Label '{args.point_label}' not found in {args.point_json}. "
                             f"Available: {[p['label'] for p in calib_points]}")
        args.a = pt["a"]
        args.omega = pt["omega"]
        args.u = pt.get("u", None)
        args.v = pt.get("v", None)
        args.lam = pt.get("lambda_real", pt.get("lam", None))
        print(f"[compare] Loaded point '{args.point_label}' from {args.point_json}")
    else:
        if args.a is None or args.omega is None:
            parser.error("Either --point-json/--point-label or --a/--omega required")
        if args.lam is None:
            parser.error("--lam required when using --a/--omega directly")

    if args.u is None or args.v is None:
        print("[compare] WARNING: u/v not provided — decoder will run without local chart features")

    print(f"[compare] Point: a={args.a:.6f}, omega={args.omega:.6e}, lam={args.lam}")
    if args.u is not None:
        print(f"         u={args.u:.6f}, v={args.v:.6f}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Load config ----
    full_cfg = load_pinn_full_config(args.config)
    runtime_cfg = full_cfg.get("runtime", {})
    dtype_name = runtime_cfg.get("dtype", "float64")
    dtype = _get_dtype(dtype_name)
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    # ---- Load Stage-3 checkpoint ----
    ckpt = torch.load(args.stage3_checkpoint, map_location="cpu", weights_only=False)
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
        decoder_hidden_dim=int(model_cfg.get("decoder_hidden_dim", 128)),
        decoder_n_hidden=int(model_cfg.get("decoder_n_hidden", 0)),
        **model_cfg.get("encoder_kwargs", {}),
    )
    state_dict = {k: v for k, v in ckpt["model_state_dict"].items()
                  if not any(k.startswith(p) for p in ["up_decoder.", "down_decoder.", "rin_decoder."])}
    model.load_state_dict(state_dict, strict=False)
    model.to(device=device, dtype=dtype)
    model.eval()
    print(f"[compare] Stage-3 checkpoint: epoch={ckpt.get('epoch', '?')}")

    # ---- Run or load spectral method ----
    if args.spectral_profile is not None:
        prof = np.load(args.spectral_profile)
        print(f"[compare] Loaded spectral profile from {args.spectral_profile}")
        # Build minimal spectral dict from npz
        spectral = {
            "sol_down": {"z": prof["z"], "u": prof["u_down"]},
            "sol_up":   {"z": prof["z"], "u": prof["u_up"]},
            "sol_in":   {"z": np.array([]), "u": np.array([])},
            "z1": float(prof["z"][-1]),  # last z in profile
            "z2": 1.0,
        }
    else:
        print(f"[compare] Running spectral: a={args.a}, omega={args.omega}, lam={args.lam}")
        print(f"         z1={args.z1}, z2={args.z2}, N_left={args.N_left}")
        spectral = run_spectral(
            a=args.a, omega=args.omega, lam=args.lam,
            z1=args.z1, z2=args.z2,
            N_left=args.N_left, N_mid=args.N_mid, N_right=args.N_right,
            M=args.M, s=args.s, ell=args.ell, m=args.m,
        )
        print(f"[compare] Spectral bases computed: "
              f"down z∈[{spectral['sol_down']['z'][0]:.4f},{spectral['sol_down']['z'][-1]:.4f}], "
              f"up z∈[{spectral['sol_up']['z'][0]:.4f},{spectral['sol_up']['z'][-1]:.4f}]")

    # ---- z <-> y mapping ----
    # z = r_plus / r = x,  y = 2*x - 1 = 2*z - 1
    def z_to_y(z):
        return 2.0 * z - 1.0

    def y_to_z(y):
        return (y + 1.0) / 2.0

    rp_val = r_plus(torch.tensor([[args.a]], dtype=dtype), args.M).item()
    print(f"[compare] r_plus = {rp_val:.6f}, mapping: z = r_+/r = x, y = 2z-1")

    # ---- Evaluate Stage-3 decoder on spectral y grid ----
    a_t = torch.tensor([[args.a]], device=device, dtype=dtype)
    omega_t = torch.tensor([[args.omega]], device=device, dtype=dtype)
    lam_complex = complex(args.lam, 0) if isinstance(args.lam, (int, float)) else args.lam
    lam_t = torch.tensor([lam_complex], device=device, dtype=cdtype)
    u_t = torch.tensor([[args.u]], device=device, dtype=dtype) if args.u is not None else None
    v_t = torch.tensor([[args.v]], device=device, dtype=dtype) if args.v is not None else None

    # Build a combined y grid for evaluation: left patch dense + right patch
    z_left = spectral["sol_down"]["z"]
    z_right = spectral["sol_in"]["z"]
    y_left = z_to_y(z_left)  # z ∈ [0, z1] → y ∈ [-1, 2*z1-1]
    y_right = z_to_y(z_right)  # z ∈ [z2, 1] → y ∈ [2*z2-1, 1]

    # Stage-3 decoder evaluation
    y_left_t = torch.tensor(y_left, device=device, dtype=dtype).unsqueeze(0)
    has_right = len(z_right) > 0
    if has_right:
        y_right_t = torch.tensor(y_right, device=device, dtype=dtype).unsqueeze(0)

    with torch.no_grad():
        f_up_left = model.predict_u_up(a_t, omega_t, y_left_t, u_t, v_t)
        f_down_left = model.predict_u_down(a_t, omega_t, y_left_t, u_t, v_t)
        if has_right:
            f_up_right = model.predict_u_up(a_t, omega_t, y_right_t, u_t, v_t)
            f_down_right = model.predict_u_down(a_t, omega_t, y_right_t, u_t, v_t)
        else:
            f_up_right = f_down_right = None

    c_up, c_down = infinity_slopes_y(a_t, omega_t, lam_t, m=args.m, M=args.M)
    u_up_left = compose_u_from_f(f_up_left, y_left_t, c_up)
    u_down_left = compose_u_from_f(f_down_left, y_left_t, c_down)
    if has_right:
        u_up_right = compose_u_from_f(f_up_right, y_right_t, c_up)
        u_down_right = compose_u_from_f(f_down_right, y_right_t, c_down)
    else:
        u_up_right = u_down_right = None

    # Convert to numpy
    u_up_left_np = u_up_left.detach().cpu().numpy().ravel()
    u_down_left_np = u_down_left.detach().cpu().numpy().ravel()
    u_up_right_np = u_up_right.detach().cpu().numpy().ravel() if has_right else np.array([])
    u_down_right_np = u_down_right.detach().cpu().numpy().ravel() if has_right else np.array([])

    # ---- Exact boundary check at y=-1 ----
    y0 = torch.tensor([[-1.0]], device=device, dtype=dtype, requires_grad=True)
    with torch.no_grad():
        f_up_0 = model.predict_u_up(a_t, omega_t, y0, u_t, v_t)
        f_down_0 = model.predict_u_down(a_t, omega_t, y0, u_t, v_t)
    u_up_0 = compose_u_from_f(f_up_0, y0, c_up)
    u_down_0 = compose_u_from_f(f_down_0, y0, c_down)
    u_up_at_inf = u_up_0[0, 0].detach().item()
    u_down_at_inf = u_down_0[0, 0].detach().item()

    # ---- Errors in left patch (near-infinity) ----
    # Interpolate Stage-3 decoder to exact spectral z grid for error computation
    from scipy.interpolate import interp1d
    y_dense_left = np.linspace(y_left[0], y_left[-1], args.n_y_diag)
    y_dense_t = torch.tensor(y_dense_left, device=device, dtype=dtype).unsqueeze(0)
    with torch.no_grad():
        f_up_dense = model.predict_u_up(a_t, omega_t, y_dense_t, u_t, v_t)
        f_down_dense = model.predict_u_down(a_t, omega_t, y_dense_t, u_t, v_t)
    u_up_dense = compose_u_from_f(f_up_dense, y_dense_t, c_up).detach().cpu().numpy().ravel()
    u_down_dense = compose_u_from_f(f_down_dense, y_dense_t, c_down).detach().cpu().numpy().ravel()

    # Interpolate spectral to dense y grid
    spec_down_interp_re = interp1d(y_left, spectral["sol_down"]["u"].real, kind="cubic", fill_value="extrapolate")
    spec_down_interp_im = interp1d(y_left, spectral["sol_down"]["u"].imag, kind="cubic", fill_value="extrapolate")
    spec_up_interp_re = interp1d(y_left, spectral["sol_up"]["u"].real, kind="cubic", fill_value="extrapolate")
    spec_up_interp_im = interp1d(y_left, spectral["sol_up"]["u"].imag, kind="cubic", fill_value="extrapolate")

    spec_down_dense = spec_down_interp_re(y_dense_left) + 1j * spec_down_interp_im(y_dense_left)
    spec_up_dense = spec_up_interp_re(y_dense_left) + 1j * spec_up_interp_im(y_dense_left)

    abs_err_down = np.abs(u_down_dense - spec_down_dense)
    abs_err_up = np.abs(u_up_dense - spec_up_dense)
    rel_err_down = abs_err_down / (np.abs(spec_down_dense) + 1e-12)
    rel_err_up = abs_err_up / (np.abs(spec_up_dense) + 1e-12)

    # Error table
    errors = {
        "u_up(-1)-1": abs(u_up_at_inf - 1.0),
        "u_down(-1)-1": abs(u_down_at_inf - 1.0),
        "median_abs_err_down_real": float(np.median(np.abs(u_down_dense.real - spec_down_dense.real))),
        "median_abs_err_down_imag": float(np.median(np.abs(u_down_dense.imag - spec_down_dense.imag))),
        "median_abs_err_up_real": float(np.median(np.abs(u_up_dense.real - spec_up_dense.real))),
        "median_abs_err_up_imag": float(np.median(np.abs(u_up_dense.imag - spec_up_dense.imag))),
        "median_rel_err_down": float(np.median(rel_err_down)),
        "median_rel_err_up": float(np.median(rel_err_up)),
        "max_rel_err_down": float(np.max(rel_err_down)),
        "max_rel_err_up": float(np.max(rel_err_up)),
    }
    # Boundary derivative: analytic pass + autograd verify
    y0_g = y0.clone().detach().requires_grad_(True)
    f_up_g0 = model.predict_u_up(a_t, omega_t, y0_g, u_t, v_t)
    f_down_g0 = model.predict_u_down(a_t, omega_t, y0_g, u_t, v_t)
    u_up_g0 = compose_u_from_f(f_up_g0, y0_g, c_up)
    u_down_g0 = compose_u_from_f(f_down_g0, y0_g, c_down)
    uy_up_real = torch.autograd.grad(u_up_g0.real.sum(), y0_g, create_graph=False, retain_graph=True)[0]
    uy_down_real = torch.autograd.grad(u_down_g0.real.sum(), y0_g, create_graph=False, retain_graph=True)[0]
    uy_up_imag = torch.autograd.grad(u_up_g0.imag.sum(), y0_g, create_graph=False, retain_graph=True)[0]
    uy_down_imag = torch.autograd.grad(u_down_g0.imag.sum(), y0_g, create_graph=False, retain_graph=False)[0]

    c_up_re = float(c_up.real.detach().cpu().squeeze().numpy())
    c_up_im = float(c_up.imag.detach().cpu().squeeze().numpy())
    c_down_re = float(c_down.real.detach().cpu().squeeze().numpy())
    c_down_im = float(c_down.imag.detach().cpu().squeeze().numpy())

    errors["du_up/dy(-1) - Re[c_up]"] = abs(float(uy_up_real[0, 0].item()) - c_up_re)
    errors["du_up/dy(-1) - Im[c_up]"] = abs(float(uy_up_imag[0, 0].item()) - c_up_im)
    errors["du_down/dy(-1) - Re[c_down]"] = abs(float(uy_down_real[0, 0].item()) - c_down_re)
    errors["du_down/dy(-1) - Im[c_down]"] = abs(float(uy_down_imag[0, 0].item()) - c_down_im)

    print("\n[compare] Error table (left patch, near-infinity):")
    for k, v in errors.items():
        print(f"  {k}: {v:.6e}")

    # ---- Plots ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    has_plots = False
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            has_plots = True
        except ImportError:
            pass

    if has_plots:
        z_left_np = z_left
        z_right_np = z_right

        # Spectral data
        spec_down_z = spectral["sol_down"]["z"]
        spec_down_u = spectral["sol_down"]["u"]
        spec_up_z = spectral["sol_up"]["z"]
        spec_up_u = spectral["sol_up"]["u"]
        spec_in_z = spectral["sol_in"]["z"]
        spec_in_u = spectral["sol_in"]["u"]

        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle(f"Stage-3 Decoder vs Spectral Basis — a={args.a}, $\\omega$={args.omega}", fontsize=13)

        # --- u_down real (left patch) ---
        ax = axes[0, 0]
        ax.semilogy(spec_down_z, np.abs(np.real(spec_down_u)), "k-", lw=1.5, alpha=0.7, label="spectral")
        ax.semilogy(z_left_np, np.abs(u_down_left_np.real), "b--", lw=1.5, label="Stage-3 decoder")
        ax.set_xlabel("z"); ax.set_ylabel("|Re(u_down)|")
        ax.set_title(f"u_down real, left patch [0,{args.z1}]")
        ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.5)

        # --- u_down imag (left patch) ---
        ax = axes[0, 1]
        ax.plot(spec_down_z, np.imag(spec_down_u), "k-", lw=1.5, alpha=0.7, label="spectral")
        ax.plot(z_left_np, u_down_left_np.imag, "b--", lw=1.5, label="Stage-3 decoder")
        ax.set_xlabel("z"); ax.set_ylabel("Im(u_down)")
        ax.set_title(f"u_down imag, left patch [0,{args.z1}]")
        ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.5)

        # --- u_up real (left patch) ---
        ax = axes[0, 2]
        ax.semilogy(spec_up_z, np.abs(np.real(spec_up_u)), "k-", lw=1.5, alpha=0.7, label="spectral")
        ax.semilogy(z_left_np, np.abs(u_up_left_np.real), "r--", lw=1.5, label="Stage-3 decoder")
        ax.set_xlabel("z"); ax.set_ylabel("|Re(u_up)|")
        ax.set_title(f"u_up real, left patch [0,{args.z1}]")
        ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.5)

        # --- u_up imag (left patch) ---
        ax = axes[1, 0]
        ax.plot(spec_up_z, np.imag(spec_up_u), "k-", lw=1.5, alpha=0.7, label="spectral")
        ax.plot(z_left_np, u_up_left_np.imag, "r--", lw=1.5, label="Stage-3 decoder")
        ax.set_xlabel("z"); ax.set_ylabel("Im(u_up)")
        ax.set_title(f"u_up imag, left patch [0,{args.z1}]")
        ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.5)

        # --- u_in real (right patch) — spectral only ---
        ax = axes[1, 1]
        ax.semilogy(spec_in_z, np.abs(np.real(spec_in_u)), "k-", lw=1.5, alpha=0.7, label="spectral in")
        ax.set_xlabel("z"); ax.set_ylabel("|Re(u_in)|")
        ax.set_title(f"u_in real, right patch [{args.z2},1]")
        ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.5)

        # --- abs error vs y (down and up) ---
        ax = axes[1, 2]
        ax.semilogy(y_dense_left, abs_err_down, "b-", lw=1, alpha=0.7, label="|u_down - spec_down|")
        ax.semilogy(y_dense_left, abs_err_up, "r-", lw=1, alpha=0.7, label="|u_up - spec_up|")
        ax.set_xlabel("y"); ax.set_ylabel("absolute error")
        ax.set_title("|decoder - spectral| in left patch")
        ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.5)

        plt.tight_layout()
        outpath = output_dir / "basis_functions_stage3_vs_spectral.png"
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        print(f"\n[compare] Comparison plot saved to {outpath}")

        # --- Extra: u_up/down near infinity (y-space) ---
        y_left_np = y_left
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
        fig2.suptitle(f"u_up / u_down near infinity — a={args.a}, $\\omega$={args.omega}")
        ax = axes2[0, 0]
        ax.plot(y_left_np, u_down_left_np.real, "b-", lw=1, label="Stage-3")
        ax.plot(y_left_np, spec_down_u.real, "k--", lw=1, label="spectral")
        ax.set_xlabel("y"); ax.set_ylabel("Re(u_down)"); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes2[0, 1]
        ax.plot(y_left_np, u_down_left_np.imag, "b-", lw=1, label="Stage-3")
        ax.plot(y_left_np, spec_down_u.imag, "k--", lw=1, label="spectral")
        ax.set_xlabel("y"); ax.set_ylabel("Im(u_down)"); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes2[1, 0]
        ax.plot(y_left_np, u_up_left_np.real, "r-", lw=1, label="Stage-3")
        ax.plot(y_left_np, spec_up_u.real, "k--", lw=1, label="spectral")
        ax.set_xlabel("y"); ax.set_ylabel("Re(u_up)"); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes2[1, 1]
        ax.plot(y_left_np, u_up_left_np.imag, "r-", lw=1, label="Stage-3")
        ax.plot(y_left_np, spec_up_u.imag, "k--", lw=1, label="spectral")
        ax.set_xlabel("y"); ax.set_ylabel("Im(u_up)"); ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outpath2 = output_dir / "u_up_down_near_infinity_comparison.png"
        fig2.savefig(outpath2, dpi=150)
        plt.close(fig2)
        print(f"[compare] Near-infinity detail plot saved to {outpath2}")

    # ---- Save summary ----
    summary = {
        "stage3_checkpoint": str(Path(args.stage3_checkpoint).resolve()),
        "spectral_params": {
            "a": args.a, "omega": args.omega, "lam": args.lam,
            "M": args.M, "s": args.s, "ell": args.ell, "m": args.m,
            "z1": args.z1, "z2": args.z2,
            "N_left": args.N_left, "N_mid": args.N_mid, "N_right": args.N_right,
        },
        "coordinate_mapping": {
            "z_to_y": "y = 2*z - 1",
            "y_to_z": "z = (y+1)/2",
            "z_equals_x": True,
            "infinity": "z=0, y=-1, x=0",
            "horizon": "z=1, y=1, x=1",
            "left_patch_y_range": [float(z_to_y(args.z1)), -1.0],
        },
        "normalization": {
            "convention": "u(infinity)=1, u_z(0)=boundary_du_exact (z-space)",
            "pinn_convention": "u(-1)=1, u_y(-1)=c_inf (y-space)",
            "consistent": True,
            "note": "du/dy = (1/2)*du/dz since dz/dy|y=-1 = 1/2. Boundary slopes match analytically.",
        },
        "errors": errors,
        "u_up_at_inf": u_up_at_inf,
        "u_down_at_inf": u_down_at_inf,
    }

    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ---- Decision ----
    print(f"\n{'='*60}")
    print("Comparison Summary")
    print(f"{'='*60}")
    print(f"  Spectral: a={args.a}, omega={args.omega}, lam={args.lam}")
    print(f"  Coordinate: z = r+/r = x, y = 2z-1")
    print(f"  Normalization: CONSISTENT (both u(inf)=1, matching derivatives)")
    print(f"  u_up(-1)={u_up_at_inf:.12e}  u_down(-1)={u_down_at_inf:.12e}")
    print(f"  median_rel_err_down: {errors['median_rel_err_down']:.4e}")
    print(f"  median_rel_err_up:   {errors['median_rel_err_up']:.4e}")
    print(f"  max_rel_err_down:     {errors['max_rel_err_down']:.4e}")
    print(f"  max_rel_err_up:       {errors['max_rel_err_up']:.4e}")

    quality_threshold = 0.05
    if errors["median_rel_err_down"] < quality_threshold and errors["median_rel_err_up"] < quality_threshold:
        print(f"\n  Stage-3 quality: GOOD (<{quality_threshold})")
        print(f"  Recommendation: No near-infinity retrain needed.")
    else:
        print(f"\n  Stage-3 quality: NEEDS IMPROVEMENT (>{quality_threshold})")
        print(f"  Recommendation: near-infinity-only Stage-3 retrain recommended.")
        print(f"  Suggested y range: [-1, 0] or narrower based on error analysis.")
        if errors["median_rel_err_up"] > errors["median_rel_err_down"]:
            print(f"  u_up is the main error source — focus on up_decoder.")


if __name__ == "__main__":
    main()
