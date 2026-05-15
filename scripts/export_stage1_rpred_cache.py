#!/usr/bin/env python3
"""
Export Stage-1 R_pred artifact from a trained autoencoder checkpoint.

Outputs:
  outputs/stage1_artifacts/patch_XXX_YYY/
    metadata.json
    stage1_best_model.pt   (copy of checkpoint)
    rpred_cache.npz
    rpred_cache_summary.md
"""
import argparse
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT.parent / "SolvingTeukolsky"))
sys.path.append(str(PROJECT_ROOT.parent / "SolvingTeukolsky" / "pybhpt"))

import numpy as np
import torch
import yaml

from config.config_loader import load_pinn_full_config
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from domain.patch_cover import load_patch_cover, load_valid_chart_points
from domain.atlas_builder import load_atlas, map_from_chart
from utils.mode import KerrMode
from physical_ansatz.residual import AuxCache, get_lambda_from_cfg
from physical_ansatz.mapping import r_plus
from physical_ansatz.transform_y import h_factor, horizon_regularity_slope, compose_reduced_shape_from_f
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives


def _get_dtype(dtype_name):
    return torch.float32 if dtype_name == "float32" else torch.float64


def _get_autoencoder_encoder_config(model):
    enc = model.encoder
    return {
        "local_coord_mode": enc.local_coord_mode,
        "a_center_local": enc.a_center_local,
        "a_half_range_local": enc.a_half_range_local,
        "omega_min_local": enc.omega_min_local,
        "omega_max_local": enc.omega_max_local,
        "u_center_local": enc.u_center_local,
        "v_center_local": enc.v_center_local,
        "u_half_range_local": enc.u_half_range_local,
        "v_half_range_local": enc.v_half_range_local,
        "M": enc.M,
        "m_mode": enc.m_mode,
    }


def main():
    parser = argparse.ArgumentParser(description="Export Stage-1 R_pred artifact")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--probe-json", type=str, default=None)
    parser.add_argument("--atlas-json", type=str, required=True)
    parser.add_argument("--patch-json", type=str, required=True)
    parser.add_argument("--patch-id", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="best_model.pt path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-param", type=int, default=64)
    parser.add_argument("--n-y", type=int, default=256)
    parser.add_argument("--r-max", type=float, default=1000.0)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    full_cfg = load_pinn_full_config(args.config)
    physics_cfg = full_cfg["physics"]
    runtime_cfg = full_cfg.get("runtime", {})
    dtype_name = runtime_cfg.get("dtype", "float64")
    dtype = _get_dtype(dtype_name)
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    problem_cfg = physics_cfg["problem"]
    M = float(problem_cfg.get("M", 1.0))
    ell = int(problem_cfg.get("l", 2))
    m_mode = int(problem_cfg.get("m", 2))
    s = int(problem_cfg.get("s", -2))

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_step = ckpt.get("step", None)
    ckpt_best_val = ckpt.get("best_val_mean", None)
    ckpt_patch_id = ckpt.get("patch_id", None)
    ckpt_patch_center = ckpt.get("patch_center", None)

    # Build model
    full_cfg_raw = ckpt.get("full_cfg", full_cfg)
    model_cfg = full_cfg_raw.get("model", full_cfg.get("model", {}))
    model_type = ckpt.get("model_type", "autoencoder")

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
    print(f"[export] Model loaded from {args.checkpoint}")
    print(f"         step={ckpt_step}, best_val_mean={ckpt_best_val}")

    encoder_config = _get_autoencoder_encoder_config(model)

    # Load patch cover and atlas
    patch_cover = load_patch_cover(args.patch_json)
    patches = [p for p in patch_cover.patches if p.patch_id == args.patch_id]
    if len(patches) != 1:
        raise ValueError(f"patch_id={args.patch_id} not found in {args.patch_json}")
    patch = patches[0]
    atlas = load_atlas(args.atlas_json)
    comp = atlas.components[patch.component_id]
    omega_chart_mode = patch_cover.meta.get("omega_chart_mode", "linear")

    # Sample (u,v) uniformly within patch rectangle, map back to (a,omega)
    if args.probe_json:
        uv_points, aw_points = load_valid_chart_points(
            probe_json=args.probe_json,
            atlas_json=args.atlas_json,
            component_id=patch.component_id,
            omega_chart_mode=omega_chart_mode,
        )
        mask = (
            (np.abs(uv_points[:, 0] - patch.u_center) <= patch.h_u)
            & (np.abs(uv_points[:, 1] - patch.v_center) <= patch.h_v)
        )
        patch_uv = uv_points[mask]
        patch_aw = aw_points[mask]
    else:
        n_grid = max(args.n_param * 4, 64)
        u_vals = np.linspace(patch.u_center - patch.h_u, patch.u_center + patch.h_u, n_grid)
        v_vals = np.linspace(patch.v_center - patch.h_v, patch.v_center + patch.h_v, n_grid)
        UU, VV = np.meshgrid(u_vals, v_vals)
        uv_flat = np.stack([UU.ravel(), VV.ravel()], axis=-1)

        aw_list = []
        uv_list = []
        for ui, vi in uv_flat:
            try:
                ai, wi = map_from_chart(comp, float(ui), float(vi), omega_chart_mode=omega_chart_mode)
                aw_list.append([float(ai), float(wi)])
                uv_list.append([float(ui), float(vi)])
            except Exception:
                continue
        patch_uv = np.array(uv_list, dtype=float)
        patch_aw = np.array(aw_list, dtype=float)

    n_avail = len(patch_uv)

    if n_avail == 0:
        raise RuntimeError(f"Patch {args.patch_id} contains no safe points.")

    # Subsample or oversample
    n_param = min(args.n_param, n_avail)
    if n_param < n_avail:
        idx = np.linspace(0, n_avail - 1, n_param, dtype=int)
    else:
        idx = np.arange(n_avail)

    a_params = patch_aw[idx, 0]
    omega_params = patch_aw[idx, 1]
    u_params = patch_uv[idx, 0]
    v_params = patch_uv[idx, 1]

    # Compute lambda for each param
    cache = AuxCache()
    lambda_list = []
    for i in range(n_param):
        lam = get_lambda_from_cfg(physics_cfg, cache,
                                   torch.tensor(float(a_params[i]), device=device, dtype=dtype),
                                   torch.tensor(float(omega_params[i]), device=device, dtype=dtype))
        lambda_list.append(complex(lam))
    lambdas = np.array(lambda_list, dtype=np.complex128)

    # Build y-grid and r-grid
    a_t = torch.tensor(a_params, device=device, dtype=dtype)
    omega_t = torch.tensor(omega_params, device=device, dtype=dtype)
    u_t = torch.tensor(u_params, device=device, dtype=dtype)
    v_t = torch.tensor(v_params, device=device, dtype=dtype)

    # r-grid: uniform from r_min to r_max
    r_min_vals = []
    for i in range(n_param):
        rp_i = r_plus(a_t[i:i+1], M).detach().cpu().item()
        r_min_vals.append(max(2.0, rp_i + 1e-4))
    r_min_global = max(r_min_vals)
    r_grid = np.linspace(r_min_global, args.r_max, args.n_y)
    r_t = torch.tensor(r_grid, device=device, dtype=dtype)

    # y from r
    rp_all = r_plus(a_t, M)  # (n_param,)
    y_grid = 2.0 * (rp_all.unsqueeze(1) / r_t.unsqueeze(0)) - 1.0  # (n_param, n_y)

    # Compute S_pred, P, h2, R_pred
    lam_t = torch.tensor(lambdas, device=device, dtype=torch.complex128)

    with torch.no_grad():
        # f_R from model (same as old PINN_MLP.forward)
        f_pred = model.predict_Rin(a_t, omega_t, y_grid, u=u_t, v=v_t)  # (n_param, n_y)

        # Transform f_R → reduced shape S_pred
        slopes = []
        for i in range(n_param):
            sl = horizon_regularity_slope(
                a_t[i:i+1], omega_t[i:i+1], lam_t[i:i+1],
                m=m_mode, M=M, s=s,
            ).squeeze(0)
            slopes.append(sl)
        slope_all = torch.stack(slopes)  # (n_param, 1)

        # compose_reduced_shape_from_f expects f of shape (n_param, n_y)
        S_pred = compose_reduced_shape_from_f(
            f=f_pred, y=y_grid, slope=slope_all,
        )  # (n_param, n_y)

        # P and h2 for each param
        P_all = []
        h2_all = []
        for i in range(n_param):
            rp_i, rm_i, _, _, _ = build_prefactor_primitives(
                r_t, a_t[i:i+1], M=M, need_rs=False,
            )
            P_i, _, _ = Leaver_prefactors(
                r_t, a_t[i:i+1], omega_t[i:i+1],
                m=m_mode, M=M, s=s, rp=rp_i, rm=rm_i,
            )
            P_all.append(P_i.squeeze(0))
            h2_i = h_factor(a_t[i:i+1], omega_t[i:i+1], m=m_mode, M=M, s=s)
            h2_all.append(h2_i.squeeze(0))

        P_mat = torch.stack(P_all)  # (n_param, n_y)
        h2_vec = torch.stack(h2_all)  # (n_param,)

        R_pred = P_mat * h2_vec.unsqueeze(1) * S_pred  # (n_param, n_y)
        R_pred_over_P = R_pred / P_mat  # (n_param, n_y)

    # Convert to numpy
    S_pred_np = S_pred.detach().cpu().numpy()
    P_np = P_mat.detach().cpu().numpy()
    h2_np = h2_vec.detach().cpu().numpy()
    R_pred_np = R_pred.detach().cpu().numpy()
    R_pred_over_P_np = R_pred_over_P.detach().cpu().numpy()
    y_np = y_grid.detach().cpu().numpy()

    # Infinity mask
    x_from_y = 0.5 * (y_np + 1.0)
    r_from_y = rp_all.detach().cpu().numpy()[:, None] / x_from_y
    mask_near_infinity = r_from_y > (0.8 * args.r_max)

    # Verify
    R_reconstructed = P_np * h2_np[:, None] * S_pred_np
    rel_diff = np.abs(R_pred_np - R_reconstructed) / np.maximum(np.abs(R_reconstructed), 1e-14)
    max_reldiff = float(np.max(rel_diff))
    assert np.all(np.isfinite(R_pred_np)), "R_pred has non-finite values!"
    print(f"[export] R = P*h2*S check: max relative diff = {max_reldiff:.2e}")
    print(f"[export] R_pred shape: {R_pred_np.shape}")
    print(f"[export] n_infinity points: {int(mask_near_infinity.sum())} / {mask_near_infinity.size}")

    # Save artifacts
    npz_path = out_dir / "rpred_cache.npz"
    np.savez_compressed(
        npz_path,
        a=a_params,
        omega=omega_params,
        u=u_params,
        v=v_params,
        y=y_np,
        r=r_grid,
        lambda_=lambdas,
        S_pred=S_pred_np,
        P=P_np,
        h2=h2_np,
        R_pred=R_pred_np,
        R_pred_over_P=R_pred_over_P_np,
        mask_near_infinity=mask_near_infinity,
    )
    print(f"[export] Saved {npz_path}")

    # Copy checkpoint
    import shutil
    ckpt_dest = out_dir / "stage1_best_model.pt"
    shutil.copy2(args.checkpoint, ckpt_dest)
    print(f"[export] Copied checkpoint to {ckpt_dest}")

    # metadata.json
    metadata = {
        "created": datetime.now().isoformat(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(ckpt_step) if ckpt_step is not None else None,
        "checkpoint_best_val_mean": float(ckpt_best_val) if ckpt_best_val is not None else None,
        "patch_id": int(args.patch_id),
        "patch_center": {
            "u": float(patch.u_center),
            "v": float(patch.v_center),
        },
        "omega_chart_mode": omega_chart_mode,
        "model_type": str(model_type),
        "encoder_config": encoder_config,
        "phyics": {
            "M": M, "l": ell, "m": m_mode, "s": s,
        },
        "scattering_convention": "R_in = B_ref * u_up * A_up + B_inc * u_down * A_down",
        "A_up": "r^3 * exp(i * omega * r_*)",
        "A_down": "r^{-1} * exp(-i * omega * r_*)",
        "n_param": int(n_param),
        "n_y": int(args.n_y),
        "r_min": float(r_min_global),
        "r_max": float(args.r_max),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[export] Wrote metadata.json")

    # summary.md
    lines = [
        "# Stage-1 R_pred Cache Summary",
        "",
        f"**Checkpoint:** `{args.checkpoint}`",
        f"**Step:** {ckpt_step}",
        f"**best_val_mean:** {ckpt_best_val}",
        f"**n_param:** {n_param}",
        f"**n_y:** {args.n_y}",
        f"**r range:** [{r_min_global:.4f}, {args.r_max:.1f}]",
        f"**a range:** [{a_params.min():.4f}, {a_params.max():.4f}]",
        f"**omega range:** [{omega_params.min():.6e}, {omega_params.max():.6e}]",
        "",
        "## Scattering Convention",
        "```",
        "R_in = B_ref * u_up * A_up + B_inc * u_down * A_down",
        "",
        "A_up   = r^3 * exp(i * omega * r_*)",
        "A_down = r^{-1} * exp(-i * omega * r_*)",
        "```",
        "",
        "## Consistency Check",
        f"- R = P*h2*S max relative diff: {max_reldiff:.2e}",
        f"- All finite: True",
        f"- Near-infinity points: {int(mask_near_infinity.sum())} / {mask_near_infinity.size}",
    ]
    with open(out_dir / "rpred_cache_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[export] Wrote rpred_cache_summary.md")
    print(f"[export] Done. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
