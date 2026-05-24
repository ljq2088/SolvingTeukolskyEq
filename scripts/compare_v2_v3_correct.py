#!/usr/bin/env python3
"""
Correct comparison: v2 vs v3 step_20000, pybhpt benchmark.
Properly transforms: f(y) → S(y) → R(r) using compose_reduced_shape_from_f.
"""
import sys
sys.path.insert(0, "/home/ljq/code/PINN/SolvingTeukolsky")
sys.path.insert(0, "/home/ljq/code/PINN/SolvingTeukolsky/pybhpt")
sys.path.insert(0, "/home/ljq/code/PINN/SolvingTeukolskyEq_autoencoder")

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config.config_loader import load_pinn_full_config

cfg = load_pinn_full_config("config/autoencoder_stage1_retrain.yaml")
physics_cfg = cfg["physics"]

from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.transform_y import (
    h_factor, compose_reduced_shape_from_f, horizon_regularity_slope,
)
from physical_ansatz.mapping import r_plus
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives
from physical_ansatz.residual import AuxCache
from physical_ansatz.residual_pinn import get_lambda_from_cfg
from pybhpt_usage.compute_solution import compute_pybhpt_solution
from dataset.sampling import sample_points_chebyshev_grid

device = torch.device("cuda")
dtype = torch.float64
OUT_DIR = Path("outputs/stage1_retrain/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load checkpoints
# ============================================================
v2_ckpt = torch.load(
    "outputs/stage1_retrain/20260523_032201_patch_000_comp_0_u_0.500_v_0.603/checkpoints/step_020000.pt",
    map_location=device)
v3_ckpt = torch.load(
    "outputs/stage1_retrain/20260523_143137_patch_000_comp_0_u_0.500_v_0.603/checkpoints/step_020000.pt",
    map_location=device)
print(f"v2: step={v2_ckpt['step']}, best_val_mean={v2_ckpt['best_val_mean']:.6f}")
print(f"v3: step={v3_ckpt['step']}, best_val_mean={v3_ckpt['best_val_mean']:.6f}")

# ============================================================
# Physics params (same for both)
# ============================================================
M = float(physics_cfg["problem"].get("M", 1.0))
l = int(physics_cfg["problem"].get("l", 2))
m_mode = int(physics_cfg["problem"].get("m", 2))
s = int(physics_cfg["problem"].get("s", -2))

# Viz params from config
viz_r_min = 2.0
viz_r_max = 1000.0
viz_n = 400

# ============================================================
# Build models
# ============================================================
def build_model(ckpt):
    enc = ckpt.get("encoder_config", {})
    pc = ckpt["patch_center"]
    model = AutoencoderTeukolskyPINN(
        hidden_dims=[128, 128, 128, 128],
        activation="silu",
        fourier_num_freqs=2,
        fourier_base_scale=1.0,
        param_embed_dim=64,
        use_film=True,
        use_residual=True,
        local_coord_mode="chart_uv",
        a_center_local=enc.get("a_center_local", 0.125),
        a_half_range_local=enc.get("a_half_range_local", 0.075),
        omega_min_local=enc.get("omega_min_local", 1e-4),
        omega_max_local=enc.get("omega_max_local", 10.0),
        u_center_local=pc["u"],
        v_center_local=pc["v"],
        u_half_range_local=enc.get("u_half_range_local", 0.125),
        v_half_range_local=enc.get("v_half_range_local", 0.125),
        M=M,
        m_mode=m_mode,
    ).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model

model_v2 = build_model(v2_ckpt)
model_v3 = build_model(v3_ckpt)
print("Both models built and weights loaded (strict match).")

# ============================================================
# Pick ref sample: patch-center nearest
# ============================================================
from domain.patch_cover import load_patch_cover, load_valid_chart_points
patch_cover = load_patch_cover("outputs/domain/patch_cover_l2_m2_logw.json")
patches = [p for p in patch_cover.patches if p.patch_id == 0]
patch = patches[0]
uv_pts, aw_pts = load_valid_chart_points(
    probe_json="outputs/domain/probe_l2_m2.json",
    atlas_json="outputs/domain/atlas_l2_m2_logw.json",
    component_id=patch.component_id,
    omega_chart_mode=patch_cover.meta.get("omega_chart_mode", "linear"),
)
mask = (np.abs(uv_pts[:, 0] - patch.u_center) <= patch.h_u) & \
       (np.abs(uv_pts[:, 1] - patch.v_center) <= patch.h_v)
uv, aw = uv_pts[mask], aw_pts[mask]
du = uv[:, 0] - patch.u_center
dv = uv[:, 1] - patch.v_center
idx = int(np.argmin(du*du + dv*dv))

a_ref = torch.tensor(float(aw[idx, 0]), device=device, dtype=dtype)
omega_ref = torch.tensor(float(aw[idx, 1]), device=device, dtype=dtype)
u_ref = torch.tensor(float(uv[idx, 0]), device=device, dtype=dtype)
v_ref = torch.tensor(float(uv[idx, 1]), device=device, dtype=dtype)
print(f"Ref: a={a_ref.item():.6f}, ω={omega_ref.item():.6f}, u={u_ref.item():.3f}, v={v_ref.item():.3f}")

# ============================================================
# Compute lambda
# ============================================================
cache = AuxCache()
lam = get_lambda_from_cfg(physics_cfg, cache, a_ref, omega_ref)
h2_val = h_factor(a_ref, omega_ref, m=m_mode, M=M, s=s)
print(f"lambda={lam.item():.6f}, h2={h2_val.item():.6f}")

# ============================================================
# Build grids
# ============================================================
rp_val = r_plus(a_ref, M)
r_min = max(viz_r_min, float(rp_val.cpu().item()) + 1e-4)
r_max = viz_r_max

# r-grid (uniform)
r_uni = torch.linspace(r_min, r_max, viz_n, device=device, dtype=dtype)
x_r = rp_val / r_uni
y_r = 2.0 * x_r - 1.0

# y-grid (chebyshev)
y_min_viz = 2.0 * float(rp_val.cpu().item()) / r_max - 1.0
y_max_viz = 2.0 * float(rp_val.cpu().item()) / r_min - 1.0
y_cheb = sample_points_chebyshev_grid(viz_n, y_min=y_min_viz, y_max=y_max_viz, device=device, dtype=dtype)
x_y = 0.5 * (y_cheb + 1.0)
r_y = rp_val / x_y

# ============================================================
# pybhpt benchmark
# ============================================================
a_sc = float(a_ref.cpu().item())
omega_sc = float(omega_ref.cpu().item())

# On r-grid
r_uni_np = r_uni.cpu().numpy()
order = np.argsort(r_uni_np)
inv_order = np.empty_like(order)
inv_order[order] = np.arange(order.size)
_, R_bhpt_r = compute_pybhpt_solution(a=a_sc, omega=omega_sc, ell=l, m=m_mode,
                                       r_grid=r_uni_np[order], timeout=30.0)
R_bhpt_r = np.asarray(R_bhpt_r, dtype=np.complex128)[inv_order]

# On y-grid
r_y_np = r_y.cpu().numpy()
order_y = np.argsort(r_y_np)
inv_order_y = np.empty_like(order_y)
inv_order_y[order_y] = np.arange(order_y.size)
_, R_bhpt_y = compute_pybhpt_solution(a=a_sc, omega=omega_sc, ell=l, m=m_mode,
                                       r_grid=r_y_np[order_y], timeout=30.0)
R_bhpt_y = np.asarray(R_bhpt_y, dtype=np.complex128)[inv_order_y]
print("pybhpt benchmark computed.")

# ============================================================
# Predict: f(y) → S(y) → R(r)
# ============================================================
def predict_all(model, a, omega, lam_, u, v, y_r_grid, r_grid, y_cheb_grid):
    """Returns R_on_r, S_on_y_cheb (both numpy complex)."""
    slope = horizon_regularity_slope(
        a=a.unsqueeze(0), omega=omega.unsqueeze(0),
        lambda_=lam_.unsqueeze(0), m=m_mode, M=M, s=s,
    ).squeeze(0)

    with torch.no_grad():
        # On r-grid
        f_r = model(a.unsqueeze(0), omega.unsqueeze(0), y_r_grid.unsqueeze(0),
                    u=u.unsqueeze(0), v=v.unsqueeze(0)).squeeze(0)
        S_r = compose_reduced_shape_from_f(f_r, y_r_grid, slope)
        rp_r, rm_r, _, _, _ = build_prefactor_primitives(r_grid, a, M=M, need_rs=False)
        P_r, _, _ = Leaver_prefactors(r_grid, a, omega, m=m_mode, M=M, s=s, rp=rp_r, rm=rm_r)
        h2_r = h_factor(a, omega, m=m_mode, M=M, s=s)
        R_r = (P_r * h2_r * S_r).cpu().numpy()

        # On cheb y-grid
        f_y = model(a.unsqueeze(0), omega.unsqueeze(0), y_cheb_grid.unsqueeze(0),
                    u=u.unsqueeze(0), v=v.unsqueeze(0)).squeeze(0)
        S_y = compose_reduced_shape_from_f(f_y, y_cheb_grid, slope).cpu().numpy()

    return R_r, S_y

R_v2_r, S_v2_y = predict_all(model_v2, a_ref, omega_ref, lam, u_ref, v_ref, y_r, r_uni, y_cheb)
R_v3_r, S_v3_y = predict_all(model_v3, a_ref, omega_ref, lam, u_ref, v_ref, y_r, r_uni, y_cheb)

# Benchmark S on cheb y-grid
rp_yv, rm_yv, _, _, _ = build_prefactor_primitives(r_y, a_ref, M=M, need_rs=False)
P_yv, _, _ = Leaver_prefactors(r_y, a_ref, omega_ref, m=m_mode, M=M, s=s, rp=rp_yv, rm=rm_yv)
h2_c = complex(h2_val.cpu().item())
S_bhpt_y = R_bhpt_y / (P_yv.cpu().numpy() * h2_c)

# Errors
v2_err = np.median(np.abs(R_v2_r - R_bhpt_r) / (np.abs(R_bhpt_r) + 1e-12))
v3_err = np.median(np.abs(R_v3_r - R_bhpt_r) / (np.abs(R_bhpt_r) + 1e-12))
print(f"v2 median relErr(R): {v2_err:.4e}")
print(f"v3 median relErr(R): {v3_err:.4e}")

# ============================================================
# Plot
# ============================================================
r_np = r_uni_np
y_np = y_cheb.cpu().numpy()

fig, axes = plt.subplots(3, 2, figsize=(15, 13))

titles_left = ["Re(R)", "Im(R)", "|R|"]
titles_right = ["Re(S)", "Im(S)", "|S|"]

for row in range(3):
    ax_l = axes[row, 0]
    ax_r = axes[row, 1]

    # Left: R(r)
    if row == 0:
        v2_l, v3_l, ref_l = np.real(R_v2_r), np.real(R_v3_r), np.real(R_bhpt_r)
    elif row == 1:
        v2_l, v3_l, ref_l = np.imag(R_v2_r), np.imag(R_v3_r), np.imag(R_bhpt_r)
    else:
        v2_l, v3_l, ref_l = np.abs(R_v2_r), np.abs(R_v3_r), np.abs(R_bhpt_r)

    ax_l.plot(r_np, ref_l, 'k-', label="pybhpt", lw=1.8)
    ax_l.plot(r_np, v2_l, 'b--', label=f"v2 (err={v2_err:.2e})", lw=1.4)
    ax_l.plot(r_np, v3_l, 'g--', label=f"v3 (err={v3_err:.2e})", lw=1.4)
    ax_l.set_ylabel(titles_left[row])
    ax_l.legend(fontsize=8)
    ax_l.grid(alpha=0.3)
    if row == 2:
        ax_l.set_xlabel("r")

    # Right: S(y)
    if row == 0:
        v2_rv, v3_rv, ref_rv = np.real(S_v2_y), np.real(S_v3_y), np.real(S_bhpt_y)
    elif row == 1:
        v2_rv, v3_rv, ref_rv = np.imag(S_v2_y), np.imag(S_v3_y), np.imag(S_bhpt_y)
    else:
        v2_rv, v3_rv, ref_rv = np.abs(S_v2_y), np.abs(S_v3_y), np.abs(S_bhpt_y)

    ax_r.plot(y_np, ref_rv, 'k-', label="pybhpt", lw=1.8)
    ax_r.plot(y_np, v2_rv, 'b--', label="v2", lw=1.4)
    ax_r.plot(y_np, v3_rv, 'g--', label="v3", lw=1.4)
    ax_r.set_ylabel(titles_right[row])
    ax_r.legend(fontsize=8)
    ax_r.grid(alpha=0.3)
    if row == 2:
        ax_r.set_xlabel("y")

fig.suptitle(
    f"patch=0 step=20000  a={a_sc:.6f}  ω={omega_sc:.6f}  benchmark=pybhpt\n"
    f"v2 best_val={v2_ckpt['best_val_mean']:.4f}  medRelErrR={v2_err:.2e}  |  "
    f"v3 best_val={v3_ckpt['best_val_mean']:.4f}  medRelErrR={v3_err:.2e}",
    fontsize=12,
)
fig.tight_layout()
sp = OUT_DIR / "compare_v2_v3_pybhpt.png"
fig.savefig(sp, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {sp}")

# ============================================================
# Per-case val from checkpoints
# ============================================================
print("\n=== Per-case PDE residual ===")
print(f"{'idx':>3s}  {'a':>8s}  {'omega':>12s}  {'v2_loss':>10s}  {'v3_loss':>10s}")
v3_cases = {c["case_index"]: c for c in v3_ckpt["best_val_cases"]}
for case in v2_ckpt["best_val_cases"]:
    i = case["case_index"]
    v2l = case["best_y_mean"]
    v3l = v3_cases.get(i, {}).get("best_y_mean", float("nan"))
    print(f"{i:3d}  {case['a']:8.4f}  {case['omega']:12.8f}  {v2l:10.6f}  {v3l:10.6f}")
print(f"\nv2 best_val_mean: {v2_ckpt['best_val_mean']:.6f}")
print(f"v3 best_val_mean: {v3_ckpt['best_val_mean']:.6f}")
print("Done.")
