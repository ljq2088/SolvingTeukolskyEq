#!/usr/bin/env python3
"""
Compare S(y) output from v2 checkpoint (step_20000) with v3 reference.

Generates reference-style plots: R(r) left, S(y) right.
Uses the same ref_sample as the v3 training for fair comparison.
"""
import sys
# IMPORTANT: PROJECT_ROOT must come LAST (inserted at position 0 last = highest priority)
sys.path.insert(0, "/home/ljq/code/PINN/SolvingTeukolsky")
sys.path.insert(0, "/home/ljq/code/PINN/SolvingTeukolsky/pybhpt")
sys.path.insert(0, "/home/ljq/code/PINN/SolvingTeukolskyEq_autoencoder")

import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config.config_loader import load_pinn_full_config
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from utils.amplitude import TeukRadAmplitudeInWithInterpolant
from utils.mode import KerrMode
from physical_ansatz.residual_pinn import get_lambda_from_cfg
from physical_ansatz.transform_y import h_factor
from physical_ansatz.mapping import r_plus
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives
from dataset.sampling import sample_points_chebyshev_grid

device = torch.device("cuda")
dtype = torch.float64

OUT_DIR = Path("outputs/stage1_retrain/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Load config
# ============================================================
cfg = load_pinn_full_config("config/autoencoder_stage1_retrain.yaml")
physics_cfg = cfg["physics"]
problem = physics_cfg["problem"]
M = float(problem.get("M", 1.0))
l = int(problem.get("l", 2))
m = int(problem.get("m", 2))
s = int(problem.get("s", -2))
viz_r_min = float(cfg.get("visualization", {}).get("viz_r_min", 2.0))
viz_r_max = float(cfg.get("visualization", {}).get("viz_r_max", 1000.0))
viz_num_points = int(cfg.get("visualization", {}).get("viz_num_points", 400))

# ============================================================
# 2. Load v2 checkpoint
# ============================================================
v2_ckpt_path = Path("outputs/stage1_retrain/20260523_032201_patch_000_comp_0_u_0.500_v_0.603/checkpoints/step_020000.pt")
v2_ckpt = torch.load(v2_ckpt_path, map_location=device)
print(f"v2: step={v2_ckpt['step']}, best_val_mean={v2_ckpt['best_val_mean']:.6f}")

# Load v3 checkpoint
v3_ckpt_path = Path("outputs/stage1_retrain/20260523_143137_patch_000_comp_0_u_0.500_v_0.603/checkpoints/step_020000.pt")
v3_ckpt = torch.load(v3_ckpt_path, map_location=device)
print(f"v3: step={v3_ckpt['step']}, best_val_mean={v3_ckpt['best_val_mean']:.6f}")

# ============================================================
# 3. Build models & load weights
# ============================================================
def build_model_from_ckpt(ckpt):
    enc_cfg = ckpt.get("encoder_config", {})
    patch_center = ckpt["patch_center"]
    model = AutoencoderTeukolskyPINN(
        hidden_dims=[128, 128, 128, 128],
        activation="silu",
        fourier_num_freqs=2,
        fourier_base_scale=1.0,
        param_embed_dim=64,
        use_film=True,
        use_residual=True,
        local_coord_mode="chart_uv",
        a_center_local=enc_cfg.get("a_center_local", 0.125),
        a_half_range_local=enc_cfg.get("a_half_range_local", 0.075),
        omega_min_local=enc_cfg.get("omega_min_local", 1.0e-4),
        omega_max_local=enc_cfg.get("omega_max_local", 10.0),
        u_center_local=patch_center["u"],
        v_center_local=patch_center["v"],
        u_half_range_local=enc_cfg.get("u_half_range_local", 0.125),
        v_half_range_local=enc_cfg.get("v_half_range_local", 0.125),
        M=M,
        m_mode=m,
    ).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model

model_v2 = build_model_from_ckpt(v2_ckpt)
model_v3 = build_model_from_ckpt(v3_ckpt)
print("Both models loaded OK")

# ============================================================
# 4. Use v3's ref_sample (or pick from validation pool)
# ============================================================
# Load val pool data
from domain.patch_cover import load_patch_cover, load_valid_chart_points

probe_json = "outputs/domain/probe_l2_m2.json"
atlas_json = "outputs/domain/atlas_l2_m2_logw.json"
patch_json = "outputs/domain/patch_cover_l2_m2_logw.json"

patch_cover = load_patch_cover(patch_json)
patches = [p for p in patch_cover.patches if p.patch_id == 0]
patch = patches[0]

uv_points, aw_points = load_valid_chart_points(
    probe_json=probe_json,
    atlas_json=atlas_json,
    component_id=patch.component_id,
    omega_chart_mode=patch_cover.meta.get("omega_chart_mode", "linear"),
)
mask = (
    (np.abs(uv_points[:, 0] - patch.u_center) <= patch.h_u)
    & (np.abs(uv_points[:, 1] - patch.v_center) <= patch.h_v)
)
patch_uv = uv_points[mask]
patch_aw = aw_points[mask]

# Pick patch-center nearest point
du = patch_uv[:, 0] - patch.u_center
dv = patch_uv[:, 1] - patch.v_center
idx = int(np.argmin(du * du + dv * dv))
a_ref = torch.tensor(float(patch_aw[idx, 0]), device=device, dtype=dtype)
omega_ref = torch.tensor(float(patch_aw[idx, 1]), device=device, dtype=dtype)
u_ref = torch.tensor(float(patch_uv[idx, 0]), device=device, dtype=dtype)
v_ref = torch.tensor(float(patch_uv[idx, 1]), device=device, dtype=dtype)
print(f"Ref sample: a={a_ref.item():.6f}, omega={omega_ref.item():.6f}, u={u_ref.item():.3f}, v={v_ref.item():.3f}")

# ============================================================
# 5. Compute reference solution (spectral)
# ============================================================
from physical_ansatz.residual import AuxCache
cache = AuxCache()
lam = get_lambda_from_cfg(physics_cfg, cache, a_ref, omega_ref)
h2 = h_factor(a_ref, omega_ref, m=m, M=M, s=s)

rp = r_plus(a_ref, M)
r_min = max(viz_r_min, float(rp.detach().cpu().item()) + 1.0e-4)
r_max = viz_r_max

# r-grid for left plots
r_uniform = torch.linspace(r_min, r_max, viz_num_points, device=device, dtype=dtype)
x_from_r = rp / r_uniform
y_from_r = 2.0 * x_from_r - 1.0

# cheb y-grid for right plots
y_min = 2.0 * float(rp.detach().cpu().item()) / r_max - 1.0
y_max_val = 2.0 * float(rp.detach().cpu().item()) / r_min - 1.0
y_cheb = sample_points_chebyshev_grid(
    n_points=viz_num_points, y_min=y_min, y_max=y_max_val,
    device=device, dtype=dtype,
)
x_from_y = 0.5 * (y_cheb + 1.0)
r_from_y = rp / x_from_y

# Spectral benchmark
a_sc = float(a_ref.cpu().item())
omega_sc = float(omega_ref.cpu().item())
mode = KerrMode(M=M, a=a_sc, omega=omega_sc, ell=l, m=m, lam=complex(lam.cpu().item()), s=s)
spectral = TeukRadAmplitudeInWithInterpolant(mode=mode, N_in=64, N_out=64, z_m=0.3)
profile = spectral.profile
R_ref_r_np = np.asarray(profile.R_of_r(r_uniform.cpu().numpy()), dtype=np.complex128)
R_ref_y_np = np.asarray(profile.R_of_r(r_from_y.cpu().numpy()), dtype=np.complex128)

# ============================================================
# 6. Predict for both models
# ============================================================
def predict(model, a, omega, lam, u, v, y_query, r_query):
    """Returns: R_pred_r_np, shape_pred_y_np, shape_ref_y_np"""
    with torch.no_grad():
        y_query = y_query.unsqueeze(0)  # (1, N)
        a_b = a.unsqueeze(0)
        omega_b = omega.unsqueeze(0)
        u_b = u.unsqueeze(0)
        v_b = v.unsqueeze(0)

        f_y = model(a_b, omega_b, y_query, u=u_b, v=v_b)  # (B, N)
        shape_y = f_y.squeeze(0)  # (N,)

        # R(r) from shape
        rp_r, rm_r, _, _, _ = build_prefactor_primitives(r_query, a, M=M, need_rs=False)
        P_r, _, _ = Leaver_prefactors(r_query, a, omega, m=m, M=M, s=s, rp=rp_r, rm=rm_r)
        h2_val = h_factor(a, omega, m=m, M=M, s=s)
        R_pred_r = P_r * h2_val * shape_y

    return R_pred_r.cpu().numpy(), shape_y.cpu().numpy()

# Compute for r-grid
rp_r, rm_r, _, _, _ = build_prefactor_primitives(r_uniform, a_ref, M=M, need_rs=False)
P_r, _, _ = Leaver_prefactors(r_uniform, a_ref, omega_ref, m=m, M=M, s=s, rp=rp_r, rm=rm_r)
h2_val_r = h_factor(a_ref, omega_ref, m=m, M=M, s=s)

# V2 prediction on r-grid
with torch.no_grad():
    f_v2_r = model_v2(a_ref.unsqueeze(0), omega_ref.unsqueeze(0), y_from_r.unsqueeze(0),
                       u=u_ref.unsqueeze(0), v=v_ref.unsqueeze(0))
    shape_v2_r = f_v2_r.squeeze(0)
    R_v2_r = (P_r * h2_val_r * shape_v2_r).cpu().numpy()

# V3 prediction on r-grid
with torch.no_grad():
    f_v3_r = model_v3(a_ref.unsqueeze(0), omega_ref.unsqueeze(0), y_from_r.unsqueeze(0),
                       u=u_ref.unsqueeze(0), v=v_ref.unsqueeze(0))
    shape_v3_r = f_v3_r.squeeze(0)
    R_v3_r = (P_r * h2_val_r * shape_v3_r).cpu().numpy()

# V2 prediction on cheb y-grid
with torch.no_grad():
    f_v2_y = model_v2(a_ref.unsqueeze(0), omega_ref.unsqueeze(0), y_cheb.unsqueeze(0),
                       u=u_ref.unsqueeze(0), v=v_ref.unsqueeze(0))
    shape_v2_y = f_v2_y.squeeze(0).cpu().numpy()

# V3 prediction on cheb y-grid
with torch.no_grad():
    f_v3_y = model_v3(a_ref.unsqueeze(0), omega_ref.unsqueeze(0), y_cheb.unsqueeze(0),
                       u=u_ref.unsqueeze(0), v=v_ref.unsqueeze(0))
    shape_v3_y = f_v3_y.squeeze(0).cpu().numpy()

# Compute S(y) reference on cheb grid
rp_y, rm_y, _, _, _ = build_prefactor_primitives(r_from_y, a_ref, M=M, need_rs=False)
P_y, _, _ = Leaver_prefactors(r_from_y, a_ref, omega_ref, m=m, M=M, s=s, rp=rp_y, rm=rm_y)
h2_y = complex(h_factor(a_ref, omega_ref, m=m, M=M, s=s).cpu().item())
shape_ref_y_np = R_ref_y_np / (P_y.cpu().numpy() * h2_y)

y_cheb_np = y_cheb.cpu().numpy()
r_uniform_np = r_uniform.cpu().numpy()

# ============================================================
# 7. Plot comparison
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Left column: R(r)
for row, (label, key) in enumerate([("Re(R)", "real"), ("Im(R)", "imag"), ("|R|", "abs")]):
    ax = axes[row, 0]
    if key == "real":
        v2_vals = np.real(R_v2_r)
        v3_vals = np.real(R_v3_r)
        ref_vals = np.real(R_ref_r_np)
    elif key == "imag":
        v2_vals = np.imag(R_v2_r)
        v3_vals = np.imag(R_v3_r)
        ref_vals = np.imag(R_ref_r_np)
    else:
        v2_vals = np.abs(R_v2_r)
        v3_vals = np.abs(R_v3_r)
        ref_vals = np.abs(R_ref_r_np)

    ax.plot(r_uniform_np, v2_vals, 'b-', label="v2 (scratch)", lw=1.6)
    ax.plot(r_uniform_np, v3_vals, 'g-', label="v3 (PINN warm)", lw=1.6)
    ax.plot(r_uniform_np, ref_vals, '--', color='orange', label="spectral ref", lw=1.0)
    ax.set_ylabel(label)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    if row == 2:
        ax.set_xlabel("r")

# Right column: S(y)
for row, (label, key) in enumerate([("Re(S)", "real"), ("Im(S)", "imag"), ("|S|", "abs")]):
    ax = axes[row, 1]
    if key == "real":
        v2_vals = np.real(shape_v2_y)
        v3_vals = np.real(shape_v3_y)
        ref_vals = np.real(shape_ref_y_np)
    elif key == "imag":
        v2_vals = np.imag(shape_v2_y)
        v3_vals = np.imag(shape_v3_y)
        ref_vals = np.imag(shape_ref_y_np)
    else:
        v2_vals = np.abs(shape_v2_y)
        v3_vals = np.abs(shape_v3_y)
        ref_vals = np.abs(shape_ref_y_np)

    ax.plot(y_cheb_np, v2_vals, 'b-', label="v2 (scratch)", lw=1.6)
    ax.plot(y_cheb_np, v3_vals, 'g-', label="v3 (PINN warm)", lw=1.6)
    ax.plot(y_cheb_np, ref_vals, '--', color='orange', label="spectral ref", lw=1.0)
    ax.set_ylabel(label)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    if row == 2:
        ax.set_xlabel("y")

# Compute errors
v2_err_R = np.median(np.abs(R_v2_r - R_ref_r_np) / (np.abs(R_ref_r_np) + 1e-12))
v3_err_R = np.median(np.abs(R_v3_r - R_ref_r_np) / (np.abs(R_ref_r_np) + 1e-12))

fig.suptitle(
    f"patch=0, step=20000, a={a_sc:.6f}, omega={omega_sc:.6f}\n"
    f"benchmark=spectral(N=64)\n"
    f"v2 (scratch) best_val={v2_ckpt['best_val_mean']:.4f}, medErrR={v2_err_R:.2e}  |  "
    f"v3 (PINN warm) best_val={v3_ckpt['best_val_mean']:.4f}, medErrR={v3_err_R:.2e}",
    fontsize=11,
)
fig.tight_layout()

save_path = OUT_DIR / "compare_v2_v3_step20000.png"
fig.savefig(save_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {save_path}")

# ============================================================
# 8. Show val from checkpoints (correct PDE residual, not |f|^2)
# ============================================================
print("\n=== Validation PDE residual (from checkpoint) ===")
print(f"{'idx':>3s}  {'a':>8s}  {'omega':>12s}  {'v2_loss':>10s}  {'v3_loss':>10s}")
v3_cases = {c["case_index"]: c for c in v3_ckpt["best_val_cases"]}

for case in v2_ckpt["best_val_cases"]:
    idx = case["case_index"]
    v2_loss = case["best_y_mean"]
    v3_case = v3_cases.get(idx, {})
    v3_loss = v3_case.get("best_y_mean", float("nan"))
    print(f"{idx:3d}  {case['a']:8.4f}  {case['omega']:12.8f}  {v2_loss:10.6f}  {v3_loss:10.6f}")

print(f"\n{'v2 best_val_mean:':>20s} {v2_ckpt['best_val_mean']:.6f}")
print(f"{'v3 best_val_mean:':>20s} {v3_ckpt['best_val_mean']:.6f}")

print("\nDone.")
