#!/usr/bin/env python3
"""加载 step_20000 checkpoint 并验证模型输出是否与 v2 一致。

生成 Image #7 风格的 rho(y) = |f(y)| 图。
"""
import sys
PROJECT_ROOT = "/home/ljq/code/PINN/SolvingTeukolskyEq_autoencoder"
# IMPORTANT: PROJECT_ROOT must come BEFORE SolvingTeukolsky so that
# model/autoencoder_pinn is found (SolvingTeukolsky also has model/ which shadows it)
sys.path.insert(0, "/home/ljq/code/PINN/SolvingTeukolsky/pybhpt")
sys.path.insert(0, "/home/ljq/code/PINN/SolvingTeukolsky")
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

device = torch.device("cuda")
dtype = torch.float64

# --- 1. Load checkpoint ---
ckpt_path = "/home/ljq/code/PINN/SolvingTeukolskyEq_autoencoder/outputs/stage1_retrain/20260523_032201_patch_000_comp_0_u_0.500_v_0.603/checkpoints/step_020000.pt"
ckpt = torch.load(ckpt_path, map_location=device)
print(f"Loaded checkpoint step={ckpt['step']}, best_val_mean={ckpt['best_val_mean']:.6f}")
print(f"Model type: {ckpt.get('model_type')}")
print(f"encoder_config: {ckpt.get('encoder_config')}")

# --- 2. Build model matching checkpoint architecture ---
from model.autoencoder_pinn import AutoencoderTeukolskyPINN

model_cfg = {
    "hidden_dims": [128, 128, 128, 128],
    "activation": "silu",
    "fourier_num_freqs": 2,
    "fourier_base_scale": 1.0,
    "param_embed_dim": 64,
    "use_film": True,
    "use_residual": True,
}

# Use encoder_config from checkpoint
enc_cfg = ckpt.get("encoder_config", {})
model = AutoencoderTeukolskyPINN(
    hidden_dims=model_cfg["hidden_dims"],
    activation=model_cfg["activation"],
    fourier_num_freqs=model_cfg["fourier_num_freqs"],
    fourier_base_scale=model_cfg["fourier_base_scale"],
    param_embed_dim=model_cfg["param_embed_dim"],
    use_film=model_cfg["use_film"],
    use_residual=model_cfg["use_residual"],
    local_coord_mode="chart_uv",
    a_center_local=enc_cfg.get("a_center_local", 0.125),
    a_half_range_local=enc_cfg.get("a_half_range_local", 0.075),
    omega_min_local=enc_cfg.get("omega_min_local", 1.0e-4),
    omega_max_local=enc_cfg.get("omega_max_local", 10.0),
    u_center_local=ckpt["patch_center"]["u"],
    v_center_local=ckpt["patch_center"]["v"],
    u_half_range_local=0.125,
    v_half_range_local=0.125,
    M=1.0,
    m_mode=2,
).to(device=device, dtype=dtype)

# --- 3. Load weights ---
state_dict = ckpt["model_state_dict"]
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing:
    print(f"WARNING: missing keys: {missing}")
if unexpected:
    print(f"WARNING: unexpected keys: {unexpected}")
if not missing and not unexpected:
    print("Weight loading: STRICT MATCH")

model.eval()

# --- 4. Check if weights actually changed from init ---
# (compare a few key params to make sure loading worked)
for name, param in model.named_parameters():
    if "encoder.base_blocks.0.linear.weight" in name:
        print(f"  {name}: mean={param.data.mean():.6e}, std={param.data.std():.6e}")
        break

# --- 5. Pick test cases from validation pool ---
# Use the same (a, omega, u, v) as checkpoint validation cases
val_cases = ckpt["best_val_cases"]
print(f"\nValidation cases in checkpoint: {len(val_cases)}")

# Pick a few representative cases: best, worst, and medium
case_best = min(val_cases, key=lambda c: c["best_y_mean"])
case_worst = max(val_cases, key=lambda c: c["best_y_mean"])
case_medium = sorted(val_cases, key=lambda c: c["best_y_mean"])[len(val_cases)//2]

test_cases = [
    ("Best val", case_best),
    ("Worst val", case_worst),
    ("Medium val", case_medium),
]

# --- 6. Compute f(y) and plot ---
from physical_ansatz.residual_pinn import compute_f_derivatives_autograd

y_grid = torch.linspace(-0.9999, 0.9999, 400, device=device, dtype=dtype)

fig, axes = plt.subplots(4, 3, figsize=(18, 22))
axes = axes.flatten()

for idx, (label, case) in enumerate(test_cases):
    a_val = torch.tensor([case["a"]], device=device, dtype=dtype)
    omega_val = torch.tensor([case["omega"]], device=device, dtype=dtype)
    u_val = torch.tensor([case["u"]], device=device, dtype=dtype)
    v_val = torch.tensor([case["v"]], device=device, dtype=dtype)

    f, fy, fyy = compute_f_derivatives_autograd(
        model, a_val, omega_val, y_grid.unsqueeze(0),
        u_batch=u_val, v_batch=v_val,
    )

    f_np = f.detach().squeeze().cpu().numpy()
    y_np = y_grid.cpu().numpy()

    # |f(y)| (rho)
    rho = np.abs(f_np)

    ax = axes[idx]
    ax.plot(y_np, rho, 'b-', linewidth=1.5)
    ax.set_title(f'{label}: a={case["a"]:.3f}, ω={case["omega"]:.4f}\n'
                 f'u={case["u"]:.3f}, v={case["v"]:.3f}, val_loss={case["best_y_mean"]:.4f}')
    ax.set_xlabel('y')
    ax.set_ylabel('|f(y)|')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Also plot phase
    phase = np.unwrap(np.angle(f_np))
    ax2 = ax.twinx()
    ax2.plot(y_np, phase, 'r-', linewidth=0.8, alpha=0.6)
    ax2.set_ylabel('phase (rad)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

# --- 7. Also plot a 3x3 grid of cases across the (a,ω) range ---
# Sort by omega to see the trend
sorted_cases = sorted(val_cases, key=lambda c: (c["omega"], c["a"]))
step = max(1, len(sorted_cases) // 9)
grid_cases = sorted_cases[::step][:9]

for idx, case in enumerate(grid_cases):
    ax_idx = idx + 3
    a_val = torch.tensor([case["a"]], device=device, dtype=dtype)
    omega_val = torch.tensor([case["omega"]], device=device, dtype=dtype)
    u_val = torch.tensor([case["u"]], device=device, dtype=dtype)
    v_val = torch.tensor([case["v"]], device=device, dtype=dtype)

    f, fy, fyy = compute_f_derivatives_autograd(
        model, a_val, omega_val, y_grid.unsqueeze(0),
        u_batch=u_val, v_batch=v_val,
    )

    rho = np.abs(f.detach().squeeze().cpu().numpy())

    ax = axes[ax_idx]
    ax.plot(y_np, rho, 'b-', linewidth=1.0)
    ax.set_title(f'a={case["a"]:.3f}, ω={case["omega"]:.4f}, loss={case["best_y_mean"]:.3f}')
    ax.set_xlabel('y')
    ax.set_ylabel('|f(y)|')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

fig.suptitle(f'Step 20000 checkpoint — |f(y)| across validation cases\n'
             f'best_val_mean={ckpt["best_val_mean"]:.6f}',
             fontsize=14)
plt.tight_layout()

out_dir = Path("outputs/stage1_retrain/diagnostics")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "verify_step20000_weights.png", dpi=120)
print(f"\nSaved: {out_dir / 'verify_step20000_weights.png'}")

# --- 8. Quick numerical check: compute val on a few cases ---
print("\n=== Quick validation check ===")
from physical_ansatz.residual_pinn import pinn_residual_loss

y_val = y_grid[:128]  # Use 128 chebyshev points for quick check
# Re-sample using chebyshev_half formula
N = 128
k = np.arange(1, N + 1)
y_cheb = 3.0 - 2.0 * np.cos(np.pi * k / (2 * N))
y_cheb = torch.tensor(y_cheb, device=device, dtype=dtype)

for label, case in test_cases:
    a_val = torch.tensor([case["a"]], device=device, dtype=dtype)
    omega_val = torch.tensor([case["omega"]], device=device, dtype=dtype)
    u_val = torch.tensor([case["u"]], device=device, dtype=dtype)
    v_val = torch.tensor([case["v"]], device=device, dtype=dtype)

    with torch.no_grad():
        loss, _ = pinn_residual_loss(
            model=model,
            cfg={"problem": {"M": 1.0, "s": -2, "m": 2}},
            a_batch=a_val.unsqueeze(0),
            omega_batch=omega_val.unsqueeze(0),
            lambda_batch=None,  # Will be computed inside
            y_interior=y_cheb,  # Use 128 chebyshev_half points
            y_boundary=torch.empty(0, device=device, dtype=dtype),
            weight_interior=1.0,
            weight_boundary=0.0,
            normalize_residual=False,
            residual_scale_eps=1.0e-12,
            return_pointwise=False,
            u_batch=u_val.unsqueeze(0),
            v_batch=v_val.unsqueeze(0),
        )
    print(f"  {label}: val={loss.item():.6f} (ckpt best={case['best_y_mean']:.6f})")

print("\nDone.")
