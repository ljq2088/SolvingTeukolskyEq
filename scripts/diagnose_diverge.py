#!/usr/bin/env python3
"""Diagnose why training diverges from step 20000 checkpoint."""
import sys
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky/pybhpt")

import torch
import numpy as np
import random

# Reproduce exact training step
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)

device = torch.device("cuda")
dtype = torch.float64

ckpt_path = "/home/ljq/code/PINN/SolvingTeukolskyEq_autoencoder/outputs/stage1_retrain/20260523_032201_patch_000_comp_0_u_0.500_v_0.603/checkpoints/step_020000.pt"
ckpt = torch.load(ckpt_path, map_location=device)
print("=== Checkpoint keys ===")
for k, v in ckpt.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    elif isinstance(v, dict):
        print(f"  {k}: dict with keys {list(v.keys())[:5]}...")
    else:
        print(f"  {k}: {v}")

model_state = ckpt["model_state_dict"]
print(f"\n=== Model state_dict: {len(model_state)} keys ===")
for k in list(model_state.keys())[:10]:
    print(f"  {k}: shape={model_state[k].shape}")
print("  ...")

# Check optimizer state
if "optimizer_state_dict" in ckpt:
    opt = ckpt["optimizer_state_dict"]
    print(f"\n=== Optimizer state ===")
    for pg_idx, pg in enumerate(opt["param_groups"]):
        print(f"  param_group[{pg_idx}]: lr={pg.get('lr')}, weight_decay={pg.get('weight_decay')}, n_params={len(pg['params'])}")
    # Check a few momentum/variance stats
    state_keys = list(opt["state"].keys())[:3]
    for k in state_keys:
        st = opt["state"][k]
        print(f"  state[{k}]: exp_avg mean={st['exp_avg'].mean().item():.6e}, std={st['exp_avg'].std().item():.6e}")
        print(f"           exp_avg_sq mean={st['exp_avg_sq'].mean().item():.6e}, std={st['exp_avg_sq'].std().item():.6e}")

print(f"\n=== Other metadata ===")
print(f"  step: {ckpt.get('step')}")
print(f"  best_val_mean: {ckpt.get('best_val_mean')}")
print(f"  global_step (alt key): {ckpt.get('global_step')}")

# Load model and check output_type
print("\n=== Building model to check output_type ===")
from config.config_loader import load_pinn_full_config
from model.autoencoder_mlp import AutoencoderTeukolskyPINN

cfg_path = "config/autoencoder_stage1_retrain.yaml"
cfg = load_pinn_full_config(cfg_path)
print(f"  Config model.hidden_dims: {cfg.get('model', {}).get('hidden_dims')}")
print(f"  Config model.activation: {cfg.get('model', {}).get('activation')}")
