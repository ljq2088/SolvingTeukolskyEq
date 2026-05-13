#!/usr/bin/env python3
"""
Minimal smoke test for AutoencoderTeukolskyPINN.

Checks:
    - import works
    - model instantiates
    - predict_Rin shape is correct
    - predict_u_up shape is correct
    - predict_u_down shape is correct
    - predict_amplitudes shape is correct
    - forward() compat with old PINN_MLP
    - no Mathematica call
    - no long training
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

torch.manual_seed(42)

B = 4   # batch
N = 64  # y points

a = torch.rand(B) * 0.9 + 0.01
omega = torch.rand(B) * 5.0 + 1.0e-4
y = torch.linspace(-1.0, 1.0, N)

print("1. Import...")
from model.autoencoder_pinn import (
    AutoencoderTeukolskyPINN,
    SharedPINNEncoder,
    TaylorComplexDecoder,
    AmplitudeNet,
    ModulatedResidualBlock,
)
print("   OK")

print("2. Instantiate model...")
model = AutoencoderTeukolskyPINN()
n_params = sum(p.numel() for p in model.parameters())
print(f"   OK — {n_params} parameters")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   trainable: {trainable}")

print("3. forward() compat...")
f = model(a, omega, y)
assert f.shape == (B, N), f"Expected ({B},{N}), got {f.shape}"
assert f.is_complex(), "Output should be complex"
print(f"   OK — shape {f.shape}, dtype {f.dtype}")

print("4. predict_Rin...")
f_rin = model.predict_Rin(a, omega, y)
assert f_rin.shape == (B * N,), f"Expected ({B * N},), got {f_rin.shape}"
print(f"   OK — shape {f_rin.shape}")

print("5. predict_u_up...")
f_up = model.predict_u_up(a, omega, y)
assert f_up.shape == (B * N,), f"Expected ({B * N},), got {f_up.shape}"
print(f"   OK — shape {f_up.shape}")

print("6. predict_u_down...")
f_dn = model.predict_u_down(a, omega, y)
assert f_dn.shape == (B * N,), f"Expected ({B * N},), got {f_dn.shape}"
print(f"   OK — shape {f_dn.shape}")

print("7. predict_amplitudes...")
Binc, Bref, raw = model.predict_amplitudes(a, omega)
assert Binc.shape == (B,), f"Expected ({B},), got {Binc.shape}"
assert Bref.shape == (B,), f"Expected ({B},), got {Bref.shape}"
assert Binc.is_complex() and Bref.is_complex(), "Should be complex"
assert "rho_inc" in raw and "phi_inc" in raw
print(f"   OK — Binc shape {Binc.shape}, Bref shape {Bref.shape}")
print(f"   Binc[0] = {Binc[0]:.4e}, Bref[0] = {Bref[0]:.4e}")

print("8. predict_asymptotic_amplitudes alias...")
Binc2, Bref2, raw2 = model.predict_asymptotic_amplitudes(a, omega)
assert torch.allclose(Binc, Binc2)
print("   OK — alias matches")

print("9. Stage-1 equivalence check...")
# Old PINN_MLP forward should produce same structure output
from model.pinn_mlp import PINN_MLP
old_model = PINN_MLP()
f_old = old_model(a, omega, y)
assert f_old.shape == f.shape, f"Shape mismatch: {f_old.shape} vs {f.shape}"
print(f"   OK — same output shape as old PINN_MLP: {f_old.shape}")

print("10. Amplitude encoder is independent...")
# Verify amplitude_net has its own param_encoder
amp_enc_params = set(id(p) for p in model.amplitude_net.param_encoder.parameters())
shared_enc_params = set(id(p) for p in model.encoder.param_encoder.parameters())
assert amp_enc_params.isdisjoint(shared_enc_params), \
    "amplitude_net.param_encoder shares params with encoder.param_encoder!"
print("   OK — AmplitudeNet has independent param_encoder")

print("11. No temporary PINN_MLP creation...")
# Verify SharedPINNEncoder has its own feature building (no PINN_MLP dep)
assert hasattr(model.encoder, "_build_param_features"), "Missing _build_param_features"
assert hasattr(model.encoder, "compute_local_coords"), "Missing compute_local_coords"
print("   OK — encoder has own feature building")

print()
print("=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)
