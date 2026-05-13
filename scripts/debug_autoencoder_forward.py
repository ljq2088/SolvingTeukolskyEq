#!/usr/bin/env python3
"""
Smoke test + Stage-1 equivalence for AutoencoderTeukolskyPINN.

Checks:
    - import works
    - model instantiates
    - predict_Rin shape (B,N)
    - predict_u_up / predict_u_down shape (B,N)
    - predict_amplitudes shape
    - forward() compat with old PINN_MLP
    - AmplitudeNet has independent param_encoder
    - No temporary PINN_MLP creation
    - autoencoder_mlp.py wrapper import works
    - Stage-1 weight-migration numerical equivalence
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

torch.manual_seed(42)

B = 4
N = 64

a = torch.rand(B) * 0.9 + 0.01
omega = torch.rand(B) * 5.0 + 1.0e-4
y = torch.linspace(-1.0, 1.0, N)


def check(desc, condition):
    assert condition, f"FAIL: {desc}"
    print(f"   OK — {desc}")


print("1. Import model.autoencoder_pinn...")
from model.autoencoder_pinn import (
    AutoencoderTeukolskyPINN, SharedPINNEncoder,
    TaylorComplexDecoder, AmplitudeNet, ModulatedResidualBlock,
    copy_pinn_mlp_to_autoencoder,
)
print("   OK")

print("2. Instantiate...")
model = AutoencoderTeukolskyPINN()
n_params = sum(p.numel() for p in model.parameters())
print(f"   OK — {n_params} params")

print("3. forward() compat shape...")
f = model(a, omega, y)
check("forward shape (B,N)", f.shape == (B, N))
check("forward complex", f.is_complex())

print("4. predict_Rin shape...")
f_rin = model.predict_Rin(a, omega, y)
check("predict_Rin shape (B,N)", f_rin.shape == (B, N))

print("5. predict_u_up shape...")
f_up = model.predict_u_up(a, omega, y)
check("predict_u_up shape (B,N)", f_up.shape == (B, N))

print("6. predict_u_down shape...")
f_dn = model.predict_u_down(a, omega, y)
check("predict_u_down shape (B,N)", f_dn.shape == (B, N))

print("7. predict_amplitudes...")
Binc, Bref, raw = model.predict_amplitudes(a, omega)
check("Binc shape (B,)", Binc.shape == (B,))
check("Bref shape (B,)", Bref.shape == (B,))
check("Binc complex", Binc.is_complex())

print("8. predict_asymptotic_amplitudes alias...")
Binc2, Bref2, _ = model.predict_asymptotic_amplitudes(a, omega)
check("alias matches", torch.allclose(Binc, Binc2))

print("9. Amplitude encoder independent...")
amp_enc_ids = set(id(p) for p in model.amplitude_net.param_encoder.parameters())
shared_enc_ids = set(id(p) for p in model.encoder.param_encoder.parameters())
check("AmplitudeNet has own param_encoder", amp_enc_ids.isdisjoint(shared_enc_ids))

print("10. No temporary PINN_MLP...")
check("encoder has _build_param_features", hasattr(model.encoder, "_build_param_features"))
check("encoder has compute_local_coords", hasattr(model.encoder, "compute_local_coords"))

print("11. autoencoder_mlp.py wrapper import...")
from model.autoencoder_mlp import AutoencoderPINN
wrapper = AutoencoderPINN()
check("wrapper is AutoencoderTeukolskyPINN", isinstance(wrapper, AutoencoderTeukolskyPINN))

print("12. Stage-1 weight-migration equivalence...")
from model.pinn_mlp import PINN_MLP
torch.manual_seed(123)
old_model = PINN_MLP()
torch.manual_seed(123)
ae_model = AutoencoderTeukolskyPINN()
copy_pinn_mlp_to_autoencoder(old_model, ae_model)

with torch.no_grad():
    f_old = old_model(a, omega, y)
    f_new = ae_model(a, omega, y)
max_err = (f_old - f_new).abs().max().item()
check(f"Stage-1 equivalence max_err={max_err:.2e}", max_err < 1e-5)
print(f"   max |f_old - f_new| = {max_err:.2e}")

print()
print("=" * 60)
print("ALL 12 CHECKS PASSED")
print("=" * 60)
