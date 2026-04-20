"""
Kerr scattering amplitude predictor.
predict_scattering_amplitudes(a, omega) -> (B_inc, B_ref) complex
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import pickle
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

class _MLP(nn.Module):
    def __init__(self, in_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,128), nn.SiLU(), nn.Linear(128,128), nn.SiLU(),
            nn.Linear(128,64), nn.SiLU(), nn.Linear(64,6))
    def forward(self, x): return self.net(x)

_cache: dict = {}

def _load(band: str):
    if band in _cache:
        return _cache[band]
    phasor_scaler = MODEL_DIR / f"scaler_band_{band}_phasor.pkl"
    phasor_model  = MODEL_DIR / f"mlp_band_{band}_phasor.pt"
    if phasor_scaler.exists() and phasor_model.exists():
        sc = pickle.load(open(phasor_scaler, 'rb'))
        m = _MLP()
        ckpt = torch.load(phasor_model, weights_only=False)
        m.load_state_dict(ckpt['model_state'])
        m.eval()
        _cache[band] = (m, sc['sx'], sc['sy_amp'], 'phasor')
    else:
        sc = pickle.load(open(MODEL_DIR / f"scaler_band_{band}.pkl", 'rb'))
        m = _MLP()
        ckpt = torch.load(MODEL_DIR / f"mlp_band_{band}.pt", weights_only=False)
        m.load_state_dict(ckpt['model_state'])
        m.eval()
        _cache[band] = (m, sc['sx'], sc['sy'], 'sincos')
    return _cache[band]

def _route(omega: float) -> str:
    if omega < 1e-3:   return 'A1'
    if omega < 0.01:   return 'A2'
    if omega < 0.5:    return 'B'
    return 'C'

def predict_scattering_amplitudes(a: float, omega: float) -> tuple[complex, complex]:
    band = _route(omega)
    model, sx, sy, fmt = _load(band)
    x = sx.transform([[a, np.log10(omega)]])
    with torch.no_grad():
        y = model(torch.tensor(x, dtype=torch.float32)).numpy()[0]
    if fmt == 'phasor':
        log_amps = sy.inverse_transform([[y[0], y[3]]])[0]
        B_inc = 10**log_amps[0] * (y[1] + 1j*y[2]) / max(np.hypot(y[1], y[2]), 1e-15)
        B_ref = 10**log_amps[1] * (y[4] + 1j*y[5]) / max(np.hypot(y[4], y[5]), 1e-15)
    else:
        y0 = sy.inverse_transform([y])[0]
        B_inc = 10**y0[0] * (y0[2] + 1j*y0[1]) / max(np.hypot(y0[1], y0[2]), 1e-15)
        B_ref = 10**y0[3] * (y0[5] + 1j*y0[4]) / max(np.hypot(y0[4], y0[5]), 1e-15)
    return complex(B_inc), complex(B_ref)
