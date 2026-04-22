"""Kerr scattering amplitude predictor for EMRI waveforms."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import pickle
from pathlib import Path

_DEFAULT_MODEL_DIR = Path(__file__).parent / "models"


class _MLP(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _phi_acc(a: float, omega: float, r_offset: float = 10.0) -> float:
    rp = 1.0 + np.sqrt(1.0 - a**2)
    rm = 1.0 - np.sqrt(1.0 - a**2)
    r = rp + r_offset
    dr = rp - rm
    rstar = r + (2.0 / dr) * (rp * np.log(abs(r - rp)) - rm * np.log(abs(r - rm)))
    return omega * rstar


def _route(omega: float) -> str:
    if omega < 1e-3:  return "A1"
    if omega < 0.01:  return "A2"
    if omega < 0.5:   return "B"
    return "C"


class KerrScatteringPredictor:
    """
    Predict Kerr scattering amplitudes B_inc, B_ref for l=m=2, s=-2.

    Parameters
    ----------
    model_dir : path-like, optional
        Directory containing model .pt and .pkl files.
        Defaults to the ``models/`` folder next to this file.
    """

    def __init__(self, model_dir=None):
        self._dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self._cache: dict = {}
        # Pre-load B_ref two-stage model for Band A1
        self._bref_model = None
        self._bref_sx = None
        self._bref_sy = None
        bref_pt  = self._dir / "mlp_band_A1_Bref.pt"
        bref_pkl = self._dir / "scaler_band_A1_Bref.pkl"
        if bref_pt.exists() and bref_pkl.exists():
            sc = pickle.load(open(bref_pkl, "rb"))
            m = _MLP([4, 512, 512, 512, 512, 1])
            ckpt = torch.load(bref_pt, weights_only=False)
            m.load_state_dict(ckpt["model_state"])
            m.eval()
            self._bref_model = m
            self._bref_sx = sc["sx"]
            self._bref_sy = sc["sy"]

    def _load(self, band: str):
        if band in self._cache:
            return self._cache[band]
        d = self._dir
        phys_pt  = d / f"mlp_band_{band}_phys.pt"
        phys_pkl = d / f"scaler_band_{band}_phys.pkl"
        if phys_pt.exists() and phys_pkl.exists():
            sc = pickle.load(open(phys_pkl, "rb"))
            m = _MLP([3, 512, 512, 512, 512, 512, 6])
            ckpt = torch.load(phys_pt, weights_only=False)
            m.load_state_dict(ckpt["model_state"])
            m.eval()
            self._cache[band] = (m, sc["sx"], sc["sy"], "phasor3")
            return self._cache[band]
        phasor_pt  = d / f"mlp_band_{band}_phasor.pt"
        phasor_pkl = d / f"scaler_band_{band}_phasor.pkl"
        if phasor_pt.exists() and phasor_pkl.exists():
            sc = pickle.load(open(phasor_pkl, "rb"))
            m = _MLP([2, 128, 128, 64, 6])
            ckpt = torch.load(phasor_pt, weights_only=False)
            m.load_state_dict(ckpt["model_state"])
            m.eval()
            self._cache[band] = (m, sc["sx"], sc["sy_amp"], "phasor")
        else:
            sc = pickle.load(open(d / f"scaler_band_{band}.pkl", "rb"))
            m = _MLP([2, 128, 128, 64, 6])
            ckpt = torch.load(d / f"mlp_band_{band}.pt", weights_only=False)
            m.load_state_dict(ckpt["model_state"])
            m.eval()
            self._cache[band] = (m, sc["sx"], sc["sy"], "sincos")
        return self._cache[band]

    def predict(self, a: float, omega: float) -> dict:
        """
        Predict scattering amplitudes.

        Parameters
        ----------
        a : float
            Kerr spin parameter, a ∈ [0, 0.999).
        omega : float
            GW frequency, omega ∈ [1e-4, 2.0].

        Returns
        -------
        dict with keys:
            B_inc : complex
            B_ref : complex
            band  : str  ('A1', 'A2', 'B', 'C')
        """
        band = _route(omega)
        model, sx, sy, fmt = self._load(band)

        if fmt == "phasor3":
            x = sx.transform([[a, np.log10(omega), _phi_acc(a, omega)]])
        else:
            x = sx.transform([[a, np.log10(omega)]])

        with torch.no_grad():
            y = model(torch.tensor(x, dtype=torch.float32)).numpy()[0]

        if fmt == "phasor3":
            y0 = sy.inverse_transform([y])[0]
            B_inc = 10**y0[0] * (y0[1] + 1j * y0[2]) / max(np.hypot(y0[1], y0[2]), 1e-15)
            if self._bref_model is not None:
                log_Binc = y0[0]
                xb = self._bref_sx.transform(
                    [[a, np.log10(omega), _phi_acc(a, omega), log_Binc]]
                )
                with torch.no_grad():
                    log_ratio_scaled = self._bref_model(
                        torch.tensor(xb, dtype=torch.float32)
                    ).numpy()
                log_ratio = self._bref_sy.inverse_transform(log_ratio_scaled)[0, 0]
                log_Bref = log_ratio + log_Binc
                # phase from stage-1 phasor3 output
                B_ref = 10**log_Bref * (y0[4] + 1j * y0[5]) / max(np.hypot(y0[4], y0[5]), 1e-15)
            else:
                B_ref = 10**y0[3] * (y0[4] + 1j * y0[5]) / max(np.hypot(y0[4], y0[5]), 1e-15)
        elif fmt == "phasor":
            log_amps = sy.inverse_transform([[y[0], y[3]]])[0]
            B_inc = 10**log_amps[0] * (y[1] + 1j * y[2]) / max(np.hypot(y[1], y[2]), 1e-15)
            B_ref = 10**log_amps[1] * (y[4] + 1j * y[5]) / max(np.hypot(y[4], y[5]), 1e-15)
        else:
            y0 = sy.inverse_transform([y])[0]
            B_inc = 10**y0[0] * (y0[2] + 1j * y0[1]) / max(np.hypot(y0[1], y0[2]), 1e-15)
            B_ref = 10**y0[3] * (y0[5] + 1j * y0[4]) / max(np.hypot(y0[4], y0[5]), 1e-15)

        return {"B_inc": complex(B_inc), "B_ref": complex(B_ref), "band": band}
