from __future__ import annotations

import torch
from torch import nn


class AmplitudeHead(nn.Module):
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 64, depth: int = 3):
        super().__init__()
        layers = []
        dim = latent_dim
        for _ in range(depth):
            layers.extend([nn.Linear(dim, hidden_dim), nn.SiLU()])
            dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(dim, 6)

    @staticmethod
    def _phase(cos_raw: torch.Tensor, sin_raw: torch.Tensor) -> torch.Tensor:
        norm = torch.clamp(torch.sqrt(cos_raw * cos_raw + sin_raw * sin_raw), min=1.0e-12)
        return (cos_raw / norm).to(torch.complex128) + 1j * (sin_raw / norm).to(torch.complex128)

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.out(self.net(latent))
        log_abs_binc = raw[:, 0].clamp(-80.0, 80.0)
        phase_binc = self._phase(raw[:, 1], raw[:, 2])
        log_abs_rho = raw[:, 3].clamp(-80.0, 80.0)
        phase_rho = self._phase(raw[:, 4], raw[:, 5])
        B_inc = torch.exp(log_abs_binc).to(torch.complex128) * phase_binc
        rho = torch.exp(log_abs_rho).to(torch.complex128) * phase_rho
        return {"B_inc": B_inc, "rho": rho, "B_ref": B_inc * rho}

