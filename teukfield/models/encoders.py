from __future__ import annotations

import torch
from torch import nn


class PhysicsFeatureEncoder(nn.Module):
    def __init__(self, in_dim: int = 10, latent_dim: int = 64, fourier_bands: int = 6):
        super().__init__()
        self.in_dim = in_dim
        self.fourier_bands = fourier_bands
        expanded = in_dim + 2 * fourier_bands * min(in_dim, 6) + 3 * min(in_dim, 6)
        self.net = nn.Sequential(
            nn.Linear(expanded, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        selected = features[..., : min(features.shape[-1], 6)]
        parts = [features]
        for k in range(self.fourier_bands):
            freq = torch.pi * (2.0**k)
            parts.extend([torch.sin(freq * selected), torch.cos(freq * selected)])
        parts.extend([selected, 2.0 * selected * selected - 1.0, 4.0 * selected**3 - 3.0 * selected])
        return self.net(torch.cat(parts, dim=-1))

