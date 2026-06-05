from __future__ import annotations

import math

import torch
from torch import nn


class FiLMPirateBlock(nn.Module):
    def __init__(self, hidden_dim: int, feature_dim: int, alpha_init: float):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.film = nn.Linear(feature_dim, 2 * hidden_dim)
        self.alpha_raw = nn.Parameter(torch.tensor(float(alpha_init)))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, h: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.film(features)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        update = torch.nn.functional.silu((1.0 + gamma) * self.linear(h) + beta)
        alpha = torch.clamp(self.alpha_raw, 0.0, 1.0)
        return (1.0 - alpha) * h + alpha * update


class FiLMPirateCorrector(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        depth: int = 4,
        fourier_bands: int = 4,
        basis: str = "in",
        z_left: float = 0.0,
        z_right: float = 1.0,
        alpha_init: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.fourier_bands = int(fourier_bands)
        self.basis = str(basis)
        self.z_left = float(z_left)
        self.z_right = float(z_right)
        in_dim = 1 + 2 * self.fourier_bands + self.feature_dim
        self.input = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([FiLMPirateBlock(hidden_dim, self.feature_dim, alpha_init) for _ in range(depth)])
        self.output = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def z_features(self, z: torch.Tensor) -> torch.Tensor:
        z_scaled = 2.0 * (z - self.z_left) / (self.z_right - self.z_left) - 1.0
        feats = [z_scaled]
        for band in range(self.fourier_bands):
            freq = math.pi * float(2**band)
            feats.extend([torch.sin(freq * z_scaled), torch.cos(freq * z_scaled)])
        return torch.cat(feats, dim=-1)

    def forward_raw(self, z: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if z.ndim == 1:
            z = z[:, None]
        if features.ndim == 1:
            features = features[None, :].expand(z.shape[0], -1)
        h = torch.nn.functional.silu(self.input(torch.cat([self.z_features(z), features], dim=-1)))
        for block in self.blocks:
            h = block(h, features)
        out = self.output(h)
        return out[..., 0].to(torch.complex128) + 1j * out[..., 1].to(torch.complex128)

    def gate(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 1:
            z = z[:, None]
        if self.basis == "in":
            return (1.0 - z).squeeze(-1) ** 2
        if self.basis in {"down", "up"}:
            return z.squeeze(-1) ** 2
        return ((z.squeeze(-1) - self.z_left) * (self.z_right - z.squeeze(-1))) ** 2

    def forward(self, z: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.gate(z).to(torch.complex128) * self.forward_raw(z, features)
