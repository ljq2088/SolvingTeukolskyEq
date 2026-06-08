from __future__ import annotations

import math

import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, omega0: float = 15.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.omega0 = omega0
        with torch.no_grad():
            bound = math.sqrt(6.0 / in_dim) / omega0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * self.linear(x))


class FiLMPirateBlock(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int, alpha_init: float = 1.0e-3, omega0: float = 15.0):
        super().__init__()
        self.core = nn.Sequential(SineLayer(hidden_dim, hidden_dim, omega0=omega0), nn.Linear(hidden_dim, hidden_dim))
        self.gamma = nn.Linear(latent_dim, hidden_dim)
        self.beta = nn.Linear(latent_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float64))

    def forward(self, h: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        gamma = 1.0 + 0.1 * torch.tanh(self.gamma(latent))
        beta = 0.1 * self.beta(latent)
        return h + self.alpha * torch.sin(gamma * self.core(h) + beta)
