from __future__ import annotations

import torch
from torch import nn

from teukfield.models.blocks import FiLMPirateBlock, SineLayer
from teukfield.models.local_windows import LocalWindowSet


class LocalSirenField(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, depth: int, alpha_init: float, omega0: float):
        super().__init__()
        self.input = SineLayer(1, hidden_dim, omega0=omega0)
        self.blocks = nn.ModuleList([FiLMPirateBlock(hidden_dim, latent_dim, alpha_init, omega0=omega0) for _ in range(depth)])
        self.output = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, xi: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        h = self.input(xi.unsqueeze(-1))
        for block in self.blocks:
            h = block(h, latent)
        out = self.output(h)
        return out[..., 0].to(torch.complex128) + 1j * out[..., 1].to(torch.complex128)


class SFieldNetwork(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        n_windows: int = 8,
        overlap: float = 0.4,
        hidden_dim: int = 96,
        depth: int = 4,
        alpha_init: float = 1.0e-3,
        omega0: float = 15.0,
    ):
        super().__init__()
        self.windows = LocalWindowSet(n_windows, overlap)
        self.locals = nn.ModuleList(
            [LocalSirenField(latent_dim, hidden_dim, depth, alpha_init, omega0) for _ in range(n_windows)]
        )

    def forward(self, y: torch.Tensor, latent: torch.Tensor, slope_y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 1:
            y = y.unsqueeze(0).expand(latent.shape[0], -1)
        weights, xi = self.windows.weights_and_coords(y)
        local_values = []
        for j, local in enumerate(self.locals):
            local_values.append(local(xi[..., j], latent.unsqueeze(1).expand(-1, y.shape[1], -1)))
        correction = torch.stack(local_values, dim=-1)
        N = torch.sum(weights.to(torch.complex128) * correction, dim=-1)
        slope = slope_y.reshape(-1, 1).to(torch.complex128)
        return 1.0 + slope * (y.to(torch.complex128) - 1.0) + ((1.0 - y).to(torch.complex128) ** 2) * N
