from __future__ import annotations

import torch
from torch import nn


class LocalWindowSet(nn.Module):
    def __init__(self, n_windows: int = 8, overlap: float = 0.4):
        super().__init__()
        centers = torch.linspace(-1.0, 1.0, n_windows, dtype=torch.float64)
        spacing = 2.0 / max(n_windows - 1, 1)
        self.register_buffer("centers", centers)
        self.width = float(spacing * (1.0 + overlap))

    def weights_and_coords(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        yy = y.unsqueeze(-1)
        dist = (yy - self.centers) / self.width
        logits = -dist * dist
        weights = torch.softmax(logits, dim=-1)
        xi = torch.clamp(dist, -1.5, 1.5)
        return weights, xi

