from __future__ import annotations

import torch


def cheb_lobatto(n: int, device=None, dtype=torch.float64) -> torch.Tensor:
    k = torch.arange(n, device=device, dtype=dtype)
    return torch.cos(torch.pi * k / (n - 1))


def map_interval(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return 0.5 * (hi - lo) * x + 0.5 * (hi + lo)

