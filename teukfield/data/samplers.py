from __future__ import annotations

import torch


def sample_patch(batch: int, cfg: dict, device="cpu", dtype=torch.float64):
    patch = cfg["patch"]
    a = patch["a_center"] + patch["a_half_width"] * (2.0 * torch.rand(batch, device=device, dtype=dtype) - 1.0)
    logw = patch["logw_center"] + patch["logw_half_width"] * (2.0 * torch.rand(batch, device=device, dtype=dtype) - 1.0)
    return a, logw


def sample_y(batch: int, n: int, lo: float = -0.98, hi: float = 0.98, device="cpu", dtype=torch.float64):
    return lo + (hi - lo) * torch.rand(batch, n, device=device, dtype=dtype)

