from __future__ import annotations

import torch


def phase_safe_complex_relerr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mag = (torch.log(torch.clamp(torch.abs(pred), min=1e-300)) - torch.log(torch.clamp(torch.abs(target), min=1e-300))) ** 2
    phase = 1.0 - torch.real(pred / torch.clamp(torch.abs(pred), min=1e-300) * torch.conj(target / torch.clamp(torch.abs(target), min=1e-300)))
    return torch.mean(mag + phase)

