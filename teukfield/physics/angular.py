from __future__ import annotations

import torch

from utils.compute_lambda_usage import compute_lambda


def lambda_spheroidal(
    a: torch.Tensor,
    omega: torch.Tensor,
    ell: int = 2,
    m: int = 2,
    s: int = -2,
) -> torch.Tensor:
    """Return angular separation constants using the existing project wrapper.

    The current wrapper is scalar/numpy based; this function preserves dtype/device
    and is intentionally kept as a boundary adapter.
    """
    values = []
    for ai, wi in zip(a.detach().cpu().reshape(-1), omega.detach().cpu().reshape(-1)):
        values.append(complex(compute_lambda(float(ai), float(wi), ell, m, s=s)))
    out = torch.tensor(values, dtype=torch.complex128, device=a.device)
    return out.reshape(a.shape)

