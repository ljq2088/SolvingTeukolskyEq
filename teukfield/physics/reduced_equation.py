from __future__ import annotations

import torch

from physical_ansatz.teukolsky_coeffs import coeffs_x
from physical_ansatz.transform_y import horizon_regularity_slope


def horizon_slope_y(
    a: torch.Tensor,
    omega: torch.Tensor,
    lambda_: torch.Tensor,
    m: int = 2,
    M: float = 1.0,
    s: int = -2,
) -> torch.Tensor:
    """Return S_y(1). Existing helper returns S_x(1); x=(y+1)/2."""
    return 0.5 * horizon_regularity_slope(a=a, omega=omega, lambda_=lambda_, m=m, M=M, s=s)


def reduced_coefficients_y(
    y: torch.Tensor,
    a: torch.Tensor,
    omega: torch.Tensor,
    lambda_: torch.Tensor,
    m: int = 2,
    M: float = 1.0,
    s: int = -2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = 0.5 * (y + 1.0)
    if x.ndim == 1:
        x_eval = x.unsqueeze(0).expand(a.numel(), -1)
    else:
        x_eval = x
    rows = []
    for i in range(a.numel()):
        A2, A1, A0 = coeffs_x(
            x=x_eval[i : i + 1],
            a=a.reshape(-1)[i : i + 1],
            omega=omega.reshape(-1)[i : i + 1],
            m=m,
            lambda_=lambda_.reshape(-1)[i : i + 1],
            s=s,
            M=M,
        )
        rows.append((4.0 * A2, 2.0 * A1, A0))
    return tuple(torch.cat([row[j] for row in rows], dim=0) for j in range(3))

