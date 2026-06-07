from __future__ import annotations

import torch


def r_plus(a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    return M + torch.sqrt(torch.clamp(M * M - a * a, min=0.0))


def r_minus(a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    return M - torch.sqrt(torch.clamp(M * M - a * a, min=0.0))


def z_from_y(y: torch.Tensor) -> torch.Tensor:
    return 0.5 * (y + 1.0)


def y_from_z(z: torch.Tensor) -> torch.Tensor:
    return 2.0 * z - 1.0


def r_from_z(z: torch.Tensor, a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    rp = r_plus(a, M)
    if rp.ndim == 1 and z.ndim > 1:
        rp = rp.unsqueeze(-1)
    return rp / torch.clamp(z, min=torch.finfo(z.dtype).tiny)


def r_from_y(y: torch.Tensor, a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    return r_from_z(z_from_y(y), a, M)


def delta(r: torch.Tensor, a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    return r * r - 2.0 * M * r + a * a


def tortoise_r(r: torch.Tensor, a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    rp = r_plus(a, M)
    rm = r_minus(a, M)
    if rp.ndim == 1 and r.ndim > 1:
        rp = rp.unsqueeze(-1)
        rm = rm.unsqueeze(-1)
    return (r * r + a * a) / delta(r, a, M)

