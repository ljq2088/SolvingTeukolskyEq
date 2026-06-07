from __future__ import annotations

import torch

from physical_ansatz.prefactor import Leaver_prefactors
from physical_ansatz.transform_y import h_factor
from teukfield.physics.coordinates import r_from_y
from utils.amplitude import A_down as np_A_down
from utils.amplitude import A_in as np_A_in
from utils.amplitude import A_up as np_A_up
from utils.mode import KerrMode


def leaver_P_h2(
    y: torch.Tensor,
    a: torch.Tensor,
    omega: torch.Tensor,
    m: int = 2,
    M: float = 1.0,
    s: int = -2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r = r_from_y(y, a, M)
    a_eval = a.unsqueeze(-1) if a.ndim == 1 and r.ndim > 1 else a
    omega_eval = omega.unsqueeze(-1) if omega.ndim == 1 and r.ndim > 1 else omega
    P, P_r, P_rr = Leaver_prefactors(r, a_eval, omega_eval, m=m, M=M, s=s)
    h2 = h_factor(a, omega, m=m, M=M, s=s).to(P.dtype)
    while h2.ndim < P.ndim:
        h2 = h2.unsqueeze(-1)
    return P, h2, P * h2


def numpy_asymptotic_factors(
    basis: str,
    a: float,
    omega: float,
    z_values,
    ell: int = 2,
    m: int = 2,
    s: int = -2,
):
    mode = KerrMode(M=1.0, a=float(a), omega=float(omega), ell=ell, m=m, s=s)
    r = mode.rp / z_values
    if basis == "in":
        return np_A_in(r, mode)
    if basis == "down":
        return np_A_down(r, mode)
    if basis == "up":
        return np_A_up(r, mode)
    raise ValueError(f"unknown basis={basis}")
