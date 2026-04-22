import torch
from .mapping import r_from_x
from .prefactor import (
    r_plus,
    delta,
    delta_r,
    V_of_r,
    build_prefactor_primitives,
    Leaver_prefactors,
)


def coeffs_x(
    x: torch.Tensor,
    a: torch.Tensor,
    omega: torch.Tensor,
    m: int,
    lambda_: torch.Tensor,
    s: int = -2,
    M: float = 1.0,
    dx_dr: torch.Tensor = None,
    d2x_dr2: torch.Tensor = None,
):
    """
    Return A2, A1, A0 in x-form for the reduced shape S(x), where

        R(r) = P(r) * S(x)

    and P is the Leaver prefactor only (Q and amplitude-ratio dependence removed).

    Equation:
        Δ R_rr + (s+1)Δ_r R_r + V R = 0

    After substituting R = P * S and dividing by P, one gets

        A2 S_xx + A1 S_x + A0 S = 0

    with
        A2 = Δ * (dx/dr)^2
        A1 = Δ * (2*(P_r/P)*dx/dr + d²x/dr²) + (s+1)Δ_r*dx/dr
        A0 = V + (s+1)Δ_r*(P_r/P) + Δ*(P_rr/P)
    """
    rp = r_plus(a, M)
    r = r_from_x(x, rp)

    Δ = delta(r, a, M)
    Δr = delta_r(r, M)
    V = V_of_r(r, a, omega, m, s, lambda_, M)

    rp_pref, rm_pref, _, _, _ = build_prefactor_primitives(
        r, a, M=M, need_rs=False
    )
    P, P_r, P_rr = Leaver_prefactors(
        r, a, omega, m, M, s, rp=rp_pref, rm=rm_pref
    )

    if dx_dr is None or d2x_dr2 is None:
        from .mapping import dx_dr_from_x, d2x_dr2_from_x
        dx_dr = dx_dr_from_x(x, a, M)
        d2x_dr2 = d2x_dr2_from_x(x, a, M)

    A2 = Δ * (dx_dr ** 2)
    A1 = Δ * (2.0 * dx_dr * (P_r / P) + d2x_dr2) + (s + 1) * Δr * dx_dr
    A0 = V + (s + 1) * Δr * (P_r / P) + Δ * (P_rr / P)
    return A2, A1, A0