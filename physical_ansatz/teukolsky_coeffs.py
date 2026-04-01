import torch
from .mapping import r_from_x
from .prefactor import *
# def q(a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
#     return a/M

# def kappa(a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
#     return torch.sqrt(1.0 - q(a, M)**2)

# def epsilon(omega: torch.Tensor,M: float = 1.0) -> torch.Tensor:
#     return 2.0 * M * omega

# def tau(a: torch.Tensor, omega: torch.Tensor, m:int,M: float = 1.0) -> torch.Tensor:
#     return (epsilon(omega, M) - m * q(a, M)) / kappa(a, M)

# def delta(r: torch.Tensor, a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
#     return r*r - 2.0*M*r + a*a

# def delta_r(r: torch.Tensor, M: float = 1.0) -> torch.Tensor:
#     return 2.0*r - 2.0*M

# def K_of_r(r: torch.Tensor, a: torch.Tensor, omega: torch.Tensor, m: int) -> torch.Tensor:
#     # omega can be complex tensor
#     return (r*r + a*a) * omega - a * m

# def V_of_r(r: torch.Tensor, a: torch.Tensor, omega: torch.Tensor, m: int, s: int, lambda_: torch.Tensor, M: float = 1.0) -> torch.Tensor:
#     Δ = delta(r, a, M)
#     K = K_of_r(r, a, omega, m)
#     i = 1j
#     return (K*K - 2.0*i*s*(r - M)*K)/Δ + 4.0*i*s*omega*r - lambda_

def coeffs_x(x: torch.Tensor, a: torch.Tensor, omega: torch.Tensor, m: int, p, R_amp: torch.Tensor, lambda_: torch.Tensor, s: int=-2, M: float = 1.0,
             dx_dr: torch.Tensor = None, d2x_dr2: torch.Tensor = None):
    """
    Return A2,A1,A0 in x-form for R'(x):
        A2 R'_xx + A1 R'_x + A0 R' = 0

    Original Teukolsky equation in r-form:
        Δ R_rr + (s+1)Δ_r R_r + V R = 0

    With R = R'(r)*U(r), where U(r) = Q(r)*P(r) is the prefactor, we get:
        Δ (R'_rr*U + 2R'_r*U_r + R'*U_rr) + (s+1)Δ_r(R'_r*U + R'*U_r) + V*U*R' = 0

    Transform to x coordinate via chain rule:
        R'_r = R'_x * dx/dr
        R'_rr = R'_xx * (dx/dr)^2 + R'_x * d²x/dr²

    Divide by U to get equation for R'(x):
        A2 R'_xx + A1 R'_x + A0 R' = 0

    where:
        A2 = Δ * (dx/dr)^2
        A1 = Δ * (2*(U_r/U)*dx/dr + d²x/dr²) + (s+1)*Δ_r*dx/dr
        A0 = V + (s+1)*Δ_r*(U_r/U) + Δ*(U_rr/U)
    """
    r = r_from_x(x, a, M)
    Δ = delta(r, a, M)
    Δr = delta_r(r, M)
    V = V_of_r(r, a, omega, m, s, lambda_, M)
    U=U_factor(r,a,omega,p,R_amp,m,s,M)
    U_r=U_factor_r(r,a,omega,p,R_amp,m,s,M)
    lnU_r=lnU_factor_r(r,a,omega,p,R_amp,m,s,M)
    U_rr=U_factor_r_r(r,a,omega,p,R_amp,m,s,M)

    # if caller did not precompute dx/dr, d2x/dr2, compute via formulas for x=r_plus/r
    if dx_dr is None or d2x_dr2 is None:
        # NOTE: prefer passing from mapping for consistency/caching
        from .mapping import dx_dr_from_x, d2x_dr2_from_x
        dx_dr = dx_dr_from_x(x, a, M)
        d2x_dr2 = d2x_dr2_from_x(x, a, M)
    #divide by U to get A2,A1,A0
    A2=Δ*(dx_dr**2)
    A1=Δ*(2*dx_dr*lnU_r+d2x_dr2)+(s+1)*Δr*dx_dr
    A0=V+(s+1)*Δr*lnU_r+Δ*U_rr/U
    return A2, A1, A0

