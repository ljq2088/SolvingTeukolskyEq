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

def coeffs_x(x: torch.Tensor, a: torch.Tensor, omega: torch.Tensor, m: int,p, R_amp: torch.Tensor, lambda_: torch.Tensor, s: int=-2, M: float = 1.0,
             dx_dr: torch.Tensor = None, d2x_dr2: torch.Tensor = None):
    """
    Return A2,A1,A0 in x-form:
        A2 R'_xx + A1 R'_x + A0 R' = 0 (1)
    Derived from r-form:
        Δ R_rr + (s+1)Δ_r R_r + V R = 0 (2)
    Replace R with R'(r)*U(r)
    and then transform to x coordinate via r=r(x), we get the above x-form with coefficients A2,A1,A0.
    From replacement of R with R'(r)*U(r), we get: 
        Δ (2R'_rU_r+R'U_rr+R'_rrU) + (s+1)Δ_r(R'_rU+R'U_r) + VU R' = 0 (3)
        Δ (2R'_x*dx_dr*U_r(r(x))+R'U_rr(r(x))+(R'_xx*dx_dr**2+R'_x*d2x_dr2)U(r(x))) + (s+1)Δ_r(R'_x*dx_dr*U+R'U_r) + VU R' = 0 (4)
        Comparing with A2 U_xx + A1 U_x + A0 U = 0, we get:
        A2 = Δ * (dx_dr**2)*U
        A1 = Δ * (2dx_dr*U_r+d2x_dr2*U)+(s+1)Δ_r*dx_dr*U
        A0 = VU+(s+1)Δ_r*U_r+Δ*U_rr

    """
    r = r_from_x(x, a, M)
    Δ = delta(r, a, M)
    Δr = delta_r(r, M)
    V = V_of_r(r, a, omega, m, s, lambda_, M)
    U=U_factor(r,a,omega,p,R_amp,m,s,M)
    U_r=U_factor_r(r,a,omega,p,R_amp,m,s,M)
    lnU_r=lnU_factor_r(r,a,omega,p,R_amp,m,s,M)
    U_rr=U_factor_rr(r,a,omega,p,R_amp,m,s,M)

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

