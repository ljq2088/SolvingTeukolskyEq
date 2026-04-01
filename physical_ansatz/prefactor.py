"""物理渐近因子/边界层因子 P(r;a,ω)，用于硬编码视界/无穷远的入射/出射行为"""
import torch
import math
from .mapping import r_plus, r_minus

#后续优化速度可以在导数中传入原来的数据，如r_star_r_r计算的时候用到r_star_r的结果，避免重复计算r_star_r
def q(a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    return a/M

def kappa(a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    return torch.sqrt(1.0 - q(a, M)**2)

def epsilon(omega: torch.Tensor,M: float = 1.0) -> torch.Tensor:
    return 2.0 * M * omega

def tau(a: torch.Tensor, omega: torch.Tensor, m:int,M: float = 1.0) -> torch.Tensor:
    return (epsilon(omega, M) - m * q(a, M)) / kappa(a, M)

def delta(r: torch.Tensor, a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    return r*r - 2.0*M*r + a*a

def delta_r(r: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    return 2.0*r - 2.0*M

def K_of_r(r: torch.Tensor, a: torch.Tensor, omega: torch.Tensor, m: int) -> torch.Tensor:
    # omega can be complex tensor
    return (r*r + a*a) * omega - a * m

def V_of_r(r: torch.Tensor, a: torch.Tensor, omega: torch.Tensor, m: int, s: int, lambda_: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    Δ = delta(r, a, M)
    K = K_of_r(r, a, omega, m)
    i = 1j
    return (K*K - 2.0*i*s*(r - M)*K)/Δ + 4.0*i*s*omega*r - lambda_

def r_star(r: torch.Tensor, a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    rp= r_plus(a, M)
    rm= r_minus(a, M)
    return r+(rp**2+a**2)/(rp-rm)*torch.log(torch.abs(r-rp)/(2*M))-(rm**2+a**2)/(rp-rm)*torch.log(torch.abs(r-rm)/(2*M))

def r_star_r(r: torch.Tensor, a: torch.Tensor, M: float = 1.0) -> torch.Tensor:

    return (r**2+a**2)/delta(r, a, M)

def r_star_r_r(r: torch.Tensor, a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    return 1/delta(r, a, M)*(2*r-r_star_r(r, a, M)*(2*r-2*M))

def rprime(r: torch.Tensor, a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    rp = r_plus(a, M)
    rm = r_minus(a, M)
    r_prime = r + (rp**2 + a**2)/(rp - rm)*torch.log(torch.abs(r - rp + 2*M*torch.exp(-r/rp+1))/(2*M)) - (rm**2 + a**2)/(rp - rm)*torch.log(torch.abs(r - rm)/(2*M))
    return r_prime


def rprime_r(r: torch.Tensor, a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    rp = r_plus(a, M)
    rm = r_minus(a, M)
    r1 = r - rp + 2*M*torch.exp(-r/rp+1)
    r2 = r - rm
    term1 = 1.0
    term2 = (rp**2 + a**2) / (rp - rm) * ((1-2*M/rp*torch.exp(-r/rp+1)) / r1)
    term3 = -(rm**2 + a**2) / (rp - rm) * (1.0 / r2)
    return term1 + term2 + term3

def rprime_r_r(r: torch.Tensor, a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    rp = r_plus(a, M)
    rm = r_minus(a, M)
    r1 = r - rp + 2*M*torch.exp(-r/rp+1)
    r2 = r - rm

    dr1_dr = 1.0 - (2*M/rp)*torch.exp(-r/rp+1)
    d2r1_dr2 = (2*M/rp**2)*torch.exp(-r/rp+1)

    term2 = (rp**2 + a**2) / (rp - rm) * ((-(dr1_dr)**2 / r1**2) + (d2r1_dr2 / r1))
    term3 = -(rm**2 + a**2) / (rp - rm) * (-1.0 / r2**2)

    return term2 + term3

def prefactor_Q(r: torch.Tensor, a: torch.Tensor, omega: torch.Tensor, p: int, R_amp: torch.Tensor, M: float = 1.0, s: int = -2) -> torch.Tensor:
    """
    Your extra (1 + S) factor:
      1 + R_amp * r^{-p+2s} * (r-r_+)^p * exp(-2 i ω r'(r))
    """
    rp = r_plus(a, M)
    i = 1j
    rp_term = (r - rp) ** p
    s_term = R_amp * (r ** (- p + 2*s)) * rp_term * torch.exp(-2.0 * i * omega * r_star(r, a, M))
    return 1.0 + s_term

def prefactor_Q_r(r: torch.Tensor, a: torch.Tensor, omega: torch.Tensor, p: int, R_amp: torch.Tensor, M: float = 1.0, s: int = -2) -> torch.Tensor:
    """
    Derivative of prefactor_Q with respect to r, needed for boundary condition checks.
    """
    rp = r_plus(a, M)
    i = 1j
    rp_term = (r - rp) ** p
    s_term = R_amp * (r ** (- p + 2*s)) * rp_term * torch.exp(-2.0 * i * omega * r_star(r, a, M))

    # Compute derivative using product rule
    ds_dr = s_term * (
        (-p + 2*s) / r +
        p / (r - rp) -
        2.0 * i * omega * r_star_r(r, a, M)
    )

    return ds_dr

def prefactor_Q_r_r(r: torch.Tensor, a: torch.Tensor, omega: torch.Tensor, p: int, R_amp: torch.Tensor, M: float = 1.0, s: int = -2) -> torch.Tensor:
    """
    Second derivative of prefactor_Q with respect to r, needed for boundary condition checks.
    """
    rp = r_plus(a, M)
    i = 1j
    rp_term = (r - rp) ** p
    s_term = R_amp * (r ** (- p + 2*s)) * rp_term * torch.exp(-2.0 * i * omega * r_star(r, a, M))

    # Compute derivative using product rule
    ds_dr = prefactor_Q_r(r, a, omega, p, R_amp, M, s)

    # Compute second derivative using product rule
    d2s_dr2 = ds_dr * (
        (-p + 2*s) / r +
        p / (r - rp) -
        2.0 * i * omega * r_star_r(r, a, M)
    )+s_term * (
        (-(-p + 2*s)) / r**2 -
        p / (r - rp)**2 -
        2.0 * i * omega * r_star_r_r(r, a, M)
    )

    return d2s_dr2

def Inf_prefactor(r: torch.Tensor, a: torch.Tensor, omega: torch.Tensor, M: float = 1.0):
    return r**3*torch.exp(1j*omega*r_star(r, a, M))

def Inf_prefactor_r(r: torch.Tensor, a: torch.Tensor, omega: torch.Tensor, M: float = 1.0):
    return (3*r**2+r**3*1j*omega*r_star_r(r, a, M))*torch.exp(1j*omega*r_star(r, a, M))

def Inf_prefactor_r_r(r: torch.Tensor, a: torch.Tensor, omega: torch.Tensor, M: float = 1.0):
    return (6*r+6*r**2*1j*omega*r_star_r(r, a, M)+r**3*1j*omega*(r_star_r_r(r, a, M)+1j*omega*r_star_r(r, a, M)*r_star_r(r, a, M)))*torch.exp(1j*omega*r_star(r, a, M))



def prefactor_P(r: torch.Tensor, a: torch.Tensor, omega: torch.Tensor, m: int, M: float = 1.0, s: int = -2) -> torch.Tensor:
    """
    The x used here is different from the x in mapping.py, it's the one used in LRR-2003-6 Eq. 116:
    """
    rp= r_plus(a, M)
    rm= r_minus(a, M)
    x= (r - rp) / (rm - rp)
    alpha=-float(s) - 1j * (epsilon(omega, M) + tau(a, omega, m, M)) / 2.0
    beta= 1j * (epsilon(omega, M) - tau(a, omega, m, M)) / 2.0
    return torch.exp(1j * epsilon(omega, M) * kappa(a, M) * x) * (-x)**(alpha) * (1.0 - x)** (beta)

def prefactor_P_r(r: torch.Tensor, a: torch.Tensor, omega: torch.Tensor, m: int, M: float = 1.0, s: int = -2) -> torch.Tensor:
    rp= r_plus(a, M)
    rm= r_minus(a, M)
    x= (r - rp) / (rm - rp)
    alpha=-float(s) - 1j * (epsilon(omega, M) + tau(a, omega, m, M)) / 2.0
    beta= 1j * (epsilon(omega, M) - tau(a, omega, m, M)) / 2.0
    
    prefactor = torch.exp(1j * epsilon(omega, M) * kappa(a, M) * x) * (-x)**(alpha) * (1.0 - x)** (beta)
    
    dx_dr = 1.0 / (rm - rp)
    
    d_prefactor_dx = prefactor * (
        1j * epsilon(omega, M) * kappa(a, M) -
        alpha / (-x) - 
        beta / (1.0 - x)
    )
    
    return d_prefactor_dx * dx_dr

def prefactor_P_r_r(r: torch.Tensor, a: torch.Tensor, omega: torch.Tensor, m: int, M: float = 1.0, s: int = -2) -> torch.Tensor:
    rp= r_plus(a, M)    
    rm= r_minus(a, M)
    x= (r - rp) / (rm - rp)
    alpha=-float(s) - 1j * (epsilon(omega, M) + tau(a, omega, m, M)) / 2.0
    beta= 1j * (epsilon(omega, M) - tau(a, omega, m, M)) / 2.0
    
    prefactor = torch.exp(1j * epsilon(omega, M) * kappa(a, M) * x) * (-x)**(alpha) * (1.0 - x)** (beta)
    
    dx_dr = 1.0 / (rm - rp)
   
    
    d_prefactor_dx = prefactor_P_r(r, a, omega, m, M, s)* (rm - rp)
    
    d2_prefactor_dx2 = (d_prefactor_dx)**2/prefactor + prefactor * (-alpha / (x)**2 -beta/(1.0 - x)**2)

    
    return d2_prefactor_dx2*dx_dr**2




def U_factor(r: torch.Tensor, a: torch.Tensor, omega: torch.Tensor,p,R_amp, m: int, s: int=-2, M: float = 1.0) -> torch.Tensor:
    """
    R=R'*U(r)
    """
    # return prefactor_P(r, a, omega, m, M, s) * prefactor_Q(r, a, omega, p, R_amp, M,s)
    return prefactor_Q(r, a, omega, p, R_amp, M,s) * Inf_prefactor(r, a, omega, M)

def U_factor_r(r: torch.Tensor, a: torch.Tensor, omega: torch.Tensor,p,R_amp, m: int, s: int=-2, M: float = 1.0) -> torch.Tensor:
    """
    R=R'*U(r)
    """
    # P = prefactor_P(r, a, omega, m, M, s)
    I=Inf_prefactor(r, a, omega, M)
    Q = prefactor_Q(r, a, omega, p, R_amp, M,s)
    
    # dP_dr = prefactor_P_r(r, a, omega, m, M, s)
    dI_dr = Inf_prefactor_r(r, a, omega, M)
    dQ_dr = prefactor_Q_r(r, a, omega, p, R_amp, M,s)
    
    # return dP_dr * Q + P * dQ_dr
    return dQ_dr * I + Q * dI_dr


def lnU_factor_r(r: torch.Tensor, a: torch.Tensor, omega: torch.Tensor,p,R_amp, m: int, s: int=-2, M: float = 1.0) -> torch.Tensor:
    """
    R=R'*U(r)
    """
    # P = prefactor_P(r, a, omega, m, M, s)
    I=Inf_prefactor(r, a, omega, M)
    Q = prefactor_Q(r, a, omega, p, R_amp, M,s)
    
    # dP_dr = prefactor_P_r(r, a, omega, m, M, s)
    dI_dr = Inf_prefactor_r(r, a, omega, M)
    dQ_dr = prefactor_Q_r(r, a, omega, p, R_amp, M,s)
    
    # return  dP_dr/P +dQ_dr/Q
    return dI_dr/I + dQ_dr/Q

def U_factor_r_r(r: torch.Tensor, a: torch.Tensor, omega: torch.Tensor,p,R_amp, m: int, s: int=-2, M: float = 1.0) -> torch.Tensor:
    """
    R=R'*U(r)
    """
    # P = prefactor_P(r, a, omega, m, M, s)
    I=Inf_prefactor(r, a, omega, M)
    Q = prefactor_Q(r, a, omega, p, R_amp, M,s)
    
    # dP_dr = prefactor_P_r(r, a, omega, m, M, s)
    dI_dr = Inf_prefactor_r(r, a, omega, M)
    dQ_dr = prefactor_Q_r(r, a, omega, p, R_amp, M,s)
    
    # d2P_dr2 = prefactor_P_r_r(r, a, omega, m, M, s)
    d2I_dr2 = Inf_prefactor_r_r(r, a, omega, M)
    d2Q_dr2 = prefactor_Q_r_r(r, a, omega, p, R_amp, M,s)
    
    # return d2P_dr2 * Q + 2.0 * dP_dr * dQ_dr + P * d2Q_dr2
    return d2Q_dr2 * I + 2.0 * dI_dr * dQ_dr + Q * d2I_dr2

