"""坐标变换：r<->z<->x 以及 ξ(x) 的映射关系"""
import torch

def r_plus(a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    return M + torch.sqrt(torch.clamp(M*M - a*a, min=0.0))

def r_minus(a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    return M - torch.sqrt(torch.clamp(M*M - a*a, min=0.0))

def r_from_x(x: torch.Tensor, a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    rp = r_plus(a, M)
    return rp / x

def dx_dr_from_x(x: torch.Tensor, a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    # dx/dr = -x^2 / r_plus
    rp = r_plus(a, M)
    return -(x**2) / rp

def d2x_dr2_from_x(x: torch.Tensor, a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    # d2x/dr2 = 2 x^3 / r_plus^2
    rp = r_plus(a, M)
    return 2.0 * (x**3) / (rp**2)