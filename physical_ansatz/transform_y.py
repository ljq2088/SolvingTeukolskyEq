"""
坐标与函数变换：从 R'(x) 到 f(y)

变换链：
1. 坐标: x = (y+1)/2,  y ∈ [-1,1] → x ∈ [0,1]
2. 函数: R'(x) = (exp(x-1)-1)*f(x) + h

目标：从 A2(x)R'_xx + A1(x)R'_x + A0(x)R' = 0
     推导 B2(y)f_yy + B1(y)f_y + B0(y)f = 0
"""
import torch
from physical_ansatz.mapping import r_plus, r_minus
def h_factor( a: torch.Tensor, omega: torch.Tensor, m: int, M: float = 1.0, s: int=-2) -> torch.Tensor:
    """
    常数项 h，确保边界条件满足 B^trans=1

    h = 1/(exp(i(ω+k)r_+)*2^(-2ik)*(r_+-r_-)^(-1+2i(ω+k)))

    其中 k = ω - m*Ω_H，Ω_H = a/(2Mr_+) 是黑洞的角速度
    """
    rp = r_plus(a, M)
    rm = r_minus(a, M)
    Ω_H = a / (2.0 * M * rp)
    k = omega - m * Ω_H
    i = 1j
    return torch.exp(-i*(omega+k)*rp) * (2.0**(2.0*i*k)) * ((rp - rm)**(1.0 - 2.0*i*(omega+k)))

def g_factor(x: torch.Tensor):
    g = torch.exp(x - 1.0) - 1.0

    # g'(x) = exp(x-1)
    g_prime = torch.exp(x - 1.0)

    # g''(x) = exp(x-1)
    g_double_prime = torch.exp(x - 1.0)

    return g, g_prime, g_double_prime


def transform_coeffs_x_to_y(
    A2: torch.Tensor,  # (B,Nx,1) 或 (B,Ny,1)
    A1: torch.Tensor,
    A0: torch.Tensor,
    y: torch.Tensor,
    h: torch.Tensor=None,
       # (B,Ny,1) 或 (Ny,) in [-1,1]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从 x 坐标的系数 A2,A1,A0 变换到 y 坐标的系数 B2,B1,B0

    变换关系：
    - x = (y+1)/2
    - dx/dy = 1/2
    - d²x/dy² = 0

    - R'(x) = (exp(x-1)-1)*f(x) + 1
    - 令 g(x) = exp(x-1)-1
    - R' = g*f + h, 其中 h是常数

    链式法则：
    - R'_x = g'*f + g*f_x
    - R'_xx = g''*f + 2*g'*f_x + g*f_xx

    其中 f_x = f_y * dy/dx = f_y * 2
          f_xx = f_yy * (dy/dx)² = f_yy * 4

    代入原方程：
    A2*(g''*f + 2*g'*f_y*2 + g*f_yy*4) + A1*(g'*f + g*f_y*2) + A0*(g*f + h) = 0

    整理得：
    B2*f_yy + B1*f_y + B0*f = -A0*h

    其中：
    B2 = 4*A2*g
    B1 = 4*A2*g' + 2*A1*g
    B0 = A2*g'' + A1*g' + A0*g
    h = exp(i(ω+k)r_+)*2^(-2ik)*(r_+-r_-)^(-1+2i(ω+k)) 是常数项，确保边界条件满足B^trans=1
    """
    if h is None:
        # 默认常数项：所有 batch、所有 y 点都取 1
        h = torch.ones_like(A0, dtype=A0.dtype, device=A0.device)
    else:
        h = h.to(dtype=A0.dtype, device=A0.device)

        # 若 h 是 (B,)：把它变成 (B,1)，以便在 y 方向广播
        if h.ndim == 1:
            h = h.unsqueeze(-1)   # (B,) -> (B,1)

        # 若 h 是标量：变成 (1,1)
        elif h.ndim == 0:
            h = h.reshape(1, 1)

    # x = (y+1)/2
    x = (y + 1.0) / 2.0

    g, g_prime, g_double_prime = g_factor(x)

    # 计算新系数
    B2 = 4.0 * A2 * g
    B1 = 4.0 * A2 * g_prime + 2.0 * A1 * g
    B0 = A2 * g_double_prime + A1 * g_prime + A0 * g

    # 右端项
    rhs = -A0 * h

    return B2, B1, B0, rhs
