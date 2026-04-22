# """
# 坐标与函数变换：从 R'(x) 到 f(y)

# 变换链：
# 1. 坐标: x = (y+1)/2,  y ∈ [-1,1] → x ∈ [0,1]
# 2. 函数: R'(x) = (exp(x-1)-1)*f(x) + h

# 目标：从 A2(x)R'_xx + A1(x)R'_x + A0(x)R' = 0
#      推导 B2(y)f_yy + B1(y)f_y + B0(y)f = 0
# """
# import torch
# from physical_ansatz.mapping import r_plus, r_minus
# def h_factor( a: torch.Tensor, omega: torch.Tensor, m: int, M: float = 1.0, s: int=-2) -> torch.Tensor:
#     """
#     常数项 h，确保边界条件满足 B^trans=1

#     h = 1/(exp(i(ω+k)r_+)*2^(-2ik)*(r_+-r_-)^(-1+2i(ω+k)))

#     其中 k = ω - m*Ω_H，Ω_H = a/(2Mr_+) 是黑洞的角速度
#     """
#     rp = r_plus(a, M)
#     rm = r_minus(a, M)
#     Ω_H = a / (2.0 * M * rp)
#     k = omega - m * Ω_H
#     i = 1j
#     return torch.exp(-i*(omega+k)*rp) * (2.0**(2.0*i*k)) * ((rp - rm)**(1.0 - 2.0*i*(omega+k)))

# def g_factor(x: torch.Tensor):
#     g = torch.exp(x - 1.0) - 1.0

#     # g'(x) = exp(x-1)
#     g_prime = torch.exp(x - 1.0)

#     # g''(x) = exp(x-1)
#     g_double_prime = torch.exp(x - 1.0)

#     return g, g_prime, g_double_prime


# def transform_coeffs_x_to_y(
#     A2: torch.Tensor,  # (B,Nx,1) 或 (B,Ny,1)
#     A1: torch.Tensor,
#     A0: torch.Tensor,
#     y: torch.Tensor,
#     h: torch.Tensor=None,
#        # (B,Ny,1) 或 (Ny,) in [-1,1]
# ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     从 x 坐标的系数 A2,A1,A0 变换到 y 坐标的系数 B2,B1,B0

#     变换关系：
#     - x = (y+1)/2
#     - dx/dy = 1/2
#     - d²x/dy² = 0

#     - R'(x) = (exp(x-1)-1)*f(x) + 1
#     - 令 g(x) = exp(x-1)-1
#     - R' = g*f + h, 其中 h是常数

#     链式法则：
#     - R'_x = g'*f + g*f_x
#     - R'_xx = g''*f + 2*g'*f_x + g*f_xx

#     其中 f_x = f_y * dy/dx = f_y * 2
#           f_xx = f_yy * (dy/dx)² = f_yy * 4

#     代入原方程：
#     A2*(g''*f + 2*g'*f_y*2 + g*f_yy*4) + A1*(g'*f + g*f_y*2) + A0*(g*f + h) = 0

#     整理得：
#     B2*f_yy + B1*f_y + B0*f = -A0*h

#     其中：
#     B2 = 4*A2*g
#     B1 = 4*A2*g' + 2*A1*g
#     B0 = A2*g'' + A1*g' + A0*g
#     h = exp(i(ω+k)r_+)*2^(-2ik)*(r_+-r_-)^(-1+2i(ω+k)) 是常数项，确保边界条件满足B^trans=1
#     """
#     if h is None:
#         # 默认常数项：所有 batch、所有 y 点都取 1
#         h = torch.ones_like(A0, dtype=A0.dtype, device=A0.device)
#     else:
#         h = h.to(dtype=A0.dtype, device=A0.device)

#         # 若 h 是 (B,)：把它变成 (B,1)，以便在 y 方向广播
#         if h.ndim == 1:
#             h = h.unsqueeze(-1)   # (B,) -> (B,1)

#         # 若 h 是标量：变成 (1,1)
#         elif h.ndim == 0:
#             h = h.reshape(1, 1)

#     # x = (y+1)/2
#     x = (y + 1.0) / 2.0

#     g, g_prime, g_double_prime = g_factor(x)

#     # 计算新系数
#     B2 = 4.0 * A2 * g
#     B1 = 4.0 * A2 * g_prime + 2.0 * A1 * g
#     B0 = A2 * g_double_prime + A1 * g_prime + A0 * g

#     # 右端项
#     rhs = -A0 * h

#     return B2, B1, B0, rhs
"""
坐标与函数变换：从网络输出 f(y) 到 reduced shape S(x)

新的 ansatz:
    R_in(r) = P(r) * S(x) * h2
其中
    S(x) = g(x) * (h1(x) * f(y) + 1) + 1

坐标:
    x = (y+1)/2,   y ∈ [-1,1],  x ∈ [0,1]

要求:
- h1(1)=0
- g(1)=0
- 且 S_x(1) = (-A0/A1)|_{x=1}
从而自动满足视界函数值与一阶导数的正则性条件。
"""
import torch
from physical_ansatz.mapping import r_plus, r_minus


def h_factor(a: torch.Tensor, omega: torch.Tensor, m: int, M: float = 1.0, s: int = -2) -> torch.Tensor:
    """
    常数因子 h2，使得 R_in 在视界处的整体归一化与原先约定一致。
    """
    rp = r_plus(a, M)
    rm = r_minus(a, M)
    Ω_H = a / (2.0 * M * rp)
    k = omega - m * Ω_H
    i = 1j
    return torch.exp(-i * (omega + k) * rp) * (2.0 ** (2.0 * i * k)) * ((rp - rm) ** (1.0 - 2.0 * i * (omega + k)))


def h1_factor(x: torch.Tensor):
    """
    h1(x) = exp(x-1) - 1
    满足 h1(1)=0
    """
    ex = torch.exp(x - 1.0)
    h1 = ex - 1.0
    h1_x = ex
    h1_xx = ex
    return h1, h1_x, h1_xx


def horizon_A1_A0_limit(
    a: torch.Tensor,
    omega: torch.Tensor,
    lambda_: torch.Tensor,
    m: int,
    M: float = 1.0,
    s: int = -2,
):
    """
    返回当前项目归一化约定下，x=1 (r=r_+) 处的
        A1^(H), A0^(H)
    这是从 coeffs_x 中 P-only 系数做视界极限得到的解析式。

    注意：
    该公式与当前项目的 Leaver_prefactors / sigma_p 定义保持一致。
    """
    rp = r_plus(a, M)
    rm = r_minus(a, M)
    d = rp - rm
    sigma_p = (2.0 * omega * rp - a * m) / d

    A1_H = (d / rp) * (s - 1.0 + 2.0j * sigma_p)
    A0_H = -(lambda_ + s + 1.0) + 2.0j * sigma_p + 2.0j * omega * rp + 8.0 * omega * rp * sigma_p
    return A1_H, A0_H


def horizon_regularity_slope(
    a: torch.Tensor,
    omega: torch.Tensor,
    lambda_: torch.Tensor,
    m: int,
    M: float = 1.0,
    s: int = -2,
):
    """
    计算
        c_H = (-A0/A1)|_{x=1}
    用于设置
        g(x) = c_H * (exp(x-1)-1)
    从而保证 reduced shape S(x) 在视界处满足一阶正则性条件。
    """
    A1_H, A0_H = horizon_A1_A0_limit(
        a=a,
        omega=omega,
        lambda_=lambda_,
        m=m,
        M=M,
        s=s,
    )
    return -A0_H / A1_H


def _broadcast_slope_to_x(slope: torch.Tensor, x: torch.Tensor):
    """
    把 slope 和 x 对齐到可广播的形状。

    典型情况：
        slope: (B,)
        x:     (N,)
    需要变成：
        slope: (B,1)
        x:     (1,N)

    也兼容：
        slope: scalar
        x:     (N,)
    或
        slope: (B,1)
        x:     (1,N)
    """
    slope = slope.to(device=x.device)

    # 不要把 complex slope 强制 cast 成 x.dtype，否则会丢掉虚部
    target_dtype = torch.promote_types(x.dtype, slope.dtype)
    slope = slope.to(dtype=target_dtype)
    x = x.to(dtype=target_dtype)

    # 最常见情形：共享 y/x 网格 + batch slope
    if slope.ndim == 1 and x.ndim == 1:
        slope = slope.unsqueeze(-1)   # (B,) -> (B,1)
        x = x.unsqueeze(0)            # (N,) -> (1,N)
        return slope, x

    # 更一般的广播
    while slope.ndim < x.ndim:
        slope = slope.unsqueeze(-1)

    while x.ndim < slope.ndim:
        x = x.unsqueeze(0)

    return slope, x


def g_factor(x: torch.Tensor, slope: torch.Tensor):
    """
    g(x) = slope * (exp(x-1)-1)
    满足:
        g(1)=0
        g_x(1)=slope

    支持：
        slope: (B,), x: (N,)   -> 输出 (B,N)
        slope: scalar, x: (N,) -> 输出 (N,)
    """
    slope, x = _broadcast_slope_to_x(slope, x)

    ex = torch.exp(x - 1.0)
    base = ex - 1.0

    g = slope * base
    g_x = slope * ex
    g_xx = slope * ex
    return g, g_x, g_xx


def compose_reduced_shape_from_f(
    f: torch.Tensor,
    y: torch.Tensor,
    slope: torch.Tensor,
):
    """
    给定网络输出 f(y)，构造
        S(x) = g(x) * (h1(x) * f(y) + 1) + 1
    """
    x = 0.5 * (y + 1.0)
    h1, _, _ = h1_factor(x)
    g, _, _ = g_factor(x, slope)
    return g * (h1 * f + 1.0) + 1.0


def transform_coeffs_x_to_y(
    A2: torch.Tensor,
    A1: torch.Tensor,
    A0: torch.Tensor,
    y: torch.Tensor,
    slope: torch.Tensor,
):
    """
    新 ansatz:
        S(x) = g(x) * (h1(x) * f(y) + 1) + 1
             = W(x) * f(y) + G(x)

    其中
        W(x) = g(x) * h1(x)
        G(x) = g(x) + 1

    且
        x = (y+1)/2
        f_x  = 2 f_y
        f_xx = 4 f_yy
    """
    x = 0.5 * (y + 1.0)

    # 若 A2/A1/A0 已经是 batch 形式，而 x 仍是 1D，则扩成 (1,N)
    if A2.ndim >= 2 and x.ndim == 1:
        x = x.unsqueeze(0)

    h1, h1_x, h1_xx = h1_factor(x)
    g, g_x, g_xx = g_factor(x, slope)

    W = g * h1
    W_x = g_x * h1 + g * h1_x
    W_xx = g_xx * h1 + 2.0 * g_x * h1_x + g * h1_xx

    G = g + 1.0
    G_x = g_x
    G_xx = g_xx

    B2 = 4.0 * A2 * W
    B1 = 4.0 * A2 * W_x + 2.0 * A1 * W
    B0 = A2 * W_xx + A1 * W_x + A0 * W
    rhs = -(A2 * G_xx + A1 * G_x + A0 * G)

    return B2, B1, B0, rhs