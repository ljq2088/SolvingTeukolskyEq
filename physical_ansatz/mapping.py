"""坐标变换：r<->z<->x 以及 ξ(x) 的映射关系"""
import torch

def r_plus(a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    return M + torch.sqrt(torch.clamp(M*M - a*a, min=0.0))

def r_minus(a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    return M - torch.sqrt(torch.clamp(M*M - a*a, min=0.0))

def r_from_x(x: torch.Tensor, rp) -> torch.Tensor:
    """
    Convert compact coordinate x to physical radius r.

    Mapping: r = r_+ / x
    - x = 1: r = r_+ (horizon)
    - x → 0: r → ∞ (infinity)

    Safety: x must be in (0, 1] to ensure r >= r_+

    Args:
        x: compact coordinate
        rp_or_a: either r_+ directly, or spin parameter a (will compute r_+)
        M: black hole mass (only used if rp_or_a is spin parameter)
    """
    
    # 安全检查：确保x在合法范围内
    x_min = torch.min(x)
    x_max = torch.max(x)

    if x_min <= 0:
        raise ValueError(f"x must be > 0, got min(x) = {x_min:.6e}")
    if x_max > 1.0:
        raise ValueError(f"x must be <= 1, got max(x) = {x_max:.6f}")

    r = rp / x

    # 二次检查：确保r >= r_+（允许微小数值误差）
    r_min = torch.min(r)
    if r_min < rp :
        raise ValueError(f"Computed r < r_+: min(r) = {r_min:.6f}, r_+ = {rp:.6f}")

    return r

def dx_dr_from_x(x: torch.Tensor, a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    # dx/dr = -x^2 / r_plus
    rp = r_plus(a, M)
    return -(x**2) / rp

def d2x_dr2_from_x(x: torch.Tensor, a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
    # d2x/dr2 = 2 x^3 / r_plus^2
    rp = r_plus(a, M)
    return 2.0 * (x**3) / (rp**2)