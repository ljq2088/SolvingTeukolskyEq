"""
Infinity basis functions for autoencoder PINN.

Defines the asymptotic factor functions A_up, A_down, the y-space boundary
ansatz u(y) = 1 + c_inf*(y+1) + (y+1)^2*f(y), and slope formulas.

Coordinate convention:
    x = r_plus / r,   y = 2*x - 1
    infinity <-> x=0, y=-1
    horizon   <-> x=1, y=1

The normalisation at infinity is:
    u_up(-1) = 1,    u_down(-1) = 1
    u'_up(-1) = c_up,  u'_down(-1) = c_down   (prime = d/dy)

IMPORTANT: analytic slope formulas give du/dx.  Conversion: du/dy = (1/2) * du/dx.
"""
import torch
from physical_ansatz.mapping import r_plus, r_minus


# ============================================================
# Asymptotic factor functions
# ============================================================
def r_star(r, a, M=1.0):
    """Tortoise coordinate.  Import from utils.amplitude if available."""
    from utils.amplitude import r_star as _r_star
    return _r_star(r, a, M)


def A_up(r, a, omega, M=1.0):
    """Outgoing asymptotic factor: r^3 * exp(+i * omega * r_*).

    Args:
        r: radial coordinate, any shape
        a: Kerr spin, broadcastable
        omega: frequency, broadcastable
    Returns complex tensor.
    """
    return r ** 3 * torch.exp(1j * omega * r_star(r, a, M))


def A_down(r, a, omega, M=1.0):
    """Incoming asymptotic factor: r^{-1} * exp(-i * omega * r_*).

    Args:
        r: radial coordinate, any shape
        a: Kerr spin, broadcastable
        omega: frequency, broadcastable
    Returns complex tensor.
    """
    return r ** (-1) * torch.exp(-1j * omega * r_star(r, a, M))


# ============================================================
# Analytic slopes at infinity (x-space: du/dx at x=0)
# ============================================================
def infinity_slope_up_x(a, omega, lambda_, m=2, M=1.0):
    """du_up/dx at x=0 (infinity).

    Formula:
        du_up/dx|_{x=0} = i/(omega * r_+) * (a*m*omega + lambda/2)
    """
    rp = r_plus(a, M)
    return 1j * (a * m * omega + 0.5 * lambda_) / (omega * rp + 1e-30)


def infinity_slope_down_x(a, omega, lambda_, m=2, M=1.0):
    """du_down/dx at x=0 (infinity).

    Formula:
        du_down/dx|_{x=0} = -2*(r_+ + r_-)/r_+ + i/(omega*r_+)*(2 - a*m*omega - lambda/2)
    """
    rp = r_plus(a, M)
    rm = r_minus(a, M)
    term1 = -2.0 * (rp + rm) / rp
    term2 = 1j * (2.0 - a * m * omega - 0.5 * lambda_) / (omega * rp + 1e-30)
    return term1 + term2


# ============================================================
# Slopes in y-space: du/dy = (1/2) * du/dx
# ============================================================
def infinity_slopes_y(a, omega, lambda_, m=2, M=1.0):
    """Return (c_up, c_down) where c = du/dy|_{y=-1}.

    These are the c_inf values used in the y-space ansatz:
        u(y) = 1 + c_inf*(y+1) + (y+1)^2 * f(y)
    """
    c_up_x = infinity_slope_up_x(a, omega, lambda_, m=m, M=M)
    c_down_x = infinity_slope_down_x(a, omega, lambda_, m=m, M=M)
    return c_up_x / 2.0, c_down_x / 2.0


# Backward-compat aliases (return du/dx, not du/dy).
infinity_slope_up = infinity_slope_up_x
infinity_slope_down = infinity_slope_down_x


# ============================================================
# Boundary ansatz: u(y) = 1 + c_inf*(y+1) + (y+1)^2 * f(y)
# ============================================================
def compose_u_from_f(f, y, c_inf):
    """Apply infinity boundary ansatz.

    Args:
        f:     (B, N) complex — free function from decoder
        y:     (B, N) or (N,) — compact coordinate in [-1, 1]
        c_inf: (B,) complex — du/dy at y=-1 from infinity_slopes_y

    Returns:
        u: (B, N) complex satisfying u(-1)=1, u_y(-1)=c_inf.
    """
    if y.ndim == 1:
        y = y.unsqueeze(0)
    if c_inf.ndim == 1:
        c_inf = c_inf.unsqueeze(-1)
    eta = y + 1.0  # eta = 0 at infinity (y=-1)
    return 1.0 + c_inf * eta + eta ** 2 * f


# ============================================================
# Coefficient transformation: x-space u-equation -> y-space f-equation
#
# Uses the ansatz:  u(x) = g(x)*(h1(x)*f(y) + 1) + 1
#   with g(x) = slope*x, h1(x) = x, x = (y+1)/2
#
# Kept for potential use; Stage 3 may prefer direct R=A*u residual.
# ============================================================
def _broadcast_slope_to_x(slope, x):
    dtype = torch.promote_types(x.dtype, slope.dtype)
    slope = slope.to(dtype=dtype, device=x.device)
    x = x.to(dtype=dtype, device=x.device)
    if slope.ndim == 1 and x.ndim == 1:
        return slope.unsqueeze(-1), x.unsqueeze(0)
    while slope.ndim < x.ndim:
        slope = slope.unsqueeze(-1)
    while x.ndim < slope.ndim:
        x = x.unsqueeze(0)
    return slope, x


def h1_inf(x):
    h1 = x
    h1_x = torch.ones_like(x, dtype=x.dtype, device=x.device)
    h1_xx = torch.zeros_like(x, dtype=x.dtype, device=x.device)
    return h1, h1_x, h1_xx


def g_inf(x, slope):
    slope, x = _broadcast_slope_to_x(slope, x)
    g = slope * x
    g_x = slope * torch.ones_like(x, dtype=slope.dtype, device=x.device)
    g_xx = torch.zeros_like(x, dtype=slope.dtype, device=x.device)
    return g, g_x, g_xx


def transform_coeffs_u_to_y(A2, A1, A0, y, slope):
    """
    Transform x-space ODE coeffs for u(x) to y-space coeffs for f(y)
    under ansatz:  u(x) = g(x)*(h1(x)*f(y) + 1) + 1.

    B2*f_yy + B1*f_y + B0*f + rhs = 0

    Kept for reference.  Stage 3 may use direct R=A*u residual instead.
    """
    x = 0.5 * (y + 1.0)
    if A2.ndim >= 2 and x.ndim == 1:
        x = x.unsqueeze(0)
    h1, h1_x, h1_xx = h1_inf(x)
    g, g_x, g_xx = g_inf(x, slope)
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
