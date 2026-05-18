"""Infinity Robin boundary constraint for S-equation at y=-1.

Based on analytic derivation in docs/instruction/inf_coeff.md.

For R = P*u with the reduced u-equation, at infinity (y=-1, x=0):
    A2(-1) = 0  (PDE degenerates to first order)
    A1(-1) = -4i * omega * r_+
    A0(-1) = 8*r_+*omega^2 - 4*a*m*omega - lambda + i*omega*(2*r_+ + 4)

The Robin condition is:
    u_y(-1) = c_inf * u(-1)
    c_inf = 1/2 + 1/r_+ - i*(8*r_+*omega^2 - 4*a*m*omega - lambda)/(4*omega*r_+)

Since S(y) and u(y) satisfy the same Robin ratio (h2 is y-independent),
    S_y(-1) = c_inf * S(-1).
"""
import torch
from physical_ansatz.mapping import r_plus


def analytic_c_inf(a, omega, lambda_, m=2, M=1.0, s=-2):
    """Compute the infinity Robin slope c_inf analytically.

    c_inf = 1/2 + 1/r_+ - i*(8*r_+*omega^2 - 4*a*m*omega - lambda)/(4*omega*r_+)

    Args:
        a: (B,) real tensor, spin parameter
        omega: (B,) real tensor, frequency (must be non-zero)
        lambda_: (B,) complex tensor, angular eigenvalue
        m, M, s: Teukolsky parameters

    Returns:
        c_inf: (B,) complex tensor — Robin slope at y=-1
    """
    rp = r_plus(a, M)  # (B,)
    num = 8.0 * rp * omega**2 - 4.0 * a * m * omega - lambda_
    denom = 4.0 * omega * rp
    c_inf = 0.5 + 1.0 / rp - 1j * num / denom
    return c_inf


def analytic_infinity_coefficients(a, omega, lambda_, m=2, M=1.0, s=-2):
    """Compute A2, A1, A0 at y=-1 analytically.

    Returns:
        dict with keys:
            A2_inf: (B,) complex — zero (degenerate PDE)
            A1_inf: (B,) complex — -4i*omega*r_+ (finite, non-zero for omega != 0)
            A0_inf: (B,) complex — finite constant
    """
    rp = r_plus(a, M)
    A2_inf = torch.zeros_like(a, dtype=lambda_.dtype)
    A1_inf = -4j * omega * rp
    A0_inf = (8.0 * rp * omega**2 - 4.0 * a * m * omega - lambda_
              + 1j * omega * (2.0 * rp + 4.0))
    return {
        "A2_inf": A2_inf,
        "A1_inf": A1_inf,
        "A0_inf": A0_inf,
    }


def infinity_robin_slope_for_S(a, omega, lambda_, m=2, M=1.0, s=-2):
    """Return c_inf — the Robin slope at y=-1. Thin wrapper around analytic_c_inf."""
    return analytic_c_inf(a, omega, lambda_, m=m, M=M, s=s)


def infinity_robin_residual(S_val, Sy_val, c_inf):
    """Compute Robin residual: S_y(-1) - c_inf * S(-1).

    Args:
        S_val: (B,) complex — S at y=-1
        Sy_val: (B,) complex — dS/dy at y=-1
        c_inf: (B,) complex — Robin slope

    Returns:
        (B,) complex residual
    """
    return Sy_val - c_inf * S_val


def infinity_robin_loss(S_val, Sy_val, c_inf, eps=1e-12):
    """Compute relative normalized Robin loss.

    L_inf = |S_y - c_inf*S|^2 / (|S_y|^2 + |c_inf*S|^2 + eps)

    Args:
        S_val: (B,) complex — S at y=-1
        Sy_val: (B,) complex — dS/dy at y=-1
        c_inf: (B,) complex — Robin slope
        eps: float, small regularization

    Returns:
        (B,) float loss per batch element
    """
    residual = infinity_robin_residual(S_val, Sy_val, c_inf)
    num = residual.real ** 2 + residual.imag ** 2
    denom = (Sy_val.real ** 2 + Sy_val.imag ** 2
             + (c_inf * S_val).real ** 2 + (c_inf * S_val).imag ** 2
             + eps)
    return num / denom


def compute_S_and_Sy_at_infinity(f_val, fy_val, slope):
    """Compute S(-1) and S_y(-1) from f(-1), f_y(-1) and the horizon slope.

    S(x) = g(x) * (h1(x) * f(y) + 1) + 1
    where x = (y+1)/2, g(x) = slope*(exp(x-1)-1), h1(x) = exp(x-1)-1

    At y=-1 (x=0):
      g(0)  = slope * (1/e - 1)
      g_x(0)= slope / e
      h1(0) = 1/e - 1
      h1_x(0)= 1/e
      S = g*(h1*f + 1) + 1
      S_x = g_x*(h1*f + 1) + g*(h1_x*f + h1*f_x)
      S_y = S_x / 2   (since dx/dy = 1/2)

    Args:
        f_val: (B,) complex — f at y=-1
        fy_val: (B,) complex — df/dy at y=-1  (note: this is f_y, not f_x)
        slope: (B,) complex — horizon regularity slope

    Returns:
        S_val: (B,) complex
        Sy_val: (B,) complex
    """
    inv_e = 1.0 / torch.e
    h1_0 = inv_e - 1.0       # h1(0)
    h1_x_0 = inv_e           # h1_x(0)
    g_0 = slope * h1_0       # g(0)
    g_x_0 = slope * inv_e    # g_x(0)

    S_val = g_0 * (h1_0 * f_val + 1.0) + 1.0

    # f_x = 2 * f_y  (chain rule: x = (y+1)/2, dx/dy = 1/2)
    fx_val = 2.0 * fy_val
    S_x_val = g_x_0 * (h1_0 * f_val + 1.0) + g_0 * (h1_x_0 * f_val + h1_0 * fx_val)
    Sy_val = S_x_val / 2.0

    return S_val, Sy_val
