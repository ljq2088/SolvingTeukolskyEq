"""
Utilities for u_basis functions used in the autoencoder PINN.

This module defines ansatz and coefficient transforms for the two radial basis
functions u_up and u_down that solve the homogeneous Teukolsky radial equation
after factoring out the asymptotic behaviors r^3 e^{+i \u03c9 r_*} and r^{-1} e^{-i \u03c9 r_*},
respectively.  See docs/02_u_basis_equations_and_boundary.md for derivations.

For each basis u(x) with x = r_+/r in [0,1], we write

    u(x) = g(x) * (h1(x) * f(y) + 1) + 1,  with y = 2 x - 1,

where g(x) and h1(x) are chosen to enforce the boundary conditions at x=0 (y=-1):

    u(0) = 1,
    u_x(0) = c_inf,

with c_inf given by analytic expressions derived from the asymptotic expansion:

    d u_up / d x |_{x=0} = i/(\u03c9 r_+) * (a m \u03c9 + \u03bb/2),
    d u_down / d x |_{x=0} = -2 (r_+ + r_-) / r_+ + i/(\u03c9 r_+) * (2 - a m \u03c9 - \u03bb/2).

We use simple choices

    h1(x) = x,
    g(x) = slope * x,

so that u(x) = slope * x * (x * f(y) + 1) + 1 automatically satisfies the boundary conditions.
The slope equals the corresponding c_inf.

The functions below provide:
- infinity_slope_up / infinity_slope_down: compute c_inf for u_up/u_down.
- h1_inf and g_inf: return h1, h1_x, h1_xx, g, g_x, g_xx used to build the ansatz.
- transform_coeffs_u_to_y: transform the x-coordinate ODE coefficients for u(x)
  into y-coordinate coefficients for f(y) using the chain rule.
"""

import torch
from physical_ansatz.mapping import r_plus, r_minus


def infinity_slope_up(a: torch.Tensor,
                      omega: torch.Tensor,
                      lambda_: torch.Tensor,
                      m: int,
                      M: float = 1.0,
                      s: int = -2) -> torch.Tensor:
    """
    Compute du_up/dx at x=0 (y=-1) for the u_up basis.

    Slope formula derived from the asymptotic expansion of the Teukolsky radial equation:

        d u_up / d x |_{x=0} = i/(\u03c9 r_+) * (a m \u03c9 + \u03bb/2).

    Args:
        a: Kerr spin parameter (B,)
        omega: frequency (B,)
        lambda_: separation constant (B,)
        m: azimuthal number
        M: mass of the black hole
        s: spin weight (default -2)

    Returns:
        slope tensor of shape (B,) with complex dtype.
    """
    rp = r_plus(a, M)
    numerator = a * m * omega + 0.5 * lambda_
    denominator = omega * rp
    eps = 1e-20
    slope = 1j * numerator / (denominator + eps)
    return slope


def infinity_slope_down(a: torch.Tensor,
                        omega: torch.Tensor,
                        lambda_: torch.Tensor,
                        m: int,
                        M: float = 1.0,
                        s: int = -2) -> torch.Tensor:
    """
    Compute du_down/dx at x=0 (y=-1) for the u_down basis.

    Slope formula derived from the asymptotic expansion:

        d u_down / d x |_{x=0} =
            -2 (r_+ + r_-) / r_+  +  i/(\u03c9 r_+) * (2 - a m \u03c9 - \u03bb/2).

    Args:
        a: Kerr spin (B,)
        omega: frequency (B,)
        lambda_: separation constant (B,)
        m: azimuthal number
        M: mass of black hole
        s: spin weight

    Returns:
        slope tensor (B,) with complex dtype.
    """
    rp = r_plus(a, M)
    rm = r_minus(a, M)
    term1 = -2.0 * (rp + rm) / rp
    numerator = 2.0 - a * m * omega - 0.5 * lambda_
    denominator = omega * rp
    eps = 1e-20
    term2 = 1j * numerator / (denominator + eps)
    return term1 + term2


def _broadcast_slope_to_x(slope: torch.Tensor, x: torch.Tensor):
    """
    Broadcast slope and x to compatible shapes.

    slope: (B,) or scalar
    x: (N,) or (B,N)
    Returns slope and x broadcast to at least 2D: (B,1) and (1,N) or (B,N).
    """
    dtype = torch.promote_types(x.dtype, slope.dtype)
    slope = slope.to(dtype=dtype, device=x.device)
    x = x.to(dtype=dtype, device=x.device)
    if slope.ndim == 1 and x.ndim == 1:
        slope = slope.unsqueeze(-1)
        x = x.unsqueeze(0)
        return slope, x
    while slope.ndim < x.ndim:
        slope = slope.unsqueeze(-1)
    while x.ndim < slope.ndim:
        x = x.unsqueeze(0)
    return slope, x


def h1_inf(x: torch.Tensor):
    """
    Basis h1(x) for the u bases.

    We choose h1(x) = x, so that h1(0) = 0 and h1_x(0) = 1.

    Returns:
        h1: same shape as x
        h1_x: same shape
        h1_xx: same shape (all zeros)
    """
    h1 = x
    h1_x = torch.ones_like(x, dtype=x.dtype, device=x.device)
    h1_xx = torch.zeros_like(x, dtype=x.dtype, device=x.device)
    return h1, h1_x, h1_xx


def g_inf(x: torch.Tensor, slope: torch.Tensor):
    """
    Basis g(x) for the u bases.

    We choose g(x) = slope * x, which satisfies g(0)=0 and g_x(0)=slope.

    slope may have shape (B,) and x shape (N,) or (B,N).
    Returns:
        g, g_x, g_xx broadcast to the combined shape.
    """
    slope, x = _broadcast_slope_to_x(slope, x)
    g = slope * x
    g_x = slope * torch.ones_like(x, dtype=slope.dtype, device=x.device)
    g_xx = torch.zeros_like(x, dtype=slope.dtype, device=x.device)
    return g, g_x, g_xx


def transform_coeffs_u_to_y(A2: torch.Tensor,
                            A1: torch.Tensor,
                            A0: torch.Tensor,
                            y: torch.Tensor,
                            slope: torch.Tensor):
    """
    Given ODE coefficients A2,A1,A0 in x-space for u(x), transform them into y-space
    coefficients B2,B1,B0 and an inhomogeneous term rhs for f(y) under the ansatz

        u(x) = g(x) * (h1(x) * f(y) + 1) + 1

    with x = (y+1)/2.

    The chain rule is similar to the S ansatz in transform_y.transform_coeffs_x_to_y:

        B2 = 4 * A2 * W
        B1 = 4 * A2 * W_x + 2 * A1 * W
        B0 = A2 * W_xx + A1 * W_x + A0 * W
        rhs = -(A2 * G_xx + A1 * G_x + A0 * G)

    where
        W = g * h1,  G = g + 1.

    Args:
        A2,A1,A0: tensors broadcastable with slope to shape (B,N) or (N,)
        y: 1D or 2D tensor of y ∈ [-1,1] with shape (N,) or (B,N)
        slope: slope tensor (B,) or scalar.

    Returns:
        B2,B1,B0,rhs tensors broadcastable to (B,N) or (N,)
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
