"""Python port of the small MatlCheb utility set.

The source MATLAB files live in ``utils/MatlCheb-main.zip``.  This module keeps
the same Chebyshev-Gauss-Lobatto node ordering as Trefethen's ``cheb.m``:
``x_j = cos(pi*j/N)``, so nodes run from ``1`` to ``-1``.
"""

from __future__ import annotations

import numpy as np


def cheb(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Chebyshev differentiation matrix and CGL nodes."""
    if N == 0:
        return np.array([[0.0]]), np.array([1.0])
    j = np.arange(N + 1)
    x = np.cos(np.pi * j / N)
    c = np.ones(N + 1)
    c[0] = c[-1] = 2.0
    c *= (-1.0) ** j
    X = np.tile(x[:, None], (1, N + 1))
    dX = X - X.T
    D = (c[:, None] / c[None, :]) / (dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D, axis=1))
    return D, x


def anmr(N: int, xB: int, kappa: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic mesh refinement from ``AnMR.m``.

    Parameters
    ----------
    xB:
        ``+1`` or ``-1``; the endpoint near which the mesh is refined.
    kappa:
        Positive refinement strength.  ``kappa=0`` falls back to ``cheb``.
    """
    if xB not in (-1, 1):
        raise ValueError("xB must be -1 or +1")
    if abs(kappa) < 1.0e-14:
        Dx, x = cheb(N)
        return Dx @ Dx, Dx, x
    Dy, y = cheb(N)
    xy = kappa * xB**2 * 2.0 * np.cosh(kappa * (1.0 - xB * y)) / np.sinh(2.0 * kappa)
    xyy = -2.0 * kappa**2 * xB**3 / np.sinh(2.0 * kappa) * np.sinh(kappa * (1.0 - xB * y))
    x = xB * (1.0 - 2.0 * np.sinh(kappa * (1.0 - xB * y)) / np.sinh(2.0 * kappa))
    Dx = Dy / xy[:, None]
    Dxx = -Dy * (xyy / xy**3)[:, None] + (Dy @ Dy) / (xy**2)[:, None]
    return Dxx, Dx, x


def real_to_cheb(values: np.ndarray) -> np.ndarray:
    """Convert values on CGL nodes to Chebyshev coefficients.

    This direct ``O(N^2)`` implementation is intentionally simple and supports
    vectors or matrices, treating each column as one function.
    """
    f = np.asarray(values)
    was_row = f.ndim == 1
    if f.ndim == 1:
        f = f[:, None]
    N = f.shape[0] - 1
    j = np.arange(N + 1)
    k = np.arange(N + 1)
    weights = np.ones(N + 1)
    weights[0] = weights[-1] = 0.5
    C = np.cos(np.pi * np.outer(k, j) / N)
    a = (2.0 / N) * (C @ (weights[:, None] * f))
    a[0, :] *= 0.5
    a[-1, :] *= 0.5
    return a[:, 0] if was_row else a


def cheb_to_real(coeffs: np.ndarray) -> np.ndarray:
    """Evaluate Chebyshev coefficients at CGL nodes."""
    a = np.asarray(coeffs)
    was_row = a.ndim == 1
    if a.ndim == 1:
        a = a[:, None]
    N = a.shape[0] - 1
    j = np.arange(N + 1)
    k = np.arange(N + 1)
    C = np.cos(np.pi * np.outer(j, k) / N)
    f = C @ a
    return f[:, 0] if was_row else f


def cheb_interpolate(coeffs: np.ndarray, x1: float, x2: float, x: np.ndarray | float) -> np.ndarray | complex:
    """Evaluate a Chebyshev expansion on ``[x1, x2]``."""
    if x1 >= x2:
        x1, x2 = x2, x1
    a = np.asarray(coeffs)
    scalar = np.ndim(x) == 0
    xq = np.asarray(x, dtype=float)
    z = 2.0 * (xq - x1) / (x2 - x1) - 1.0
    z = np.clip(z, -1.0, 1.0)
    theta = np.arccos(z)
    if a.ndim == 1:
        n = np.arange(a.shape[0])
        out = np.sum(a[:, None] * np.cos(np.outer(n, np.ravel(theta))), axis=0)
    else:
        n = np.arange(a.shape[0])
        out = a.T @ np.cos(np.outer(n, np.ravel(theta)))
    out = out.reshape(xq.shape if a.ndim == 1 else (*xq.shape, a.shape[1]))
    if scalar:
        return complex(np.ravel(out)[0])
    return out


def cheb_tail_ratio(coeffs: np.ndarray, tail: int = 8) -> float:
    """Crude spectral convergence diagnostic."""
    a = np.asarray(coeffs)
    mag = np.linalg.norm(a.reshape(a.shape[0], -1), axis=1)
    return float(np.max(mag[-tail:]) / max(np.max(mag), 1.0e-300))
