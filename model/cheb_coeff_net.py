"""
Chebyshev Coefficient Network for machine-precision Teukolsky solving.

Core idea:
    f(y; a,omega) = sum_{n=0}^{N-1} c_n(a,omega) * T_n(y)

- T_n(y): Chebyshev polynomials (y in [-1,1])
- c_n(a,omega): complex coefficients output by a parameter network
- Derivatives via exact Chebyshev recurrence (no autograd in y-direction)

Spectral convergence: N=64 yields ~1e-14 for smooth functions.
Inference: O(N) per y-point via Clenshaw recurrence, < 1us per point.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ===========================================================================
# Chebyshev basis & Clenshaw evaluation
# ===========================================================================

def clenshaw_evaluate(coeff: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Clenshaw recurrence for f(y) = sum_{k=0}^{N-1} c_k T_k(y).

    Args:
        coeff: (Nc,) or (B, Nc) complex
        y: scalar or (Ny,)

    Returns:
        scalar / (Ny,) / (B,) / (B, Ny) complex
    """
    if coeff.ndim not in (1, 2):
        raise ValueError(f"coeff must be 1D or 2D, got shape {tuple(coeff.shape)}")

    coeff_was_1d = coeff.ndim == 1
    if coeff_was_1d:
        coeff = coeff.unsqueeze(0)  # (1, Nc)

    y_was_scalar = y.ndim == 0
    y = y.reshape(-1).to(device=coeff.device, dtype=coeff.real.dtype)

    B, Nc = coeff.shape
    Ny = y.numel()
    device = coeff.device

    b_kp1 = torch.zeros((B, Ny), dtype=coeff.dtype, device=device)
    b_kp2 = torch.zeros((B, Ny), dtype=coeff.dtype, device=device)

    for k in range(Nc - 1, 0, -1):
        c_k = coeff[:, k:k + 1]  # (B, 1)
        b_k = 2.0 * y * b_kp1 - b_kp2 + c_k
        b_kp2 = b_kp1
        b_kp1 = b_k

    out = y * b_kp1 - b_kp2 + coeff[:, 0:1]  # (B, Ny)

    if coeff_was_1d:
        out = out[0]
    if y_was_scalar:
        out = out[..., 0]

    return out


def cheb_derivative_coeffs(coeff: torch.Tensor) -> torch.Tensor:
    """
    Convert Chebyshev coefficients of f to coefficients of f'.

    Given f = sum_{n=0}^{N-1} c_n T_n,
    compute c'_n such that f' = sum_{n=0}^{N-2} c'_n T_n.

    Recurrence:
        c'_{N} = c'_{N-1} = 0
        c'_{n-1} = c'_{n+1} + 2 n c_n    for n = N-1, N-2, ..., 2
        c'_0 = c'_2 / 2 + c_1            (special case for n=1)

    Args:
        coeff: (B, N) complex

    Returns:
        coeff': (B, N) complex (last entry zero, f' is degree N-1)
    """
    B, N = coeff.shape
    device = coeff.device
    dtype = coeff.dtype

    cp = torch.zeros_like(coeff)

    for n in range(N - 1, 1, -1):
        cp_np1 = cp[:, n + 1] if n + 1 < N else 0.0
        cp[:, n - 1] = cp_np1 + 2.0 * n * coeff[:, n]

    # Special case n=1: c'_0 = c'_2/2 + c_1
    cp2 = cp[:, 2] if N > 2 else 0.0
    cp[:, 0] = cp2 / 2.0 + coeff[:, 1]

    return cp


def clenshaw_evaluate_with_derivatives(
    coeff: torch.Tensor, y: torch.Tensor
):
    """
    Evaluate f(y), f'(y), f''(y) using Clenshaw + Chebyshev derivative recurrences.

    Args:
        coeff: (B, N) complex Chebyshev coefficients
        y: (Ny,) real in [-1, 1]

    Returns:
        f:   (B, Ny) complex
        fy:  (B, Ny) complex  -- df/dy
        fyy: (B, Ny) complex  -- d^2f/dy^2
    """
    # Convert coefficients to derivative coefficients
    coeff_y = cheb_derivative_coeffs(coeff)       # (B, N)
    coeff_yy = cheb_derivative_coeffs(coeff_y)    # (B, N)

    f = clenshaw_evaluate(coeff, y)
    fy = clenshaw_evaluate(coeff_y, y)
    fyy = clenshaw_evaluate(coeff_yy, y)

    return f, fy, fyy


def cheb_basis_matrix(y: torch.Tensor, order: int) -> torch.Tensor:
    """
    Tmat[j, k] = T_k(y_j) for k=0..order.

    Args:
        y: (Ny,)
        order: int

    Returns:
        Tmat: (Ny, order+1)
    """
    Ny = y.numel()
    Tmat = torch.empty((Ny, order + 1), dtype=y.dtype, device=y.device)
    Tmat[:, 0] = 1.0
    if order >= 1:
        Tmat[:, 1] = y
    for k in range(1, order):
        Tmat[:, k + 1] = 2.0 * y * Tmat[:, k] - Tmat[:, k - 1]
    return Tmat


def cheb_derivative_basis_matrix(y: torch.Tensor, order: int) -> torch.Tensor:
    """
    T'_mat[j, k] = T'_k(y_j) for k=0..order.

    Uses recurrence: T'_0=0, T'_1=1, T'_k = 2*T_{k-1} + 2*y*T'_{k-1} - T'_{k-2}
    """
    Ny = y.numel()
    Tp = torch.empty((Ny, order + 1), dtype=y.dtype, device=y.device)
    Tp[:, 0] = 0.0
    if order >= 1:
        Tp[:, 1] = 1.0
    T = cheb_basis_matrix(y, max(1, order))
    for k in range(2, order + 1):
        # T'_k = 2 * T_{k-1} + 2*y * T'_{k-1} - T'_{k-2}
        Tp[:, k] = 2.0 * T[:, k - 1] + 2.0 * y * Tp[:, k - 1] - Tp[:, k - 2]
    return Tp


# ===========================================================================
# Coefficient network
# ===========================================================================

def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class ChebCoeffNet(nn.Module):
    """
    Chebyshev Coefficient Network.

    Maps (a, omega) -> N complex Chebyshev coefficients c_n(a,omega).
    Then evaluates f(y) = sum c_n T_n(y) via Clenshaw recurrence.

    Compatible with the same forward(a, omega, y, u=None, v=None) interface
    as PINN_MLP, and provides exact f_y, f_yy without autograd.
    """

    def __init__(
        self,
        N: int = 64,
        hidden_dims: list[int] | None = None,
        activation: str = "silu",
        param_embed_dim: int = 64,
        # chart_uv mode
        local_coord_mode: str = "raw_aw",
        u_center_local: float = 0.5,
        v_center_local: float = 0.5,
        u_half_range_local: float = 0.12,
        v_half_range_local: float = 0.12,
        a_center_local: float = 0.5,
        a_half_range_local: float = 0.499,
        omega_min_local: float = 1.0e-4,
        omega_max_local: float = 10.0,
        M: float = 1.0,
        m_mode: int = 2,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 256, 256, 128]

        self.N = int(N)
        self.hidden_dims = list(hidden_dims)
        self.activation_name = activation

        self.local_coord_mode = str(local_coord_mode)
        self.u_center_local = float(u_center_local)
        self.v_center_local = float(v_center_local)
        self.u_half_range_local = float(u_half_range_local)
        self.v_half_range_local = float(v_half_range_local)
        self.a_center_local = float(a_center_local)
        self.a_half_range_local = float(a_half_range_local)
        self.omega_min_local = float(omega_min_local)
        self.omega_max_local = float(omega_max_local)
        self.M = float(M)
        self.m_mode = int(m_mode)

        # Parameter features: same 10D as PINN_MLP
        self.param_in_dim = 10

        # Parameter encoder
        layers = []
        prev_dim = self.param_in_dim
        layers.append(nn.Linear(prev_dim, param_embed_dim))
        layers.append(_make_activation(activation))
        prev_dim = param_embed_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(_make_activation(activation))
            prev_dim = hd
        self.param_encoder = nn.Sequential(*layers)

        # Coefficient head: outputs 2*N real values (real + imag parts)
        self.coeff_head = nn.Linear(prev_dim, 2 * self.N)

        self._init_weights()

    def _init_weights(self):
        """Xavier for encoder, small-init for coeff head."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if module is self.coeff_head:
                    nn.init.xavier_normal_(module.weight, gain=0.01)
                    nn.init.zeros_(module.bias)
                else:
                    nn.init.xavier_normal_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Parameter normalisation
    # ------------------------------------------------------------------
    def _normalize_raw_aw(self, a, omega):
        alpha = (a - self.a_center_local) / self.a_half_range_local
        omega_safe = torch.clamp(omega, min=1.0e-12)
        logw = torch.log10(omega_safe)
        logw_min = math.log10(self.omega_min_local)
        logw_max = math.log10(self.omega_max_local)
        xi = 2.0 * (logw - logw_min) / (logw_max - logw_min) - 1.0
        return alpha, xi

    def _normalize_chart_uv(self, u, v):
        alpha = (u - self.u_center_local) / self.u_half_range_local
        xi = (v - self.v_center_local) / self.v_half_range_local
        return alpha, xi

    def compute_local_coords(self, a, omega, u=None, v=None):
        if self.local_coord_mode == "raw_aw":
            return self._normalize_raw_aw(a, omega)
        if u is None or v is None:
            raise ValueError("chart_uv mode requires u, v")
        return self._normalize_chart_uv(u, v)

    def _build_param_features(self, a, omega, u=None, v=None):
        alpha, xi = self.compute_local_coords(a, omega, u, v)
        M = self.M
        m_mode = self.m_mode

        spin_gap = torch.sqrt(torch.clamp(1.0 - (a / M) ** 2, min=1.0e-12))
        r_plus = M * (1.0 + spin_gap)
        Omega_H = a / (2.0 * M * r_plus)
        k = omega - m_mode * Omega_H
        omega_safe = torch.clamp(omega, min=1.0e-12)
        log10_omega = torch.log10(omega_safe)

        feats = torch.cat(
            [
                alpha,
                xi,
                alpha ** 2,
                xi ** 2,
                alpha * xi,
                r_plus,
                spin_gap,
                Omega_H,
                k,
                log10_omega,
            ],
            dim=-1,
        )
        return feats, alpha, xi

    # ------------------------------------------------------------------
    # Coefficient computation
    # ------------------------------------------------------------------
    def compute_coefficients(self, a, omega, u=None, v=None):
        """
        Compute Chebyshev coefficients c_n(a,omega) as complex tensor.

        Args:
            a: (B,) or (B,1)
            omega: (B,) or (B,1)
            u, v: optional (B,) or (B,1)

        Returns:
            coeff: (B, N) complex
        """
        if a.ndim == 1:
            a = a.unsqueeze(-1)
        if omega.ndim == 1:
            omega = omega.unsqueeze(-1)
        if u is not None and u.ndim == 1:
            u = u.unsqueeze(-1)
        if v is not None and v.ndim == 1:
            v = v.unsqueeze(-1)

        feats, _, _ = self._build_param_features(a, omega, u, v)
        h = self.param_encoder(feats)  # (B, hidden_dim)
        raw = self.coeff_head(h)       # (B, 2*N)

        coeff_re = raw[:, 0:self.N]
        coeff_im = raw[:, self.N:]
        return torch.complex(coeff_re, coeff_im)

    # ------------------------------------------------------------------
    # Forward (compatible with PINN_MLP)
    # ------------------------------------------------------------------
    def forward(self, a, omega, y, u=None, v=None):
        """
        Args:
            a:     (B,) or (B,1)
            omega: (B,) or (B,1)
            y:     (N,) or (B,N)
            u, v:  optional (B,) or (B,1)

        Returns:
            f: (B,N) complex
        """
        model_param = next(self.parameters())
        target_device = model_param.device
        target_dtype = model_param.dtype

        a = a.to(device=target_device, dtype=target_dtype)
        omega = omega.to(device=target_device, dtype=target_dtype)
        y = y.to(device=target_device, dtype=target_dtype)
        if u is not None:
            u = u.to(device=target_device, dtype=target_dtype)
        if v is not None:
            v = v.to(device=target_device, dtype=target_dtype)

        if a.ndim == 1:
            a = a.unsqueeze(-1)
        if omega.ndim == 1:
            omega = omega.unsqueeze(-1)
        if y.ndim == 1:
            y = y.unsqueeze(0).expand(a.shape[0], -1)
        if u is not None and u.ndim == 1:
            u = u.unsqueeze(-1)
        if v is not None and v.ndim == 1:
            v = v.unsqueeze(-1)

        B, Ny = y.shape
        y_eval = y[0]  # (Ny,) — all rows identical after expansion

        coeff = self.compute_coefficients(a, omega, u, v)  # (B, N)
        f = clenshaw_evaluate(coeff, y_eval)               # (B, Ny)
        return f

    # ------------------------------------------------------------------
    # Forward with exact derivatives (no autograd in y)
    # ------------------------------------------------------------------
    def forward_with_derivatives(self, a, omega, y, u=None, v=None):
        """
        Compute f(y), df/dy, d^2f/dy^2 using exact Chebyshev derivatives.

        Args:
            a, omega: (B,) or (B,1)
            y: (N,) or (B,N)
            u, v: optional

        Returns:
            f:   (B,N) complex
            fy:  (B,N) complex -- df/dy
            fyy: (B,N) complex -- d^2f/dy^2
        """
        model_param = next(self.parameters())
        target_device = model_param.device
        target_dtype = model_param.dtype

        a = a.to(device=target_device, dtype=target_dtype)
        omega = omega.to(device=target_device, dtype=target_dtype)
        y = y.to(device=target_device, dtype=target_dtype)
        if u is not None:
            u = u.to(device=target_device, dtype=target_dtype)
        if v is not None:
            v = v.to(device=target_device, dtype=target_dtype)

        if a.ndim == 1:
            a = a.unsqueeze(-1)
        if omega.ndim == 1:
            omega = omega.unsqueeze(-1)
        if y.ndim == 1:
            y = y.unsqueeze(0).expand(a.shape[0], -1)
        if u is not None and u.ndim == 1:
            u = u.unsqueeze(-1)
        if v is not None and v.ndim == 1:
            v = v.unsqueeze(-1)

        B, Ny = y.shape
        y_eval = y[0]

        coeff = self.compute_coefficients(a, omega, u, v)

        f, fy, fyy = clenshaw_evaluate_with_derivatives(coeff, y_eval)
        return f, fy, fyy
