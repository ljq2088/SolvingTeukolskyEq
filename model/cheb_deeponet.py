"""
Chebyshev DeepONet learning f(y) within the full ansatz.

Architecture:
  f(y; p) = h_base(y) + alpha·h_a(y) + xi·h_omega(y) + (alpha²+xi²)·h_nl(y)

where each h_*(y) is a Chebyshev expansion: sum_n c_n^* · T_n(y).

The full ansatz (enforced in training pipeline):
  S(x) = g(x) * (h1(x) * f(y) + 1) + 1
  R_in(r) = P(r) * h2 * S(x)

Hard BC enforced by construction:
  - S(1) = 1  (g(1) = 0)
  - S_x(1) = c_H = -A0/A1|_x=1  (g_x(1) = c_H, h1(1) = 0)

Key advantages over PINN_MLP:
  - Full BC: both value and derivative at horizon enforced exactly
  - Spectral convergence in y (N=48 Chebyshev)
  - Exact f_y, f_yy via Chebyshev recurrence (no autograd noise)

Compatible with PINN_MLP via output_type="f" detection in training pipeline.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ===========================================================================
# Chebyshev basis & Clenshaw evaluation (from cheb_coeff_net.py)
# ===========================================================================

def clenshaw_evaluate(coeff: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Clenshaw recurrence: f(y) = sum_{k=0}^{N-1} c_k T_k(y).

    Args:
        coeff: (B, N) complex
        y: (Ny,) real in [-1, 1]
    Returns:
        (B, Ny) complex
    """
    if coeff.ndim == 1:
        coeff = coeff.unsqueeze(0)
    y_was_scalar = y.ndim == 0
    y = y.reshape(-1).to(device=coeff.device, dtype=coeff.real.dtype)
    B, Nc = coeff.shape
    Ny = y.numel()

    b_kp1 = torch.zeros((B, Ny), dtype=coeff.dtype, device=coeff.device)
    b_kp2 = torch.zeros((B, Ny), dtype=coeff.dtype, device=coeff.device)

    for k in range(Nc - 1, 0, -1):
        b_k = 2.0 * y * b_kp1 - b_kp2 + coeff[:, k:k + 1]
        b_kp2, b_kp1 = b_kp1, b_k

    out = y * b_kp1 - b_kp2 + coeff[:, 0:1]
    if y_was_scalar:
        out = out[..., 0]
    return out


def cheb_derivative_coeffs(coeff: torch.Tensor) -> torch.Tensor:
    """Convert coefficients of f to coefficients of f'.

    c'_{N} = c'_{N-1} = 0
    c'_{n-1} = c'_{n+1} + 2 n c_n  (n = N-1, ..., 2)
    c'_0 = c'_2/2 + c_1
    """
    B, N = coeff.shape
    cp = torch.zeros_like(coeff)
    for n in range(N - 1, 1, -1):
        cp_np1 = cp[:, n + 1] if n + 1 < N else 0.0
        cp[:, n - 1] = cp_np1 + 2.0 * n * coeff[:, n]
    cp2 = cp[:, 2] if N > 2 else 0.0
    cp[:, 0] = cp2 / 2.0 + coeff[:, 1]
    return cp


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


class ChebDeepONet(nn.Module):
    """Chebyshev DeepONet with hard boundary constraints & Taylor expansion.

    Compatible interface with PINN_MLP for drop-in replacement.
    """

    def __init__(
        self,
        N: int = 48,
        hidden_dims: list[int] | None = None,
        activation: str = "silu",
        param_embed_dim: int = 128,
        # chart_uv mode
        local_coord_mode: str = "chart_uv",
        u_center_local: float = 0.5,
        v_center_local: float = 0.5,
        u_half_range_local: float = 0.12,
        v_half_range_local: float = 0.12,
        # raw_aw mode backward compat
        a_center_local: float = 0.5,
        a_half_range_local: float = 0.499,
        omega_min_local: float = 1.0e-4,
        omega_max_local: float = 10.0,
        # physics
        M: float = 1.0,
        m_mode: int = 2,
        # Taylor structure: if False, single head (no base/a/omega/nl split)
        use_taylor: bool = True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 256, 256, 128]

        self.N = int(N)
        self.hidden_dims = list(hidden_dims)
        self.activation_name = activation
        self.use_taylor = bool(use_taylor)

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
        self.output_type = "f"  # uses original f-PDE path; full BC via g*(h1*f+1)+1 ansatz

        # Parameter features: [α, ξ, α², ξ², αξ, r+, spin_gap, Ω_H, k, log₁₀(ω),
        #                       sin(πα), cos(πα), sin(πξ), cos(πξ)]
        self.param_in_dim = 14

        # ---- Parameter encoder ----
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
        self.encoder_out_dim = prev_dim

        # ---- Coefficient heads ----
        n_heads = 4 if use_taylor else 1
        self.n_heads = n_heads
        for i in range(n_heads):
            head = nn.Linear(prev_dim, 2 * self.N)  # 2N = real + imag
            self.add_module(f"coeff_head_{i}", head)

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "coeff_head" in name:
                    nn.init.xavier_normal_(module.weight, gain=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    nn.init.xavier_normal_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Parameter normalisation (identical to PINN_MLP)
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

        feats = torch.cat([
            alpha, xi,
            alpha ** 2, xi ** 2, alpha * xi,
            r_plus, spin_gap, Omega_H, k, log10_omega,
            torch.sin(math.pi * alpha), torch.cos(math.pi * alpha),
            torch.sin(math.pi * xi), torch.cos(math.pi * xi),
        ], dim=-1)
        return feats, alpha, xi

    # ------------------------------------------------------------------
    # Coefficient computation
    # ------------------------------------------------------------------
    def compute_coefficients(self, a, omega, u=None, v=None):
        """Compute Chebyshev coefficients. Returns (B, n_heads, N) complex."""
        if a.ndim == 1:
            a = a.unsqueeze(-1)
        if omega.ndim == 1:
            omega = omega.unsqueeze(-1)
        if u is not None and u.ndim == 1:
            u = u.unsqueeze(-1)
        if v is not None and v.ndim == 1:
            v = v.unsqueeze(-1)

        feats, _, _ = self._build_param_features(a, omega, u, v)
        h = self.param_encoder(feats)  # (B, enc_dim)

        coeffs = []
        for i in range(self.n_heads):
            head = getattr(self, f"coeff_head_{i}")
            raw = head(h)  # (B, 2*N)
            coeffs.append(torch.complex(raw[:, :self.N], raw[:, self.N:]))
        return torch.stack(coeffs, dim=1)  # (B, n_heads, N)

    # ------------------------------------------------------------------
    # Forward — returns f(y) = h(y) directly (no hard BC in model)
    # Full ansatz S = g*(h1*f + 1) + 1 is applied in training pipeline.
    # ------------------------------------------------------------------
    def forward(self, a, omega, y, u=None, v=None):
        """
        f(y; p) = h_base(y) + alpha·h_a(y) + xi·h_omega(y) + (alpha²+xi²)·h_nl(y)

        Args:
            a:     (B,) or (B,1)
            omega: (B,) or (B,1)
            y:     (N,) or (B,N)
            u, v:  optional (B,) or (B,1)
        Returns:
            f: (B,N) complex — Chebyshev expansion evaluated at y
        """
        model_param = next(self.parameters())
        device = model_param.device
        dtype = model_param.dtype

        a = a.to(device=device, dtype=dtype)
        omega = omega.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)
        if u is not None:
            u = u.to(device=device, dtype=dtype)
        if v is not None:
            v = v.to(device=device, dtype=dtype)

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
        y_eval = y[0]  # (Ny,) — all rows identical

        _, alpha, xi = self._build_param_features(a, omega, u, v)
        coeffs = self.compute_coefficients(a, omega, u, v)

        # Taylor assemble: f = c_base + alpha*c_a + xi*c_omega + (alpha²+xi²)*c_nl
        f = 0.0
        for i in range(self.n_heads):
            h_i = clenshaw_evaluate(coeffs[:, i, :], y_eval)  # (B, Ny)
            if i == 0:
                f = h_i if not self.use_taylor else h_i
            elif i == 1:
                f = f + alpha.squeeze(-1).unsqueeze(-1) * h_i
            elif i == 2:
                f = f + xi.squeeze(-1).unsqueeze(-1) * h_i
            elif i == 3:
                rho2 = alpha.squeeze(-1) ** 2 + xi.squeeze(-1) ** 2
                f = f + rho2.unsqueeze(-1) * h_i
        return f

    # ------------------------------------------------------------------
    # Forward with exact Chebyshev derivatives — returns f, f_y, f_yy
    # ------------------------------------------------------------------
    def forward_with_derivatives(self, a, omega, y, u=None, v=None):
        """Compute f(y), f_y(y), f_yy(y) with exact Chebyshev derivatives.

        f = c_base + alpha*c_a + xi*c_omega + (alpha²+xi²)*c_nl
        """
        model_param = next(self.parameters())
        device = model_param.device
        dtype = model_param.dtype

        a = a.to(device=device, dtype=dtype)
        omega = omega.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)
        if u is not None:
            u = u.to(device=device, dtype=dtype)
        if v is not None:
            v = v.to(device=device, dtype=dtype)

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

        _, alpha, xi = self._build_param_features(a, omega, u, v)
        coeffs = self.compute_coefficients(a, omega, u, v)

        h_vals = []
        hy_vals = []
        hyy_vals = []
        for i in range(self.n_heads):
            coeff_i = coeffs[:, i, :]
            coeff_y = cheb_derivative_coeffs(coeff_i)
            coeff_yy = cheb_derivative_coeffs(coeff_y)

            h_vals.append(clenshaw_evaluate(coeff_i, y_eval))
            hy_vals.append(clenshaw_evaluate(coeff_y, y_eval))
            hyy_vals.append(clenshaw_evaluate(coeff_yy, y_eval))

        # Taylor combine
        f = h_vals[0] if self.use_taylor else h_vals[0]
        fy = hy_vals[0] if self.use_taylor else hy_vals[0]
        fyy = hyy_vals[0] if self.use_taylor else hyy_vals[0]
        if self.use_taylor:
            alpha_b = alpha.squeeze(-1).unsqueeze(-1)
            xi_b = xi.squeeze(-1).unsqueeze(-1)
            rho2 = alpha_b ** 2 + xi_b ** 2

            f = (h_vals[0] + alpha_b * h_vals[1]
                 + xi_b * h_vals[2] + rho2 * h_vals[3])
            fy = (hy_vals[0] + alpha_b * hy_vals[1]
                  + xi_b * hy_vals[2] + rho2 * hy_vals[3])
            fyy = (hyy_vals[0] + alpha_b * hyy_vals[1]
                   + xi_b * hyy_vals[2] + rho2 * hyy_vals[3])

        return f, fy, fyy
