"""
Enhanced PINN_MLP with:
- Multi-scale Fourier features
- Optional self-attention
- Larger capacity
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from .enhanced_modules import MultiScaleFourierFeature1D, SelfAttention1D, TransformerEncoderLayer1D


def _make_activation(name: str):
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "silu" or name == "swish":
        return nn.SiLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unknown activation: {name}")


class FiLMBlock(nn.Module):
    """FiLM block with conditional modulation."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        cond_dim: int,
        activation: str = "gelu",
        use_film: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.cond_dim = int(cond_dim)
        self.use_film = bool(use_film)
        self.use_residual = bool(use_residual)

        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.act = _make_activation(activation)

        if self.use_film:
            self.gamma = nn.Linear(self.cond_dim, self.out_dim)
            self.beta = nn.Linear(self.cond_dim, self.out_dim)

        if self.use_residual and self.in_dim != self.out_dim:
            self.skip = nn.Linear(self.in_dim, self.out_dim)
        else:
            self.skip = nn.Identity() if self.use_residual else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)

        if self.use_film:
            gamma = 1.0 + 0.1 * torch.tanh(self.gamma(cond))
            beta = 0.1 * self.beta(cond)
            h = gamma * h + beta

        h = self.act(h)

        if self.use_residual:
            h = h + self.skip(x)

        return h


class PINN_MLP_Enhanced(nn.Module):
    """
    Enhanced PINN_MLP with multi-scale Fourier features and optional attention.

    Improvements over base PINN_MLP:
    1. Multi-scale Fourier features (multiple frequency scales)
    2. Optional self-attention along y-direction
    3. Larger default capacity
    4. Better parameter embedding
    """

    def __init__(
        self,
        hidden_dims: list[int] = None,
        activation: str = "gelu",
        output_activation: str = None,

        # Multi-scale Fourier features
        fourier_num_freqs: int = 8,
        fourier_base_scale: float = 1.0,
        fourier_scales: list[float] = None,

        param_embed_dim: int = 128,
        use_film: bool = True,
        use_residual: bool = True,

        # Attention mechanism
        use_attention: bool = False,
        attention_heads: int = 4,
        attention_layers: int = 1,

        # Local coordinate mode
        local_coord_mode: str = "chart_uv",

        # Raw (a, omega) normalization
        a_center_local: float = 0.5,
        a_half_range_local: float = 0.499,
        omega_min_local: float = 1.0e-4,
        omega_max_local: float = 10.0,

        # Chart (u, v) normalization
        u_center_local: float = 0.5,
        v_center_local: float = 0.5,
        u_half_range_local: float = 0.12,
        v_half_range_local: float = 0.12,

        # Physical constants
        M: float = 1.0,
        m_mode: int = 2,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 512, 512, 256]

        if fourier_scales is None:
            fourier_scales = [0.5, 1.0, 2.0, 4.0]

        self.hidden_dims = list(hidden_dims)
        self.activation_name = activation
        self.output_activation = output_activation
        self.use_film = bool(use_film)
        self.use_residual = bool(use_residual)
        self.use_attention = bool(use_attention)

        self.local_coord_mode = str(local_coord_mode)
        if self.local_coord_mode not in ("raw_aw", "chart_uv"):
            raise ValueError(f"Unsupported local_coord_mode={self.local_coord_mode}")

        # Store normalization parameters
        self.a_center_local = float(a_center_local)
        self.a_half_range_local = float(a_half_range_local)
        self.omega_min_local = float(omega_min_local)
        self.omega_max_local = float(omega_max_local)

        self.u_center_local = float(u_center_local)
        self.v_center_local = float(v_center_local)
        self.u_half_range_local = float(u_half_range_local)
        self.v_half_range_local = float(v_half_range_local)

        self.M = float(M)
        self.m_mode = int(m_mode)

        # Multi-scale Fourier features for y
        self.fourier = MultiScaleFourierFeature1D(
            num_frequencies=fourier_num_freqs,
            base_scale=fourier_base_scale,
            scales=fourier_scales,
            include_input=True,
        )
        y_feat_dim = self.fourier.out_dim

        # Parameter encoder: (a, omega) or (u, v) -> embedding
        # Input: 2D (alpha, xi) or (u_norm, v_norm)
        # Add sin/cos encoding for better periodicity
        param_input_dim = 2 + 4  # raw + sin/cos for each

        self.param_encoder = nn.Sequential(
            nn.Linear(param_input_dim, param_embed_dim),
            _make_activation(activation),
            nn.Linear(param_embed_dim, param_embed_dim),
            _make_activation(activation),
        )

        # Main network: FiLM blocks
        self.blocks = nn.ModuleList()
        prev_dim = y_feat_dim

        for hd in hidden_dims:
            self.blocks.append(
                FiLMBlock(
                    in_dim=prev_dim,
                    out_dim=hd,
                    cond_dim=param_embed_dim,
                    activation=activation,
                    use_film=use_film,
                    use_residual=use_residual,
                )
            )
            prev_dim = hd

        # Optional attention layers
        if self.use_attention:
            self.attention_layers = nn.ModuleList([
                TransformerEncoderLayer1D(
                    hidden_dim=prev_dim,
                    num_heads=attention_heads,
                    ff_dim=prev_dim * 2,
                    dropout=0.0,
                    activation=activation,
                )
                for _ in range(attention_layers)
            ])
        else:
            self.attention_layers = None

        # Output head: complex-valued
        self.head = nn.Linear(prev_dim, 2)  # real + imag

        if output_activation:
            self.out_act = _make_activation(output_activation)
        else:
            self.out_act = None

    def _normalize_params(self, a: torch.Tensor, omega: torch.Tensor):
        """Normalize (a, omega) to local coordinates."""
        if self.local_coord_mode == "raw_aw":
            # alpha in [-1, 1]
            alpha = (a - self.a_center_local) / self.a_half_range_local

            # xi: log-scale for omega
            omega_safe = torch.clamp(omega.real, min=1.0e-12)
            logw = torch.log10(omega_safe)
            logw_min = math.log10(self.omega_min_local)
            logw_max = math.log10(self.omega_max_local)
            xi = 2.0 * (logw - logw_min) / (logw_max - logw_min) - 1.0

            return alpha, xi

        else:  # chart_uv
            # Assume a, omega are already in (u, v) chart coordinates
            u = a
            v = omega.real

            u_norm = (u - self.u_center_local) / self.u_half_range_local
            v_norm = (v - self.v_center_local) / self.v_half_range_local

            return u_norm, v_norm

    def forward(
        self,
        a: torch.Tensor,
        omega: torch.Tensor,
        y: torch.Tensor,
        u: torch.Tensor = None,
        v: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            a: (B,) spin parameter
            omega: (B,) frequency (complex)
            y: (B, M) or (M,) collocation points in [-1, 1]
            u, v: optional chart coordinates (if local_coord_mode == "chart_uv")

        Returns:
            f: (B, M) complex output
        """
        # Handle input shapes
        if a.dim() == 0:
            a = a.unsqueeze(0)
        if omega.dim() == 0:
            omega = omega.unsqueeze(0)

        B = a.shape[0]

        if y.dim() == 1:
            y = y.unsqueeze(0).expand(B, -1)

        M = y.shape[1]

        # Use chart coordinates if provided
        if self.local_coord_mode == "chart_uv" and u is not None and v is not None:
            if u.dim() == 0:
                u = u.unsqueeze(0)
            if v.dim() == 0:
                v = v.unsqueeze(0)
            alpha, xi = self._normalize_params(u, v)
        else:
            alpha, xi = self._normalize_params(a, omega)

        # Parameter encoding with sin/cos
        alpha = alpha.view(B, 1)
        xi = xi.view(B, 1)

        param_feats = torch.cat([
            alpha, xi,
            torch.sin(math.pi * alpha), torch.cos(math.pi * alpha),
            torch.sin(math.pi * xi), torch.cos(math.pi * xi),
        ], dim=-1)  # (B, 6)

        param_code = self.param_encoder(param_feats)  # (B, param_embed_dim)

        # Fourier features for y
        y_feats = self.fourier(y.unsqueeze(-1))  # (B, M, y_feat_dim)

        # FiLM blocks
        h = y_feats
        for block in self.blocks:
            # Expand param_code to match y points
            param_code_expanded = param_code.unsqueeze(1).expand(B, M, -1)
            h = block(h, param_code_expanded)  # (B, M, hidden_dim)

        # Optional attention
        if self.use_attention:
            for attn_layer in self.attention_layers:
                h = attn_layer(h)  # (B, M, hidden_dim)

        # Output head
        out = self.head(h)  # (B, M, 2)

        if self.out_act:
            out = self.out_act(out)

        # Convert to complex
        f = torch.complex(out[..., 0], out[..., 1])  # (B, M)

        return f
