"""
Enhanced modules for PINN_MLP:
- Multi-scale Fourier Features
- Self-Attention for y-direction
"""
import math
import torch
import torch.nn as nn


class MultiScaleFourierFeature1D(nn.Module):
    """
    Multi-scale Fourier features for 1D coordinate y.
    Combines features at different scales: [0.5, 1.0, 2.0, 4.0] * base_scale
    """
    def __init__(
        self,
        num_frequencies: int = 8,
        base_scale: float = 1.0,
        scales: list[float] = None,
        include_input: bool = True,
    ):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        self.base_scale = float(base_scale)
        self.include_input = bool(include_input)

        if scales is None:
            scales = [0.5, 1.0, 2.0, 4.0]
        self.scales = scales

        # Create frequency banks for each scale
        all_freqs = []
        for scale in scales:
            if num_frequencies > 0:
                freqs = (2.0 ** torch.arange(num_frequencies, dtype=torch.float32)) * math.pi * base_scale * scale
                all_freqs.append(freqs)

        if all_freqs:
            all_freqs = torch.cat(all_freqs)
        else:
            all_freqs = torch.empty(0, dtype=torch.float32)

        self.register_buffer("freqs", all_freqs)

        # Output dimension
        out_dim = 0
        if self.include_input:
            out_dim += 1
        out_dim += 2 * len(all_freqs)  # sin + cos for each frequency
        self.out_dim = out_dim

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (M, 1) or (B, M, 1)
        return: (M, out_dim) or (B, M, out_dim)
        """
        feats = []
        if self.include_input:
            feats.append(y)

        if len(self.freqs) > 0:
            # y: (..., 1), freqs: (K,) -> arg: (..., K)
            arg = y * self.freqs.unsqueeze(0)
            feats.append(torch.sin(arg))
            feats.append(torch.cos(arg))

        return torch.cat(feats, dim=-1)


class SelfAttention1D(nn.Module):
    """
    Self-attention along y-direction.
    Allows the model to attend to different y positions.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        assert hidden_dim % num_heads == 0, f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, M, hidden_dim) where B is batch, M is number of y points
        return: (B, M, hidden_dim)
        """
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        return x


class TransformerEncoderLayer1D(nn.Module):
    """
    Transformer encoder layer for y-direction processing.
    Includes self-attention + feedforward with residual connections.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        ff_dim: int = None,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        if ff_dim is None:
            ff_dim = 4 * hidden_dim

        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU() if activation == "gelu" else nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, M, hidden_dim)
        return: (B, M, hidden_dim)
        """
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feedforward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x
