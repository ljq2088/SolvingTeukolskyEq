"""
改进版 PINN 网络：(a, ω, y) -> f(y)

核心改动：
1. y 分支：Fourier feature，缓解谱偏置
2. (a, ω) 分支：单独参数编码，不再和 y 粗暴直接拼接
3. 隐层：FiLM 条件调制，让参数去调制每层表示
4. 残差连接：减少深层饱和和训练退化
5. 保持原 forward 接口不变，方便无缝接入现有 trainer / residual loss
"""

import math
import torch
import torch.nn as nn


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


class FourierFeature1D(nn.Module):
    """
    对 1D 坐标 y 做 Fourier feature:
        [y, sin(w_k y), cos(w_k y)]
    """
    def __init__(
        self,
        num_frequencies: int = 8,
        scale: float = 1.0,
        include_input: bool = True,
    ):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        self.scale = float(scale)
        self.include_input = bool(include_input)

        if self.num_frequencies > 0:
            # 频率取 2^k * pi * scale
            freqs = (2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)) * math.pi * self.scale
        else:
            freqs = torch.empty(0, dtype=torch.float32)

        self.register_buffer("freqs", freqs)

        out_dim = 0
        if self.include_input:
            out_dim += 1
        out_dim += 2 * self.num_frequencies
        self.out_dim = out_dim

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (M, 1)
        return: (M, out_dim)
        """
        feats = []
        if self.include_input:
            feats.append(y)

        if self.num_frequencies > 0:
            # y: (M,1), freqs: (K,) -> arg: (M,K)
            arg = y * self.freqs.unsqueeze(0)
            feats.append(torch.sin(arg))
            feats.append(torch.cos(arg))

        return torch.cat(feats, dim=-1)


class FiLMBlock(nn.Module):
    """
    一层条件调制块：
        Linear -> FiLM(gamma, beta from param code) -> Act -> Residual
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        cond_dim: int,
        activation: str = "silu",
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
        else:
            self.gamma = None
            self.beta = None

        if self.use_residual:
            if self.in_dim == self.out_dim:
                self.skip = nn.Identity()
            else:
                self.skip = nn.Linear(self.in_dim, self.out_dim)
        else:
            self.skip = None

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (M, in_dim)
        cond: (M, cond_dim)
        """
        h = self.linear(x)

        if self.use_film:
            # 初始附近接近 identity modulation
            gamma = 1.0 + 0.1 * torch.tanh(self.gamma(cond))
            beta = 0.1 * self.beta(cond)
            h = gamma * h + beta

        h = self.act(h)

        if self.use_residual:
            h = h + self.skip(x)

        return h


class PINN_MLP(nn.Module):
    """
    改进版 PINN MLP:
      - y 走 Fourier feature
      - (a, ω) 走 parameter encoder
      - hidden layers 使用 FiLM 条件调制
      - 输出仍然是复数 f(y)
    """
    def __init__(
        self,
        hidden_dims=[128, 128, 128, 128],
        activation="silu",
        output_activation=None,
        fourier_num_freqs: int = 2,
        fourier_scale: float = 1.0,
        param_embed_dim: int = 64,
        use_film: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()

        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one element.")

        self.hidden_dims = list(hidden_dims)
        self.activation_name = activation
        self.output_activation = output_activation
        self.use_film = bool(use_film)
        self.use_residual = bool(use_residual)

        # ---- y 分支：Fourier feature ----
        self.y_encoder = FourierFeature1D(
            num_frequencies=fourier_num_freqs,
            scale=fourier_scale,
            include_input=True,
        )

        # ---- 参数分支：(a, ω) -> latent code ----
        self.param_encoder = nn.Sequential(
            nn.Linear(2, param_embed_dim),
            _make_activation(activation),
            nn.Linear(param_embed_dim, param_embed_dim),
            _make_activation(activation),
        )

        # ---- 输入投影 ----
        self.input_proj = nn.Linear(self.y_encoder.out_dim, self.hidden_dims[0])
        self.input_act = _make_activation(activation)

        if self.use_film:
            self.input_gamma = nn.Linear(param_embed_dim, self.hidden_dims[0])
            self.input_beta = nn.Linear(param_embed_dim, self.hidden_dims[0])
        else:
            self.input_gamma = None
            self.input_beta = None

        # ---- 隐层 ----
        blocks = []
        prev_dim = self.hidden_dims[0]
        for out_dim in self.hidden_dims[1:]:
            blocks.append(
                FiLMBlock(
                    in_dim=prev_dim,
                    out_dim=out_dim,
                    cond_dim=param_embed_dim,
                    activation=activation,
                    use_film=self.use_film,
                    use_residual=self.use_residual,
                )
            )
            prev_dim = out_dim
        self.blocks = nn.ModuleList(blocks)

        # ---- 输出层 ----
        self.out_layer = nn.Linear(prev_dim, 2)

        if self.output_activation == "tanh":
            self.out_act = nn.Tanh()
        else:
            self.out_act = None

        self._init_weights()

    def _init_weights(self):
        """
        初始化策略：
        - 普通线性层: Xavier
        - 输出层: 更小 gain，避免初始复振幅太大
        - FiLM gamma/beta: 全 0，使初始时接近“不调制”
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if module is self.out_layer:
                    nn.init.xavier_normal_(module.weight, gain=0.05)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif "gamma" in name or "beta" in name:
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    nn.init.xavier_normal_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, a, omega, y):
        """
        Args:
            a:     (B,) 或 (B,1)
            omega: (B,) 或 (B,1)
            y:     (N,) 或 (B,N)

        Returns:
            f_complex: (B,N) 复数张量
        """
        if a.ndim == 1:
            a = a.unsqueeze(-1)          # (B,1)
        if omega.ndim == 1:
            omega = omega.unsqueeze(-1)  # (B,1)

        if y.ndim == 1:
            y = y.unsqueeze(0).expand(a.shape[0], -1)  # (B,N)

        B, N = y.shape

        # ---- 参数编码 ----
        param_in = torch.cat([a, omega], dim=-1)         # (B,2)
        param_code = self.param_encoder(param_in)        # (B,C)
        param_code = param_code.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)  # (B*N,C)

        # ---- y Fourier feature ----
        y_flat = y.reshape(-1, 1)                        # (B*N,1)
        y_feat = self.y_encoder(y_flat)                  # (B*N,F)

        # ---- 首层 ----
        h = self.input_proj(y_feat)
        if self.use_film:
            gamma0 = 1.0 + 0.1 * torch.tanh(self.input_gamma(param_code))
            beta0 = 0.1 * self.input_beta(param_code)
            h = gamma0 * h + beta0
        h = self.input_act(h)

        # ---- 隐层 ----
        for block in self.blocks:
            h = block(h, param_code)

        # ---- 输出 ----
        out = self.out_layer(h)
        if self.out_act is not None:
            out = self.out_act(out)

        out = out.reshape(B, N, 2)
        f_re = out[..., 0]
        f_im = out[..., 1]
        f_complex = torch.complex(f_re, f_im)

        return f_complex