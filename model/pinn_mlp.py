"""
PINN MLP网络：(a, ω, y) → f(y)

"""
import torch
import torch.nn as nn


class PINN_MLP(nn.Module):
    """
    简单的MLP网络，输入 (a, ω, y)，输出 f(y)

    参数范围：
        a ∈ [0.09, 0.11] (a=0.1±0.01)
        ω ∈ [0.09, 0.11] (ω=0.1±0.01)
        y ∈ [-1, 1]

    输出：f(y) 复数
    """
    def __init__(
        self,
        hidden_dims=[64, 64, 64, 64],
        activation='tanh',
        output_activation=None,
    ):
        super().__init__()

        # 输入：3维 (a, ω, y)
        # 输出：2维 (Re[f], Im[f])

        layers = []
        in_dim = 3

        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'silu':
                layers.append(nn.SiLU())
            in_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(in_dim, 2))
        if output_activation == 'tanh':
            layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

        # Xavier初始化
        self._init_weights()

    def _init_weights(self):
        linear_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]

        for i, m in enumerate(linear_layers):
            if i < len(linear_layers) - 1:
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                # 最后一层压小，避免初始输出和高阶导过大
                nn.init.xavier_normal_(m.weight, gain=0.05)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, a, omega, y):
        """
        Args:
            a: (B,) 或 (B, 1)
            omega: (B,) 或 (B, 1)
            y: (N,) 或 (B, N)

        Returns:
            f_complex: (B, N) 复数张量
        """
        # 确保维度正确
        if a.ndim == 1:
            a = a.unsqueeze(-1)  # (B, 1)
        if omega.ndim == 1:
            omega = omega.unsqueeze(-1)  # (B, 1)

        if y.ndim == 1:
            # y: (N,) -> (1, N) -> (B, N)
            y = y.unsqueeze(0).expand(a.shape[0], -1)

        B, N = y.shape

        # 扩展 a, omega 到每个 y 点
        a_expanded = a.unsqueeze(1).expand(B, N, 1)  # (B, N, 1)
        omega_expanded = omega.unsqueeze(1).expand(B, N, 1)  # (B, N, 1)
        y_expanded = y.unsqueeze(-1)  # (B, N, 1)

        # 拼接输入
        x = torch.cat([a_expanded, omega_expanded, y_expanded], dim=-1)  # (B, N, 3)

        # 前向传播
        x_flat = x.reshape(-1, 3)  # (B*N, 3)
        out_flat = self.net(x_flat)  # (B*N, 2)
        out = out_flat.reshape(B, N, 2)  # (B, N, 2)

        # 转换为复数
        f_re = out[..., 0]  # (B, N)
        f_im = out[..., 1]  # (B, N)
        f_complex = torch.complex(f_re, f_im)

        return f_complex
