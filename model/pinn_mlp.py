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
    小范围变参数版 PINN:
        f(y;a,ω) = f_base(y) + α f_a(y,p) + ξ f_omega(y,p) + (α^2+ξ^2) f_nl(y,p)

    其中当前目标物理范围:
        a ∈ [0.001, 0.999]
        ω ∈ [1e-4, 10]

    设计思想：
    1. y 仍然走 FourierFeature1D，保留对振荡结构的表达能力
    2. 参数分支不只吃 (a, ω)，而是吃局部 patch 上更合理的物理特征 p(a,ω)
    3. 主干拆成两条：
       - base_trunk: 只依赖 y，学习参数无关的基解表示
       - response_trunk: 依赖 y + 参数调制，学习参数响应表示
    4. 输出拆成四头：
       - base_head
       - a_head
       - omega_head
       - nl_head
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

        # =====================================================
        # 局部坐标模式
        #   raw_aw   : 旧模式，alpha/xi 从 raw (a, omega) 归一化得到
        #   chart_uv : 新模式，alpha/xi 从 atlas patch 的 (u, v) 局部坐标得到
        # =====================================================
        local_coord_mode: str = "raw_aw",

        # ---- 旧模式：raw (a, omega) 的局部归一化参数 ----
        a_center_local: float = 0.5,
        a_half_range_local: float = 0.499,
        omega_min_local: float = 1.0e-4,
        omega_max_local: float = 10.0,

        # ---- 新模式：chart patch 的局部中心与半宽 ----
        u_center_local: float = 0.5,
        v_center_local: float = 0.5,
        u_half_range_local: float = 0.12,
        v_half_range_local: float = 0.12,

        # ---- 物理常数 ----
        M: float = 1.0,
        m_mode: int = 2,
    ):
        super().__init__()

        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one element.")

        self.hidden_dims = list(hidden_dims)
        self.activation_name = activation
        self.output_activation = output_activation
        self.use_film = bool(use_film)
        self.use_residual = bool(use_residual)

        self.local_coord_mode = str(local_coord_mode)
        if self.local_coord_mode not in ("raw_aw", "chart_uv"):
            raise ValueError(
                f"Unsupported local_coord_mode={self.local_coord_mode}, "
                f"must be 'raw_aw' or 'chart_uv'."
            )

        # ---- 旧模式参数 ----
        self.a_center_local = float(a_center_local)
        self.a_half_range_local = float(a_half_range_local)
        self.omega_min_local = float(omega_min_local)
        self.omega_max_local = float(omega_max_local)

        # ---- 新模式参数 ----
        self.u_center_local = float(u_center_local)
        self.v_center_local = float(v_center_local)
        self.u_half_range_local = float(u_half_range_local)
        self.v_half_range_local = float(v_half_range_local)

        self.M = float(M)
        self.m_mode = int(m_mode)

        # ---- y 分支：Fourier feature ----
        self.y_encoder = FourierFeature1D(
            num_frequencies=fourier_num_freqs,
            scale=fourier_scale,
            include_input=True,
        )

        # 参数特征:
        # [alpha, xi, alpha^2, xi^2, alpha*xi, r_plus, sqrt(1-a^2/M^2), Omega_H, k, log10_omega]
        self.param_in_dim = 10

        # ---- 参数编码器 ----
        self.param_encoder = nn.Sequential(
            nn.Linear(self.param_in_dim, param_embed_dim),
            _make_activation(activation),
            nn.Linear(param_embed_dim, param_embed_dim),
            _make_activation(activation),
        )

        # =========================================================
        # 1) base trunk: 只走 y 特征，不受参数调制
        # =========================================================
        self.base_input_proj = nn.Linear(self.y_encoder.out_dim, self.hidden_dims[0])
        self.base_input_act = _make_activation(activation)

        base_blocks = []
        prev_dim = self.hidden_dims[0]
        for out_dim in self.hidden_dims[1:]:
            # base_trunk 不用 FiLM，只保留 residual
            base_blocks.append(
                FiLMBlock(
                    in_dim=prev_dim,
                    out_dim=out_dim,
                    cond_dim=param_embed_dim,   # 占位即可，不会用到
                    activation=activation,
                    use_film=False,
                    use_residual=self.use_residual,
                )
            )
            prev_dim = out_dim
        self.base_blocks = nn.ModuleList(base_blocks)
        self.base_out_dim = prev_dim

        # =========================================================
        # 2) response trunk: y + 参数 FiLM 调制
        # =========================================================
        self.resp_input_proj = nn.Linear(self.y_encoder.out_dim, self.hidden_dims[0])
        self.resp_input_act = _make_activation(activation)

        if self.use_film:
            self.resp_input_gamma = nn.Linear(param_embed_dim, self.hidden_dims[0])
            self.resp_input_beta = nn.Linear(param_embed_dim, self.hidden_dims[0])
        else:
            self.resp_input_gamma = None
            self.resp_input_beta = None

        resp_blocks = []
        prev_dim = self.hidden_dims[0]
        for out_dim in self.hidden_dims[1:]:
            resp_blocks.append(
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
        self.resp_blocks = nn.ModuleList(resp_blocks)
        self.resp_out_dim = prev_dim

        # =========================================================
        # 3) 融合层：concat(base, resp) -> fusion
        # =========================================================
        fusion_in_dim = self.base_out_dim + self.resp_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_in_dim),
            _make_activation(activation),
        )
        self.fusion_dim = fusion_in_dim

        # =========================================================
        # 4) 四个输出头
        # =========================================================
        self.base_head = nn.Linear(self.fusion_dim, 2)
        self.a_head = nn.Linear(self.fusion_dim, 2)
        self.omega_head = nn.Linear(self.fusion_dim, 2)
        self.nl_head = nn.Linear(self.fusion_dim, 2)

        if self.output_activation == "tanh":
            self.out_act = nn.Tanh()
        else:
            self.out_act = None

        self._init_weights()

    # ---------------------------------------------------------
    # 参数特征
    # ---------------------------------------------------------
    def _normalize_params_raw_aw(self, a: torch.Tensor, omega: torch.Tensor):
        """
        旧模式：
            alpha 从 a 的局部线性归一化得到
            xi    从 omega 的对数域归一化得到

        当前默认目标范围:
            a ∈ [0.001, 0.999]
            ω ∈ [1e-4, 10]
        """
        alpha = (a - self.a_center_local) / self.a_half_range_local

        omega_safe = torch.clamp(omega, min=1.0e-12)
        logw = torch.log10(omega_safe)
        logw_min = math.log10(self.omega_min_local)
        logw_max = math.log10(self.omega_max_local)
        xi = 2.0 * (logw - logw_min) / (logw_max - logw_min) - 1.0

        return alpha, xi


    def _normalize_params_chart_uv(self, u: torch.Tensor, v: torch.Tensor):
        """
        新模式：
            alpha = (u - u_c) / h_u
            xi    = (v - v_c) / h_v
        """
        alpha = (u - self.u_center_local) / self.u_half_range_local
        xi = (v - self.v_center_local) / self.v_half_range_local
        return alpha, xi


    def compute_local_coords(
        self,
        a: torch.Tensor,
        omega: torch.Tensor,
        u: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
    ):
        """
        统一入口：
        - raw_aw   -> 从 raw (a, omega) 算 alpha, xi
        - chart_uv -> 从 (u, v) 算 alpha, xi
        """
        if self.local_coord_mode == "raw_aw":
            return self._normalize_params_raw_aw(a, omega)

        if u is None or v is None:
            raise ValueError(
                "local_coord_mode='chart_uv' requires u and v to be provided."
            )

        return self._normalize_params_chart_uv(u, v)

    def _build_param_features(
        self,
        a: torch.Tensor,
        omega: torch.Tensor,
        u: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
    ):
        """
        p(a,ω) = [α, ξ, α², ξ², α ξ, r_+, sqrt(1-a²/M²), Ω_H, k, log10(ω)]

        说明:
        - local_coord_mode='chart_uv' 时，alpha/xi 来自 patch 的 (u, v)
        - 物理特征中的 omega 仍使用新的 log10_omega，以统一宽频率范围拟合
        """
        alpha, xi = self.compute_local_coords(a=a, omega=omega, u=u, v=v)

        M = self.M
        m_mode = self.m_mode

        # 为避免极端情况下 sqrt 负数，做一次 clamp
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

    # ---------------------------------------------------------
    # 初始化
    # ---------------------------------------------------------
    def _init_weights(self):
        """
        初始化策略：
        1. 普通线性层 Xavier
        2. 四个头：
           - base_head 小幅 Xavier
           - a_head / omega_head / nl_head 零初始化
        3. 所有 FiLM gamma/beta 全零，保证初始时接近 identity modulation
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if module in [self.base_head, self.a_head, self.omega_head, self.nl_head]:
                    if module is self.base_head:
                        nn.init.xavier_normal_(module.weight, gain=0.05)
                    else:
                        nn.init.zeros_(module.weight)
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

    # ---------------------------------------------------------
    # forward
    # ---------------------------------------------------------
    def forward(self, a, omega, y, u=None, v=None):
        """
        Args:
            a:     (B,) or (B,1)
            omega: (B,) or (B,1)
            y:     (N,) or (B,N)

        Returns:
            f_complex: (B,N) complex tensor
        """
        # ---- 统一 device / dtype 到模型参数 ----
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
        B, N = y.shape

        # ---- 参数特征 ----
        param_feats, alpha, xi = self._build_param_features(
            a=a,
            omega=omega,
            u=u,
            v=v,
        )
        param_code = self.param_encoder(param_feats)                   # (B,C)
        param_code = (
            param_code.unsqueeze(1)
            .expand(B, N, -1)
            .reshape(B * N, -1)
        )

        alpha = alpha.unsqueeze(1).expand(B, N, -1).reshape(B * N, 1)
        xi = xi.unsqueeze(1).expand(B, N, -1).reshape(B * N, 1)
        rho2 = alpha ** 2 + xi ** 2

        # ---- y 特征 ----
        y_flat = y.reshape(-1, 1)
        y_feat = self.y_encoder(y_flat)

        # =====================================================
        # base trunk
        # =====================================================
        hb = self.base_input_proj(y_feat)
        hb = self.base_input_act(hb)
        for block in self.base_blocks:
            hb = block(hb, param_code)   # use_film=False，所以 cond 不会被实际使用

        # =====================================================
        # response trunk
        # =====================================================
        hr = self.resp_input_proj(y_feat)
        if self.use_film:
            gamma0 = 1.0 + 0.1 * torch.tanh(self.resp_input_gamma(param_code))
            beta0 = 0.1 * self.resp_input_beta(param_code)
            hr = gamma0 * hr + beta0
        hr = self.resp_input_act(hr)

        for block in self.resp_blocks:
            hr = block(hr, param_code)

        # =====================================================
        # 融合
        # =====================================================
        h = torch.cat([hb, hr], dim=-1)
        h = self.fusion(h)

        # =====================================================
        # 四头输出
        # =====================================================
        out_base = self.base_head(h)
        out_a = self.a_head(h)
        out_omega = self.omega_head(h)
        out_nl = self.nl_head(h)

        if self.out_act is not None:
            out_base = self.out_act(out_base)
            out_a = self.out_act(out_a)
            out_omega = self.out_act(out_omega)
            out_nl = self.out_act(out_nl)

        out = out_base + alpha * out_a + xi * out_omega + rho2 * out_nl

        out = out.reshape(B, N, 2)
        f_re = out[..., 0]
        f_im = out[..., 1]
        f_complex = torch.complex(f_re, f_im)

        return f_complex
