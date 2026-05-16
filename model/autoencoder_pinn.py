"""
Autoencoder-style PINN: shared physics encoder + multiple physics decoders.

Architecture:
    Inputs: (a, omega, y, u, v)
    SharedPINNEncoder -> shared latent h + (alpha, xi, rho2)
        |
        +-- RinDecoder    -> f_R(y)  (Stage 1: original incoming solution)
        +-- UpDecoder     -> f_up(y) (Stage 3: outgoing infinity basis)
        +-- DownDecoder   -> f_dn(y) (Stage 3: incoming infinity basis)
    AmplitudeNet(p) -> B_inc, B_ref    (Stage 2: parameter-only)

Old PINN_MLP.forward() equivalence:
    h, alpha, xi, rho2 = encoder.forward_features(...)
    f_R = rin_decoder(h, alpha, xi, rho2)   # matches old output exactly
"""

import math
import torch
import torch.nn as nn

from .enhanced_modules import MultiScaleFourierFeature1D
from .pinn_mlp import FiLMBlock, _make_activation


# ============================================================
# ModulatedResidualBlock — ported from mature PINN_MLP
# ============================================================
class ModulatedResidualBlock(nn.Module):
    """Gated FiLM residual block for parameter-conditioned tokens."""

    def __init__(self, hidden_dim: int, cond_dim: int, activation: str = "silu"):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta = nn.Linear(cond_dim, hidden_dim)
        self.gate = nn.Linear(cond_dim, hidden_dim)
        self.act = _make_activation(activation)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z = self.norm(h)
        gamma = 1.0 + 0.1 * torch.tanh(self.gamma(cond))
        beta = 0.1 * self.beta(cond)
        gate = torch.sigmoid(self.gate(cond))
        z = gamma * z + beta
        z = self.linear2(self.act(self.linear1(z)))
        return h + gate * z


# ============================================================
# AmplitudeNet — absolute amplitude surrogate (parameter-only)
# ============================================================
class AmplitudeNet(nn.Module):
    """
    Absolute amplitude surrogate:
        p(a, omega, u, v) -> B_inc, B_ref

    Output: B = exp(rho) * exp(i phi)
    Does NOT depend on y. Has its own param_encoder.
    """

    def __init__(
        self,
        param_in_dim: int,
        hidden_dim: int = 128,
        n_blocks: int = 3,
        activation: str = "silu",
    ):
        super().__init__()
        self.param_encoder = nn.Sequential(
            nn.Linear(param_in_dim, hidden_dim),
            _make_activation(activation),
            nn.Linear(hidden_dim, hidden_dim),
            _make_activation(activation),
        )
        self.inc_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.ref_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.inc_blocks = nn.ModuleList([
            ModulatedResidualBlock(hidden_dim, hidden_dim, activation=activation)
            for _ in range(n_blocks)
        ])
        self.ref_blocks = nn.ModuleList([
            ModulatedResidualBlock(hidden_dim, hidden_dim, activation=activation)
            for _ in range(n_blocks)
        ])
        self.out_inc = nn.Linear(hidden_dim, 2)
        self.out_ref = nn.Linear(hidden_dim, 2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.inc_token, mean=0.0, std=1.0e-3)
        nn.init.normal_(self.ref_token, mean=0.0, std=1.0e-3)

    def forward(self, param_feats: torch.Tensor):
        """
        Args:
            param_feats: (B, param_in_dim)
        Returns:
            Binc: (B,) complex
            Bref: (B,) complex
            raw: dict with rho_inc, phi_inc, rho_ref, phi_ref
        """
        B = param_feats.shape[0]
        cond = self.param_encoder(param_feats)
        h_inc = self.inc_token.expand(B, -1)
        h_ref = self.ref_token.expand(B, -1)
        for block in self.inc_blocks:
            h_inc = block(h_inc, cond)
        for block in self.ref_blocks:
            h_ref = block(h_ref, cond)

        out_inc = self.out_inc(h_inc)
        out_ref = self.out_ref(h_ref)
        rho_inc, phi_inc = out_inc[:, 0], out_inc[:, 1]
        rho_ref, phi_ref = out_ref[:, 0], out_ref[:, 1]

        cdtype = torch.complex128 if param_feats.dtype == torch.float64 else torch.complex64
        Binc = (torch.exp(rho_inc) * torch.exp(1j * phi_inc)).to(cdtype)
        Bref = (torch.exp(rho_ref) * torch.exp(1j * phi_ref)).to(cdtype)

        raw = {"rho_inc": rho_inc, "phi_inc": phi_inc, "rho_ref": rho_ref, "phi_ref": phi_ref}
        return Binc, Bref, raw


# ============================================================
# SharedPINNEncoder
# ============================================================
class SharedPINNEncoder(nn.Module):
    """
    Shared encoder: everything before the output heads of the old PINN_MLP.

    Contains:
        - y Fourier feature encoder
        - parameter feature builder (pure math, no trained params)
        - param_encoder (MLP)
        - base trunk (y only)
        - response trunk (y + FiLM from param_code)
        - fusion (concat base+resp -> h)

    forward_features() returns:
        h:       (B*N, fusion_dim)  — shared latent
        alpha:   (B*N, 1)           — normalised param offset
        xi:      (B*N, 1)
        rho2:    (B*N, 1)           — alpha^2 + xi^2
        param_feats: (B, param_in_dim) — raw param features (for AmplitudeNet)
    """

    def __init__(
        self,
        hidden_dims: list = None,
        activation: str = "silu",
        param_embed_dim: int = 64,
        fourier_num_freqs: int = 2,
        fourier_base_scale: float = 1.0,
        fourier_scales: list = None,
        use_film: bool = True,
        use_residual: bool = True,
        local_coord_mode: str = "raw_aw",
        a_center_local: float = 0.5,
        a_half_range_local: float = 0.499,
        omega_min_local: float = 1.0e-4,
        omega_max_local: float = 10.0,
        u_center_local: float = 0.5,
        v_center_local: float = 0.5,
        u_half_range_local: float = 0.12,
        v_half_range_local: float = 0.12,
        M: float = 1.0,
        m_mode: int = 2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 128, 128]
        if fourier_scales is None:
            fourier_scales = [1.0]

        self.hidden_dims = list(hidden_dims)
        self.use_film = bool(use_film)
        self.use_residual = bool(use_residual)
        self.local_coord_mode = str(local_coord_mode)

        # ---- local coordinate config ----
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

        # ---- y Fourier encoder ----
        self.y_encoder = MultiScaleFourierFeature1D(
            num_frequencies=fourier_num_freqs,
            base_scale=fourier_base_scale,
            scales=list(fourier_scales),
            include_input=True,
        )

        # ---- param_encoder ----
        self.param_in_dim = 14
        self.param_encoder = nn.Sequential(
            nn.Linear(self.param_in_dim, param_embed_dim),
            _make_activation(activation),
            nn.Linear(param_embed_dim, param_embed_dim),
            _make_activation(activation),
        )

        # ---- base trunk (no FiLM) ----
        self.base_input_proj = nn.Linear(self.y_encoder.out_dim, hidden_dims[0])
        self.base_input_act = _make_activation(activation)
        base_blocks = []
        prev_dim = hidden_dims[0]
        for out_dim in hidden_dims[1:]:
            base_blocks.append(FiLMBlock(
                in_dim=prev_dim, out_dim=out_dim, cond_dim=param_embed_dim,
                activation=activation, use_film=False, use_residual=self.use_residual,
            ))
            prev_dim = out_dim
        self.base_blocks = nn.ModuleList(base_blocks)
        self.base_out_dim = prev_dim

        # ---- response trunk (with FiLM) ----
        self.resp_input_proj = nn.Linear(self.y_encoder.out_dim, hidden_dims[0])
        self.resp_input_act = _make_activation(activation)
        if self.use_film:
            self.resp_input_gamma = nn.Linear(param_embed_dim, hidden_dims[0])
            self.resp_input_beta = nn.Linear(param_embed_dim, hidden_dims[0])
        else:
            self.resp_input_gamma = None
            self.resp_input_beta = None
        resp_blocks = []
        prev_dim = hidden_dims[0]
        for out_dim in hidden_dims[1:]:
            resp_blocks.append(FiLMBlock(
                in_dim=prev_dim, out_dim=out_dim, cond_dim=param_embed_dim,
                activation=activation, use_film=self.use_film, use_residual=self.use_residual,
            ))
            prev_dim = out_dim
        self.resp_blocks = nn.ModuleList(resp_blocks)
        self.resp_out_dim = prev_dim

        # ---- fusion ----
        fusion_in_dim = self.base_out_dim + self.resp_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_in_dim),
            _make_activation(activation),
        )
        self.fusion_dim = fusion_in_dim

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "gamma" in name or "beta" in name:
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    nn.init.xavier_normal_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    # ---- local coordinate helpers (pure math, no trained params) ----
    def _normalize_params_raw_aw(self, a, omega):
        alpha = (a - self.a_center_local) / self.a_half_range_local
        omega_safe = torch.clamp(omega, min=1.0e-12)
        logw = torch.log10(omega_safe)
        logw_min = math.log10(self.omega_min_local)
        logw_max = math.log10(self.omega_max_local)
        xi = 2.0 * (logw - logw_min) / (logw_max - logw_min) - 1.0
        return alpha, xi

    def _normalize_params_chart_uv(self, u, v):
        alpha = (u - self.u_center_local) / self.u_half_range_local
        xi = (v - self.v_center_local) / self.v_half_range_local
        return alpha, xi

    def compute_local_coords(self, a, omega, u=None, v=None):
        if self.local_coord_mode == "raw_aw":
            return self._normalize_params_raw_aw(a, omega)
        if u is None or v is None:
            raise ValueError("local_coord_mode='chart_uv' requires u and v.")
        return self._normalize_params_chart_uv(u, v)

    def _build_param_features(self, a, omega, u=None, v=None):
        """
        p = [alpha, xi, alpha^2, xi^2, alpha*xi, r_+, sqrt(1-a^2/M^2),
             Omega_H, k, log10(omega),
             sin(pi*alpha), cos(pi*alpha), sin(pi*xi), cos(pi*xi)]
        """
        alpha, xi = self.compute_local_coords(a=a, omega=omega, u=u, v=v)
        M = self.M
        spin_gap = torch.sqrt(torch.clamp(1.0 - (a / M) ** 2, min=1.0e-12))
        r_plus = M * (1.0 + spin_gap)
        Omega_H = a / (2.0 * M * r_plus)
        k = omega - self.m_mode * Omega_H
        omega_safe = torch.clamp(omega, min=1.0e-12)
        log10_omega = torch.log10(omega_safe)
        feats = torch.cat([
            alpha, xi, alpha ** 2, xi ** 2, alpha * xi,
            r_plus, spin_gap, Omega_H, k, log10_omega,
            torch.sin(math.pi * alpha), torch.cos(math.pi * alpha),
            torch.sin(math.pi * xi), torch.cos(math.pi * xi),
        ], dim=-1)
        return feats, alpha, xi

    # ---- main forward ----
    def forward_features(self, a, omega, y, u=None, v=None):
        """
        Returns:
            h:          (B*N, fusion_dim)
            alpha:      (B*N, 1)
            xi:         (B*N, 1)
            rho2:       (B*N, 1)
            param_feats: (B, param_in_dim)
        """
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

        # param features
        param_feats, alpha, xi = self._build_param_features(a=a, omega=omega, u=u, v=v)
        param_code = self.param_encoder(param_feats)
        param_code_exp = param_code.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)

        alpha = alpha.unsqueeze(1).expand(B, N, -1).reshape(B * N, 1)
        xi = xi.unsqueeze(1).expand(B, N, -1).reshape(B * N, 1)
        rho2 = alpha ** 2 + xi ** 2

        # y features
        y_flat = y.reshape(-1, 1)
        y_feat = self.y_encoder(y_flat)

        # base trunk
        hb = self.base_input_proj(y_feat)
        hb = self.base_input_act(hb)
        for block in self.base_blocks:
            hb = block(hb, param_code_exp)

        # response trunk
        hr = self.resp_input_proj(y_feat)
        if self.use_film:
            gamma0 = 1.0 + 0.1 * torch.tanh(self.resp_input_gamma(param_code_exp))
            beta0 = 0.1 * self.resp_input_beta(param_code_exp)
            hr = gamma0 * hr + beta0
        hr = self.resp_input_act(hr)
        for block in self.resp_blocks:
            hr = block(hr, param_code_exp)

        # fusion
        h = torch.cat([hb, hr], dim=-1)
        h = self.fusion(h)

        return h, alpha, xi, rho2, param_feats


# ============================================================
# TaylorComplexDecoder — reproduces old 4-head output structure
# ============================================================
class TaylorComplexDecoder(nn.Module):
    """
    Four-head Taylor expansion decoder:
        f = f_base + alpha*f_a + xi*f_omega + rho2*f_nl
    where each head outputs (real, imag) -> complex.

    Optionally includes a shared hidden trunk before the heads for
    increased expressivity.

    Args:
        fusion_dim: input latent dimension from encoder
        hidden_dim: width of hidden layers (default 128)
        n_hidden: number of hidden layers before heads (0 = original linear heads)
    """

    def __init__(self, fusion_dim: int, hidden_dim: int = 128, n_hidden: int = 0):
        super().__init__()
        self.n_hidden = n_hidden
        if n_hidden > 0:
            layers = []
            in_dim = fusion_dim
            for i in range(n_hidden):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.SiLU())
                in_dim = hidden_dim
            self.trunk = nn.Sequential(*layers)
            head_in = hidden_dim
        else:
            self.trunk = nn.Identity()
            head_in = fusion_dim
        self.base_head = nn.Linear(head_in, 2)
        self.a_head = nn.Linear(head_in, 2)
        self.omega_head = nn.Linear(head_in, 2)
        self.nl_head = nn.Linear(head_in, 2)
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if module is self.base_head:
                    nn.init.xavier_normal_(module.weight, gain=0.05)
                elif any(module is h for h in [self.a_head, self.omega_head, self.nl_head]):
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, h, alpha, xi, rho2):
        """
        Args:
            h:     (B*N, fusion_dim)
            alpha: (B*N, 1)
            xi:    (B*N, 1)
            rho2:  (B*N, 1)
        Returns:
            f_complex: (B*N,) complex
        """
        h = self.trunk(h)
        out = (self.base_head(h)
               + alpha * self.a_head(h)
               + xi * self.omega_head(h)
               + rho2 * self.nl_head(h))
        return torch.complex(out[..., 0], out[..., 1])


# ============================================================
# AutoencoderTeukolskyPINN — top-level wrapper
# ============================================================
class AutoencoderTeukolskyPINN(nn.Module):
    """
    Autoencoder-style Teukolsky PINN.

    Components:
        encoder:        SharedPINNEncoder
        rin_decoder:    TaylorComplexDecoder  (Stage 1)
        up_decoder:     TaylorComplexDecoder  (Stage 3)
        down_decoder:   TaylorComplexDecoder  (Stage 3)
        amplitude_net:  AmplitudeNet          (Stage 2)

    Backward compat: forward(a, omega, y) returns f_R(y) same as old PINN_MLP.
    """

    def __init__(
        self,
        hidden_dims: list = None,
        activation: str = "silu",
        param_embed_dim: int = 64,
        fourier_num_freqs: int = 2,
        fourier_base_scale: float = 1.0,
        fourier_scales: list = None,
        use_film: bool = True,
        use_residual: bool = True,
        local_coord_mode: str = "raw_aw",
        a_center_local: float = 0.5,
        a_half_range_local: float = 0.499,
        omega_min_local: float = 1.0e-4,
        omega_max_local: float = 10.0,
        u_center_local: float = 0.5,
        v_center_local: float = 0.5,
        u_half_range_local: float = 0.12,
        v_half_range_local: float = 0.12,
        M: float = 1.0,
        m_mode: int = 2,
        amp_hidden_dim: int = 128,
        amp_n_blocks: int = 3,
        decoder_hidden_dim: int = 128,
        decoder_n_hidden: int = 0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 128, 128]

        encoder_kwargs = dict(
            hidden_dims=hidden_dims, activation=activation,
            param_embed_dim=param_embed_dim,
            fourier_num_freqs=fourier_num_freqs,
            fourier_base_scale=fourier_base_scale,
            fourier_scales=fourier_scales,
            use_film=use_film, use_residual=use_residual,
            local_coord_mode=local_coord_mode,
            a_center_local=a_center_local, a_half_range_local=a_half_range_local,
            omega_min_local=omega_min_local, omega_max_local=omega_max_local,
            u_center_local=u_center_local, v_center_local=v_center_local,
            u_half_range_local=u_half_range_local, v_half_range_local=v_half_range_local,
            M=M, m_mode=m_mode,
        )
        self.encoder = SharedPINNEncoder(**encoder_kwargs)
        self.rin_decoder = TaylorComplexDecoder(self.encoder.fusion_dim, decoder_hidden_dim, decoder_n_hidden)
        self.up_decoder = TaylorComplexDecoder(self.encoder.fusion_dim, decoder_hidden_dim, decoder_n_hidden)
        self.down_decoder = TaylorComplexDecoder(self.encoder.fusion_dim, decoder_hidden_dim, decoder_n_hidden)
        self.amplitude_net = AmplitudeNet(
            param_in_dim=self.encoder.param_in_dim,
            hidden_dim=amp_hidden_dim,
            n_blocks=amp_n_blocks,
            activation=activation,
        )

    def _to_model_dtype(self, a, omega, y=None, u=None, v=None):
        """Cast inputs to model device/dtype."""
        p = next(self.parameters())
        dev, dt = p.device, p.dtype
        a = a.to(device=dev, dtype=dt)
        omega = omega.to(device=dev, dtype=dt)
        out = [a, omega]
        for t in [y, u, v]:
            out.append(t.to(device=dev, dtype=dt) if t is not None else None)
        return tuple(out)

    def _predict_decoder(self, decoder, a, omega, y, u=None, v=None):
        """Run encoder + one decoder, return (B, N) complex."""
        a, omega, y, u, v = self._to_model_dtype(a, omega, y, u, v)
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
        h, alpha, xi, rho2, _ = self.encoder.forward_features(a, omega, y, u, v)
        f_flat = decoder(h, alpha, xi, rho2)
        return f_flat.reshape(B, N)

    # ---- public predict methods (all return (B,N) complex) ----
    def predict_Rin(self, a, omega, y, u=None, v=None):
        """Return f_R(y) complex, shape (B,N). Same as old PINN_MLP.forward."""
        return self._predict_decoder(self.rin_decoder, a, omega, y, u, v)

    def predict_u_up(self, a, omega, y, u=None, v=None):
        """Return f_up(y) complex, shape (B,N). Free function inside u_up ansatz."""
        return self._predict_decoder(self.up_decoder, a, omega, y, u, v)

    def predict_u_down(self, a, omega, y, u=None, v=None):
        """Return f_down(y) complex, shape (B,N). Free function inside u_down ansatz."""
        return self._predict_decoder(self.down_decoder, a, omega, y, u, v)

    def predict_amplitudes(self, a, omega, u=None, v=None):
        """Return (Binc, Bref, raw) — pure network, no spectral dependency."""
        a, omega, _, u, v = self._to_model_dtype(a, omega, None, u, v)
        if a.ndim == 1:
            a = a.unsqueeze(-1)
        if omega.ndim == 1:
            omega = omega.unsqueeze(-1)
        if u is not None and u.ndim == 1:
            u = u.unsqueeze(-1)
        if v is not None and v.ndim == 1:
            v = v.unsqueeze(-1)
        param_feats, _, _ = self.encoder._build_param_features(a=a, omega=omega, u=u, v=v)
        return self.amplitude_net(param_feats)

    def predict_asymptotic_amplitudes(self, a, omega, u=None, v=None):
        """Alias for predict_amplitudes, matching old API."""
        return self.predict_amplitudes(a, omega, u, v)

    # ---- backward-compat forward ----
    def forward(self, a, omega, y, u=None, v=None):
        """
        Stage-1 compatible forward: returns complex f_R(y).
        Shape: (B, N) complex.  Equivalent to old PINN_MLP.forward.
        """
        return self._predict_decoder(self.rin_decoder, a, omega, y, u, v)


# ============================================================
# Weight migration helper — copy old PINN_MLP weights -> AutoencoderTeukolskyPINN
# ============================================================
def copy_pinn_mlp_to_autoencoder(old_model, ae_model):
    """
    Copy weights from a trained PINN_MLP into AutoencoderTeukolskyPINN.

    Maps:
        old y_encoder  -> ae.encoder.y_encoder
        old param_encoder -> ae.encoder.param_encoder
        old base_input_proj / base_input_act / base_blocks -> ae.encoder.*
        old resp_input_proj / resp_input_act / resp_input_gamma / resp_input_beta
            / resp_blocks -> ae.encoder.*
        old fusion -> ae.encoder.fusion
        old base_head -> ae.rin_decoder.base_head
        old a_head    -> ae.rin_decoder.a_head
        old omega_head -> ae.rin_decoder.omega_head
        old nl_head   -> ae.rin_decoder.nl_head
    """
    _copy_state(old_model.y_encoder, ae_model.encoder.y_encoder)
    _copy_state(old_model.param_encoder, ae_model.encoder.param_encoder)
    _copy_state(old_model.base_input_proj, ae_model.encoder.base_input_proj)
    _copy_state(old_model.base_input_act, ae_model.encoder.base_input_act)
    for old_block, new_block in zip(old_model.base_blocks, ae_model.encoder.base_blocks):
        _copy_state(old_block, new_block)
    _copy_state(old_model.resp_input_proj, ae_model.encoder.resp_input_proj)
    _copy_state(old_model.resp_input_act, ae_model.encoder.resp_input_act)
    if old_model.resp_input_gamma is not None and ae_model.encoder.resp_input_gamma is not None:
        _copy_state(old_model.resp_input_gamma, ae_model.encoder.resp_input_gamma)
        _copy_state(old_model.resp_input_beta, ae_model.encoder.resp_input_beta)
    for old_block, new_block in zip(old_model.resp_blocks, ae_model.encoder.resp_blocks):
        _copy_state(old_block, new_block)
    _copy_state(old_model.fusion, ae_model.encoder.fusion)
    _copy_state(old_model.base_head, ae_model.rin_decoder.base_head)
    _copy_state(old_model.a_head, ae_model.rin_decoder.a_head)
    _copy_state(old_model.omega_head, ae_model.rin_decoder.omega_head)
    _copy_state(old_model.nl_head, ae_model.rin_decoder.nl_head)


def _copy_state(src, dst):
    """Copy state dict from src module to dst module (strict)."""
    dst.load_state_dict(src.state_dict())
