"""
Autoencoder variant of the PINN model.

This module defines an AutoencoderPINN network that shares the same encoder (Fourier y-encoder, param-encoder, base trunk and response trunk) as `PINN_MLP` but decouples the decoder into multiple heads:

- `rin_head`: decodes the latent representation into the standard Teukolsky function R_in (via the same ansatz as the original PINN).
- `up_head` and `down_head`: decode the latent representation into the two basis functions u↑ and u↓, which correspond to R/(r^3 e^{i ω r*}) and R/(r^{-1} e^{-i ω r*}) respectively.
- `amp_head`: predicts asymptotic amplitude corrections (log-magnitude and phase) for B^inc and B^ref solely from parameter features, reusing the same param-encoding and FiLM modulations.

During stage 1 training, only `rin_head` is trained to replicate the original PINN behaviour (R_in). The encoder weights are transferred from a pretrained PINN_MLP. Stage 2 trains `amp_head` using external teacher amplitudes (spectral for ω in [1e-4,1e-1], GSN for ω in [1e-1,10]). Stage 3 trains `up_head` and `down_head` to satisfy the second-order ODE for the rescaled functions u↑ and u↓ with appropriate boundary conditions at infinity. Stage 4 performs joint fine-tuning of all components with a combined loss on R_in and the scattering reconstruction.

This file provides only the class definitions and skeleton methods; training logic is implemented elsewhere.
"""

import torch
import torch.nn as nn
from .enhanced_modules import MultiScaleFourierFeature1D, FiLMBlock, _make_activation
from .pinn_mlp import PINN_MLP


class AutoencoderPINN(nn.Module):
    def __init__(
        self,
        hidden_dims=[128, 128, 128, 128],
        activation="silu",
        param_embed_dim=64,
        local_coord_mode="raw_aw",
        fourier_num_freqs=2,
        fourier_base_scale=1.0,
        fourier_scales=None,
        M=1.0,
        m_mode=2,
    ):
        super().__init__()
        # copy arguments
        self.hidden_dims = list(hidden_dims)
        self.param_embed_dim = param_embed_dim
        self.activation_name = activation
        self.local_coord_mode = local_coord_mode
        self.M = float(M)
        self.m_mode = int(m_mode)

        # encoder: same as in PINN_MLP
        self.y_encoder = MultiScaleFourierFeature1D(
            num_frequencies=fourier_num_freqs,
            base_scale=fourier_base_scale,
            scales=fourier_scales or [1.0],
            include_input=True,
        )
        # parameter feature dimension (see PINN_MLP._build_param_features for details)
        self.param_in_dim = 14
        # parameter encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(self.param_in_dim, param_embed_dim),
            _make_activation(activation),
            nn.Linear(param_embed_dim, param_embed_dim),
            _make_activation(activation),
        )
        # base trunk (no FiLM)
        self.base_input_proj = nn.Linear(self.y_encoder.out_dim, self.hidden_dims[0])
        self.base_input_act = _make_activation(activation)
        base_blocks = []
        prev_dim = self.hidden_dims[0]
        for out_dim in self.hidden_dims[1:]:
            base_blocks.append(
                FiLMBlock(
                    in_dim=prev_dim,
                    out_dim=out_dim,
                    cond_dim=param_embed_dim,
                    activation=activation,
                    use_film=False,
                    use_residual=True,
                )
            )
            prev_dim = out_dim
        self.base_blocks = nn.ModuleList(base_blocks)
        self.base_out_dim = prev_dim

        # response trunk (FiLM-modulated)
        self.resp_input_proj = nn.Linear(self.y_encoder.out_dim, self.hidden_dims[0])
        self.resp_input_act = _make_activation(activation)
        self.resp_input_gamma = nn.Linear(param_embed_dim, self.hidden_dims[0])
        self.resp_input_beta = nn.Linear(param_embed_dim, self.hidden_dims[0])
        resp_blocks = []
        prev_dim = self.hidden_dims[0]
        for out_dim in self.hidden_dims[1:]:
            resp_blocks.append(
                FiLMBlock(
                    in_dim=prev_dim,
                    out_dim=out_dim,
                    cond_dim=param_embed_dim,
                    activation=activation,
                    use_film=True,
                    use_residual=True,
                )
            )
            prev_dim = out_dim
        self.resp_blocks = nn.ModuleList(resp_blocks)
        self.resp_out_dim = prev_dim

        # fusion layer
        fusion_in_dim = self.base_out_dim + self.resp_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_in_dim),
            _make_activation(activation),
        )
        self.fusion_dim = fusion_in_dim

        # decoders: each outputs 2-dim real/imag
        self.rin_head = nn.Linear(self.fusion_dim, 2)
        self.up_head = nn.Linear(self.fusion_dim, 2)
        self.down_head = nn.Linear(self.fusion_dim, 2)

        # amplitude network: predicts log-magnitude and phase for B^inc and B^ref
        self.amp_head = nn.Sequential(
            nn.Linear(param_embed_dim, 64),
            _make_activation(activation),
            nn.Linear(64, 4),
        )

        # initialize weights similar to PINN_MLP
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # small gain for decoder heads; last layer of amp zero init
                if module in [self.rin_head, self.up_head, self.down_head]:
                    nn.init.xavier_normal_(module.weight, gain=0.05)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif module is self.amp_head[-1]:
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

    def compute_local_coords(self, a, omega, u=None, v=None):
        """
        Compute local coordinates (alpha, xi) from raw (a, omega) or chart (u, v).
        This uses a temporary PINN_MLP instance for convenience.
        """
        tmp = PINN_MLP(local_coord_mode=self.local_coord_mode)
        return tmp.compute_local_coords(a=a, omega=omega, u=u, v=v)

    def build_param_features(self, a, omega, u=None, v=None):
        """
        Build the 14-dimensional parameter feature vector p(a, omega).
        This uses a temporary PINN_MLP instance for convenience.
        """
        tmp = PINN_MLP(local_coord_mode=self.local_coord_mode)
        return tmp._build_param_features(a=a, omega=omega, u=u, v=v)

    def forward(self, a, omega, y, u=None, v=None):
        """
        Forward pass returns:
          f_rin_complex: (B,N) complex tensor for R_in
          f_up_complex:  (B,N) complex tensor for the up-going basis function u^\u2191
          f_down_complex:(B,N) complex tensor for the down-going basis function u^\u2193
          dB_inc: (B,) complex tensor of amplitude correction for B^inc
          dB_ref: (B,) complex tensor of amplitude correction for B^ref
        """
        # unify shapes and devices similar to PINN_MLP
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
        B, N = y.shape

        # parameter features and code
        param_feats, alpha, xi = self.build_param_features(a=a, omega=omega, u=u, v=v)
        param_code = self.param_encoder(param_feats)  # (B,C)
        # expand to match y dimension
        param_code_exp = param_code.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)

        alpha = alpha.unsqueeze(1).expand(B, N, -1).reshape(B * N, 1)
        xi = xi.unsqueeze(1).expand(B, N, -1).reshape(B * N, 1)

        # y features
        y_flat = y.reshape(-1, 1)
        y_feat = self.y_encoder(y_flat)

        # base trunk
        hb = self.base_input_proj(y_feat)
        hb = self.base_input_act(hb)
        for block in self.base_blocks:
            hb = block(hb, param_code_exp)  # cond unused when use_film=False

        # response trunk
        hr = self.resp_input_proj(y_feat)
        gamma0 = 1.0 + 0.1 * torch.tanh(self.resp_input_gamma(param_code_exp))
        beta0 = 0.1 * self.resp_input_beta(param_code_exp)
        hr = gamma0 * hr + beta0
        hr = self.resp_input_act(hr)
        for block in self.resp_blocks:
            hr = block(hr, param_code_exp)

        # fusion
        h = torch.cat([hb, hr], dim=-1)
        h = self.fusion(h)

        # decoders
        out_rin = self.rin_head(h)  # (B*N,2)
        out_up = self.up_head(h)
        out_down = self.down_head(h)

        out_rin = out_rin.reshape(B, N, 2)
        out_up = out_up.reshape(B, N, 2)
        out_down = out_down.reshape(B, N, 2)

        f_rin_complex = torch.complex(out_rin[..., 0], out_rin[..., 1])
        f_up_complex = torch.complex(out_up[..., 0], out_up[..., 1])
        f_down_complex = torch.complex(out_down[..., 0], out_down[..., 1])

        # amplitude corrections: param_code -> 4 numbers per batch
        amp_out = self.amp_head(param_code)  # (B,4)
        dB_inc_log_mag = amp_out[:, 0]
        dB_inc_phase = amp_out[:, 1]
        dB_ref_log_mag = amp_out[:, 2]
        dB_ref_phase = amp_out[:, 3]
        dB_inc = torch.exp(dB_inc_log_mag) * torch.exp(1j * dB_inc_phase)
        dB_ref = torch.exp(dB_ref_log_mag) * torch.exp(1j * dB_ref_phase)

        return f_rin_complex, f_up_complex, f_down_complex, dB_inc, dB_ref
