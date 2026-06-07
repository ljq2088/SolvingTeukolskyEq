from __future__ import annotations

import torch
from torch import nn

from teukfield.models.amplitude_head import AmplitudeHead
from teukfield.models.encoders import PhysicsFeatureEncoder
from teukfield.models.s_field import SFieldNetwork
from teukfield.physics.angular import lambda_spheroidal
from teukfield.physics.coordinates import r_minus, r_plus
from teukfield.physics.factors import leaver_P_h2
from teukfield.physics.reduced_equation import horizon_slope_y


def build_physics_features(a: torch.Tensor, logw: torch.Tensor, lambda_: torch.Tensor, m: int = 2) -> torch.Tensor:
    omega = torch.pow(torch.tensor(10.0, dtype=a.dtype, device=a.device), logw)
    rp = r_plus(a)
    rm = r_minus(a)
    kappa = torch.sqrt(torch.clamp(1.0 - a * a, min=1.0e-30))
    omega_h = a / (rp * rp + a * a)
    k = omega - m * omega_h
    return torch.stack(
        [a, logw, omega, a * omega, rp, rm, kappa, omega_h, k, lambda_.real.to(a.dtype)],
        dim=-1,
    )


class TeukfieldAmpNet(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        fourier_bands: int = 6,
        n_windows: int = 8,
        window_overlap: float = 0.4,
        local_hidden_dim: int = 96,
        local_depth: int = 4,
        amp_hidden_dim: int = 64,
        amp_depth: int = 3,
        m: int = 2,
        s: int = -2,
    ):
        super().__init__()
        self.m = m
        self.s = s
        self.encoder = PhysicsFeatureEncoder(10, latent_dim, fourier_bands)
        self.s_field = SFieldNetwork(latent_dim, n_windows, window_overlap, local_hidden_dim, local_depth)
        self.amp_head = AmplitudeHead(latent_dim, amp_hidden_dim, amp_depth)

    def forward(self, y: torch.Tensor, a: torch.Tensor, logw: torch.Tensor, lambda_: torch.Tensor | None = None):
        omega = torch.pow(torch.tensor(10.0, dtype=a.dtype, device=a.device), logw)
        if lambda_ is None:
            lambda_ = lambda_spheroidal(a, omega, m=self.m, s=self.s)
        features = build_physics_features(a, logw, lambda_, self.m)
        latent = self.encoder(features)
        slope_y = horizon_slope_y(a, omega, lambda_, m=self.m, s=self.s)
        S = self.s_field(y, latent, slope_y)
        _, _, Ph2 = leaver_P_h2(y if y.ndim > 1 else y.unsqueeze(0).expand(a.numel(), -1), a, omega, m=self.m, s=self.s)
        amp = self.amp_head(latent)
        return {"S": S, "R": Ph2 * S, "Ph2": Ph2, "lambda": lambda_, **amp}

