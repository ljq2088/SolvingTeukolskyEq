"""
Export trained PINN_MLP patch models to TorchScript for fast inference.

Usage:
    python3 scripts/export_torchscript.py \
        --checkpoint outputs/atlas_patch_train_phase1/.../best_model.pt \
        --output model_patch5.pt \
        --test-a 0.5 --test-omega 0.8 \
        --benchmark
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physical_ansatz.prefactor import Leaver_prefactors
from physical_ansatz.transform_y import compose_reduced_shape_from_f, h_factor, horizon_regularity_slope
from utils.mode import KerrMode


class RInPredictor(nn.Module):
    """
    End-to-end R_in predictor for a single patch.

    Takes (a, omega, y, u, v, lambda_val) and returns R_in on the given y grid.

    All hparams (u_c, v_c, h_u, h_v, M, m, s) are baked in at construction.
    """

    def __init__(self, pinn_mlp: nn.Module, M: float = 1.0, m_mode: int = 2, s_mode: int = -2):
        super().__init__()
        self.pinn_mlp = pinn_mlp
        self.M = float(M)
        self.m_mode = int(m_mode)
        self.s_mode = int(s_mode)

    def forward(
        self,
        a: torch.Tensor,         # scalar
        omega: torch.Tensor,     # scalar
        y: torch.Tensor,         # (N,) compactified radial coordinate
        u: torch.Tensor,         # scalar chart u
        v: torch.Tensor,         # scalar chart v
        lambda_val: torch.Tensor,  # scalar angular eigenvalue (complex)
    ) -> torch.Tensor:
        # MLP forward: f(y)
        f = self.pinn_mlp(a, omega, y, u=u, v=v)

        # Compute slope for horizon regularity
        slope = horizon_regularity_slope(
            a=a.unsqueeze(0), omega=omega.unsqueeze(0),
            lambda_=lambda_val.unsqueeze(0),
            m=self.m_mode, M=self.M, s=self.s_mode,
        )

        # S = g * (h1 * f + 1) + 1
        S = compose_reduced_shape_from_f(f=f, y=y.unsqueeze(0), slope=slope)

        # r from y: r = rp / x = rp / (0.5*(y+1))
        x = 0.5 * (y + 1.0)
        rp = self._r_plus(a)
        r = rp / x

        # Leaver prefactor
        P, _, _ = Leaver_prefactors(r=r, a=a, omega=omega, m=self.m_mode, M=self.M, s=self.s_mode)

        # Horizon normalization
        h2 = h_factor(a=a, omega=omega, m=self.m_mode, M=self.M, s=self.s_mode)

        R = P * S * h2
        return R.squeeze(0)

    @staticmethod
    def _r_plus(a: torch.Tensor, M: float = 1.0) -> torch.Tensor:
        spin_gap = torch.sqrt(torch.clamp(1.0 - (a / M) ** 2, min=1e-12))
        return M * (1.0 + spin_gap)


def load_checkpoint(
    ckpt_path: str,
    device: str = "cpu",
    u_center: float | None = None,
    v_center: float | None = None,
    h_u: float = 0.4,
    h_v: float = 0.4,
):
    """Load a PINN_MLP model from checkpoint, returning (model, meta)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]

    # Infer model params from state dict
    hidden_dim = state["base_input_proj.weight"].shape[0]
    num_blocks = sum(1 for k in state if k.startswith("base_blocks.") and k.endswith(".linear.weight"))
    hidden_dims = [hidden_dim] + [state[f"base_blocks.{i}.linear.weight"].shape[0] for i in range(num_blocks)]

    fourier_num_freqs = state["y_encoder.freqs"].shape[0] if "y_encoder.freqs" in state else 2
    param_embed_dim = state["param_encoder.0.weight"].shape[0]
    use_film = "resp_blocks.0.gamma.weight" in state
    use_residual = any("base_blocks" in k and "skip" in k for k in state)

    # Use patch_center from checkpoint if not explicitly provided
    pc = ckpt.get("patch_center", {})
    if u_center is None:
        u_center = float(pc.get("u", 0.5))
    if v_center is None:
        v_center = float(pc.get("v", 0.5))

    from model.pinn_mlp import PINN_MLP
    model = PINN_MLP(
        hidden_dims=hidden_dims,
        activation="silu",
        fourier_num_freqs=fourier_num_freqs,
        fourier_scale=1.0,
        param_embed_dim=param_embed_dim,
        use_film=use_film,
        use_residual=use_residual,
        local_coord_mode="chart_uv",
        u_center_local=u_center,
        v_center_local=v_center,
        u_half_range_local=h_u,
        v_half_range_local=h_v,
    )
    model.load_state_dict(state)
    model.eval()
    meta = {"patch_id": ckpt.get("patch_id"), "u_center": u_center, "v_center": v_center,
            "h_u": h_u, "h_v": h_v}
    return model, meta


def export_model(ckpt_path: str, output_path: str, device: str = "cpu",
                 u_center: float | None = None, v_center: float | None = None,
                 h_u: float = 0.4, h_v: float = 0.4):
    """Export model to TorchScript via tracing. Returns (predictor, meta)."""
    model, meta = load_checkpoint(ckpt_path, device, u_center, v_center, h_u, h_v)
    predictor = RInPredictor(model, M=1.0, m_mode=2, s_mode=-2)
    predictor.eval()
    predictor.to(device)

    # Trace with representative inputs
    a = torch.tensor([0.5], device=device, dtype=torch.float64)
    omega = torch.tensor([0.8], device=device, dtype=torch.float64)
    y = torch.linspace(-0.99, 0.99, 400, device=device, dtype=torch.float64)
    u = torch.tensor([0.5], device=device, dtype=torch.float64)
    v = torch.tensor([1.0], device=device, dtype=torch.float64)
    lam = torch.tensor(-0.05 + 0.0j, device=device, dtype=torch.complex128)

    traced = torch.jit.trace(predictor, (a, omega, y, u, v, lam))
    traced.save(output_path)
    print(f"Exported to {output_path}")

    # Verify
    loaded = torch.jit.load(output_path)
    with torch.no_grad():
        out1 = predictor(a, omega, y, u, v, lam)
        out2 = loaded(a, omega, y, u, v, lam)
    max_diff = (out1 - out2).abs().max().item()
    print(f"Verification max diff: {max_diff:.2e}")
    return predictor, meta


def benchmark(model, device: str = "cpu", n_warmup: int = 50, n_repeat: int = 500):
    """Benchmark inference latency."""
    a = torch.tensor([0.5], device=device, dtype=torch.float64)
    omega = torch.tensor([0.8], device=device, dtype=torch.float64)
    y = torch.linspace(-0.99, 0.99, 400, device=device, dtype=torch.float64)
    u = torch.tensor([0.5], device=device, dtype=torch.float64)
    v = torch.tensor([1.0], device=device, dtype=torch.float64)
    lam = torch.tensor(-0.05 + 0.0j, device=device, dtype=torch.complex128)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(a, omega, y, u, v, lam)

    # Benchmark
    torch.cuda.synchronize() if "cuda" in device else None
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        with torch.no_grad():
            _ = model(a, omega, y, u, v, lam)
    torch.cuda.synchronize() if "cuda" in device else None
    elapsed = time.perf_counter() - t0

    avg_us = elapsed / n_repeat * 1e6
    per_point_ns = avg_us * 1000 / 400
    print(f"Benchmark ({device}): {avg_us:.1f} us/call ({n_repeat} calls, 400 r-points)")
    print(f"  Per r-point: {per_point_ns:.1f} ns")
    print(f"  Throughput: {1e6/avg_us:.0f} calls/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="model_rin.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--u-center", type=float, default=None)
    parser.add_argument("--v-center", type=float, default=None)
    parser.add_argument("--h-u", type=float, default=0.4)
    parser.add_argument("--h-v", type=float, default=0.4)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--n-r", type=int, default=400)
    parser.add_argument("--r-min", type=float, default=2.0)
    parser.add_argument("--r-max", type=float, default=1000.0)
    args = parser.parse_args()

    predictor, meta = export_model(
        args.checkpoint, args.output, args.device,
        u_center=args.u_center, v_center=args.v_center,
        h_u=args.h_u, h_v=args.h_v,
    )
    print(f"Patch {meta['patch_id']}: u_c={meta['u_center']}, v_c={meta['v_center']}, "
          f"h_u={meta['h_u']}, h_v={meta['h_v']}")

    if args.benchmark:
        print("\n--- Original model ---")
        benchmark(predictor, args.device)

        loaded = torch.jit.load(args.output)
        print("\n--- Traced model ---")
        benchmark(loaded, args.device)


if __name__ == "__main__":
    main()
