from __future__ import annotations
import sys

sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
import torch

from model.pinn_mlp import PINN_MLP


def main():
    device = "cpu"
    dtype = torch.float64

    # ---------------------------------------------------------
    # 1) 旧模式检查：raw_aw
    # ---------------------------------------------------------
    model_raw = PINN_MLP(
        hidden_dims=[64, 64],
        activation="silu",
        fourier_num_freqs=2,
        fourier_scale=1.0,
        param_embed_dim=32,
        use_film=True,
        use_residual=True,
        local_coord_mode="raw_aw",
        a_center_local=0.5,
        a_half_range_local=0.499,
        omega_min_local=1.0e-4,
        omega_max_local=10.0,
        M=1.0,
        m_mode=2,
    ).to(device=device, dtype=dtype)

    a = torch.tensor([[0.5]], dtype=dtype)
    omega = torch.tensor([[1.0]], dtype=dtype)

    alpha_raw, xi_raw = model_raw.compute_local_coords(a=a, omega=omega)
    print("=" * 80)
    print("[raw_aw mode]")
    print(f"alpha_raw = {alpha_raw.item():.6f}")
    print(f"xi_raw    = {xi_raw.item():.6f}")

    # ---------------------------------------------------------
    # 2) 新模式检查：chart_uv
    # ---------------------------------------------------------
    model_chart = PINN_MLP(
        hidden_dims=[64, 64],
        activation="silu",
        fourier_num_freqs=2,
        fourier_scale=1.0,
        param_embed_dim=32,
        use_film=True,
        use_residual=True,
        local_coord_mode="chart_uv",
        u_center_local=0.40,
        v_center_local=0.65,
        u_half_range_local=0.12,
        v_half_range_local=0.10,
        M=1.0,
        m_mode=2,
    ).to(device=device, dtype=dtype)

    a_dummy = torch.tensor([[0.7]], dtype=dtype)
    omega_dummy = torch.tensor([[0.2]], dtype=dtype)

    # 中心点 -> (0,0)
    u0 = torch.tensor([[0.40]], dtype=dtype)
    v0 = torch.tensor([[0.65]], dtype=dtype)
    alpha0, xi0 = model_chart.compute_local_coords(
        a=a_dummy, omega=omega_dummy, u=u0, v=v0
    )

    # 四条边界
    uL = torch.tensor([[0.40 - 0.12]], dtype=dtype)
    uR = torch.tensor([[0.40 + 0.12]], dtype=dtype)
    vD = torch.tensor([[0.65 - 0.10]], dtype=dtype)
    vU = torch.tensor([[0.65 + 0.10]], dtype=dtype)

    alphaL, _ = model_chart.compute_local_coords(
        a=a_dummy, omega=omega_dummy, u=uL, v=v0
    )
    alphaR, _ = model_chart.compute_local_coords(
        a=a_dummy, omega=omega_dummy, u=uR, v=v0
    )
    _, xiD = model_chart.compute_local_coords(
        a=a_dummy, omega=omega_dummy, u=u0, v=vD
    )
    _, xiU = model_chart.compute_local_coords(
        a=a_dummy, omega=omega_dummy, u=u0, v=vU
    )

    print("=" * 80)
    print("[chart_uv mode]")
    print(f"center: alpha={alpha0.item():.6f}, xi={xi0.item():.6f}")
    print(f"left  : alpha={alphaL.item():.6f}")
    print(f"right : alpha={alphaR.item():.6f}")
    print(f"down  : xi   ={xiD.item():.6f}")
    print(f"up    : xi   ={xiU.item():.6f}")

    # ---------------------------------------------------------
    # 3) forward shape 检查
    # ---------------------------------------------------------
    y = torch.linspace(-1.0, 1.0, 17, dtype=dtype)
    out = model_chart(
        a=a_dummy.squeeze(-1),
        omega=omega_dummy.squeeze(-1),
        y=y,
        u=u0.squeeze(-1),
        v=v0.squeeze(-1),
    )

    print("=" * 80)
    print("[forward check]")
    print(f"output shape = {tuple(out.shape)}")
    print(f"dtype        = {out.dtype}")
    print("=" * 80)


if __name__ == "__main__":
    main()
