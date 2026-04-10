from __future__ import annotations

import os
import sys
import yaml
import torch

# 让脚本能从项目根目录导入
sys.path.insert(0, os.getcwd())

from dataset.grids import chebyshev_grid_bundle
from physical_ansatz.residual import (
    AuxCache,
    get_lambda_from_cfg,
    get_ramp_and_p_from_cfg,
)
from physical_ansatz.teukolsky_coeffs import coeffs_x
from physical_ansatz.transform_y import transform_coeffs_x_to_y


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    # -------------------------------------------------
    # 1. 基本设置
    # -------------------------------------------------
    cfg = load_cfg("config/teukolsky_radial.yaml")

    device = torch.device("cpu")
    dtype_real = torch.float64
    dtype_cplx = torch.complex128

    # 固定参数点
    a0 = 0.1
    omega0 = 0.1

    # 谱阶数
    order = 128

    # 去掉边界层点数（左右各 n_boundary_drop 个点不进入 operator）
    n_boundary_drop = 3

    # -------------------------------------------------
    # 2. 构造谱网格
    # -------------------------------------------------
    grid = chebyshev_grid_bundle(order=order, dtype=dtype_real, device=device)

    y_nodes = grid.y_nodes   # (Ny,)
    D = grid.D               # (Ny,Ny)
    D2 = grid.D2             # (Ny,Ny)
    Tmat = grid.Tmat         # (Ny,Nc), Nc = order+1

    Ny = y_nodes.shape[0]
    Nc = Tmat.shape[1]

    print("===== spectral setup =====")
    print(f"order = {order}")
    print(f"Ny    = {Ny}")
    print(f"Nc    = {Nc}")
    print("==========================\n")

    # -------------------------------------------------
    # 3. 单样本 batch 参数
    # -------------------------------------------------
    a_batch = torch.tensor([a0], dtype=dtype_real, device=device)
    omega_batch = torch.tensor([omega0], dtype=dtype_real, device=device)

    # -------------------------------------------------
    # 4. 预计算 lambda, R_amp, p
    # -------------------------------------------------
    cache = AuxCache()

    lam_list = []
    ramp_list = []
    p_val = None

    for i in range(a_batch.shape[0]):
        lam_i = get_lambda_from_cfg(cfg, cache, a_batch[i], omega_batch[i])
        p_i, ramp_i = get_ramp_and_p_from_cfg(cfg, cache, a_batch[i], omega_batch[i])

        lam_list.append(lam_i)
        ramp_list.append(ramp_i)

        if p_val is None:
            p_val = p_i
        else:
            if p_i != p_val:
                raise ValueError(f"Inconsistent p across batch: {p_val} vs {p_i}")

    lambda_batch = torch.stack(lam_list).to(device=device, dtype=dtype_cplx)
    ramp_batch   = torch.stack(ramp_list).to(device=device, dtype=dtype_cplx)
    p = int(p_val or 1)

    print("===== physical parameters =====")
    print(f"a      = {a0}")
    print(f"omega  = {omega0}")
    print(f"lambda = {lambda_batch[0].item()}")
    print(f"R_amp  = {ramp_batch[0].item()}")
    print(f"p      = {p}")
    print("================================\n")

    # -------------------------------------------------
    # 5. 全节点上预计算 T, DT, D2T
    #
    #    f    = T b
    #    fy   = D T b
    #    fyy  = D2 T b
    # -------------------------------------------------
    Tmat_c = Tmat.to(dtype=dtype_cplx, device=device)
    D_c    = D.to(dtype=dtype_cplx, device=device)
    D2_c   = D2.to(dtype=dtype_cplx, device=device)

    DT = D_c @ Tmat_c       # (Ny, Nc)
    D2T = D2_c @ Tmat_c     # (Ny, Nc)

    # -------------------------------------------------
    # 6. 只在内点上构造 operator 系数
    #
    #    注意：不能先在端点上算 A_i/B_i 再删点，
    #    要先决定 interior slice，再把这些点送进 operator
    # -------------------------------------------------
    if n_boundary_drop < 0:
        raise ValueError("n_boundary_drop must be >= 0")
    if 2 * n_boundary_drop >= Ny:
        raise ValueError(
            f"Too many boundary points dropped: n_boundary_drop={n_boundary_drop}, Ny={Ny}"
        )

    if n_boundary_drop > 0:
        sl = slice(n_boundary_drop, -n_boundary_drop)
    else:
        sl = slice(None)

    y_used = y_nodes[sl]   # (Ny_used,)
    Ny_used = y_used.shape[0]

    # 只把 interior 节点送进 coeffs_x / transform
    y = y_used[None, :, None]                    # (1,Ny_used,1)
    x = (y + 1.0) / 2.0                          # (1,Ny_used,1)

    a = a_batch.reshape(1, 1, 1)
    omega = omega_batch.reshape(1, 1, 1)
    lam = lambda_batch.reshape(1, 1, 1)
    ramp = ramp_batch.reshape(1, 1, 1)

    prob = cfg["problem"]
    m = int(prob["m"])
    s = int(prob.get("s", -2))
    M = float(prob.get("M", 1.0))

    A2, A1, A0 = coeffs_x(x, a, omega, m, p, ramp, lam, s=s, M=M)
    B2, B1, B0, rhs = transform_coeffs_x_to_y(A2, A1, A0, y)

    # 去掉 batch 维和末尾 singleton 维
    B2v = B2[0, :, 0]       # (Ny_used,)
    B1v = B1[0, :, 0]
    B0v = B0[0, :, 0]
    rhsv = rhs[0, :, 0]     # (Ny_used,)

    # -------------------------------------------------
    # 7. 在同样的 interior 点上截取 T, DT, D2T
    # -------------------------------------------------
    T_used   = Tmat_c[sl, :]    # (Ny_used, Nc)
    DT_used  = DT[sl, :]        # (Ny_used, Nc)
    D2T_used = D2T[sl, :]       # (Ny_used, Nc)

    # -------------------------------------------------
    # 8. 组装线性系统
    #
    #    Mmat b = rhs
    #
    #    每一行 j:
    #    B2_j * (D2T)_j + B1_j * (DT)_j + B0_j * T_j
    # -------------------------------------------------
    Mmat = (
        B2v[:, None] * D2T_used
        + B1v[:, None] * DT_used
        + B0v[:, None] * T_used
    )   # (Ny_used, Nc)

    # -------------------------------------------------
    # 9. 直接谱法求解
    #
    #    因为当前只是 interior residual，没有完整边界闭合，
    #    所以一般是过定约系统，用最小二乘更稳
    # -------------------------------------------------
    sol = torch.linalg.lstsq(Mmat, rhsv)
    coeff = sol.solution   # (Nc,) complex

    # -------------------------------------------------
    # 10. 回代检查 residual
    # -------------------------------------------------
    f_used = T_used @ coeff
    fy_used = DT_used @ coeff
    fyy_used = D2T_used @ coeff

    res = B2v * fyy_used + B1v * fy_used + B0v * f_used - rhsv

    loss_res = torch.mean(res.real**2 + res.imag**2).item()
    res_abs_max = torch.max(torch.abs(res)).item()

    coeff_re = coeff.real.detach().cpu().reshape(1, -1)
    coeff_im = coeff.imag.detach().cpu().reshape(1, -1)

    coeff_abs_max = torch.max(torch.abs(coeff)).item()

    print("===== direct spectral solve summary =====")
    print(f"Ny_used      = {Ny_used}")
    print(f"Mmat shape   = {tuple(Mmat.shape)}")
    print(f"loss_res     = {loss_res:.6e}")
    print(f"res_abs_max  = {res_abs_max:.6e}")
    print(f"coeff_abs_max= {coeff_abs_max:.6e}")
    print("=========================================\n")

    # -------------------------------------------------
    # 11. 保存
    # -------------------------------------------------
    os.makedirs("outputs", exist_ok=True)
    save_path = f"outputs/direct_spectral_a{a0}_w{omega0}_o{order}.pt"

    torch.save(
        {
            "a": a0,
            "omega": omega0,
            "order": order,
            "n_boundary_drop": n_boundary_drop,
            "coeff_re": coeff_re,
            "coeff_im": coeff_im,
            "lambda": lambda_batch.detach().cpu(),
            "R_amp": ramp_batch.detach().cpu(),
            "p": p,
            "loss_res": loss_res,
            "res_abs_max": res_abs_max,
        },
        save_path,
    )

    print(f"Saved direct spectral coefficients to:\n  {save_path}")


if __name__ == "__main__":
    main()