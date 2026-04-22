from __future__ import annotations
import sys
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky/pybhpt")
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免 Qt 错误
import matplotlib.pyplot as plt



from config.config_loader import load_pinn_full_config
from compute_solution import compute_pybhpt_solution
from mma.rin_sampler import MathematicaRinSampler
from physical_ansatz.prefactor import (
    build_prefactor_primitives,
    Leaver_prefactors,
    prefactor_Q,
    U_prefactor,
)
from physical_ansatz.mapping import r_plus
from physical_ansatz.residual import AuxCache, get_ramp_and_p_from_cfg


def sample_points_chebyshev_grid(
    n_points: int,
    y_min: float = -0.9999,
    y_max: float = 0.9999,
) -> np.ndarray:
    if n_points <= 0:
        return np.empty((0,), dtype=float)
    if n_points == 1:
        return np.array([0.5 * (y_min + y_max)], dtype=float)

    k = np.arange(n_points, dtype=float)
    y_ref = np.cos(np.pi * k / (n_points - 1))
    y = 0.5 * (y_max - y_min) * y_ref + 0.5 * (y_max + y_min)
    return np.sort(y)


def eval_mma_with_fallback(
    sampler: MathematicaRinSampler,
    *,
    s: int,
    l: int,
    m: int,
    a: float,
    omega: float,
    r_query: np.ndarray,
) -> np.ndarray:
    try:
        return sampler.evaluate_rin_at_points_direct(
            s=s,
            l=l,
            m=m,
            a=a,
            omega=omega,
            r_query=r_query,
        )
    except Exception:
        return sampler.evaluate_rin_at_points(
            s=s,
            l=l,
            m=m,
            a=a,
            omega=omega,
            r_query=r_query,
        )


def main():
    cfg_path = Path(__file__).resolve().parents[2] / "config" / "pinn_config.yaml"
    full_cfg = load_pinn_full_config(str(cfg_path))
    physics_cfg = full_cfg["physics"]
    mma_cfg = full_cfg.get("mathematica", full_cfg.get("train", {}).get("mathematica", {}))
    print("A: config loaded")
    # -----------------------------
    # 刚才的参数
    # -----------------------------
    a_val = 0.581667
    omega_val = 1.0
    ell = 2
    m = 2
    s = -2
    M = float(physics_cfg["problem"].get("M", 1.0))

    # -----------------------------
    # 可视化网格
    # 左列: R(r) 用均匀 r 网格
    # 右列: R/U 用 y 上的 Chebyshev 网格
    # -----------------------------
    dtype = torch.float64
    device = torch.device("cpu")

    a_t = torch.tensor(a_val, device=device, dtype=dtype)
    omega_t = torch.tensor(omega_val, device=device, dtype=dtype)

    rp = r_plus(a_t, M)
    viz_num_points = 200  # 从 500 减少到 200
    viz_r_min = 2.0
    viz_r_max = 1000.0     # 从 1000 减少到 100

    r_min = max(viz_r_min, float(rp.detach().cpu().item()) + 1.0e-4)
    r_max = viz_r_max

    y_min = 2.0 * float(rp.detach().cpu().item()) / r_max - 1.0
    y_max = 2.0 * float(rp.detach().cpu().item()) / r_min - 1.0

    y_grid = sample_points_chebyshev_grid(
        n_points=viz_num_points,
        y_min=y_min,
        y_max=y_max,
    )
    y_t = torch.tensor(y_grid, device=device, dtype=dtype)

    x_t = 0.5 * (y_t + 1.0)
    r_t = rp / x_t
    r_np_y = r_t.detach().cpu().numpy()
    r_np_uniform = np.linspace(r_min, r_max, viz_num_points, dtype=float)
    print("B: before pybhpt")
    # -----------------------------
    # pybhpt 解
    # -----------------------------
    _, R_pybhpt_r = compute_pybhpt_solution(
        a=a_val,
        omega=omega_val,
        ell=ell,
        m=m,
        r_grid=r_np_uniform,
        timeout=60.0,  # 增加到 60 秒
    )
    _, R_pybhpt_y = compute_pybhpt_solution(
        a=a_val,
        omega=omega_val,
        ell=ell,
        m=m,
        r_grid=r_np_y,
        timeout=60.0,
    )

    R_pybhpt_y_t = torch.as_tensor(R_pybhpt_y, device=device, dtype=torch.complex128)
    print("C: after pybhpt")
    # -----------------------------
    # 取当前项目里的 p 和 R_amp
    # -----------------------------
    cache = AuxCache()
    p_val, ramp_val = get_ramp_and_p_from_cfg(
        physics_cfg,
        cache,
        a_t,
        omega_t,
    )
    ramp_t = ramp_val.to(device=device, dtype=torch.complex128)
    print("E: after ramp")
    # -----------------------------
    # 构造 U = P * Q
    # -----------------------------
    rp_p, rm_p, rs_p, rs_r_p, rs_rr_p = build_prefactor_primitives(r_t, a_t, M=M)

    P, P_r, P_rr = Leaver_prefactors(
        r_t,
        a_t,
        omega_t,
        m=m,
        M=M,
        s=s,
        rp=rp_p,
        rm=rm_p,
    )

    Q, Q_r, Q_rr = prefactor_Q(
        r_t,
        a_t,
        omega_t,
        p=int(p_val),
        R_amp=ramp_t,
        M=M,
        s=s,
        rp=rp_p,
        rs=rs_p,
        rs_r=rs_r_p,
        rs_rr=rs_rr_p,
    )

    U, _, _ = U_prefactor(P, P_r, P_rr, Q, Q_r, Q_rr)

    # -----------------------------
    # 除以 U_factor
    # -----------------------------
    R_div_U = R_pybhpt_y_t / U

    # -----------------------------
    # 画图
    # -----------------------------
    print("F: before savefig")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex="col")

    # 左列：原始 pybhpt R
    axes[0, 0].plot(r_np_uniform, np.real(R_pybhpt_r), label="Re(R_pybhpt)")
    axes[1, 0].plot(r_np_uniform, np.imag(R_pybhpt_r), label="Im(R_pybhpt)")
    axes[2, 0].plot(r_np_uniform, np.abs(R_pybhpt_r), label="|R_pybhpt|")

    axes[0, 0].set_ylabel("Re(R)")
    axes[1, 0].set_ylabel("Im(R)")
    axes[2, 0].set_ylabel("|R|")
    axes[2, 0].set_xlabel("r")

    # 右列：除以 U 之后
    R_div_U_np = R_div_U.detach().cpu().numpy()
    U_np = U.detach().cpu().numpy()

    axes[0, 1].plot(y_grid, np.real(R_div_U_np), label="Re(R/U)")
    axes[1, 1].plot(y_grid, np.imag(R_div_U_np), label="Im(R/U)")
    axes[2, 1].plot(y_grid, np.abs(R_div_U_np), label="|R/U|")

    axes[0, 1].set_ylabel("Re(R/U)")
    axes[1, 1].set_ylabel("Im(R/U)")
    axes[2, 1].set_ylabel("|R/U|")
    axes[2, 1].set_xlabel("y")

    for ax in axes.ravel():
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(
        f"pybhpt solution and pybhpt/U\n"
        f"a={a_val}, omega={omega_val}, ell={ell}, m={m}, p={p_val}"
    )
    fig.tight_layout()

    out_path = Path(__file__).resolve().parents[1] / "outputs" / "pybhpt_solution_fixed_divU.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    sampler = None
    try:
        print("G: before mma")
        sampler = MathematicaRinSampler(
            kernel_path=mma_cfg["kernel_path"],
            wl_path_win=mma_cfg["wl_path_win"],
            timeout_sec=60.0,
        )
        R_mma_r = eval_mma_with_fallback(
            sampler,
            s=s,
            l=ell,
            m=m,
            a=a_val,
            omega=omega_val,
            r_query=r_np_uniform,
        )
        R_mma_y = eval_mma_with_fallback(
            sampler,
            s=s,
            l=ell,
            m=m,
            a=a_val,
            omega=omega_val,
            r_query=r_np_y,
        )
        print("H: after mma")

        R_mma_y_t = torch.as_tensor(R_mma_y, device=device, dtype=torch.complex128)
        R_mma_div_U = R_mma_y_t / U
        R_mma_div_U_np = R_mma_div_U.detach().cpu().numpy()

        fig_mma, axes_mma = plt.subplots(3, 2, figsize=(12, 10), sharex="col")

        axes_mma[0, 0].plot(r_np_uniform, np.real(R_mma_r), label="Re(R_mma)")
        axes_mma[1, 0].plot(r_np_uniform, np.imag(R_mma_r), label="Im(R_mma)")
        axes_mma[2, 0].plot(r_np_uniform, np.abs(R_mma_r), label="|R_mma|")

        axes_mma[0, 0].set_ylabel("Re(R)")
        axes_mma[1, 0].set_ylabel("Im(R)")
        axes_mma[2, 0].set_ylabel("|R|")
        axes_mma[2, 0].set_xlabel("r")

        axes_mma[0, 1].plot(y_grid, np.real(R_mma_div_U_np), label="Re(R_mma/U)")
        axes_mma[1, 1].plot(y_grid, np.imag(R_mma_div_U_np), label="Im(R_mma/U)")
        axes_mma[2, 1].plot(y_grid, np.abs(R_mma_div_U_np), label="|R_mma/U|")

        axes_mma[0, 1].set_ylabel("Re(R/U)")
        axes_mma[1, 1].set_ylabel("Im(R/U)")
        axes_mma[2, 1].set_ylabel("|R/U|")
        axes_mma[2, 1].set_xlabel("y")

        for ax in axes_mma.ravel():
            ax.grid(alpha=0.3)
            ax.legend()

        fig_mma.suptitle(
            f"mma solution and mma/U\n"
            f"a={a_val}, omega={omega_val}, ell={ell}, m={m}, p={p_val}"
        )
        fig_mma.tight_layout()

        out_path_mma = Path(__file__).resolve().parents[1] / "outputs" / "mma_solution_fixed_divU.png"
        out_path_mma.parent.mkdir(parents=True, exist_ok=True)
        fig_mma.savefig(out_path_mma, dpi=180, bbox_inches="tight")
        plt.close(fig_mma)
        print(f"[saved] {out_path_mma}")
    finally:
        if sampler is not None:
            sampler.close()

    # 额外输出：检查 U/Q 是否病态
    print(f"[saved] {out_path}")
    print(f"[info] min |Q| = {np.min(np.abs(Q.detach().cpu().numpy())):.6e}")
    print(f"[info] min |U| = {np.min(np.abs(U_np)):.6e}")
    idx_q = int(np.argmin(np.abs(Q.detach().cpu().numpy())))
    idx_u = int(np.argmin(np.abs(U_np)))
    print(
        f"[info] argmin |Q|: idx={idx_q}, y={y_grid[idx_q]:.6f}, "
        f"r={r_np_y[idx_q]:.6f}, Q={Q.detach().cpu().numpy()[idx_q]}"
    )
    print(
        f"[info] argmin |U|: idx={idx_u}, y={y_grid[idx_u]:.6f}, "
        f"r={r_np_y[idx_u]:.6f}, U={U_np[idx_u]}"
    )


if __name__ == "__main__":
    main()
