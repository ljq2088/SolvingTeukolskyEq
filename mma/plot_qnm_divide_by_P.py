"""
用 Mathematica 计算 QNM 径向函数，然后除以 P(r) 得到剩余部分
QNM: 只有入射和出射波，ω 是复数本征值，由 Mathematica 自动求解
参数: s=-2, l=2, m=2, a=0.1, n=0 (基态)
r 范围: 2.1M → 1000M
输出: mma/qnm_divide_by_P/ 下按参数分类的图
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physical_ansatz.prefactor import prefactor_P

# ── 参数 ──────────────────────────────────────────────
S      = -2
L      = 2
M_MODE = 2
A      = 0.1
N_OVT  = 0       # 泛音阶数
RMIN   = 2.1     # 避开 r_plus 奇点
RMAX   = 1000.0
NPTS   = 500
M      = 1.0

KERNEL_PATH  = r"/mnt/f/mma/WolframKernel.exe"
WL_PATH_WIN  = r"F:/EMRI/Radial_flow/QNM.wl"

# OUT_DIR 在获取到 omega 之后再确定（因为 omega 是本征值，事先不知道）
_BASE_DIR = os.path.join(os.path.dirname(__file__), "qnm_divide_by_P")


# ── 调用 Mathematica ──────────────────────────────────
def sample_qnm(rmin, rmax, npts):
    """
    返回:
        omega: complex  QNM 本征频率
        r:     (N,)     r 网格
        R:     (N,)     复数径向函数
    """
    session = WolframLanguageSession(kernel=KERNEL_PATH)
    try:
        session.evaluate(wlexpr(rf'Get["{WL_PATH_WIN}"]'))

        # 1) 取本征频率
        freq = session.evaluate(
            wlexpr(f'GetQNMFrequency[{S}, {L}, {M_MODE}, {N_OVT}, {A:.16g}]')
        )
        omega = complex(freq["ReFrequency"], freq["ImFrequency"])
        print(f"QNM ω = {omega.real:.8f} {omega.imag:+.8f}i")

        # 2) 采样径向函数
        result = session.evaluate(
            wlexpr(
                f'SampleQNMRadialOnGrid[{S}, {L}, {M_MODE}, {N_OVT}, '
                f'{A:.16g}, {rmin:.16g}, {rmax:.16g}, {npts}]'
            )
        )
        arr = np.array(result, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"Unexpected shape: {arr.shape}")
        r = arr[:, 0]
        R = arr[:, 1] + 1j * arr[:, 2]
        return omega, r, R
    finally:
        session.terminate()


# ── 计算 P(r; ω 复数) ────────────────────────────────
def compute_P(r_np, omega_cplx):
    """
    P(r) 中的 ω 是复数本征值
    """
    device = torch.device("cpu")
    r_t     = torch.tensor(r_np, dtype=torch.float64, device=device)
    a_t     = torch.tensor(A,    dtype=torch.float64, device=device)
    # omega 是复数，用 complex128
    omega_t = torch.tensor(omega_cplx, dtype=torch.complex128, device=device)

    P_t = prefactor_P(r_t, a_t, omega_t, M_MODE, M, S)
    return P_t.detach().cpu().numpy()


# ── 绘图 ─────────────────────────────────────────────
def make_plots(r, R_over_P, omega, out_dir):
    wr, wi = omega.real, omega.imag
    title = (
        rf"$R_{{QNM}}/P$: $l={L},\ m={M_MODE},\ a={A},\ s={S}$"
        "\n"
        rf"$\omega = {wr:.5f}{wi:+.5f}i$  (n={N_OVT})"
    )

    # 图1: 普通坐标 Re/Im
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(r, R_over_P.real, lw=1.0, label="Re")
    axes[0].set_ylabel(r"$\mathrm{Re}[R/P]$")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(r, R_over_P.imag, lw=1.0, color="C1", label="Im")
    axes[1].set_xlabel(r"$r / M$")
    axes[1].set_ylabel(r"$\mathrm{Im}[R/P]$")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "RoverP_linear.png"), dpi=200)
    plt.close()

    # 图2: 普通坐标 模长
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(r, np.abs(R_over_P), lw=1.0, color="C2")
    ax.set_xlabel(r"$r / M$")
    ax.set_ylabel(r"$|R/P|$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "RoverP_linear_abs.png"), dpi=200)
    plt.close()

    # 图3: log r 坐标 Re/Im
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].semilogx(r, R_over_P.real, lw=1.0, label="Re")
    axes[0].set_ylabel(r"$\mathrm{Re}[R/P]$")
    axes[0].set_title(title)
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].semilogx(r, R_over_P.imag, lw=1.0, color="C1", label="Im")
    axes[1].set_xlabel(r"$r / M$")
    axes[1].set_ylabel(r"$\mathrm{Im}[R/P]$")
    axes[1].grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "RoverP_logr.png"), dpi=200)
    plt.close()

    # 图4: log r 坐标 模长
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogx(r, np.abs(R_over_P), lw=1.0, color="C2")
    ax.set_xlabel(r"$r / M$")
    ax.set_ylabel(r"$|R/P|$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "RoverP_logr_abs.png"), dpi=200)
    plt.close()

    print(f"Saved 4 figures to: {out_dir}")


# ── 主函数 ────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Parameters: s={S}, l={L}, m={M_MODE}, n={N_OVT}, a={A}")
    print(f"r range: [{RMIN}, {RMAX}], npts={NPTS}\n")

    omega, r, R = sample_qnm(RMIN, RMAX, NPTS)

    # 确定输出目录（包含 omega 本征值）
    out_dir = os.path.join(
        _BASE_DIR,
        f"s{S}",
        f"l{L}_m{M_MODE}_n{N_OVT}",
        f"a{A:.4g}",
        f"omega_{omega.real:.5f}{omega.imag:+.5f}i",
        f"r{RMIN:.4g}_to_{RMAX:.4g}",
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"Got {len(r)} points, r ∈ [{r[0]:.3f}, {r[-1]:.3f}]")
    print(f"|R| range: [{np.abs(R).min():.4e}, {np.abs(R).max():.4e}]")

    P = compute_P(r, omega)
    print(f"|P| range: [{np.abs(P).min():.4e}, {np.abs(P).max():.4e}]")

    R_over_P = R / P
    print(f"|R/P| range: [{np.abs(R_over_P).min():.4e}, {np.abs(R_over_P).max():.4e}]")

    make_plots(r, R_over_P, omega, out_dir)
    print("Done.")
