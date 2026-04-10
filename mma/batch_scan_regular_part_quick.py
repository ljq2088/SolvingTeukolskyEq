"""
快速测试版本：只扫描少量参数组合
参数范围:
  - a: [0.1, 0.5]
  - ω: [0.01, 0.1]
  - l, m: [(2,2), (3,2)]
  - r: 2M 到 100M (在 x=rp/r 上均匀采样)
输出: mma/batch_regular_part/ 下按参数分类
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg")  # 前端交互显示
import matplotlib.pyplot as plt
from itertools import product

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physical_ansatz.prefactor import U_factor
from physical_ansatz.mapping import r_plus
from physical_ansatz.residual import AuxCache, get_ramp_and_p_from_cfg
import yaml

# ── 全局配置 ──────────────────────────────────────────
S = -2
M = 1.0
RMIN = 2.0
RMAX = 100.0  # 快速测试用较小范围
NPTS_X = 200  # 快速测试用较少点

KERNEL_PATH = r"/mnt/f/mma/WolframKernel.exe"
WL_PATH_WIN = r"F:/EMRI/Radial_flow/Radial_Function.wl"
CFG_PATH = "config/teukolsky_radial.yaml"

_BASE_DIR = os.path.join(os.path.dirname(__file__), "batch_regular_part")

# ── 参数网格（快速测试）──────────────────────────────
A_VALUES = [0.1, 0.5]
OMEGA_VALUES = [0.01, 0.1]
LM_PAIRS = [(2, 2), (3, 2)]  # (l, m) 对

# ── 在 x 坐标上均匀采样，转换回 r ────────────────────
def make_r_grid_from_x(rp_val, rmin, rmax, npts_x):
    """
    在 x = rp/r 上均匀采样，然后转换回 r
    x ∈ [rp/rmax, rp/rmin]，靠近视界（r→rp, x→1）采点密
    """
    x_min = rp_val / rmax
    x_max = rp_val / rmin
    x_grid = np.linspace(x_min, x_max, npts_x)
    r_grid = rp_val / x_grid
    return r_grid

# ── 调用 Mathematica 获取 R_in ────────────────────────
def sample_rin(session, s, l, m, a, omega, r_grid):
    """
    使用已打开的 session，在给定 r_grid 上采样 R_in
    """
    npts = len(r_grid)
    rmin = r_grid[0]
    rmax = r_grid[-1]

    expr = (
        f"SampleRinOnGrid[{s}, {l}, {m}, "
        f"{a:.16g}, {omega:.16g}, "
        f"{rmin:.16g}, {rmax:.16g}, {npts}]"
    )
    result = session.evaluate(wlexpr(expr))
    arr = np.array(result, dtype=float)

    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Unexpected Mathematica output shape: {arr.shape}")

    r_mma = arr[:, 0]
    Rin = arr[:, 1] + 1j * arr[:, 2]
    return r_mma, Rin

# ── 计算 U(r) ─────────────────────────────────────────
def compute_U(r_np, a_val, omega_val, m_val, s_val, cfg):
    device = torch.device("cpu")
    dtype_real = torch.float64

    r_t = torch.tensor(r_np, dtype=dtype_real, device=device)
    a_t = torch.tensor(a_val, dtype=dtype_real, device=device)
    omega_t = torch.tensor(omega_val, dtype=dtype_real, device=device)

    cache = AuxCache()
    p, ramp_t = get_ramp_and_p_from_cfg(cfg, cache, a_t, omega_t)

    U_t = U_factor(r_t, a_t, omega_t, p, ramp_t, m_val, s_val, M)
    return U_t.detach().cpu().numpy(), p, complex(ramp_t.item())

# ── 绘图并保存 ────────────────────────────────────────
def plot_and_save(r, Rprime, s, l, m, a, omega, p, ramp, out_dir, show=True):
    """
    生成2张图并保存（快速版本）
    """
    title_base = rf"$R'(r) = R_{{in}}/U$: $l={l},\ m={m},\ a={a},\ \omega={omega},\ s={s}$"
    subtitle = rf"$p={p},\ R_{{amp}}={ramp.real:.3f}{ramp.imag:+.3f}i$"

    # ── 图1: 线性坐标，Re/Im ──────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(r, Rprime.real, lw=1.0, label=r"$\mathrm{Re}[R']$")
    axes[0].set_ylabel(r"$\mathrm{Re}[R']$")
    axes[0].legend()
    axes[0].set_title(title_base + "\n" + subtitle)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(r, Rprime.imag, lw=1.0, color="C1", label=r"$\mathrm{Im}[R']$")
    axes[1].set_xlabel(r"$r / M$")
    axes[1].set_ylabel(r"$\mathrm{Im}[R']$")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Rprime_linear_re_im.png"), dpi=150)
    if show:
        plt.show(block=False)
        plt.pause(1.0)
    plt.close()

    # ── 图2: 对数坐标，模长 ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogx(r, np.abs(Rprime), lw=1.0, color="C2", label=r"$|R'|$")
    ax.set_xlabel(r"$r / M$")
    ax.set_ylabel(r"$|R'|$")
    ax.set_title(title_base + "\n" + subtitle)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Rprime_logr_abs.png"), dpi=150)
    if show:
        plt.show(block=False)
        plt.pause(1.0)
    plt.close()

# ── 主函数 ────────────────────────────────────────────
def main():
    # 加载配置
    with open(CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # 打开 Mathematica session（复用）
    print("Starting Mathematica session...")
    session = WolframLanguageSession(kernel=KERNEL_PATH)
    try:
        session.evaluate(wlexpr(rf'Get["{WL_PATH_WIN}"]'))
        print("Mathematica loaded.\n")

        # 生成所有参数组合
        param_list = list(product(A_VALUES, OMEGA_VALUES, LM_PAIRS))
        total = len(param_list)
        print(f"Total parameter combinations: {total}\n")

        for idx, (a, omega, (l, m)) in enumerate(param_list, 1):
            print(f"[{idx}/{total}] Processing: a={a}, ω={omega}, l={l}, m={m}")

            # 计算 r_+ 并生成 r 网格
            a_t = torch.tensor(a, dtype=torch.float64)
            rp_val = float(r_plus(a_t, M).item())
            r_grid = make_r_grid_from_x(rp_val, RMIN, RMAX, NPTS_X)

            # 调用 Mathematica 获取 R_in
            try:
                r_mma, Rin = sample_rin(session, S, l, m, a, omega, r_grid)
            except Exception as e:
                print(f"  ERROR calling Mathematica: {e}")
                continue

            # 计算 U(r)
            try:
                U, p, ramp = compute_U(r_mma, a, omega, m, S, cfg)
            except Exception as e:
                print(f"  ERROR computing U: {e}")
                continue

            # 计算 R' = R_in / U
            Rprime = Rin / U

            # 创建输出目录
            out_dir = os.path.join(
                _BASE_DIR,
                f"s{S}",
                f"l{l}_m{m}",
                f"a{a:.4g}_omega{omega:.4g}",
                f"r{RMIN:.4g}_to_{RMAX:.4g}",
            )
            os.makedirs(out_dir, exist_ok=True)

            # 绘图并保存（全部显示）
            plot_and_save(r_mma, Rprime, S, l, m, a, omega, p, ramp, out_dir, show=True)

            print(f"  |R'| range: [{np.abs(Rprime).min():.4e}, {np.abs(Rprime).max():.4e}]")
            print(f"  Saved to: {out_dir}\n")

    finally:
        session.terminate()
        print("Mathematica session terminated.")

if __name__ == "__main__":
    plt.ion()  # 开启交互模式
    main()
    print("\nAll done.")

