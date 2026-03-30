"""
用 Mathematica 计算 R_in，然后除以 U(r) 得到 regular 部分 R'
参数: l=2, m=2, a=0.1, ω=0.1, s=-2
r 范围: 2M → 10M
输出: mma/regular_part/ 下按参数分类的图
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

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physical_ansatz.prefactor import U_factor
from physical_ansatz.mapping import r_plus
from physical_ansatz.residual import AuxCache, get_ramp_and_p_from_cfg
import yaml

# ── 参数 ──────────────────────────────────────────────
S      = -2
L      = 2
M_MODE = 2
A      = 0.1
OMEGA  = 0.1
RMIN   = 2.0
RMAX   = 1000.0
NPTS   = 500
M      = 1.0

KERNEL_PATH  = r"/mnt/f/mma/WolframKernel.exe"
WL_PATH_WIN  = r"F:/EMRI/Radial_flow/Radial_Function.wl"
CFG_PATH     = "config/teukolsky_radial.yaml"

_BASE_DIR = os.path.join(os.path.dirname(__file__), "regular_part")
OUT_DIR = os.path.join(
    _BASE_DIR,
    f"s{S}",
    f"l{L}_m{M_MODE}",
    f"a{A:.4g}_omega{OMEGA:.4g}",
    f"r{RMIN:.4g}_to_{RMAX:.4g}",
)
os.makedirs(OUT_DIR, exist_ok=True)

# ── 调用 Mathematica ──────────────────────────────────
def sample_rin(rmin, rmax, npts):
    session = WolframLanguageSession(kernel=KERNEL_PATH)
    try:
        session.evaluate(wlexpr(rf'Get["{WL_PATH_WIN}"]'))
        expr = (
            f"SampleRinOnGrid[{S}, {L}, {M_MODE}, "
            f"{A:.16g}, {OMEGA:.16g}, "
            f"{rmin:.16g}, {rmax:.16g}, {npts}]"
        )
        result = session.evaluate(wlexpr(expr))
        arr = np.array(result, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"Unexpected Mathematica output shape: {arr.shape}")
        r   = arr[:, 0]
        Rin = arr[:, 1] + 1j * arr[:, 2]
        return r, Rin
    finally:
        session.terminate()

# ── 计算 U(r) ─────────────────────────────────────────
def compute_U(r_np, cfg):
    device = torch.device("cpu")
    dtype_real = torch.float64
    
    r_t = torch.tensor(r_np, dtype=dtype_real, device=device)
    a_t = torch.tensor(A, dtype=dtype_real, device=device)
    omega_t = torch.tensor(OMEGA, dtype=dtype_real, device=device)
    
    cache = AuxCache()
    p, ramp_t = get_ramp_and_p_from_cfg(cfg, cache, a_t, omega_t)
    
    U_t = U_factor(r_t, a_t, omega_t, p, ramp_t, M_MODE, S, M)
    return U_t.detach().cpu().numpy(), p, complex(ramp_t.item())

# ── 绘图 ─────────────────────────────────────────────
def make_plots(r, Rprime, p, ramp):
    title_base = rf"$R'(r) = R_{{in}}/U$: $l={L},\ m={M_MODE},\ a={A},\ \omega={OMEGA},\ s={S}$"
    subtitle = rf"$p={p},\ R_{{amp}}={ramp.real:.3f}{ramp.imag:+.3f}i$"

    # ── 图1: 普通坐标，Re/Im ──────────────────────────
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
    plt.savefig(os.path.join(OUT_DIR, "Rprime_linear_re_im.png"), dpi=200)
    plt.close()

    # ── 图2: 普通坐标，模长 ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(r, np.abs(Rprime), lw=1.0, color="C2", label=r"$|R'|$")
    ax.set_xlabel(r"$r / M$")
    ax.set_ylabel(r"$|R'|$")
    ax.set_title(title_base + "\n" + subtitle)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Rprime_linear_abs.png"), dpi=200)
    plt.close()

    # ── 图3: 对数坐标（x轴），Re/Im ──────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].semilogx(r, Rprime.real, lw=1.0, label=r"$\mathrm{Re}[R']$")
    axes[0].set_ylabel(r"$\mathrm{Re}[R']$")
    axes[0].legend()
    axes[0].set_title(title_base + "\n" + subtitle)
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogx(r, Rprime.imag, lw=1.0, color="C1", label=r"$\mathrm{Im}[R']$")
    axes[1].set_xlabel(r"$r / M$")
    axes[1].set_ylabel(r"$\mathrm{Im}[R']$")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Rprime_logr_re_im.png"), dpi=200)
    plt.close()

    # ── 图4: 对数坐标（x轴），模长 ───────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogx(r, np.abs(Rprime), lw=1.0, color="C2", label=r"$|R'|$")
    ax.set_xlabel(r"$r / M$")
    ax.set_ylabel(r"$|R'|$")
    ax.set_title(title_base + "\n" + subtitle)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Rprime_logr_abs.png"), dpi=200)
    plt.close()

    print(f"Saved 4 figures to: {OUT_DIR}")

# ── 主函数 ────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Parameters: s={S}, l={L}, m={M_MODE}, a={A}, ω={OMEGA}")
    print(f"r range: [{RMIN}, {RMAX}], npts={NPTS}")
    
    # 1. 从 Mathematica 获取 R_in
    print("\nCalling Mathematica...")
    r, Rin = sample_rin(RMIN, RMAX, NPTS)
    print(f"Got {len(r)} points")
    print(f"|R_in| range: [{np.abs(Rin).min():.4e}, {np.abs(Rin).max():.4e}]")
    
    # 2. 加载配置并计算 U(r)
    print("\nComputing U(r)...")
    with open(CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    
    U, p, ramp = compute_U(r, cfg)
    print(f"p = {p}")
    print(f"R_amp = {ramp}")
    print(f"|U| range: [{np.abs(U).min():.4e}, {np.abs(U).max():.4e}]")
    
    # 3. 计算 R' = R_in / U
    Rprime = Rin / U
    print(f"\n|R'| range: [{np.abs(Rprime).min():.4e}, {np.abs(Rprime).max():.4e}]")
    
    # 4. 绘图
    print("\nGenerating plots...")
    make_plots(r, Rprime, p, ramp)
    print("Done.")
