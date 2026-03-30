"""
用 Mathematica 计算 R_in，然后只除以 P(r) 得到 R_in/P
参数: l=2, m=2, a=0.1, ω=0.1, s=-2
r 范围: 2M → 10M
输出: mma/divide_by_P/ 下按参数分类的图
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

from physical_ansatz.prefactor import prefactor_P

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

_BASE_DIR = os.path.join(os.path.dirname(__file__), "divide_by_P")
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
            raise ValueError(f"Unexpected shape: {arr.shape}")
        r   = arr[:, 0]
        Rin = arr[:, 1] + 1j * arr[:, 2]
        return r, Rin
    finally:
        session.terminate()

# ── 计算 P(r) ─────────────────────────────────────────
def compute_P(r_np):
    device = torch.device("cpu")
    dtype_real = torch.float64
    
    r_t = torch.tensor(r_np, dtype=dtype_real, device=device)
    a_t = torch.tensor(A, dtype=dtype_real, device=device)
    omega_t = torch.tensor(OMEGA, dtype=dtype_real, device=device)
    
    P_t = prefactor_P(r_t, a_t, omega_t, M_MODE, M, S)
    return P_t.detach().cpu().numpy()

# ── 绘图 ─────────────────────────────────────────────
def make_plots(r, R_over_P):
    title = rf"$R_{{in}}/P$: $l={L},\ m={M_MODE},\ a={A},\ \omega={OMEGA},\ s={S}$"

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(r, R_over_P.real, lw=1.0)
    axes[0].set_ylabel(r"$\mathrm{Re}[R_{in}/P]$")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(r, R_over_P.imag, lw=1.0, color="C1")
    axes[1].set_xlabel(r"$r / M$")
    axes[1].set_ylabel(r"$\mathrm{Im}[R_{in}/P]$")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Rin_over_P_linear.png"), dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogx(r, R_over_P.real, lw=1.0, label="Re")
    ax.semilogx(r, R_over_P.imag, lw=1.0, label="Im")
    ax.set_xlabel(r"$r / M$")
    ax.set_ylabel(r"$R_{in}/P$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Rin_over_P_logr.png"), dpi=200)
    plt.close()
    print(f"Saved to: {OUT_DIR}")

if __name__ == "__main__":
    print(f"Parameters: s={S}, l={L}, m={M_MODE}, a={A}, ω={OMEGA}")
    r, Rin = sample_rin(RMIN, RMAX, NPTS)
    print(f"Got {len(r)} points")
    
    P = compute_P(r)
    print(f"|P| range: [{np.abs(P).min():.4e}, {np.abs(P).max():.4e}]")
    
    R_over_P = Rin / P
    print(f"|R/P| range: [{np.abs(R_over_P).min():.4e}, {np.abs(R_over_P).max():.4e}]")
    
    make_plots(r, R_over_P)
    print("Done.")
