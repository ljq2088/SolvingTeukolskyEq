"""
用 Mathematica 计算并绘制 R_in 径向函数
参数: l=2, m=2, a=0.1, ω=0.1, s=-2
r 范围: 2M → 1000M
输出: mma/figures/ 下的普通坐标和对数坐标图
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr

# ── 参数 ──────────────────────────────────────────────
S      = -2
L      = 2
M_MODE = 2
A      = 0.1
OMEGA  = 0.1
RMIN   = 2.0
RMAX   = 10.0
NPTS   = 500

KERNEL_PATH  = r"/mnt/f/mma/WolframKernel.exe"
WL_PATH_WIN  = r"F:/EMRI/Radial_flow/Radial_Function.wl"

_BASE_DIR = os.path.join(os.path.dirname(__file__), "figures")
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

# ── 绘图 ─────────────────────────────────────────────
def make_plots(r, Rin):
    title_base = rf"$R_{{in}}$: $l={L},\ m={M_MODE},\ a={A},\ \omega={OMEGA},\ s={S}$"

    # ── 图1: 普通坐标，Re/Im ──────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(r, Rin.real, lw=1.0, label=r"$\mathrm{Re}[R_{in}]$")
    axes[0].set_ylabel(r"$\mathrm{Re}[R_{in}]$")
    axes[0].legend()
    axes[0].set_title(title_base + "  (linear scale)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(r, Rin.imag, lw=1.0, color="C1", label=r"$\mathrm{Im}[R_{in}]$")
    axes[1].set_xlabel(r"$r / M$")
    axes[1].set_ylabel(r"$\mathrm{Im}[R_{in}]$")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "Rin_linear_re_im.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")

    # ── 图2: 普通坐标，模长 ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(r, np.abs(Rin), lw=1.0, color="C2", label=r"$|R_{in}|$")
    ax.set_xlabel(r"$r / M$")
    ax.set_ylabel(r"$|R_{in}|$")
    ax.set_title(title_base + "  (linear scale, magnitude)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "Rin_linear_abs.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")

    # ── 图3: 对数坐标（x轴），Re/Im ──────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].semilogx(r, Rin.real, lw=1.0, label=r"$\mathrm{Re}[R_{in}]$")
    axes[0].set_ylabel(r"$\mathrm{Re}[R_{in}]$")
    axes[0].legend()
    axes[0].set_title(title_base + "  (log r scale)")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogx(r, Rin.imag, lw=1.0, color="C1", label=r"$\mathrm{Im}[R_{in}]$")
    axes[1].set_xlabel(r"$r / M$")
    axes[1].set_ylabel(r"$\mathrm{Im}[R_{in}]$")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "Rin_logr_re_im.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")

    # ── 图4: 对数坐标（x轴），模长 ───────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogx(r, np.abs(Rin), lw=1.0, color="C2", label=r"$|R_{in}|$")
    ax.set_xlabel(r"$r / M$")
    ax.set_ylabel(r"$|R_{in}|$")
    ax.set_title(title_base + "  (log r scale, magnitude)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "Rin_logr_abs.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")

    # ── 图5: log-log，模长（看渐近行为）──────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.loglog(r, np.abs(Rin), lw=1.0, color="C3", label=r"$|R_{in}|$")
    ax.set_xlabel(r"$r / M$")
    ax.set_ylabel(r"$|R_{in}|$")
    ax.set_title(title_base + "  (log-log scale, magnitude)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "Rin_loglog_abs.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    print(f"Calling Mathematica: s={S}, l={L}, m={M_MODE}, a={A}, ω={OMEGA}")
    print(f"r range: [{RMIN}, {RMAX}], npts={NPTS}")
    r, Rin = sample_rin(RMIN, RMAX, NPTS)
    print(f"Got {len(r)} points, r[0]={r[0]:.4f}, r[-1]={r[-1]:.4f}")
    print(f"|R_in| range: [{np.abs(Rin).min():.4e}, {np.abs(Rin).max():.4e}]")
    make_plots(r, Rin)
    print("Done.")
