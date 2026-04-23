from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mode import KerrMode
from utils.amplitude_profile_three_patch import TeukRadAmplitudeIn3PatchWithProfile


def y_from_r(r: np.ndarray, rp: float) -> np.ndarray:
    return 2.0 * rp / r - 1.0


def r_from_y(y: np.ndarray, rp: float) -> np.ndarray:
    return 2.0 * rp / (1.0 + y)


def make_y_grid(n: int, kind: str = "uniform") -> np.ndarray:
    if n < 2:
        raise ValueError("n must be >= 2")

    if kind == "uniform":
        return np.linspace(-1.0, 1.0, n)

    if kind == "chebyshev":
        k = np.arange(n, dtype=float)
        y = np.cos(np.pi * k / (n - 1))
        return np.sort(y)

    raise ValueError(f"Unknown y-grid kind: {kind}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot three-patch interpolated in-mode profile in r and reduced Psi(y)."
    )
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--omega", type=float, default=0.1)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--lam", type=float, default=None)

    parser.add_argument("--N-left", dest="N_left", type=int, default=64)
    parser.add_argument("--N-mid", dest="N_mid", type=int, default=96)
    parser.add_argument("--N-right", dest="N_right", type=int, default=64)

    parser.add_argument("--z1", type=float, default=0.1)
    parser.add_argument("--z2", type=float, default=0.9)

    parser.add_argument("--n-r", type=int, default=800)
    parser.add_argument("--n-y", type=int, default=800)
    parser.add_argument(
        "--y-grid",
        type=str,
        default="uniform",
        choices=["uniform", "chebyshev"],
    )
    parser.add_argument("--r-min", type=float, default=None, help="Default: r_+ + 1e-4")
    parser.add_argument("--r-max", type=float, default=1000.0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    mode = KerrMode(
        M=1.0,
        a=args.a,
        omega=args.omega,
        ell=args.ell,
        m=args.m,
        lam=args.lam,
        s=args.s,
    )

    amp = TeukRadAmplitudeIn3PatchWithProfile(
        mode,
        N_left=args.N_left,
        N_mid=args.N_mid,
        N_right=args.N_right,
        z1=args.z1,
        z2=args.z2,
    )
    profile = amp.profile

    rp = mode.rp
    r_min = args.r_min if args.r_min is not None else (rp + 1.0e-4)
    r_max = args.r_max

    # full R_in on uniform r-grid
    r_uniform = np.linspace(r_min, r_max, args.n_r)
    R_r = profile.R_of_r(r_uniform)

    # reduced Psi on y-grid
    y_grid = make_y_grid(args.n_y, kind=args.y_grid)
    z_grid = 0.5 * (y_grid + 1.0)
    psi_y = profile.psi_of_z(z_grid)

    # segment boundaries
    r_z1 = rp / args.z1
    r_z2 = rp / args.z2
    y_z1 = 2.0 * args.z1 - 1.0
    y_z2 = 2.0 * args.z2 - 1.0

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex="col")

    # left column: full R_in on uniform r-grid
    axes[0, 0].plot(r_uniform, R_r.real, lw=1.5)
    axes[1, 0].plot(r_uniform, R_r.imag, lw=1.5)
    axes[2, 0].plot(r_uniform, np.abs(R_r), lw=1.5)

    for ax in axes[:, 0]:
        ax.axvline(r_z1, ls="--", lw=1.0, alpha=0.6)
        ax.axvline(r_z2, ls="--", lw=1.0, alpha=0.6)

    axes[0, 0].set_ylabel("Re(R_in)")
    axes[1, 0].set_ylabel("Im(R_in)")
    axes[2, 0].set_ylabel("|R_in|")
    axes[2, 0].set_xlabel("r")
    axes[0, 0].set_title(
        f"three-patch full R_in on uniform r-grid\n"
        f"segments at z={args.z1:.3f},{args.z2:.3f}"
    )

    # right column: reduced Psi = R_in / P_Leaver on y-grid
    axes[0, 1].plot(y_grid, psi_y.real, lw=1.5)
    axes[1, 1].plot(y_grid, psi_y.imag, lw=1.5)
    axes[2, 1].plot(y_grid, np.abs(psi_y), lw=1.5)

    for ax in axes[:, 1]:
        ax.axvline(y_z1, ls="--", lw=1.0, alpha=0.6)
        ax.axvline(y_z2, ls="--", lw=1.0, alpha=0.6)

    axes[0, 1].set_ylabel(r"Re($R_{\rm in}/P_{\rm Leaver}$)")
    axes[1, 1].set_ylabel(r"Im($R_{\rm in}/P_{\rm Leaver}$)")
    axes[2, 1].set_ylabel(r"$|R_{\rm in}/P_{\rm Leaver}|$")
    axes[2, 1].set_xlabel("y")
    axes[0, 1].set_title(f"three-patch reduced profile on {args.y_grid} y-grid")

    for ax in axes.ravel():
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Three-patch in-mode interpolant\n"
        f"a={args.a}, omega={args.omega}, ell={args.ell}, m={args.m}, "
        f"N_left={args.N_left}, N_mid={args.N_mid}, N_right={args.N_right}, "
        f"z1={args.z1}, z2={args.z2}"
    )
    fig.tight_layout()

    out_dir = PROJECT_ROOT / "benchmark" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.out is None:
        out_path = out_dir / (
            f"inmode_profile_interp_3patch_a{args.a:.6f}_w{args.omega:.6f}"
            f"_Nl{args.N_left}_Nm{args.N_mid}_Nr{args.N_right}"
            f"_z1{args.z1:.3f}_z2{args.z2:.3f}.png"
        )
    else:
        out_path = Path(args.out)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {out_path}")
    print(f"[info] B_inc={amp.B_inc}")
    print(f"[info] B_ref={amp.B_ref}")
    print(f"[info] B_trans={amp.B_trans}")


if __name__ == "__main__":
    main()