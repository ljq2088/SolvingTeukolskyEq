from __future__ import annotations

import os

# ---- 必须在 numpy / matplotlib 之前 ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.mode import KerrMode
from utils.amplitude_profile import TeukRadAmplitudeInWithProfile


def y_from_r(r: np.ndarray, rp: float) -> np.ndarray:
    return 2.0 * rp / r - 1.0


def r_from_y(y: np.ndarray, rp: float) -> np.ndarray:
    return 2.0 * rp / (1.0 + y)

def make_y_grid(n: int, kind: str = "uniform") -> np.ndarray:
    """
    Build y-grid on [-1, 1].
    kind:
        - uniform
        - chebyshev
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    if kind == "uniform":
        return np.linspace(-1.0, 1.0, n)

    if kind == "chebyshev":
        k = np.arange(n, dtype=float)
        y = np.cos(np.pi * k / (n - 1))
        return np.sort(y)

    raise ValueError(f"Unknown y-grid kind: {kind}")

def _lambda_worker(queue, a, omega, ell, m, s):
    try:
        from utils.mode import KerrMode
        mode_tmp = KerrMode(
            M=1.0,
            a=float(a),
            omega=float(omega),
            ell=int(ell),
            m=int(m),
            lam=None,
            s=int(s),
        )
        lam = mode_tmp.lambda_value
        queue.put({"ok": True, "lam_re": float(complex(lam).real), "lam_im": float(complex(lam).imag)})
    except Exception as e:
        queue.put({"ok": False, "err": str(e)})


def resolve_lambda_with_timeout(a, omega, ell, m, s, timeout=30.0):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_lambda_worker, args=(q, a, omega, ell, m, s), daemon=True)
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join(1.0)
        raise TimeoutError(
            f"lambda computation timed out after {timeout:.1f}s "
            f"(a={a}, omega={omega}, ell={ell}, m={m}, s={s})"
        )

    if q.empty():
        raise RuntimeError("lambda subprocess returned no result")

    res = q.get()
    if not res["ok"]:
        raise RuntimeError(f"lambda subprocess failed: {res['err']}")

    return complex(res["lam_re"], res["lam_im"])


def main():
    parser = argparse.ArgumentParser(
        description="Plot interpolated in-mode profile in r- and y-uniform samples."
    )
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)

    parser.add_argument("--lam-re", type=float, default=None, help="Optional precomputed lambda real part")
    parser.add_argument("--lam-im", type=float, default=0.0, help="Optional precomputed lambda imag part")
    parser.add_argument("--lambda-timeout", type=float, default=30.0)

    parser.add_argument("--N-in", type=int, default=128)
    parser.add_argument("--N-out", type=int, default=128)
    parser.add_argument("--z-m", type=float, default=0.3)

    parser.add_argument("--n-r", type=int, default=600)
    parser.add_argument("--n-y", type=int, default=600)
    parser.add_argument(
        "--y-grid",
        type=str,
        default="uniform",
        choices=["uniform", "chebyshev"],
        help="Sampling grid for reduced function Psi(y)=R_in/P_Leaver on y in [-1,1].",
    )
    parser.add_argument("--r-min", type=float, default=None, help="Default: r_+ + 1e-4")
    parser.add_argument("--r-max", type=float, default=1000.0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    print("[stage] resolve lambda", flush=True)
    if args.lam_re is not None:
        lam = complex(args.lam_re, args.lam_im)
        print(f"[info] using user-provided lambda = {lam}", flush=True)
    else:
        lam = resolve_lambda_with_timeout(
            args.a, args.omega, args.ell, args.m, args.s, timeout=args.lambda_timeout
        )
        print(f"[info] resolved lambda = {lam}", flush=True)

    print("[stage] construct mode", flush=True)
    mode = KerrMode(
        M=1.0,
        a=args.a,
        omega=args.omega,
        ell=args.ell,
        m=args.m,
        lam=lam,
        s=args.s,
    )

    print("[stage] build amplitude/profile object", flush=True)
    amp = TeukRadAmplitudeInWithProfile(
        mode,
        N_in=args.N_in,
        N_out=args.N_out,
        z_m=args.z_m,
    )

    print("[stage] solve spectral patches and build interpolant", flush=True)
    profile = amp.profile

    rp = mode.rp
    r_min = args.r_min if args.r_min is not None else (rp + 1.0e-4)
    r_max = args.r_max

    print("[stage] sample full R_in on uniform r-grid", flush=True)
    r_uniform = np.linspace(r_min, r_max, args.n_r)
    R_r = profile.R_of_r(r_uniform)

    print(f"[stage] sample reduced profile Psi=R_in/P on {args.y_grid} y-grid", flush=True)
    y_grid = make_y_grid(args.n_y, kind=args.y_grid)
    z_grid = 0.5 * (y_grid + 1.0)
    psi_y = profile.psi_of_z(z_grid)

    print("[stage] plotting", flush=True)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex="col")

    # -------------------------------------------------
    # left column: reconstructed full R_in on uniform r
    # -------------------------------------------------
    axes[0, 0].plot(r_uniform, R_r.real, lw=1.6)
    axes[1, 0].plot(r_uniform, R_r.imag, lw=1.6)
    axes[2, 0].plot(r_uniform, np.abs(R_r), lw=1.6)

    axes[0, 0].set_ylabel("Re(R_in)")
    axes[1, 0].set_ylabel("Im(R_in)")
    axes[2, 0].set_ylabel("|R_in|")
    axes[2, 0].set_xlabel("r")
    axes[0, 0].set_title(f"full R_in on uniform r-grid\nr in [{r_min:.6g}, {r_max:.6g}]")

    # -------------------------------------------------
    # right column: reduced function Psi = R_in / P_Leaver
    # on y-grid (uniform or chebyshev)
    # -------------------------------------------------
    axes[0, 1].plot(y_grid, psi_y.real, lw=1.6)
    axes[1, 1].plot(y_grid, psi_y.imag, lw=1.6)
    axes[2, 1].plot(y_grid, np.abs(psi_y), lw=1.6)

    axes[0, 1].set_ylabel(r"Re($R_{\rm in}/P_{\rm Leaver}$)")
    axes[1, 1].set_ylabel(r"Im($R_{\rm in}/P_{\rm Leaver}$)")
    axes[2, 1].set_ylabel(r"$|R_{\rm in}/P_{\rm Leaver}|$")
    axes[2, 1].set_xlabel("y")
    axes[0, 1].set_title(f"reduced profile on {args.y_grid} y-grid")

    for ax in axes.ravel():
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"In-mode interpolant via Leaver-reduced profile\n"
        f"a={args.a}, omega={args.omega}, ell={args.ell}, m={args.m}, "
        f"N_in={args.N_in}, N_out={args.N_out}, z_m={args.z_m}, y_grid={args.y_grid}"
    )
    fig.tight_layout()

    out_dir = ROOT / "benchmark" / "outputs"/"R_in_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        out_path = out_dir / (
            f"inmode_profile_interp_a{args.a:.6f}_w{args.omega:.6f}"
            f"_Nin{args.N_in}_Nout{args.N_out}_zm{args.z_m:.3f}.png"
        )
    else:
        out_path = Path(args.out)

    print("[stage] save figure", flush=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {out_path}", flush=True)
    print(f"[info] B_inc={amp.B_inc}", flush=True)
    print(f"[info] B_ref={amp.B_ref}", flush=True)
    print(f"[info] B_trans={amp.B_trans}", flush=True)


if __name__ == "__main__":
    main()