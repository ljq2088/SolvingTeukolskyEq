"""
compute_pybhpt_solution.py — callable wrapper for pybhpt RadialTeukolsky.

Normalization: R_in ~ B^trans * Delta^2 * exp(-i*k*r*) as r -> r+
where B^trans ≈ 1 (validated for a∈[0.1,0.9], omega∈[0.01,1.0], l=m=2, s=-2).
"""
import argparse
import math
import multiprocessing as mp
import numpy as np


def build_chebyshev_x_grid(
    n_points: int = 1000,
    x_min: float = 1.0e-4,
    x_max: float = 1.0 - 1.0e-4,
) -> np.ndarray:
    """Chebyshev nodes on x=r_+/r in the open interval (0, 1), endpoints avoided."""
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if not (0.0 < x_min < x_max < 1.0):
        raise ValueError(f"Require 0 < x_min < x_max < 1, got {x_min}, {x_max}")

    k = np.arange(n_points, dtype=float)
    x_cheb_std = np.cos(np.pi * (k + 0.5) / n_points)  # avoids endpoints
    x = 0.5 * (x_max + x_min) + 0.5 * (x_max - x_min) * x_cheb_std
    return np.sort(x.astype(np.float64))


def x_to_r_grid(x_grid: np.ndarray, a: float) -> np.ndarray:
    rp = 1.0 + math.sqrt(1.0 - a * a)
    x_grid = np.asarray(x_grid, dtype=float)
    return rp / x_grid


def _worker(queue, a, omega, ell, m, r_grid):
    try:
        from pybhpt.radial import RadialTeukolsky
        rad = RadialTeukolsky(s=-2, j=ell, m=m, a=a, omega=omega, r=r_grid.tolist())
        rad.solve()
        vals = np.array(rad.radialsolutions('In'), dtype=complex)
        queue.put({'ok': True, 're': vals.real.tolist(), 'im': vals.imag.tolist()})
    except Exception as e:
        queue.put({'ok': False, 'err': str(e)[:120]})


def compute_pybhpt_solution(a, omega, ell=2, m=2, r_grid=None, timeout=30.0):
    """
    Compute pybhpt In-mode radial solution R_in(r) for Kerr s=-2 Teukolsky.

    Parameters
    ----------
    a : float
        Kerr spin parameter (M=1 units), a ∈ [0, 0.999).
    omega : float
        GW frequency (M=1 units).
    ell : int
        Spheroidal harmonic l (default 2).
    m : int
        Azimuthal mode number (default 2).
    r_grid : array-like or None
        Radial evaluation points. Default: near-horizon grid r ∈ [r+ + 1e-4, r+ + 1.0].
    timeout : float
        Subprocess timeout in seconds (default 30).

    Returns
    -------
    r_values : np.ndarray, shape (N,), dtype float64
        Radial coordinate values.
    R_in_values : np.ndarray, shape (N,), dtype complex128
        R_in solution values.

    Raises
    ------
    RuntimeError
        If pybhpt fails or times out.

    Notes
    -----
    Normalization: R_in ~ B^trans * Delta^2 * exp(-i*k*r*) as r -> r+,
    with B^trans ≈ 1 (horizon-normalized). This differs from null-cheb convention
    where R_in ~ B_inc * r^{-1} * exp(-i*omega*r*) at infinity.
    B_ref (reflection amplitude) is directly comparable between pybhpt and null-cheb.
    B_inc is NOT directly comparable (normalization factor varies with a, omega).
    """
    rp = 1.0 + math.sqrt(1.0 - a * a)
    if r_grid is None:
        r_grid = np.linspace(rp + 1e-4, rp + 1.0, 20)
    r_grid = np.asarray(r_grid, dtype=float)

    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(q, a, omega, ell, m, r_grid), daemon=True)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join(1.0)
        raise RuntimeError(f"pybhpt timeout after {timeout}s for a={a}, omega={omega}")
    if q.empty():
        raise RuntimeError(f"pybhpt returned no result for a={a}, omega={omega}")
    res = q.get()
    if not res['ok']:
        raise RuntimeError(f"pybhpt failed for a={a}, omega={omega}: {res['err']}")
    R_in = np.array(res['re']) + 1j * np.array(res['im'])
    return r_grid, R_in


def plot_solution_on_x_grid(
    a: float,
    omega: float,
    ell: int = 2,
    m: int = 2,
    n_points: int = 1000,
    x_min: float = 1.0e-4,
    x_max: float = 1.0 - 1.0e-4,
    timeout: float = 30.0,
    out_path: str = "pybhpt_solution_on_x_chebyshev.png",
):
    import matplotlib.pyplot as plt

    x_grid = build_chebyshev_x_grid(n_points=n_points, x_min=x_min, x_max=x_max)
    r_grid = x_to_r_grid(x_grid, a)
    _, R_in = compute_pybhpt_solution(a, omega, ell=ell, m=m, r_grid=r_grid, timeout=timeout)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(x_grid, R_in.real, lw=1.0)
    axes[0].set_ylabel("Re(R_in)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x_grid, R_in.imag, lw=1.0)
    axes[1].set_ylabel("Im(R_in)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(x_grid, np.abs(R_in), lw=1.0)
    axes[2].set_xlabel(r"$x=r_+/r$")
    axes[2].set_ylabel(r"$|R_{in}|$")
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        f"pybhpt R_in on Chebyshev x-grid\n"
        f"a={a}, omega={omega}, l={ell}, m={m}, n={n_points}, x∈[{x_min}, {x_max}]"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return x_grid, r_grid, R_in, out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--omega", type=float, default=0.3)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--plot-x-cheb", action="store_true")
    parser.add_argument("--n-points", type=int, default=1000)
    parser.add_argument("--x-min", type=float, default=1.0e-4)
    parser.add_argument("--x-max", type=float, default=1.0 - 1.0e-4)
    parser.add_argument("--out", type=str, default="pybhpt_solution_on_x_chebyshev.png")
    args = parser.parse_args()

    if args.plot_x_cheb:
        x, r, R, out_path = plot_solution_on_x_grid(
            a=args.a,
            omega=args.omega,
            ell=args.ell,
            m=args.m,
            n_points=args.n_points,
            x_min=args.x_min,
            x_max=args.x_max,
            timeout=args.timeout,
            out_path=args.out,
        )
        print(
            f"a={args.a}, omega={args.omega}: plotted {len(x)} Chebyshev x-points, "
            f"x∈[{x[0]:.6e}, {x[-1]:.6e}], saved -> {out_path}"
        )
    else:
        r, R = compute_pybhpt_solution(args.a, args.omega, ell=args.ell, m=args.m, timeout=args.timeout)
        print(f"a={args.a}, omega={args.omega}: {len(r)} points, |R_in[0]|={abs(R[0]):.4e}")
