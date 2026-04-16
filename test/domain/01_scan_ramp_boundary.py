from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
PINN_ROOT = REPO_ROOT.parent / "PINN" / "SolvingTeukolsky"

for path in (SRC_ROOT, REPO_ROOT, PINN_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from domain.safe_eval import try_compute_lambda_status, try_compute_ramp_status


@dataclass
class ScanSettings:
    l: int
    m: int
    s: int
    a_min: float
    a_max: float
    n_a: int
    omega_min: float
    n_omega: int
    omega_scale: float
    eta_max: float
    k_margin: float
    r_match: float
    n_cheb: int
    fill_gap_bins: int


def update_progress(prefix: str, done: int, total: int) -> None:
    width = 32
    frac = 0.0 if total <= 0 else done / total
    n_fill = min(width, max(0, int(round(width * frac))))
    bar = "#" * n_fill + "-" * (width - n_fill)
    sys.stdout.write(f"\r{prefix} [{bar}] {done}/{total} ({100.0 * frac:5.1f}%)")
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def r_plus_of_a(a: float, M: float = 1.0) -> float:
    return M + math.sqrt(max(M * M - a * a, 0.0))


def omega_horizon(a: float, M: float = 1.0) -> float:
    rp = r_plus_of_a(a, M=M)
    return a / (2.0 * M * rp)


def k_horizon(a: float, omega: float, m: int, M: float = 1.0) -> float:
    return omega - m * omega_horizon(a, M=M)


def build_a_grid(a_min: float, a_max: float, n_a: int) -> np.ndarray:
    eps_min = 1.0 - a_max
    eps_max = 1.0 - a_min
    eps_grid = np.geomspace(eps_min, eps_max, n_a)
    a_grid = 1.0 - eps_grid[::-1]
    a_grid[0] = a_min
    a_grid[-1] = a_max
    return a_grid


def eta_from_omega(omega: np.ndarray | float, omega_scale: float) -> np.ndarray | float:
    return (2.0 / math.pi) * np.arctan(np.asarray(omega) / omega_scale)


def omega_from_eta(eta: np.ndarray | float, omega_scale: float) -> np.ndarray | float:
    return omega_scale * np.tan(0.5 * math.pi * np.asarray(eta))


def build_eta_grid(omega_min: float, n_omega: int, omega_scale: float, eta_max: float) -> np.ndarray:
    eta_min = float(eta_from_omega(omega_min, omega_scale))
    return np.linspace(eta_min, eta_max, n_omega)


def centers_to_edges(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    if vals.ndim != 1 or len(vals) < 2:
        raise ValueError("Need at least two grid centers to build edges.")
    mids = 0.5 * (vals[:-1] + vals[1:])
    left = vals[0] - 0.5 * (vals[1] - vals[0])
    right = vals[-1] + 0.5 * (vals[-1] - vals[-2])
    edges = np.concatenate([[left], mids, [right]])
    return edges


def classify_lambda(a: float, omega: float, l: int, m: int, s: int) -> tuple[bool, str]:
    st = try_compute_lambda_status(a=a, omega=omega, l=l, m=m, s=s)
    return st.ok, st.code


def classify_ramp(
    a: float,
    omega: float,
    l: int,
    m: int,
    s: int,
    r_match: float,
    n_cheb: int,
) -> tuple[bool, str]:
    lam_status = try_compute_lambda_status(a=a, omega=omega, l=l, m=m, s=s)
    if not lam_status.ok:
        return False, lam_status.code

    st = try_compute_ramp_status(
        a=a,
        omega=omega,
        l=l,
        m=m,
        s=s,
        lambda_sep=lam_status.value,
        r_match=r_match,
        n_cheb=n_cheb,
    )
    return st.ok, st.code


def scan_mode(
    mode_name: str,
    classify_fn: Callable[[float, float], tuple[bool, str]],
    a_vals: np.ndarray,
    omega_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Counter, dict[str, list[dict[str, float | str]]]]:
    safe = np.zeros((len(a_vals), len(omega_vals)), dtype=bool)
    codes = np.empty((len(a_vals), len(omega_vals)), dtype=object)
    counts: Counter = Counter()
    failure_examples: dict[str, list[dict[str, float | str]]] = {}
    total = len(a_vals) * len(omega_vals)
    done = 0

    for j, omega in enumerate(omega_vals):
        for i, a in enumerate(a_vals):
            ok, code = classify_fn(float(a), float(omega))
            safe[i, j] = ok
            codes[i, j] = code
            counts[str(code)] += 1
            if not ok:
                lst = failure_examples.setdefault(str(code), [])
                if len(lst) < 12:
                    lst.append({"a": float(a), "omega": float(omega), "code": str(code)})
            done += 1
            update_progress(f"[{mode_name}]", done, total)

    return safe, codes, counts, failure_examples


def fill_small_false_gaps(mask_1d: np.ndarray, max_gap_bins: int) -> np.ndarray:
    """
    对 1D 布尔数组，把中间长度 <= max_gap_bins 的 False 小洞填成 True。
    """
    arr = mask_1d.astype(bool).copy()
    n = len(arr)
    i = 0
    while i < n:
        if arr[i]:
            i += 1
            continue
        j = i
        while j < n and not arr[j]:
            j += 1

        left_true = (i - 1 >= 0 and arr[i - 1])
        right_true = (j < n and arr[j])
        gap_len = j - i

        if left_true and right_true and gap_len <= max_gap_bins:
            arr[i:j] = True

        i = j
    return arr


def extract_boundary_from_mask(
    safe_mask: np.ndarray,
    omega_vals: np.ndarray,
    fill_gap_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对每个 a-index，提取 safe omega 区间的下/上边界。
    返回:
        omega_low, omega_high, valid_col
    """
    n_a, n_omega = safe_mask.shape
    omega_low = np.full(n_a, np.nan, dtype=float)
    omega_high = np.full(n_a, np.nan, dtype=float)
    valid_col = np.zeros(n_a, dtype=bool)

    for i in range(n_a):
        row = safe_mask[i, :]
        row_filled = fill_small_false_gaps(row, fill_gap_bins)

        idx = np.flatnonzero(row_filled)
        if idx.size == 0:
            continue

        valid_col[i] = True
        omega_low[i] = float(omega_vals[idx[0]])
        omega_high[i] = float(omega_vals[idx[-1]])

    return omega_low, omega_high, valid_col


def setup_omega_ticks(ax: plt.Axes, omega_scale: float) -> None:
    tick_omegas = np.array([1.0e-6, 1.0e-4, 1.0e-2, 1.0, 10.0, 100.0], dtype=float)
    tick_etas = eta_from_omega(tick_omegas, omega_scale)
    mask = (tick_etas >= ax.get_ylim()[0]) & (tick_etas <= ax.get_ylim()[1])
    ax.set_yticks(tick_etas[mask])
    labels = []
    for w in tick_omegas[mask]:
        if w < 1.0e-3:
            labels.append(f"{w:.0e}")
        elif w < 1.0:
            labels.append(f"{w:.2g}")
        elif w < 10.0:
            labels.append(f"{w:.0f}")
        else:
            labels.append(f"{w:.0f}")
    ax.set_yticklabels(labels)


def plot_boundary(
    out_png: Path,
    settings: ScanSettings,
    a_vals: np.ndarray,
    eta_vals: np.ndarray,
    lambda_safe: np.ndarray,
    ramp_safe: np.ndarray,
    omega_low: np.ndarray,
    omega_high: np.ndarray,
    valid_col: np.ndarray,
) -> None:
    a_edges = centers_to_edges(a_vals)
    eta_edges = centers_to_edges(eta_vals)
    cmap = ListedColormap(["#d9d9d9", "#1b9e77"])

    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6.5), sharey=True, constrained_layout=True
    )

    mode_items = [
        (axes[0], "lambda", lambda_safe, r"$\lambda(a,\omega)$ finite"),
        (axes[1], "ramp", ramp_safe, r"$R_{\rm amp}$ finite + boundary"),
    ]

    k0 = np.array([float(settings.m) * omega_horizon(float(a)) for a in a_vals], dtype=float)
    k0_eta = eta_from_omega(k0, settings.omega_scale)
    km_minus = np.maximum(k0 - settings.k_margin, settings.omega_min)
    km_plus = k0 + settings.k_margin
    km_minus_eta = eta_from_omega(km_minus, settings.omega_scale)
    km_plus_eta = eta_from_omega(km_plus, settings.omega_scale)

    for ax, name, safe_mask, title in mode_items:
        ax.pcolormesh(
            a_edges,
            eta_edges,
            safe_mask.T.astype(int),
            cmap=cmap,
            shading="auto",
            vmin=0,
            vmax=1,
        )
        ax.plot(a_vals, k0_eta, color="#2c3e50", lw=1.4, label=r"$k_H=0$")
        ax.plot(a_vals, km_minus_eta, color="#2c3e50", lw=0.9, ls="--", alpha=0.7, label=rf"$|k_H|={settings.k_margin:g}$")
        ax.plot(a_vals, km_plus_eta, color="#2c3e50", lw=0.9, ls="--", alpha=0.7)

        if name == "ramp":
            eta_low = eta_from_omega(np.where(valid_col, omega_low, np.nan), settings.omega_scale)
            eta_high = eta_from_omega(np.where(valid_col, omega_high, np.nan), settings.omega_scale)
            ax.plot(a_vals, eta_low, color="#e7298a", lw=1.8, label=r"$\omega_{\rm low}(a)$")
            ax.plot(a_vals, eta_high, color="#66a61e", lw=1.8, label=r"$\omega_{\rm high}(a)$")

        safe_n = int(np.count_nonzero(safe_mask))
        total_n = safe_mask.size
        ax.text(
            0.02, 0.02,
            f"safe={safe_n}/{total_n}\nfail={total_n-safe_n}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("a")
        ax.set_xlim(a_vals.min(), a_vals.max())
        ax.set_ylim(eta_vals.min(), eta_vals.max())
        setup_omega_ticks(ax, settings.omega_scale)
        ax.grid(alpha=0.15)

    axes[0].set_ylabel(r"compactified $\omega$: $\eta=\frac{2}{\pi}\arctan(\omega/\omega_s)$")
    axes[1].legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.suptitle(
        (
            f"R_amp feasible-domain boundary  "
            f"(l,m,s)=({settings.l},{settings.m},{settings.s}),  "
            f"a∈[{settings.a_min:.0e},{settings.a_max:.6f}],  "
            f"ω∈[{settings.omega_min:.0e},∞)"
        ),
        fontsize=14,
    )
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan R_amp feasible domain and extract boundary.")
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--a-min", type=float, default=1.0e-5)
    parser.add_argument("--a-max", type=float, default=0.999999)
    parser.add_argument("--n-a", type=int, default=64)
    parser.add_argument("--omega-min", type=float, default=1.0e-6)
    parser.add_argument("--n-omega", type=int, default=96)
    parser.add_argument("--omega-scale", type=float, default=1.0)
    parser.add_argument("--eta-max", type=float, default=0.995)
    parser.add_argument("--k-margin", type=float, default=1.0e-2)
    parser.add_argument("--r-match", type=float, default=8.0)
    parser.add_argument("--n-cheb", type=int, default=32)
    parser.add_argument("--fill-gap-bins", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tests" / "outputs" / "ramp_boundary_scan",
    )
    args = parser.parse_args()

    settings = ScanSettings(
        l=args.l,
        m=args.m,
        s=args.s,
        a_min=args.a_min,
        a_max=args.a_max,
        n_a=args.n_a,
        omega_min=args.omega_min,
        n_omega=args.n_omega,
        omega_scale=args.omega_scale,
        eta_max=args.eta_max,
        k_margin=args.k_margin,
        r_match=args.r_match,
        n_cheb=args.n_cheb,
        fill_gap_bins=args.fill_gap_bins,
    )

    a_vals = build_a_grid(args.a_min, args.a_max, args.n_a)
    eta_vals = build_eta_grid(args.omega_min, args.n_omega, args.omega_scale, args.eta_max)
    omega_vals = np.asarray(omega_from_eta(eta_vals, args.omega_scale), dtype=float)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lambda_safe, lambda_codes, lambda_counts, lambda_fail = scan_mode(
        "lambda",
        lambda a, omega: classify_lambda(a, omega, args.l, args.m, args.s),
        a_vals, omega_vals,
    )

    ramp_safe, ramp_codes, ramp_counts, ramp_fail = scan_mode(
        "ramp",
        lambda a, omega: classify_ramp(
            a, omega, args.l, args.m, args.s,
            r_match=args.r_match,
            n_cheb=args.n_cheb,
        ),
        a_vals, omega_vals,
    )

    omega_low, omega_high, valid_col = extract_boundary_from_mask(
        safe_mask=ramp_safe,
        omega_vals=omega_vals,
        fill_gap_bins=args.fill_gap_bins,
    )

    out_png = out_dir / "ramp_boundary_scan.png"
    plot_boundary(
        out_png=out_png,
        settings=settings,
        a_vals=a_vals,
        eta_vals=eta_vals,
        lambda_safe=lambda_safe,
        ramp_safe=ramp_safe,
        omega_low=omega_low,
        omega_high=omega_high,
        valid_col=valid_col,
    )

    np.savez_compressed(
        out_dir / "ramp_boundary_data.npz",
        a_vals=a_vals,
        eta_vals=eta_vals,
        omega_vals=omega_vals,
        lambda_safe=lambda_safe,
        ramp_safe=ramp_safe,
        omega_low=omega_low,
        omega_high=omega_high,
        valid_col=valid_col,
    )

    summary = {
        "settings": asdict(settings),
        "counts": {
            "lambda": {str(k): int(v) for k, v in lambda_counts.items()},
            "ramp": {str(k): int(v) for k, v in ramp_counts.items()},
        },
        "failure_examples": {
            "lambda": lambda_fail,
            "ramp": ramp_fail,
        },
        "boundary": {
            "n_valid_columns": int(np.count_nonzero(valid_col)),
            "n_invalid_columns": int(valid_col.size - np.count_nonzero(valid_col)),
        },
        "output_png": str(out_png),
        "output_npz": str(out_dir / "ramp_boundary_data.npz"),
    }
    with open(out_dir / "ramp_boundary_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"saved figure  -> {out_png}")
    print(f"saved data    -> {out_dir / 'ramp_boundary_data.npz'}")
    print(f"saved summary -> {out_dir / 'ramp_boundary_summary.json'}")
    print(f"[lambda] safe={int(lambda_safe.sum())}/{lambda_safe.size}, counts={dict(lambda_counts)}")
    print(f"[ramp]   safe={int(ramp_safe.sum())}/{ramp_safe.size}, counts={dict(ramp_counts)}")


if __name__ == "__main__":
    main()