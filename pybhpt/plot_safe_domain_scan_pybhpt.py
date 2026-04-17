from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import signal
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parents[3]
MATCHER_ROOT = CODE_ROOT / "radial_flow" / "spec_flow_method_Kerr" / "kerr_matcher_project"
MATCHER_SRC_ROOT = MATCHER_ROOT / "src"

for path in (PROJECT_ROOT, MATCHER_ROOT, MATCHER_SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from mma.rin_sampler import MathematicaRinSampler
from utils.amplitude_ratio import compute_amplitude_ratio
from utils.compute_lambda import compute_lambda


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
    rin_ref_mode: str
    mma_kernel_path: str
    mma_wl_path_win: str
    mma_timeout_sec: float
    pybhpt_timeout_sec: float


class HardTimeoutError(TimeoutError):
    pass


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


def r_minus_of_a(a: float, M: float = 1.0) -> float:
    return M - math.sqrt(max(M * M - a * a, 0.0))


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
    return np.concatenate([[left], mids, [right]])


def build_reference_radii(a: float, mode: str) -> np.ndarray:
    rp = r_plus_of_a(a)
    rm = r_minus_of_a(a)
    gap = max(rp - rm, 1.0e-8)

    if mode == "relative":
        raw = [
            rp + max(1.0e-4, 1.0e-2 * gap),
            rp + max(5.0e-4, 5.0e-2 * gap),
            1.05 * rp,
            1.5 * rp,
            max(8.0, 2.0 * rp),
            max(20.0, 5.0 * rp),
        ]
    elif mode == "fixed":
        raw = [
            rp + max(1.0e-4, 1.0e-2 * gap),
            max(2.2, 1.05 * rp),
            4.0,
            8.0,
            20.0,
            50.0,
        ]
    else:
        raise ValueError(f"Unknown rin_ref_mode={mode}")

    refs = sorted({float(r) for r in raw if r > rp})
    return np.asarray(refs, dtype=float)


def run_with_hard_timeout(timeout_sec: float, fn: Callable[[], np.ndarray]) -> np.ndarray:
    if timeout_sec <= 0.0:
        return fn()

    def _handle_timeout(signum, frame):
        raise HardTimeoutError(f"Python-side timeout after {timeout_sec:.3f} s")

    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout_sec)
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _pybhpt_worker(queue, s: int, l: int, m: int, a: float, omega: float, r_refs: np.ndarray):
    try:
        from pybhpt.radial import RadialTeukolsky

        rad = RadialTeukolsky(s=s, j=l, m=m, a=a, omega=omega, r=r_refs)
        rad.solve()

        vals = np.asarray(rad.radialsolutions("In"), dtype=np.complex128)
        lam = complex(getattr(rad, "eigenvalue", np.nan))
        queue.put(
            {
                "ok": True,
                "vals_real": vals.real.tolist(),
                "vals_imag": vals.imag.tolist(),
                "lambda_real": float(lam.real),
                "lambda_imag": float(lam.imag),
            }
        )
    except Exception as exc:
        queue.put(
            {
                "ok": False,
                "err_type": type(exc).__name__,
                "err": str(exc),
            }
        )


def solve_pybhpt_with_timeout(
    s: int,
    l: int,
    m: int,
    a: float,
    omega: float,
    r_refs: np.ndarray,
    timeout_sec: float,
):
    if timeout_sec <= 0.0:
        from pybhpt.radial import RadialTeukolsky

        rad = RadialTeukolsky(s=s, j=l, m=m, a=a, omega=omega, r=r_refs)
        rad.solve()
        vals = np.asarray(rad.radialsolutions("In"), dtype=np.complex128)
        lam = complex(getattr(rad, "eigenvalue", np.nan))
        return vals, lam

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_pybhpt_worker,
        args=(queue, s, l, m, a, omega, np.asarray(r_refs, dtype=float)),
        daemon=True,
    )
    proc.start()
    proc.join(timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join(1.0)
        raise HardTimeoutError(f"pybhpt subprocess timeout after {timeout_sec:.3f} s")

    if queue.empty():
        raise RuntimeError("pybhpt subprocess exited without returning a result")

    result = queue.get()
    if not result.get("ok", False):
        err_type = result.get("err_type", "RuntimeError")
        err = result.get("err", "")
        raise RuntimeError(f"{err_type}: {err}")

    vals = np.asarray(result["vals_real"], dtype=float) + 1j * np.asarray(result["vals_imag"], dtype=float)
    lam = complex(result["lambda_real"], result["lambda_imag"])
    return vals, lam


def should_reset_mma_session(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    transport_markers = (
        "socket exception",
        "failed to start",
        "socket operation aborted",
        "failed to read any message from socket",
        "wstp",
        "linkobject",
        "transport",
        "connection",
        "broken pipe",
    )
    return any(marker in text for marker in transport_markers)


def classify_lambda(a: float, omega: float, l: int, m: int, s: int) -> tuple[bool, str]:
    try:
        lam = complex(compute_lambda(a, omega, l, m, s))
        if not (math.isfinite(lam.real) and math.isfinite(lam.imag)):
            return False, "lambda_nonfinite"
        return True, "ok"
    except Exception as exc:
        return False, f"lambda:{type(exc).__name__}"


def classify_ramp(
    a: float,
    omega: float,
    l: int,
    m: int,
    s: int,
    r_match: float,
    n_cheb: int,
) -> tuple[bool, str]:
    try:
        lam = complex(compute_lambda(a, omega, l, m, s))
        if not (math.isfinite(lam.real) and math.isfinite(lam.imag)):
            return False, "lambda_nonfinite"
        out = compute_amplitude_ratio(
            a,
            omega,
            l,
            m,
            lambda_sep=lam,
            r_match=r_match,
            n_cheb=n_cheb,
            s=s,
        )
        ratio = complex(out["ratio"])
        if not (math.isfinite(ratio.real) and math.isfinite(ratio.imag)):
            return False, "ramp_nonfinite"
        return True, "ok"
    except Exception as exc:
        return False, f"ramp:{type(exc).__name__}"


def classify_mma(
    sampler: MathematicaRinSampler,
    a: float,
    omega: float,
    l: int,
    m: int,
    s: int,
    k_margin: float,
    rin_ref_mode: str,
    hard_timeout_sec: float,
    function_name: str,
    code_prefix: str,
) -> tuple[bool, str]:
    try:
        if abs(k_horizon(a, omega, m)) < k_margin:
            return False, f"{code_prefix}_k_horizon_margin"
        r_refs = build_reference_radii(a, mode=rin_ref_mode)
        vals = run_with_hard_timeout(
            hard_timeout_sec,
            lambda: sampler.evaluate_rin_at_points_direct(
                s=s,
                l=l,
                m=m,
                a=a,
                omega=omega,
                r_query=r_refs,
                function_name=function_name,
            ),
        )
        vals = np.asarray(vals, dtype=np.complex128)
        if not np.all(np.isfinite(vals.real) & np.isfinite(vals.imag)):
            return False, f"{code_prefix}_nonfinite"
        return True, "ok"
    except Exception as exc:
        if should_reset_mma_session(exc):
            sampler.close()
        return False, f"{code_prefix}:{type(exc).__name__}"


def classify_pybhpt(
    a: float,
    omega: float,
    l: int,
    m: int,
    s: int,
    k_margin: float,
    rin_ref_mode: str,
    hard_timeout_sec: float,
) -> tuple[bool, str]:
    try:
        if abs(k_horizon(a, omega, m)) < k_margin:
            return False, "pybhpt_k_horizon_margin"

        r_refs = build_reference_radii(a, mode=rin_ref_mode)
        vals, lam = solve_pybhpt_with_timeout(
            s=s,
            l=l,
            m=m,
            a=a,
            omega=omega,
            r_refs=r_refs,
            timeout_sec=hard_timeout_sec,
        )
        if vals.shape[0] != r_refs.shape[0]:
            return False, "pybhpt_length_mismatch"
        if not np.all(np.isfinite(vals.real) & np.isfinite(vals.imag)):
            return False, "pybhpt_nonfinite"

        if not (math.isfinite(lam.real) and math.isfinite(lam.imag)):
            return False, "pybhpt_lambda_nonfinite"

        return True, "ok"
    except Exception as exc:
        return False, f"pybhpt:{type(exc).__name__}"


def _partial_state_paths(output_dir: Path, mode_name: str) -> tuple[Path, Path]:
    return (
        output_dir / f"partial_{mode_name}.npz",
        output_dir / f"partial_{mode_name}_failures.json",
    )


def load_partial_state(
    output_dir: Path,
    mode_name: str,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[dict[str, float | str]]]]:
    npz_path, failure_path = _partial_state_paths(output_dir, mode_name)

    safe = np.zeros(shape, dtype=bool)
    done_mask = np.zeros(shape, dtype=bool)
    codes = np.full(shape, "", dtype=object)
    failure_examples: dict[str, list[dict[str, float | str]]] = {}

    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        safe = np.asarray(data["safe"], dtype=bool)
        done_mask = np.asarray(data["done_mask"], dtype=bool)
        codes = np.asarray(data["codes"], dtype=object)

    if failure_path.exists():
        with open(failure_path, "r", encoding="utf-8") as f:
            failure_examples = json.load(f)

    return safe, done_mask, codes, failure_examples


def save_partial_state(
    output_dir: Path,
    mode_name: str,
    safe: np.ndarray,
    done_mask: np.ndarray,
    codes: np.ndarray,
    failure_examples: dict[str, list[dict[str, float | str]]],
) -> None:
    npz_path, failure_path = _partial_state_paths(output_dir, mode_name)
    np.savez_compressed(npz_path, safe=safe, done_mask=done_mask, codes=codes)
    with open(failure_path, "w", encoding="utf-8") as f:
        json.dump(failure_examples, f, ensure_ascii=False, indent=2)


def remove_partial_state(output_dir: Path, mode_name: str) -> None:
    for path in _partial_state_paths(output_dir, mode_name):
        if path.exists():
            path.unlink()


def scan_mode(
    mode_name: str,
    classify_fn: Callable[[float, float], tuple[bool, str]],
    a_vals: np.ndarray,
    omega_vals: np.ndarray,
    output_dir: Path,
    resume: bool,
) -> tuple[np.ndarray, np.ndarray, Counter, dict[str, list[dict[str, float | str]]]]:
    shape = (len(a_vals), len(omega_vals))
    if resume:
        safe, done_mask, codes, failure_examples = load_partial_state(output_dir, mode_name, shape)
    else:
        safe = np.zeros(shape, dtype=bool)
        done_mask = np.zeros(shape, dtype=bool)
        codes = np.full(shape, "", dtype=object)
        failure_examples = {}

    counts: Counter = Counter(str(code) for code in codes[done_mask].tolist())
    total = len(a_vals) * len(omega_vals)
    done = int(done_mask.sum())

    for j, omega in enumerate(omega_vals):
        for i, a in enumerate(a_vals):
            if done_mask[i, j]:
                update_progress(f"[{mode_name}]", done, total)
                continue
            ok, code = classify_fn(float(a), float(omega))
            safe[i, j] = ok
            codes[i, j] = code
            done_mask[i, j] = True
            counts[code] += 1
            if not ok:
                lst = failure_examples.setdefault(str(code), [])
                if len(lst) < 12:
                    lst.append({"a": float(a), "omega": float(omega), "code": str(code)})
            done += 1
            if done % max(32, len(a_vals)) == 0 or i == len(a_vals) - 1:
                save_partial_state(output_dir, mode_name, safe, done_mask, codes, failure_examples)
            update_progress(f"[{mode_name}]", done, total)

    save_partial_state(output_dir, mode_name, safe, done_mask, codes, failure_examples)
    return safe, codes, counts, failure_examples


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
        else:
            labels.append(f"{w:.0f}")
    ax.set_yticklabels(labels)


def plot_four_modes(
    out_png: Path,
    settings: ScanSettings,
    a_vals: np.ndarray,
    eta_vals: np.ndarray,
    mode_data: list[tuple[str, np.ndarray, Counter]],
) -> None:
    a_edges = centers_to_edges(a_vals)
    eta_edges = centers_to_edges(eta_vals)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(20, 11),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()
    cmap = ListedColormap(["#d9d9d9", "#1b9e77"])

    k0 = np.array([float(settings.m) * omega_horizon(float(a)) for a in a_vals], dtype=float)
    k0_eta = eta_from_omega(k0, settings.omega_scale)
    km_minus = np.maximum(k0 - settings.k_margin, settings.omega_min)
    km_plus = k0 + settings.k_margin
    km_minus_eta = eta_from_omega(km_minus, settings.omega_scale)
    km_plus_eta = eta_from_omega(km_plus, settings.omega_scale)

    titles = {
        "lambda": r"$\lambda(a,\omega)$ finite",
        "ramp": r"$R_{\rm amp}$ finite",
        "mma_num": r"MMA Numerical $R_{\rm in}(r_{\rm ref})$ finite",
        "mma_mst": r"MMA MST $R_{\rm in}(r_{\rm ref})$ finite",
        "pybhpt": r"pybhpt $R_{\rm in}(r_{\rm ref})$ finite",
    }

    for ax, (name, safe_mask, counts) in zip(axes_flat, mode_data):
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
        ax.set_title(titles[name], fontsize=12)
        ax.set_xlim(a_vals.min(), a_vals.max())
        ax.set_ylim(eta_vals.min(), eta_vals.max())
        setup_omega_ticks(ax, settings.omega_scale)

        safe_n = int(np.count_nonzero(safe_mask))
        total_n = safe_mask.size
        ax.text(
            0.02,
            0.02,
            f"safe={safe_n}/{total_n}\nfail={total_n - safe_n}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

    for ax in axes_flat[len(mode_data):]:
        ax.axis("off")

    axes[0, 0].set_ylabel(r"compactified $\omega$: $\eta=\frac{2}{\pi}\arctan(\omega/\omega_s)$")
    axes[1, 0].set_ylabel(r"compactified $\omega$: $\eta=\frac{2}{\pi}\arctan(\omega/\omega_s)$")
    axes[1, 0].set_xlabel("a")
    axes[1, 1].set_xlabel("a")
    axes[1, 2].set_xlabel("a")
    axes[0, 0].legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.suptitle(
        (
            f"Safe-domain scan with aligned $R_{{in}}$ existence tests  "
            f"(l,m,s)=({settings.l},{settings.m},{settings.s}),  "
            f"a∈[{settings.a_min:.0e},{settings.a_max:.6f}],  "
            f"ω∈[{settings.omega_min:.0e},∞)"
        ),
        fontsize=14,
    )
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan five safe domains: lambda, ramp, MMA Numerical R_in, MMA MST R_in, and pybhpt R_in. "
            "For MMA and pybhpt, only existence/finiteness at shared reference radii is tested; "
            "normalization is not compared."
        )
    )
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--a-min", type=float, default=1.0e-5)
    parser.add_argument("--a-max", type=float, default=0.999999)
    parser.add_argument("--n-a", type=int, default=48)
    parser.add_argument("--omega-min", type=float, default=1.0e-6)
    parser.add_argument("--n-omega", type=int, default=72)
    parser.add_argument("--omega-scale", type=float, default=1.0)
    parser.add_argument("--eta-max", type=float, default=0.995)
    parser.add_argument("--k-margin", type=float, default=1.0e-2)
    parser.add_argument("--r-match", type=float, default=8.0)
    parser.add_argument("--n-cheb", type=int, default=32)
    parser.add_argument("--rin-ref-mode", choices=["relative", "fixed"], default="relative")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["lambda", "ramp", "mma_num", "mma_mst", "pybhpt"],
        default=["lambda", "ramp", "mma_num", "mma_mst", "pybhpt"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "pybhpt" / "outputs" / "safe_domain_scan",
    )
    parser.add_argument(
        "--mma-kernel-path",
        type=str,
        default="/mnt/f/mma/WolframKernel.exe",
    )
    parser.add_argument(
        "--mma-wl-path-win",
        type=str,
        default="F:/EMRI/Radial_flow/Radial_Function.wl",
    )
    parser.add_argument("--mma-timeout-sec", type=float, default=8.0)
    parser.add_argument("--pybhpt-timeout-sec", type=float, default=20.0)
    parser.add_argument("--no-resume", action="store_true")
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
        rin_ref_mode=args.rin_ref_mode,
        mma_kernel_path=args.mma_kernel_path,
        mma_wl_path_win=args.mma_wl_path_win,
        mma_timeout_sec=args.mma_timeout_sec,
        pybhpt_timeout_sec=args.pybhpt_timeout_sec,
    )

    a_vals = build_a_grid(args.a_min, args.a_max, args.n_a)
    eta_vals = build_eta_grid(args.omega_min, args.n_omega, args.omega_scale, args.eta_max)
    omega_vals = np.asarray(omega_from_eta(eta_vals, args.omega_scale), dtype=float)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[scan] modes={args.modes}, resume={not args.no_resume}, "
        f"mma_timeout={args.mma_timeout_sec}s, pybhpt_timeout={args.pybhpt_timeout_sec}s"
    )

    mode_results: list[tuple[str, np.ndarray, Counter]] = []
    mode_code_grids: dict[str, np.ndarray] = {}
    mode_fail_examples: dict[str, dict[str, list[dict[str, float | str]]]] = {}

    mma_sampler = None
    try:
        if "mma_num" in args.modes or "mma_mst" in args.modes:
            mma_sampler = MathematicaRinSampler(
                kernel_path=args.mma_kernel_path,
                wl_path_win=args.mma_wl_path_win,
                timeout_sec=args.mma_timeout_sec,
            )

        for mode_name in args.modes:
            if mode_name == "lambda":
                classify_fn = lambda a, omega: classify_lambda(a, omega, args.l, args.m, args.s)
            elif mode_name == "ramp":
                classify_fn = lambda a, omega: classify_ramp(
                    a,
                    omega,
                    args.l,
                    args.m,
                    args.s,
                    r_match=args.r_match,
                    n_cheb=args.n_cheb,
                )
            elif mode_name == "mma_num":
                if mma_sampler is None:
                    raise RuntimeError("Mathematica sampler was not initialized.")
                classify_fn = lambda a, omega: classify_mma(
                    mma_sampler,
                    a,
                    omega,
                    args.l,
                    args.m,
                    args.s,
                    k_margin=args.k_margin,
                    rin_ref_mode=args.rin_ref_mode,
                    hard_timeout_sec=max(2.0 * args.mma_timeout_sec, args.mma_timeout_sec + 5.0),
                    function_name="SampleRinAtPoints",
                    code_prefix="mma_num",
                )
            elif mode_name == "mma_mst":
                if mma_sampler is None:
                    raise RuntimeError("Mathematica sampler was not initialized.")
                classify_fn = lambda a, omega: classify_mma(
                    mma_sampler,
                    a,
                    omega,
                    args.l,
                    args.m,
                    args.s,
                    k_margin=args.k_margin,
                    rin_ref_mode=args.rin_ref_mode,
                    hard_timeout_sec=max(2.0 * args.mma_timeout_sec, args.mma_timeout_sec + 5.0),
                    function_name="SampleRinAtPointsMST",
                    code_prefix="mma_mst",
                )
            elif mode_name == "pybhpt":
                classify_fn = lambda a, omega: classify_pybhpt(
                    a,
                    omega,
                    args.l,
                    args.m,
                    args.s,
                    k_margin=args.k_margin,
                    rin_ref_mode=args.rin_ref_mode,
                    hard_timeout_sec=args.pybhpt_timeout_sec,
                )
            else:
                raise ValueError(f"Unknown mode {mode_name}")

            safe_mask, code_grid, counts, failure_examples = scan_mode(
                mode_name=mode_name,
                classify_fn=classify_fn,
                a_vals=a_vals,
                omega_vals=omega_vals,
                output_dir=out_dir,
                resume=(not args.no_resume),
            )
            mode_results.append((mode_name, safe_mask, counts))
            mode_code_grids[mode_name] = code_grid
            mode_fail_examples[mode_name] = failure_examples
            remove_partial_state(out_dir, mode_name)

    finally:
        if mma_sampler is not None:
            mma_sampler.close()

    name_to_result = {name: (mask, counts) for name, mask, counts in mode_results}
    plot_modes = []
    for name in ("lambda", "ramp", "mma_num", "mma_mst", "pybhpt"):
        if name in name_to_result:
            plot_modes.append((name, *name_to_result[name]))
        else:
            plot_modes.append(
                (
                    name,
                    np.zeros((len(a_vals), len(omega_vals)), dtype=bool),
                    Counter({"not_scanned": len(a_vals) * len(omega_vals)}),
                )
            )

    png_path = out_dir / "safe_domain_five_modes.png"
    plot_four_modes(
        out_png=png_path,
        settings=settings,
        a_vals=a_vals,
        eta_vals=eta_vals,
        mode_data=plot_modes,
    )

    np.savez_compressed(
        out_dir / "safe_domain_scan_data.npz",
        a_vals=a_vals,
        eta_vals=eta_vals,
        omega_vals=omega_vals,
        **{f"{name}_safe": mask for name, mask, _ in mode_results},
    )

    summary = {
        "settings": asdict(settings),
        "output_png": str(png_path),
        "counts": {
            name: {str(k): int(v) for k, v in counts.items()}
            for name, _, counts in mode_results
        },
        "failure_examples": mode_fail_examples,
        "normalization_note": (
            "For modes 'mma_num', 'mma_mst', and 'pybhpt', the scan only tests whether R_in can be computed "
            "as finite values at shared reference radii. It does not compare absolute normalization."
        ),
        "omega_tick_examples": [1.0e-6, 1.0e-4, 1.0e-2, 1.0, 10.0, 100.0],
    }
    with open(out_dir / "safe_domain_scan_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"saved figure -> {png_path}")
    print(f"saved summary -> {out_dir / 'safe_domain_scan_summary.json'}")
    for name, mask, counts in mode_results:
        print(f"[{name}] safe={int(mask.sum())}/{mask.size}, counts={dict(counts)}")


if __name__ == "__main__":
    main()
