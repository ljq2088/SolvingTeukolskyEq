from __future__ import annotations
import random
import sys

sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset.sampling import sample_points_chebyshev_grid
from domain.atlas_builder import map_from_chart
from inference.atlas_predictor import AtlasPredictor
from mma.rin_sampler import MathematicaRinSampler
from physical_ansatz.mapping import r_plus
from physical_ansatz.transform_y import g_factor, h_factor
from physical_ansatz.prefactor import Leaver_prefactors, prefactor_Q, U_prefactor
from physical_ansatz.residual import AuxCache, get_ramp_and_p_from_cfg


def default_parameter_bounds(predictor: AtlasPredictor) -> tuple[float, float, float, float]:
    comp = predictor.component
    a_min = float(comp.a_support[0])
    a_max = float(comp.a_support[-1])
    omega_min = float(min(comp.omega_lower))
    omega_max = float(max(comp.omega_upper))
    return a_min, a_max, omega_min, omega_max


def build_y_grid(n_y: int, y_min: float, y_max: float, mode: str, device, dtype):
    if mode == "chebyshev":
        return sample_points_chebyshev_grid(
            n_points=n_y,
            y_min=y_min,
            y_max=y_max,
            device=device,
            dtype=dtype,
        )
    return torch.linspace(y_min, y_max, n_y, device=device, dtype=dtype)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_recoverable_mma_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "failed to parse mathematica",
        "timed out",
        "mathematica evaluation returned",
    )
    return any(marker in text for marker in markers)


def should_reset_mma_session(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "failed to communicate with kernel",
        "socket exception",
        "failed to start",
        "socket operation aborted",
        "failed to read any message from socket",
        "broken pipe",
        "wstp",
        "linkobject",
        "transport",
        "connection",
    )
    return any(marker in text for marker in markers)


def evaluate_one_point(
    predictor: AtlasPredictor,
    sampler: MathematicaRinSampler,
    a_eval: float,
    omega_eval: float,
    n_r: int,
    n_y: int,
    y_grid_mode: str,
    viz_r_min: float,
    viz_r_max: float,
):
    full_cfg = predictor.full_cfg
    physics_cfg = predictor.physics_cfg
    problem_cfg = physics_cfg["problem"]

    M = float(problem_cfg.get("M", 1.0))
    l = int(problem_cfg.get("l", 2))
    m = int(problem_cfg.get("m", 2))
    s = int(problem_cfg.get("s", -2))

    dtype = predictor.dtype
    device = predictor.device

    a_t = torch.tensor(float(a_eval), device=device, dtype=dtype)
    omega_t = torch.tensor(float(omega_eval), device=device, dtype=dtype)

    rp = r_plus(a_t, M)
    r_min = max(float(viz_r_min), float(rp.detach().cpu().item()) + 1.0e-4)
    r_max = float(viz_r_max)

    r_grid = torch.linspace(r_min, r_max, n_r, device=device, dtype=dtype)
    x_grid = rp / r_grid
    y_grid_from_r = 2.0 * x_grid - 1.0

    y_min = 2.0 * float(rp.detach().cpu().item()) / r_max - 1.0
    y_max = 2.0 * float(rp.detach().cpu().item()) / r_min - 1.0
    y_grid = build_y_grid(n_y, y_min, y_max, y_grid_mode, device, dtype)
    x_grid_from_y = 0.5 * (y_grid + 1.0)
    r_grid_from_y = rp / x_grid_from_y

    f_pred_r = predictor.predict_f(a=float(a_eval), omega=float(omega_eval), y=y_grid_from_r)
    f_pred_y = predictor.predict_f(a=float(a_eval), omega=float(omega_eval), y=y_grid)

    g_val, _, _ = g_factor(x_grid)
    h = h_factor(a_t, omega_t, m=m, M=M, s=s)
    Rprime_pred_r = g_val * f_pred_r + h

    g_val_y, _, _ = g_factor(x_grid_from_y)
    Rprime_pred_y = g_val_y * f_pred_y + h

    cache = AuxCache()
    p, ramp = get_ramp_and_p_from_cfg(physics_cfg, cache, a_t, omega_t)
    ramp_t = ramp.to(device=device, dtype=torch.complex128)

    P, P_r, P_rr = Leaver_prefactors(r_grid, a_t, omega_t, m=m, M=M, s=s)
    Q, Q_r, Q_rr = prefactor_Q(
        r_grid,
        a_t,
        omega_t,
        p=int(p),
        R_amp=ramp_t,
        M=M,
        s=s,
    )
    U, _, _ = U_prefactor(P, P_r, P_rr, Q, Q_r, Q_rr)
    R_pred = U * Rprime_pred_r

    P_y, P_y_r, P_y_rr = Leaver_prefactors(r_grid_from_y, a_t, omega_t, m=m, M=M, s=s)
    Q_y, Q_y_r, Q_y_rr = prefactor_Q(
        r_grid_from_y,
        a_t,
        omega_t,
        p=int(p),
        R_amp=ramp_t,
        M=M,
        s=s,
    )
    U_y, _, _ = U_prefactor(P_y, P_y_r, P_y_rr, Q_y, Q_y_r, Q_y_rr)

    try:
        R_mma = sampler.evaluate_rin_at_points_direct(
            s=s,
            l=l,
            m=m,
            a=float(a_eval),
            omega=float(omega_eval),
            r_query=r_grid.detach().cpu().numpy(),
        )
        R_mma_y = sampler.evaluate_rin_at_points_direct(
            s=s,
            l=l,
            m=m,
            a=float(a_eval),
            omega=float(omega_eval),
            r_query=r_grid_from_y.detach().cpu().numpy(),
        )
    except Exception as exc:
        if not is_recoverable_mma_error(exc):
            raise
        if should_reset_mma_session(exc):
            sampler.close()
        R_mma = sampler.evaluate_rin_at_points_direct(
            s=s,
            l=l,
            m=m,
            a=float(a_eval),
            omega=float(omega_eval),
            r_query=r_grid.detach().cpu().numpy(),
        )
        R_mma_y = sampler.evaluate_rin_at_points_direct(
            s=s,
            l=l,
            m=m,
            a=float(a_eval),
            omega=float(omega_eval),
            r_query=r_grid_from_y.detach().cpu().numpy(),
        )

    R_mma_t = torch.as_tensor(R_mma, device=device, dtype=torch.complex128)
    Rprime_mma_y = torch.as_tensor(R_mma_y, device=device, dtype=torch.complex128) / U_y

    eps = 1.0e-12
    rel_R_l2 = torch.linalg.norm(R_pred - R_mma_t) / (torch.linalg.norm(R_mma_t) + eps)
    rel_Rprime_l2 = torch.linalg.norm(Rprime_pred_y - Rprime_mma_y) / (torch.linalg.norm(Rprime_mma_y) + eps)
    abs_R_max = torch.max(torch.abs(R_pred - R_mma_t))
    abs_Rprime_max = torch.max(torch.abs(Rprime_pred_y - Rprime_mma_y))
    rel_R_pointwise = torch.abs(R_pred - R_mma_t) / (torch.abs(R_mma_t) + eps)
    rel_Rprime_pointwise = torch.abs(Rprime_pred_y - Rprime_mma_y) / (torch.abs(Rprime_mma_y) + eps)
    rel_R_pointwise_mean = torch.mean(rel_R_pointwise)
    rel_Rprime_pointwise_mean = torch.mean(rel_Rprime_pointwise)

    summary = {
        "a": float(a_eval),
        "omega": float(omega_eval),
        "n_r": int(n_r),
        "n_y": int(n_y),
        "y_grid_mode": str(y_grid_mode),
        "r_min": float(r_min),
        "r_max": float(r_max),
        "rel_R_l2": float(rel_R_l2.detach().cpu().item()),
        "rel_Rprime_l2": float(rel_Rprime_l2.detach().cpu().item()),
        "rel_R_pointwise_mean": float(rel_R_pointwise_mean.detach().cpu().item()),
        "rel_Rprime_pointwise_mean": float(rel_Rprime_pointwise_mean.detach().cpu().item()),
        "abs_R_max": float(abs_R_max.detach().cpu().item()),
        "abs_Rprime_max": float(abs_Rprime_max.detach().cpu().item()),
    }

    arrays = {
        "r_grid": r_grid.detach().cpu().numpy(),
        "y_grid": y_grid.detach().cpu().numpy(),
        "r_grid_from_y": r_grid_from_y.detach().cpu().numpy(),
        "R_pred": R_pred.detach().cpu().numpy(),
        "R_mma": np.asarray(R_mma),
        "Rprime_pred_y": Rprime_pred_y.detach().cpu().numpy(),
        "Rprime_mma_y": Rprime_mma_y.detach().cpu().numpy(),
        "rel_R_pointwise": rel_R_pointwise.detach().cpu().numpy(),
        "rel_Rprime_pointwise": rel_Rprime_pointwise.detach().cpu().numpy(),
    }
    return summary, arrays


def save_comparison_figure(out_path: Path, arrays: dict, summary: dict) -> None:
    r_np = arrays["r_grid"]
    y_np = arrays["y_grid"]
    R_pred_np = arrays["R_pred"]
    R_mma_np = arrays["R_mma"]
    Rprime_pred_np = arrays["Rprime_pred_y"]
    Rprime_mma_np = arrays["Rprime_mma_y"]
    rel_R_np = arrays["rel_R_pointwise"]
    rel_Rprime_np = arrays["rel_Rprime_pointwise"]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=False)

    axes[0, 0].plot(r_np, np.real(R_pred_np), label="TeukolskySolver Re(R)", lw=1.6)
    axes[0, 0].plot(r_np, np.real(R_mma_np), "--", label="MMA Re(R)", lw=1.0)
    axes[0, 0].set_ylabel("Re(R)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[1, 0].plot(r_np, np.imag(R_pred_np), label="TeukolskySolver Im(R)", lw=1.6)
    axes[1, 0].plot(r_np, np.imag(R_mma_np), "--", label="MMA Im(R)", lw=1.0)
    axes[1, 0].set_ylabel("Im(R)")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[2, 0].plot(r_np, np.abs(R_pred_np), label="TeukolskySolver |R|", lw=1.6)
    axes[2, 0].plot(r_np, np.abs(R_mma_np), "--", label="MMA |R|", lw=1.0)
    axes[2, 0].set_ylabel("|R|")
    axes[2, 0].set_xlabel("r")
    axes[2, 0].legend()
    axes[2, 0].grid(alpha=0.3)

    axes[0, 1].plot(y_np, np.real(Rprime_pred_np), label="TeukolskySolver Re(R')", lw=1.6)
    axes[0, 1].plot(y_np, np.real(Rprime_mma_np), "--", label="MMA Re(R')", lw=1.0)
    axes[0, 1].set_ylabel("Re(R')")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 1].plot(y_np, np.imag(Rprime_pred_np), label="TeukolskySolver Im(R')", lw=1.6)
    axes[1, 1].plot(y_np, np.imag(Rprime_mma_np), "--", label="MMA Im(R')", lw=1.0)
    axes[1, 1].set_ylabel("Im(R')")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    axes[2, 1].plot(y_np, np.abs(Rprime_pred_np), label="TeukolskySolver |R'|", lw=1.6)
    axes[2, 1].plot(y_np, np.abs(Rprime_mma_np), "--", label="MMA |R'|", lw=1.0)
    axes[2, 1].set_ylabel("|R'|")
    axes[2, 1].set_xlabel("y")
    axes[2, 1].legend()
    axes[2, 1].grid(alpha=0.3)

    axes[0, 2].plot(
        r_np,
        rel_R_np,
        label=f"mean={summary['rel_R_pointwise_mean']:.3e}",
        lw=1.4,
        color="#d95f02",
    )
    axes[0, 2].set_ylabel("RelErr(R)")
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    axes[1, 2].plot(
        y_np,
        rel_Rprime_np,
        label=f"mean={summary['rel_Rprime_pointwise_mean']:.3e}",
        lw=1.4,
        color="#7570b3",
    )
    axes[1, 2].set_ylabel("RelErr(R')")
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)

    axes[2, 2].axis("off")
    axes[2, 2].text(
        0.02,
        0.95,
        "\n".join(
            [
                f"mean rel err R   = {summary['rel_R_pointwise_mean']:.3e}",
                f"mean rel err R'  = {summary['rel_Rprime_pointwise_mean']:.3e}",
                f"L2 rel err R     = {summary['rel_R_l2']:.3e}",
                f"L2 rel err R'    = {summary['rel_Rprime_l2']:.3e}",
                f"max abs err R    = {summary['abs_R_max']:.3e}",
                f"max abs err R'   = {summary['abs_Rprime_max']:.3e}",
            ]
        ),
        transform=axes[2, 2].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    fig.suptitle(
        f"TeukolskySolver vs MMA\n"
        f"a={summary['a']:.6f}, omega={summary['omega']:.6f}, "
        f"mean_rel_R={summary['rel_R_pointwise_mean']:.3e}, "
        f"mean_rel_R'={summary['rel_Rprime_pointwise_mean']:.3e}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def repair_missing_figures(out_dir: Path) -> int:
    repaired = 0
    for sample_dir in sorted(out_dir.glob("sample_*")):
        if not sample_dir.is_dir():
            continue
        npz_path = sample_dir / "benchmark_data.npz"
        summary_path = sample_dir / "summary.json"
        fig_path = sample_dir / "teukolsky_solver_vs_mma.png"
        if not npz_path.exists() or not summary_path.exists() or fig_path.exists():
            continue

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            npz = np.load(npz_path, allow_pickle=False)
            arrays = {key: npz[key] for key in npz.files}
            save_comparison_figure(fig_path, arrays, summary)
            repaired += 1
        except Exception as exc:
            print(f"[repair] failed for {sample_dir.name}: {type(exc).__name__}: {exc}")
    return repaired


def save_scatter_figure(out_path: Path, records: list[dict], a_min: float, a_max: float, omega_min: float, omega_max: float) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    success = [rec for rec in records if rec["status"] == "ok"]
    failed = [rec for rec in records if rec["status"] != "ok"]

    if failed:
        ax.scatter(
            [rec["a"] for rec in failed],
            [rec["omega"] for rec in failed],
            s=24,
            c="#d95f02",
            marker="x",
            label=f"failed ({len(failed)})",
        )
    if success:
        ax.scatter(
            [rec["a"] for rec in success],
            [rec["omega"] for rec in success],
            s=28,
            c="#1b9e77",
            label=f"ok ({len(success)})",
        )

    ax.set_xlim(a_min, a_max)
    ax.set_ylim(omega_min, omega_max)
    ax.set_xlabel("a")
    ax.set_ylabel("omega")
    ax.set_title("Random Parameter Scan for TeukolskySolver vs MMA")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Random parameter scan benchmark for TeukolskySolver vs MMA.")
    parser.add_argument("--registry-json", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--a-min", type=float, default=None)
    parser.add_argument("--a-max", type=float, default=None)
    parser.add_argument("--omega-min", type=float, default=None)
    parser.add_argument("--omega-max", type=float, default=None)
    parser.add_argument("--n-r", type=int, default=300)
    parser.add_argument("--n-y", type=int, default=800)
    parser.add_argument("--y-grid-mode", choices=["chebyshev", "uniform"], default="chebyshev")
    parser.add_argument("--viz-r-min", type=float, default=2.0)
    parser.add_argument("--viz-r-max", type=float, default=200.0)
    parser.add_argument("--out-dir", type=str, default="outputs/teukolsky_solver_random_scan")
    parser.add_argument("--repair-only", action="store_true")
    args = parser.parse_args()

    set_global_seed(args.seed)

    registry_path = Path(args.registry_json).resolve()
    predictor = AtlasPredictor(str(registry_path), device=args.device)
    full_cfg = predictor.full_cfg
    mma_cfg = full_cfg["train"]["mathematica"]

    a0, a1, w0, w1 = default_parameter_bounds(predictor)
    a_min = float(a0 if args.a_min is None else args.a_min)
    a_max = float(a1 if args.a_max is None else args.a_max)
    omega_min = float(w0 if args.omega_min is None else args.omega_min)
    omega_max = float(w1 if args.omega_max is None else args.omega_max)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.repair_only:
        repaired = repair_missing_figures(out_dir)
        print(f"[repair] repaired {repaired} figure(s) under {out_dir}")
        return

    rng = np.random.default_rng(args.seed)
    sampler = MathematicaRinSampler(
        kernel_path=mma_cfg["kernel_path"],
        wl_path_win=mma_cfg["wl_path_win"],
    )

    records = []
    try:
        for idx in range(args.n_samples):
            u = float(rng.uniform(0.0, 1.0))
            v = float(rng.uniform(0.0, 1.0))
            a_eval, omega_eval = map_from_chart(predictor.component, u, v)

            if not (a_min <= a_eval <= a_max and omega_min <= omega_eval <= omega_max):
                record = {
                    "index": int(idx),
                    "a": float(a_eval),
                    "omega": float(omega_eval),
                    "status": "failed",
                    "error": "sample_outside_user_bounds",
                }
                records.append(record)
                continue

            sample_dir = out_dir / f"sample_{idx:03d}_a_{a_eval:.6f}_omega_{omega_eval:.6f}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            record = {
                "index": int(idx),
                "a": a_eval,
                "omega": omega_eval,
                "status": "ok",
            }

            try:
                summary, arrays = evaluate_one_point(
                    predictor=predictor,
                    sampler=sampler,
                    a_eval=a_eval,
                    omega_eval=omega_eval,
                    n_r=args.n_r,
                    n_y=args.n_y,
                    y_grid_mode=args.y_grid_mode,
                    viz_r_min=args.viz_r_min,
                    viz_r_max=args.viz_r_max,
                )
                with open(sample_dir / "summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                np.savez_compressed(sample_dir / "benchmark_data.npz", **arrays)
                save_comparison_figure(sample_dir / "teukolsky_solver_vs_mma.png", arrays, summary)
                record.update(summary)
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                with open(sample_dir / "error.json", "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)

            records.append(record)
            with open(out_dir / "scan_records.json", "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
    finally:
        sampler.close()

    save_scatter_figure(
        out_path=out_dir / "random_scatter.png",
        records=records,
        a_min=a_min,
        a_max=a_max,
        omega_min=omega_min,
        omega_max=omega_max,
    )
    repaired = repair_missing_figures(out_dir)

    summary = {
        "registry_json": str(registry_path),
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "a_min": a_min,
        "a_max": a_max,
        "omega_min": omega_min,
        "omega_max": omega_max,
        "n_r": int(args.n_r),
        "n_y": int(args.n_y),
        "y_grid_mode": str(args.y_grid_mode),
        "n_success": int(sum(rec["status"] == "ok" for rec in records)),
        "n_failed": int(sum(rec["status"] != "ok" for rec in records)),
        "repaired_missing_figures": int(repaired),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved scatter -> {out_dir / 'random_scatter.png'}")
    print(f"saved records -> {out_dir / 'scan_records.json'}")


if __name__ == "__main__":
    main()
