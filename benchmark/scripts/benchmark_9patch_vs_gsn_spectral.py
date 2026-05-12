"""
Benchmark 9 atlas-patch PINN models against GSN and spectral methods for R_in.

Compares Re(R_in), Im(R_in), |R_in| in r-space and y-space (Leaver factor removed).
Produces 6-subplot figures, records a-ω-r relative error matrices, and computes
comprehensive statistics to identify underperforming patches.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
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
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.config_loader import load_pinn_full_config
from domain.patch_cover import load_patch_cover, load_valid_chart_points
from model.pinn_mlp import PINN_MLP
from model.cheb_coeff_net import ChebCoeffNet
from physical_ansatz.prefactor import Leaver_prefactors
from physical_ansatz.transform_y import (
    compose_reduced_shape_from_f,
    h_factor,
    horizon_regularity_slope,
)
from utils.mode import KerrMode
from utils.amplitude_profile_three_patch import TeukRadAmplitudeIn3PatchWithProfile

GSN_SCRIPT = ROOT / "benchmark" / "scripts" / "gsn_ref_rin.jl"
JULIA_BIN = "/home/ljq/julia-1.10.7/bin/julia"


def _get_dtype(dtype_name: str):
    return torch.float32 if str(dtype_name).lower() == "float32" else torch.float64


def slug_float_short(x: float) -> str:
    return f"{x:.6g}".replace("+", "").replace("-", "m").replace(".", "p")


def safe_rel_err(x: np.ndarray, y: np.ndarray, floor: float = 1.0e-14) -> np.ndarray:
    return np.abs(x - y) / np.maximum(np.abs(y), floor)


def resolve_maybe_relative(path_str: str, repo_root: Path, registry_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    for cand in [repo_root / p, registry_dir / p]:
        if cand.resolve().exists():
            return cand.resolve()
    return (repo_root / p).resolve()


def gsn_eval_rin(
    s: int, l: int, m: int, a: float, omega: float,
    r_grid: np.ndarray, M: float = 1.0, timeout: float = 300.0,
) -> np.ndarray | None:
    """Call GSN.jl to evaluate R_in on r_grid. Returns complex array or None on failure."""
    r_str = ",".join(f"{rv:.16g}" for rv in r_grid)
    cmd = [
        JULIA_BIN, "--project=" + str(Path("/home/ljq/code/GSN/GeneralizedSasakiNakamura.jl")),
        str(GSN_SCRIPT),
        f"--s={s}", f"--l={l}", f"--m={m}",
        f"--a={a}", f"--omega={omega}", f"--M={M}",
        f"--r-list={r_str}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            print(f"  [gsn] FAILED: {proc.stderr[-200:]}", file=sys.stderr, flush=True)
            return None
        for line in proc.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                rec = json.loads(line)
                return np.asarray(rec["Rin_re"]) + 1j * np.asarray(rec["Rin_im"])
        return None
    except Exception as e:
        print(f"  [gsn] exception: {e}", file=sys.stderr, flush=True)
        return None


def build_model_for_patch(full_cfg: dict, patch, device: torch.device, model_type: str = "pinn_mlp", cheb_N: int = 64):
    physics_cfg = full_cfg["physics"]
    train_cfg = full_cfg["train"]
    model_cfg = train_cfg.get("model", {})
    runtime_cfg = train_cfg.get("runtime", {})
    dtype = _get_dtype(runtime_cfg.get("dtype", "float64"))
    problem_cfg = physics_cfg["problem"]
    M = float(problem_cfg.get("M", 1.0))
    m_mode = int(problem_cfg.get("m", 2))

    if model_type == "cheb":
        model = ChebCoeffNet(
            N=cheb_N,
            hidden_dims=model_cfg.get("hidden_dims", [128, 256, 256, 128]),
            activation=model_cfg.get("activation", "silu"),
            param_embed_dim=model_cfg.get("param_embed_dim", 64),
            local_coord_mode="chart_uv",
            u_center_local=float(patch.u_center),
            v_center_local=float(patch.v_center),
            u_half_range_local=float(patch.h_u),
            v_half_range_local=float(patch.h_v),
            M=M,
            m_mode=m_mode,
        ).to(device=device, dtype=dtype)
    else:
        model = PINN_MLP(
            hidden_dims=model_cfg.get("hidden_dims", [128, 128, 128, 128]),
            activation=model_cfg.get("activation", "silu"),
            fourier_num_freqs=model_cfg.get("fourier_num_freqs", 2),
            fourier_scale=model_cfg.get("fourier_scale", 1.0),
            param_embed_dim=model_cfg.get("param_embed_dim", 64),
            use_film=model_cfg.get("use_film", True),
            use_residual=model_cfg.get("use_residual", True),
            local_coord_mode="chart_uv",
            a_center_local=model_cfg.get("a_center_local", 0.125),
            a_half_range_local=model_cfg.get("a_half_range_local", 0.075),
            omega_min_local=model_cfg.get("omega_min_local", 1.0e-4),
            omega_max_local=model_cfg.get("omega_max_local", 10.0),
            u_center_local=float(patch.u_center),
            v_center_local=float(patch.v_center),
            u_half_range_local=float(patch.h_u),
            v_half_range_local=float(patch.h_v),
            M=M,
            m_mode=m_mode,
        ).to(device=device, dtype=dtype)
    return model, dtype


def load_checkpoint_into_model(model, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return ckpt.get("model_type", "pinn_mlp"), ckpt.get("cheb_N")


def evaluate_model_rin(
    model, dtype, device: torch.device, physics_cfg: dict,
    a: float, omega: float, u: float, v: float,
    r_grid: np.ndarray, y_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (r_grid, R_model_on_r, R_over_P_on_y)."""
    problem_cfg = physics_cfg["problem"]
    M = float(problem_cfg.get("M", 1.0))
    s_mode = int(problem_cfg.get("s", -2))
    ell = int(problem_cfg.get("l", 2))
    m_mode = int(problem_cfg.get("m", 2))

    mode = KerrMode(M=M, a=float(a), omega=float(omega), ell=ell, m=m_mode, lam=None, s=s_mode)
    lam = mode.lambda_value
    rp = mode.rp

    a_t = torch.tensor([a], device=device, dtype=dtype)
    omega_t = torch.tensor([omega], device=device, dtype=dtype)
    u_t = torch.tensor([u], device=device, dtype=dtype)
    v_t = torch.tensor([v], device=device, dtype=dtype)

    # --- r-space ---
    y_from_r = 2.0 * (rp / r_grid) - 1.0
    y_r_t = torch.tensor(y_from_r, device=device, dtype=dtype)
    with torch.no_grad():
        f_r = model(a_t, omega_t, y_r_t, u=u_t, v=v_t)[0]

    lam_t = torch.tensor([lam], device=device, dtype=torch.complex128)

    with torch.no_grad():
        slope = horizon_regularity_slope(a=a_t, omega=omega_t, lambda_=lam_t, m=m_mode, M=M, s=s_mode)
        S_r = compose_reduced_shape_from_f(f=f_r.unsqueeze(0), y=y_r_t.unsqueeze(0), slope=slope).squeeze(0)
        r_t = torch.tensor(r_grid, device=device, dtype=dtype)
        P_r, _, _ = Leaver_prefactors(r=r_t, a=a_t.squeeze(0), omega=omega_t.squeeze(0), m=m_mode, M=M, s=s_mode)
        h2_r = h_factor(a=a_t.squeeze(0), omega=omega_t.squeeze(0), m=m_mode, M=M, s=s_mode)
        R_model_r = (P_r * S_r * h2_r).detach().cpu().numpy()

    # --- y-space ---
    y_t = torch.tensor(y_grid, device=device, dtype=dtype)
    with torch.no_grad():
        f_y = model(a_t, omega_t, y_t, u=u_t, v=v_t)[0]
        S_y = compose_reduced_shape_from_f(f=f_y.unsqueeze(0), y=y_t.unsqueeze(0), slope=slope).squeeze(0)
        x_t = 0.5 * (y_t + 1.0)
        r_from_y = rp / x_t
        r_y_t = torch.tensor(r_from_y, device=device, dtype=dtype)
        P_y, _, _ = Leaver_prefactors(r=r_y_t, a=a_t.squeeze(0), omega=omega_t.squeeze(0), m=m_mode, M=M, s=s_mode)
        h2_y = h_factor(a=a_t.squeeze(0), omega=omega_t.squeeze(0), m=m_mode, M=M, s=s_mode)
        R_over_P_y = (S_y * h2_y).detach().cpu().numpy()

    return R_model_r, R_over_P_y, r_from_y.detach().cpu().numpy()


def compute_spectral_rin(
    physics_cfg: dict, a: float, omega: float,
    r_grid: np.ndarray, y_grid: np.ndarray,
    N_left: int, N_mid: int, N_right: int,
    z1: float, z2: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Returns (R_spec_on_r, R_spec_over_P_on_y) or None on failure.
    """
    problem_cfg = physics_cfg["problem"]
    M = float(problem_cfg.get("M", 1.0))
    s_mode = int(problem_cfg.get("s", -2))
    ell = int(problem_cfg.get("l", 2))
    m_mode = int(problem_cfg.get("m", 2))

    try:
        mode = KerrMode(M=M, a=float(a), omega=float(omega), ell=ell, m=m_mode, lam=None, s=s_mode)
        spec = TeukRadAmplitudeIn3PatchWithProfile(
            mode, N_left=N_left, N_mid=N_mid, N_right=N_right, z1=z1, z2=z2,
        )
        R_spec_r = spec.profile.R_of_r(r_grid)

        rp = mode.rp
        z_grid = 0.5 * (y_grid + 1.0)
        r_from_y = rp / z_grid

        P_y, _, _ = Leaver_prefactors(
            r=torch.tensor(r_from_y),
            a=torch.tensor(a, dtype=torch.float64),
            omega=torch.tensor(omega, dtype=torch.float64),
            m=m_mode, M=M, s=s_mode,
        )
        R_spec_over_P_y = spec.profile.R_of_z(z_grid) / P_y.detach().cpu().numpy()
        return (
            np.asarray(R_spec_r, dtype=np.complex128),
            np.asarray(R_spec_over_P_y, dtype=np.complex128),
        )
    except Exception as e:
        print(f"  [spectral] FAILED: {e}", file=sys.stderr, flush=True)
        return None


def compute_gsn_over_P_y(
    R_gsn_r: np.ndarray, r_grid: np.ndarray, r_from_y: np.ndarray,
    a: float, omega: float, m_mode: int, M: float, s_mode: int,
) -> np.ndarray:
    """Interpolate GSN R_in to r_from_y, then divide by Leaver prefactor."""
    # Linear interpolation of complex R_gsn from r_grid to r_from_y
    R_real = np.interp(r_from_y, r_grid, R_gsn_r.real)
    R_imag = np.interp(r_from_y, r_grid, R_gsn_r.imag)
    R_gsn_at_y = R_real + 1j * R_imag

    P_y, _, _ = Leaver_prefactors(
        r=torch.tensor(r_from_y),
        a=torch.tensor(a, dtype=torch.float64),
        omega=torch.tensor(omega, dtype=torch.float64),
        m=m_mode, M=M, s=s_mode,
    )
    return R_gsn_at_y / P_y.detach().cpu().numpy()


def make_6panel_figure(
    r_grid: np.ndarray,
    y_grid: np.ndarray,
    R_model: np.ndarray,   # on r-grid
    R_gsn: np.ndarray,     # on r-grid
    R_spec: np.ndarray,    # on r-grid
    Psi_model: np.ndarray,  # R/P on y-grid
    Psi_gsn: np.ndarray,    # R/P on y-grid
    Psi_spec: np.ndarray,   # R/P on y-grid
    meta_title: str,
    out_path: Path,
):
    """6-subplot figure: 3 rows (real/imag/mag) × 2 cols (r-space / y-space)."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex="col")

    colors = {"GSN": "#2196F3", "Model": "#FF5722", "Spectral": "#4CAF50"}
    lw_ref = 1.5
    lw_model = 1.2
    lw_spec = 1.0

    # Row 1: Real part
    axes[0, 0].plot(r_grid, R_gsn.real, color=colors["GSN"], linewidth=lw_ref, label="GSN")
    axes[0, 0].plot(r_grid, R_model.real, color=colors["Model"], linewidth=lw_model, linestyle="--", label="Model")
    axes[0, 0].plot(r_grid, R_spec.real, color=colors["Spectral"], linewidth=lw_spec, linestyle=":", label="Spectral")
    axes[0, 0].set_ylabel("Re(R_in)")
    axes[0, 0].legend(fontsize=8, loc="upper right")

    axes[0, 1].plot(y_grid, Psi_gsn.real, color=colors["GSN"], linewidth=lw_ref, label="GSN")
    axes[0, 1].plot(y_grid, Psi_model.real, color=colors["Model"], linewidth=lw_model, linestyle="--", label="Model")
    axes[0, 1].plot(y_grid, Psi_spec.real, color=colors["Spectral"], linewidth=lw_spec, linestyle=":", label="Spectral")
    axes[0, 1].set_ylabel("Re(R_in / P)")
    axes[0, 1].legend(fontsize=8, loc="upper right")

    # Row 2: Imaginary part
    axes[1, 0].plot(r_grid, R_gsn.imag, color=colors["GSN"], linewidth=lw_ref, label="GSN")
    axes[1, 0].plot(r_grid, R_model.imag, color=colors["Model"], linewidth=lw_model, linestyle="--", label="Model")
    axes[1, 0].plot(r_grid, R_spec.imag, color=colors["Spectral"], linewidth=lw_spec, linestyle=":", label="Spectral")
    axes[1, 0].set_ylabel("Im(R_in)")

    axes[1, 1].plot(y_grid, Psi_gsn.imag, color=colors["GSN"], linewidth=lw_ref, label="GSN")
    axes[1, 1].plot(y_grid, Psi_model.imag, color=colors["Model"], linewidth=lw_model, linestyle="--", label="Model")
    axes[1, 1].plot(y_grid, Psi_spec.imag, color=colors["Spectral"], linewidth=lw_spec, linestyle=":", label="Spectral")
    axes[1, 1].set_ylabel("Im(R_in / P)")

    # Row 3: Magnitude
    axes[2, 0].plot(r_grid, np.abs(R_gsn), color=colors["GSN"], linewidth=lw_ref, label="GSN")
    axes[2, 0].plot(r_grid, np.abs(R_model), color=colors["Model"], linewidth=lw_model, linestyle="--", label="Model")
    axes[2, 0].plot(r_grid, np.abs(R_spec), color=colors["Spectral"], linewidth=lw_spec, linestyle=":", label="Spectral")
    axes[2, 0].set_ylabel("|R_in|")
    axes[2, 0].set_xlabel("r")

    axes[2, 1].plot(y_grid, np.abs(Psi_gsn), color=colors["GSN"], linewidth=lw_ref, label="GSN")
    axes[2, 1].plot(y_grid, np.abs(Psi_model), color=colors["Model"], linewidth=lw_model, linestyle="--", label="Model")
    axes[2, 1].plot(y_grid, np.abs(Psi_spec), color=colors["Spectral"], linewidth=lw_spec, linestyle=":", label="Spectral")
    axes[2, 1].set_ylabel("|R_in / P|")
    axes[2, 1].set_xlabel("y")

    for ax in axes.ravel():
        ax.grid(alpha=0.3, linewidth=0.5)

    fig.suptitle(meta_title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sample_patch_params(
    patch_aw: np.ndarray,
    patch_uv: np.ndarray,
    patch,
    n_samples: int,
) -> list[tuple[float, float, float, float]]:
    """Select (a, omega, u, v) samples from a patch's valid points."""
    n = len(patch_aw)
    if n == 0:
        return []

    du = patch_uv[:, 0] - float(patch.u_center)
    dv = patch_uv[:, 1] - float(patch.v_center)
    center_idx = int(np.argmin(du * du + dv * dv))

    order = np.lexsort((patch_aw[:, 0], patch_aw[:, 1]))
    if n <= n_samples:
        idxs = list(order)
    else:
        qidx = np.linspace(0, n - 1, n_samples, dtype=int)
        idxs = [int(order[i]) for i in qidx]

    if center_idx not in idxs:
        idxs[len(idxs) // 2] = center_idx if len(idxs) >= n_samples else idxs.append(center_idx)

    idxs = sorted(set(idxs), key=lambda i: (patch_aw[i, 1], patch_aw[i, 0]))
    return [(float(patch_aw[i, 0]), float(patch_aw[i, 1]), float(patch_uv[i, 0]), float(patch_uv[i, 1])) for i in idxs]


def compute_statistics(all_errors: list[dict]) -> dict:
    """
    Compute comprehensive statistics from per-patch error records.
    Each record: {patch_id, a, omega, median_rel, max_rel, p90_rel, p99_rel, median_rel_Psi, ...}
    """
    import math

    patches = sorted(set(r["patch_id"] for r in all_errors))

    stats = {"per_patch": {}, "overall": {}, "ranking": []}

    # Per-patch stats
    for pid in patches:
        recs = [r for r in all_errors if r["patch_id"] == pid]
        if not recs:
            continue
        median_vals = [r["median_rel_R"] for r in recs if not math.isnan(r["median_rel_R"])]
        max_vals = [r["max_rel_R"] for r in recs if not math.isnan(r["max_rel_R"])]
        p90_vals = [r["p90_rel_R"] for r in recs if not math.isnan(r["p90_rel_R"])]

        stats["per_patch"][pid] = {
            "n_samples": len(recs),
            "n_valid": len(median_vals),
            "median_of_median": float(np.median(median_vals)) if median_vals else np.nan,
            "worst_median": float(np.max(median_vals)) if median_vals else np.nan,
            "best_median": float(np.min(median_vals)) if median_vals else np.nan,
            "median_of_max": float(np.median(max_vals)) if max_vals else np.nan,
            "worst_max": float(np.max(max_vals)) if max_vals else np.nan,
            "median_of_p90": float(np.median(p90_vals)) if p90_vals else np.nan,
        }

    # Ranking by median_of_median (lower is better)
    ranked = sorted(
        [(pid, s["median_of_median"], s["worst_max"], s["n_valid"])
         for pid, s in stats["per_patch"].items()],
        key=lambda x: (x[1] if not math.isnan(x[1]) else float("inf")),
    )
    stats["ranking"] = ranked

    # Overall stats
    all_median = [r["median_rel_R"] for r in all_errors if not math.isnan(r["median_rel_R"])]
    all_max = [r["max_rel_R"] for r in all_errors if not math.isnan(r["max_rel_R"])]
    if all_median:
        stats["overall"] = {
            "n_total_samples": len(all_errors),
            "n_valid": len(all_median),
            "grand_median_rel": float(np.median(all_median)),
            "grand_p90_rel": float(np.percentile(all_median, 90)),
            "grand_p99_rel": float(np.percentile(all_median, 99)),
            "global_max_rel": float(np.max(all_max)),
        }

    return stats


def print_statistics_report(stats: dict):
    """Print a human-readable statistics report."""
    print("\n" + "=" * 80)
    print("BENCHMARK STATISTICS REPORT")
    print("=" * 80)

    overall = stats.get("overall", {})
    if overall:
        print(f"\nOverall (across all patches):")
        print(f"  Valid samples:     {overall['n_valid']} / {overall['n_total_samples']}")
        print(f"  Grand median rel:  {overall['grand_median_rel']:.4e}")
        print(f"  90th percentile:   {overall['grand_p90_rel']:.4e}")
        print(f"  99th percentile:   {overall['grand_p99_rel']:.4e}")
        print(f"  Global max rel:    {overall['global_max_rel']:.4e}")

    print(f"\nPatch ranking (by median relative error in R, lower = better):")
    print(f"  {'Rank':>4s}  {'Patch':>6s}  {'Median(median)':>16s}  {'Worst max':>14s}  {'N valid':>8s}  {'Assessment':>20s}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*16}  {'-'*14}  {'-'*8}  {'-'*20}")

    for rank, (pid, med_med, worst_max, n_valid) in enumerate(stats["ranking"], 1):
        if np.isnan(med_med):
            assessment = "NO DATA"
        elif med_med < 0.01:
            assessment = "EXCELLENT"
        elif med_med < 0.05:
            assessment = "GOOD"
        elif med_med < 0.10:
            assessment = "FAIR — consider retrain"
        elif med_med < 0.30:
            assessment = "POOR — needs retrain"
        else:
            assessment = "VERY POOR — must retrain"
        print(f"  {rank:4d}  patch_{pid:03d}  {med_med:16.4e}  {worst_max:14.4e}  {n_valid:8d}  {assessment:>20s}")

    print(f"\nBottom 3 patches most in need of retraining:")
    worst = sorted(stats["ranking"], key=lambda x: (x[1] if not np.isnan(x[1]) else float("inf")), reverse=True)[:3]
    for pid, med_med, worst_max, _ in worst:
        print(f"  patch_{pid:03d}: median_err={med_med:.4e}, worst_max={worst_max:.4e}")

    print(f"\nTop 3 best-performing patches:")
    best = stats["ranking"][:3]
    for pid, med_med, _, _ in best:
        print(f"  patch_{pid:03d}: median_err={med_med:.4e}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark 9 atlas-patch PINN models vs GSN and spectral methods for R_in."
    )
    parser.add_argument("--registry-json", type=str,
                        default="outputs/atlas_multipatch_train_coarse040_retrain_rel1e4/atlas_registry_coarse040_retrain_rel1e4.json")
    parser.add_argument("--cfg", type=str, default="config/pinn_config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ckpt-kind", type=str, default="best", choices=["best", "latest"])
    parser.add_argument("--n-samples-per-patch", type=int, default=6)
    parser.add_argument("--n-y", type=int, default=400)
    parser.add_argument("--n-r", type=int, default=400)
    parser.add_argument("--r-max", type=float, default=1000.0)
    parser.add_argument("--r-eps", type=float, default=1.0e-4)
    parser.add_argument("--y-eps", type=float, default=0.01)
    parser.add_argument("--N-left", type=int, default=64)
    parser.add_argument("--N-mid", type=int, default=128)
    parser.add_argument("--N-right", type=int, default=64)
    parser.add_argument("--z1", type=float, default=0.1)
    parser.add_argument("--z2", type=float, default=0.9)
    parser.add_argument("--gsn-timeout", type=float, default=300.0)
    parser.add_argument("--out-dir-name", type=str, default="atlas_9patch_vs_gsn_spectral")
    parser.add_argument("--skip-gsn", action="store_true")
    parser.add_argument("--skip-spectral", action="store_true")
    parser.add_argument("--max-patches", type=int, default=None)
    args = parser.parse_args()

    repo_root = ROOT.resolve()
    registry_path = resolve_maybe_relative(args.registry_json, repo_root, repo_root)
    cfg_path = resolve_maybe_relative(args.cfg, repo_root, repo_root)
    registry_dir = registry_path.parent

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    full_cfg = load_pinn_full_config(str(cfg_path))
    physics_cfg = full_cfg["physics"]
    problem_cfg = physics_cfg["problem"]
    M = float(problem_cfg.get("M", 1.0))
    ell = int(problem_cfg.get("l", 2))
    m_mode = int(problem_cfg.get("m", 2))
    s_mode = int(problem_cfg.get("s", -2))

    patch_json = resolve_maybe_relative(registry["patch_json"], repo_root, registry_dir)
    atlas_json = resolve_maybe_relative(registry["atlas_json"], repo_root, registry_dir)
    probe_json = resolve_maybe_relative(registry["probe_json"], repo_root, registry_dir)

    patch_cover = load_patch_cover(str(patch_json))
    uv_points, aw_points = load_valid_chart_points(
        probe_json=str(probe_json), atlas_json=str(atlas_json),
        component_id=int(patch_cover.component_id),
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_root = repo_root / "benchmark" / "outputs" / args.out_dir_name
    out_root.mkdir(parents=True, exist_ok=True)

    record_map = {int(r["patch_id"]): r for r in registry.get("patch_records", [])}

    # Common y-grid (same for all samples)
    y_grid = np.linspace(-1.0 + args.y_eps, 1.0 - args.y_eps, args.n_y)

    print("=" * 80)
    print(f"[info] registry: {registry_path}")
    print(f"[info] output dir: {out_root}")
    print(f"[info] n_r={args.n_r}, n_y={args.n_y}, r_max={args.r_max}")
    print(f"[info] GSN script: {GSN_SCRIPT}")
    print(f"[info] skip_gsn={args.skip_gsn}, skip_spectral={args.skip_spectral}")
    print("=" * 80)

    all_error_records = []  # Per-sample error summaries
    all_full_errors = []    # Full (a, omega, r) error matrices

    patches_to_process = list(patch_cover.patches)
    if args.max_patches:
        patches_to_process = patches_to_process[:args.max_patches]

    for patch in patches_to_process:
        patch_id = int(patch.patch_id)
        rec = record_map.get(patch_id, None)
        if rec is None:
            print(f"[skip] patch {patch_id}: not in registry")
            continue

        ckpt_key = "best_model_path" if args.ckpt_kind == "best" else "latest_model_path"
        ckpt_path = resolve_maybe_relative(rec[ckpt_key], repo_root, registry_dir)
        if not ckpt_path.exists():
            print(f"[skip] patch {patch_id}: ckpt not found -> {ckpt_path}")
            continue

        patch_dir = out_root / f"patch_{patch_id:03d}_u_{patch.u_center:.3f}_v_{patch.v_center:.3f}"
        patch_dir.mkdir(parents=True, exist_ok=True)

        mask = (
            (np.abs(uv_points[:, 0] - patch.u_center) <= patch.h_u)
            & (np.abs(uv_points[:, 1] - patch.v_center) <= patch.h_v)
        )
        patch_uv = uv_points[mask]
        patch_aw = aw_points[mask]

        if len(patch_aw) == 0:
            print(f"[skip] patch {patch_id}: no safe points")
            continue

        # Peek at checkpoint to detect model type
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        model_type = ckpt.get("model_type", "pinn_mlp")
        cheb_N = ckpt.get("cheb_N", 64)
        model, dtype = build_model_for_patch(full_cfg, patch, device=device,
                                             model_type=model_type, cheb_N=cheb_N)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
        model.eval()

        samples = sample_patch_params(patch_aw, patch_uv, patch, args.n_samples_per_patch)

        print("-" * 80)
        print(f"[patch {patch_id:03d}] ckpt={ckpt_path}, n_safe={len(patch_aw)}, n_samples={len(samples)}")

        # Per-patch error accumulation
        patch_err_rel = []

        for isamp, (a, omega, u, v) in enumerate(samples):
            # Per-sample r_grid based on actual r_+
            sample_rp = KerrMode(M=M, a=a, omega=omega, ell=ell, m=m_mode, lam=None, s=s_mode).rp
            r_min_s = sample_rp + args.r_eps
            r_grid = np.linspace(r_min_s, args.r_max, args.n_r)

            print(f"  [{isamp+1}/{len(samples)}] a={a:.6f}, ω={omega:.6e}", end="", flush=True)

            # Model evaluation
            t0 = time.time()
            R_model_r, Psi_model, r_from_y = evaluate_model_rin(
                model, dtype, device, physics_cfg,
                a, omega, u, v, r_grid, y_grid,
            )
            dt_model = time.time() - t0
            print(f" [model:{dt_model:.1f}s]", end="", flush=True)

            # GSN evaluation
            R_gsn_r = None
            if not args.skip_gsn:
                t0 = time.time()
                R_gsn_r = gsn_eval_rin(s_mode, ell, m_mode, a, omega, r_grid, M=M, timeout=args.gsn_timeout)
                dt_gsn = time.time() - t0
                status = "OK" if R_gsn_r is not None else "FAIL"
                print(f" [GSN:{dt_gsn:.1f}s/{status}]", end="", flush=True)
            else:
                print(f" [GSN:SKIP]", end="", flush=True)

            # Spectral evaluation
            R_spec_r = None
            Psi_spec = None
            if not args.skip_spectral:
                t0 = time.time()
                spec_result = compute_spectral_rin(
                    physics_cfg, a, omega, r_grid, y_grid,
                    N_left=args.N_left, N_mid=args.N_mid, N_right=args.N_right,
                    z1=args.z1, z2=args.z2,
                )
                dt_spec = time.time() - t0
                if spec_result is not None:
                    R_spec_r, Psi_spec = spec_result
                    print(f" [spec:{dt_spec:.1f}s/OK]", end="", flush=True)
                else:
                    print(f" [spec:{dt_spec:.1f}s/FAIL]", end="", flush=True)
            else:
                print(f" [spec:SKIP]", end="", flush=True)
            print()

            # Compute Psi for GSN (R/P on y-grid via interpolation)
            Psi_gsn = None
            if R_gsn_r is not None:
                Psi_gsn = compute_gsn_over_P_y(R_gsn_r, r_grid, r_from_y, a, omega, m_mode, M, s_mode)

            # Compute Psi for model over the y-grid r values for GSN comparison
            # Actually model already computed Psi_model on y_grid

            # Error metrics (model vs GSN, with GSN as reference)
            if R_gsn_r is not None and len(R_gsn_r) == len(R_model_r):
                rel_err_R = safe_rel_err(R_model_r, R_gsn_r)
                median_rel_R = float(np.median(rel_err_R))
                max_rel_R = float(np.max(rel_err_R))
                p90_rel_R = float(np.percentile(rel_err_R, 90))
                p99_rel_R = float(np.percentile(rel_err_R, 99))
            else:
                rel_err_R = np.full_like(R_model_r.real, np.nan)
                median_rel_R = max_rel_R = p90_rel_R = p99_rel_R = np.nan

            if Psi_gsn is not None and len(Psi_gsn) == len(Psi_model):
                rel_err_Psi = safe_rel_err(Psi_model, Psi_gsn)
                median_rel_Psi = float(np.median(rel_err_Psi))
                max_rel_Psi = float(np.max(rel_err_Psi))
            else:
                median_rel_Psi = max_rel_Psi = np.nan

            # Model vs spectral error
            if R_spec_r is not None and len(R_spec_r) == len(R_model_r):
                rel_err_model_vs_spec = safe_rel_err(R_model_r, R_spec_r)
                median_rel_vs_spec = float(np.median(rel_err_model_vs_spec))
            else:
                median_rel_vs_spec = np.nan

            # Record per-sample summary
            error_rec = {
                "patch_id": patch_id,
                "sample_id": isamp,
                "a": a, "omega": omega, "u": u, "v": v,
                "median_rel_R": median_rel_R,
                "max_rel_R": max_rel_R,
                "p90_rel_R": p90_rel_R,
                "p99_rel_R": p99_rel_R,
                "median_rel_Psi": median_rel_Psi,
                "max_rel_Psi": max_rel_Psi,
                "median_rel_vs_spec": median_rel_vs_spec,
            }
            all_error_records.append(error_rec)

            # Store full (a, omega, r) error for this sample
            if R_gsn_r is not None:
                all_full_errors.append({
                    "patch_id": patch_id,
                    "sample_id": isamp,
                    "a": a, "omega": omega,
                    "r_grid": r_grid.copy(),
                    "R_model": R_model_r.copy(),
                    "R_gsn": R_gsn_r.copy(),
                    "rel_err": rel_err_R.copy(),
                })

            # --- Plotting ---
            if R_gsn_r is not None:
                meta_title = (
                    f"Patch {patch_id:03d}  a={a:.6g}  ω={omega:.6g}  u={u:.4f}  v={v:.4f}\n"
                    f"R_in: model vs GSN vs Spectral  |  median rel err = {median_rel_R:.3e}"
                )
                fig_name = (
                    f"patch_{patch_id:03d}_s{isamp:02d}_"
                    f"a{slug_float_short(a)}_w{slug_float_short(omega)}.png"
                )
                fig_path = patch_dir / fig_name

                make_6panel_figure(
                    r_grid=r_grid,
                    y_grid=y_grid,
                    R_model=R_model_r,
                    R_gsn=R_gsn_r,
                    R_spec=R_spec_r if R_spec_r is not None else np.full_like(R_model_r, np.nan),
                    Psi_model=Psi_model,
                    Psi_gsn=Psi_gsn if Psi_gsn is not None else np.full_like(Psi_model, np.nan),
                    Psi_spec=Psi_spec if Psi_spec is not None else np.full_like(Psi_model, np.nan),
                    meta_title=meta_title,
                    out_path=fig_path,
                )
                error_rec["figure"] = str(fig_path)

        patch_medians = [r['median_rel_R'] for r in all_error_records if r['patch_id']==patch_id and not np.isnan(r['median_rel_R'])]
        patch_summary = np.median(patch_medians) if patch_medians else float('nan')
        print(f"  [patch {patch_id:03d}] done. median(median_rel)={patch_summary:.4e}")

    # --- Compute and save statistics ---
    stats = compute_statistics(all_error_records)
    print_statistics_report(stats)

    # Save statistics JSON
    stats_path = out_root / "statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[saved] statistics -> {stats_path}")

    # Save per-sample error CSV
    csv_path = out_root / f"errors_summary_{args.ckpt_kind}.csv"
    if all_error_records:
        keys = [k for k in all_error_records[0].keys() if k != "figure"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_error_records)
        print(f"[saved] error summary -> {csv_path}")

    # Save full error matrices (compressed npz)
    npz_path = out_root / "full_error_matrices.npz"
    npz_data = {}
    for i, rec in enumerate(all_full_errors):
        prefix = f"patch{rec['patch_id']:03d}_s{rec['sample_id']:02d}"
        npz_data[f"{prefix}_a"] = rec["a"]
        npz_data[f"{prefix}_omega"] = rec["omega"]
        npz_data[f"{prefix}_r_grid"] = rec["r_grid"]
        npz_data[f"{prefix}_R_model_re"] = rec["R_model"].real
        npz_data[f"{prefix}_R_model_im"] = rec["R_model"].imag
        npz_data[f"{prefix}_R_gsn_re"] = rec["R_gsn"].real
        npz_data[f"{prefix}_R_gsn_im"] = rec["R_gsn"].imag
        npz_data[f"{prefix}_rel_err"] = rec["rel_err"]
    np.savez_compressed(npz_path, **npz_data)
    print(f"[saved] full error matrices -> {npz_path}")

    print("\n" + "=" * 80)
    print(f"[done] All figures saved under: {out_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
