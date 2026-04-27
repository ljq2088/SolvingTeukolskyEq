from __future__ import annotations

import argparse
import csv
import json
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
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.config_loader import load_pinn_full_config
from domain.patch_cover import load_patch_cover, load_valid_chart_points
from model.pinn_mlp import PINN_MLP
from physical_ansatz.prefactor import Leaver_prefactors
from physical_ansatz.transform_y import (
    compose_reduced_shape_from_f,
    h_factor,
    horizon_regularity_slope,
)
from utils.mode import KerrMode
from utils.amplitude_profile_three_patch import TeukRadAmplitudeIn3PatchWithProfile
from mma.rin_sampler import MathematicaRinSampler
from pybhpt_usage.compute_solution import compute_pybhpt_solution


def _get_dtype(dtype_name: str):
    if str(dtype_name).lower() == "float32":
        return torch.float32
    return torch.float64


def slug_float(x: float) -> str:
    s = f"{x:.8g}"
    s = s.replace("+", "")
    s = s.replace("-", "m")
    s = s.replace(".", "p")
    return s


def safe_rel_err(x: np.ndarray, y: np.ndarray, floor: float = 1.0e-14) -> np.ndarray:
    return np.abs(x - y) / np.maximum(np.abs(y), floor)


def resolve_maybe_relative(path_str: str, repo_root: Path, registry_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    cand1 = (repo_root / p).resolve()
    if cand1.exists():
        return cand1
    cand2 = (registry_dir / p).resolve()
    if cand2.exists():
        return cand2
    return cand1


def build_model_for_patch(full_cfg: dict, patch, device: torch.device):
    physics_cfg = full_cfg["physics"]
    train_cfg = full_cfg["train"]
    model_cfg = train_cfg.get("model", {})
    runtime_cfg = train_cfg.get("runtime", {})
    dtype = _get_dtype(runtime_cfg.get("dtype", "float64"))

    problem_cfg = physics_cfg["problem"]
    M = float(problem_cfg.get("M", 1.0))
    m_mode = int(problem_cfg.get("m", 2))

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


def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()


def choose_patch_samples(
    patch_aw: np.ndarray,
    patch_uv: np.ndarray,
    patch,
    n_samples: int,
) -> list[tuple[float, float, float, float]]:
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
        if len(idxs) < n_samples:
            idxs.append(center_idx)
        else:
            idxs[len(idxs) // 2] = center_idx

    idxs = sorted(set(idxs), key=lambda i: (patch_aw[i, 1], patch_aw[i, 0]))

    samples = []
    for i in idxs:
        a = float(patch_aw[i, 0])
        omega = float(patch_aw[i, 1])
        u = float(patch_uv[i, 0])
        v = float(patch_uv[i, 1])
        samples.append((a, omega, u, v))
    return samples


def evaluate_model_on_y_grid(
    model: torch.nn.Module,
    dtype,
    device: torch.device,
    physics_cfg: dict,
    a: float,
    omega: float,
    u: float,
    v: float,
    y_grid: np.ndarray,
):
    """
    在统一的 y-grid 上返回:
      r_grid, P_grid, R_model, Psi_model
    其中 Psi_model = R_model / P_grid
    """
    problem_cfg = physics_cfg["problem"]
    M = float(problem_cfg.get("M", 1.0))
    s_mode = int(problem_cfg.get("s", -2))
    ell = int(problem_cfg.get("l", 2))
    m_mode = int(problem_cfg.get("m", 2))

    mode = KerrMode(
        M=M,
        a=float(a),
        omega=float(omega),
        ell=ell,
        m=m_mode,
        lam=None,
        s=s_mode,
    )
    lam = mode.lambda_value
    rp = mode.rp

    a_t = torch.tensor([a], device=device, dtype=dtype)
    omega_t = torch.tensor([omega], device=device, dtype=dtype)
    u_t = torch.tensor([u], device=device, dtype=dtype)
    v_t = torch.tensor([v], device=device, dtype=dtype)
    y_t = torch.tensor(y_grid, device=device, dtype=dtype)

    with torch.no_grad():
        f = model(a_t, omega_t, y_t, u=u_t, v=v_t)[0]

    lam_t = torch.tensor([lam], device=device, dtype=torch.complex128)

    with torch.no_grad():
        slope = horizon_regularity_slope(
            a=a_t,
            omega=omega_t,
            lambda_=lam_t,
            m=m_mode,
            M=M,
            s=s_mode,
        )

        S = compose_reduced_shape_from_f(
            f=f.unsqueeze(0),
            y=y_t.unsqueeze(0),
            slope=slope,
        ).squeeze(0)

        x_t = 0.5 * (y_t + 1.0)
        r_t = torch.tensor(rp, device=device, dtype=dtype) / x_t

        P, _, _ = Leaver_prefactors(
            r=r_t,
            a=a_t.squeeze(0),
            omega=omega_t.squeeze(0),
            m=m_mode,
            M=M,
            s=s_mode,
        )
        h2 = h_factor(
            a=a_t.squeeze(0),
            omega=omega_t.squeeze(0),
            m=m_mode,
            M=M,
            s=s_mode,
        )

        R_model = P * S * h2
        Psi_model = R_model / P

    return (
        r_t.detach().cpu().numpy(),
        P.detach().cpu().numpy(),
        R_model.detach().cpu().numpy(),
        Psi_model.detach().cpu().numpy(),
    )


def evaluate_model_on_r_grid(
    model: torch.nn.Module,
    dtype,
    device: torch.device,
    physics_cfg: dict,
    a: float,
    omega: float,
    u: float,
    v: float,
    r_grid: np.ndarray,
) -> np.ndarray:
    problem_cfg = physics_cfg["problem"]
    M = float(problem_cfg.get("M", 1.0))
    s_mode = int(problem_cfg.get("s", -2))
    ell = int(problem_cfg.get("l", 2))
    m_mode = int(problem_cfg.get("m", 2))

    mode = KerrMode(
        M=M,
        a=float(a),
        omega=float(omega),
        ell=ell,
        m=m_mode,
        lam=None,
        s=s_mode,
    )
    lam = mode.lambda_value
    rp = mode.rp

    y_grid = 2.0 * (rp / r_grid) - 1.0

    a_t = torch.tensor([a], device=device, dtype=dtype)
    omega_t = torch.tensor([omega], device=device, dtype=dtype)
    u_t = torch.tensor([u], device=device, dtype=dtype)
    v_t = torch.tensor([v], device=device, dtype=dtype)
    y_t = torch.tensor(y_grid, device=device, dtype=dtype)

    with torch.no_grad():
        f = model(a_t, omega_t, y_t, u=u_t, v=v_t)[0]

    lam_t = torch.tensor([lam], device=device, dtype=torch.complex128)

    with torch.no_grad():
        slope = horizon_regularity_slope(
            a=a_t,
            omega=omega_t,
            lambda_=lam_t,
            m=m_mode,
            M=M,
            s=s_mode,
        )

        S = compose_reduced_shape_from_f(
            f=f.unsqueeze(0),
            y=y_t.unsqueeze(0),
            slope=slope,
        ).squeeze(0)

        r_t = torch.tensor(r_grid, device=device, dtype=dtype)
        P, _, _ = Leaver_prefactors(
            r=r_t,
            a=a_t.squeeze(0),
            omega=omega_t.squeeze(0),
            m=m_mode,
            M=M,
            s=s_mode,
        )
        h2 = h_factor(
            a=a_t.squeeze(0),
            omega=omega_t.squeeze(0),
            m=m_mode,
            M=M,
            s=s_mode,
        )

        R_model = P * S * h2

    return R_model.detach().cpu().numpy()


def build_reference_on_r_grid(
    physics_cfg: dict,
    a: float,
    omega: float,
    r_grid: np.ndarray,
    loww_cut: float,
    loww_backend: str,
    mma_sampler: MathematicaRinSampler | None,
    pybhpt_timeout: float,
    N_left: int | None = None,
    N_mid: int | None = None,
    N_right: int | None = None,
    z1: float | None = None,
    z2: float | None = None,
):
    """
    在 r_grid 上生成参考解:
      omega < loww_cut  -> mma 或 pybhpt
      else              -> three-patch spectral (adaptive parameters)
    返回:
      ref_backend, R_ref
    """
    problem_cfg = physics_cfg["problem"]
    M = float(problem_cfg.get("M", 1.0))
    s_mode = int(problem_cfg.get("s", -2))
    ell = int(problem_cfg.get("l", 2))
    m_mode = int(problem_cfg.get("m", 2))

    mode = KerrMode(
        M=M,
        a=float(a),
        omega=float(omega),
        ell=ell,
        m=m_mode,
        lam=None,
        s=s_mode,
    )

    if float(omega) < float(loww_cut):
        if loww_backend == "mma":
            if mma_sampler is not None:
                try:
                    R_ref = mma_sampler.evaluate_rin_at_points_direct(
                        s=s_mode,
                        l=ell,
                        m=m_mode,
                        a=float(a),
                        omega=float(omega),
                        r_query=r_grid,
                    )
                    return "mma", np.asarray(R_ref, dtype=np.complex128)
                except Exception:
                    pass

            try:
                _, R_ref = compute_pybhpt_solution(
                    a=float(a),
                    omega=float(omega),
                    ell=ell,
                    m=m_mode,
                    r_grid=r_grid,
                    timeout=pybhpt_timeout,
                )
                return "pybhpt-fallback", np.asarray(R_ref, dtype=np.complex128)
            except Exception:
                pass

        elif loww_backend == "pybhpt":
            try:
                _, R_ref = compute_pybhpt_solution(
                    a=float(a),
                    omega=float(omega),
                    ell=ell,
                    m=m_mode,
                    r_grid=r_grid,
                    timeout=pybhpt_timeout,
                )
                return "pybhpt", np.asarray(R_ref, dtype=np.complex128)
            except Exception:
                pass

        else:
            raise ValueError(f"Unknown loww_backend={loww_backend}")

    spec = TeukRadAmplitudeIn3PatchWithProfile(
        mode,
        N_left=N_left if N_left is not None else 64,
        N_mid=N_mid if N_mid is not None else 96,
        N_right=N_right if N_right is not None else 64,
        z1=z1,
        z2=z2,
    )
    R_ref = spec.profile.R_of_r(r_grid)
    backend = "spectral3patch"
    if float(omega) < float(loww_cut):
        backend = "spectral3patch-fallback"
    return backend, np.asarray(R_ref, dtype=np.complex128)


def build_reference_on_y_grid(
    physics_cfg: dict,
    a: float,
    omega: float,
    r_grid_from_y: np.ndarray,
    z_grid: np.ndarray,
    P_grid: np.ndarray,
    loww_cut: float,
    loww_backend: str,
    mma_sampler: MathematicaRinSampler | None,
    pybhpt_timeout: float,
    N_left: int | None = None,
    N_mid: int | None = None,
    N_right: int | None = None,
    z1: float | None = None,
    z2: float | None = None,
):
    ref_backend, R_ref = build_reference_on_r_grid(
        physics_cfg=physics_cfg,
        a=a,
        omega=omega,
        r_grid=r_grid_from_y,
        loww_cut=loww_cut,
        loww_backend=loww_backend,
        mma_sampler=mma_sampler,
        pybhpt_timeout=pybhpt_timeout,
        N_left=N_left, N_mid=N_mid, N_right=N_right,
        z1=z1, z2=z2,
    )
    Psi_ref = R_ref / P_grid
    return ref_backend, np.asarray(Psi_ref, dtype=np.complex128)


def make_compare_figure(
    y_grid: np.ndarray,
    r_grid: np.ndarray,
    R_model: np.ndarray,
    R_ref: np.ndarray,
    Psi_model: np.ndarray,
    Psi_ref: np.ndarray,
    relR: np.ndarray,
    relPsi: np.ndarray,
    meta_title: str,
    out_path: Path,
):
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex="col")

    # full R vs r
    axes[0, 0].plot(r_grid, R_ref.real, label="ref")
    axes[0, 0].plot(r_grid, R_model.real, ls="--", label="model")
    axes[0, 0].set_ylabel("Re(R_in)")

    axes[1, 0].plot(r_grid, R_ref.imag)
    axes[1, 0].plot(r_grid, R_model.imag, ls="--")
    axes[1, 0].set_ylabel("Im(R_in)")

    axes[2, 0].plot(r_grid, np.abs(R_ref))
    axes[2, 0].plot(r_grid, np.abs(R_model), ls="--")
    axes[2, 0].set_ylabel("|R_in|")

    axes[3, 0].plot(r_grid, relR)
    axes[3, 0].set_yscale("log")
    axes[3, 0].set_ylabel("rel err of R")
    axes[3, 0].set_xlabel("r")

    # reduced vs y
    axes[0, 1].plot(y_grid, Psi_ref.real, label="ref")
    axes[0, 1].plot(y_grid, Psi_model.real, ls="--", label="model")
    axes[0, 1].set_ylabel("Re(Psi)")

    axes[1, 1].plot(y_grid, Psi_ref.imag)
    axes[1, 1].plot(y_grid, Psi_model.imag, ls="--")
    axes[1, 1].set_ylabel("Im(Psi)")

    axes[2, 1].plot(y_grid, np.abs(Psi_ref))
    axes[2, 1].plot(y_grid, np.abs(Psi_model), ls="--")
    axes[2, 1].set_ylabel("|Psi|")

    axes[3, 1].plot(y_grid, relPsi)
    axes[3, 1].set_yscale("log")
    axes[3, 1].set_ylabel("rel err of Psi")

    axes[3, 0].set_xlabel("r")
    axes[3, 1].set_xlabel("y")

    for ax in axes.ravel():
        ax.grid(alpha=0.3)

    axes[0, 0].legend(fontsize=9)
    axes[0, 1].legend(fontsize=9)
    fig.suptitle(meta_title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare patch PINN models against hybrid reference R_in: low-w uses MMA/pybhpt, otherwise three-patch spectral."
    )
    parser.add_argument(
        "--registry-json",
        type=str,
        default="outputs/atlas_multipatch_train_coarse040_retrain_rel1e4/atlas_registry_coarse040_retrain_rel1e4.json",
    )
    parser.add_argument("--cfg", type=str, default="config/pinn_config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ckpt-kind", type=str, default="best", choices=["best", "latest"])
    parser.add_argument("--n-samples-per-patch", type=int, default=6)
    parser.add_argument("--n-y", type=int, default=500)
    parser.add_argument("--y-eps", type=float, default=1.0e-2)
    parser.add_argument("--n-r", type=int, default=500)
    parser.add_argument("--r-max", type=float, default=1000.0)
    parser.add_argument("--r-eps", type=float, default=1.0e-4)
    parser.add_argument("--loww-cut", type=float, default=1.0e-2)
    parser.add_argument("--loww-backend", type=str, default="mma", choices=["mma", "pybhpt"])
    parser.add_argument("--pybhpt-timeout", type=float, default=10.0)
    parser.add_argument("--N-left", type=int, default=None,
                        help="override adaptive N_left (default: auto)")
    parser.add_argument("--N-mid", type=int, default=None,
                        help="override adaptive N_mid (default: auto ≈ 38ω)")
    parser.add_argument("--N-right", type=int, default=None,
                        help="override adaptive N_right (default: auto)")
    parser.add_argument("--z1", type=float, default=None,
                        help="override adaptive z1 (default: auto)")
    parser.add_argument("--z2", type=float, default=None,
                        help="override adaptive z2 (default: auto)")
    parser.add_argument(
        "--out-dir-name",
        type=str,
        default="atlas_patch_model_vs_hybrid_ref_rin",
    )
    args = parser.parse_args()

    repo_root = ROOT.resolve()
    registry_path = resolve_maybe_relative(args.registry_json, repo_root, repo_root)
    cfg_path = resolve_maybe_relative(args.cfg, repo_root, repo_root)
    registry_dir = registry_path.parent

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    full_cfg = load_pinn_full_config(str(cfg_path))
    physics_cfg = full_cfg["physics"]
    train_cfg = full_cfg["train"]
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
        probe_json=str(probe_json),
        atlas_json=str(atlas_json),
        component_id=int(patch_cover.component_id),
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # mathematica sampler
    mma_sampler = None
    if args.loww_backend == "mma":
        mma_cfg = train_cfg.get("mathematica", {})
        if not mma_cfg:
            raise RuntimeError("mathematica config not found in cfg")
        mma_sampler = MathematicaRinSampler(
            kernel_path=str(mma_cfg["kernel_path"]),
            wl_path_win=str(mma_cfg["wl_path_win"]),
            timeout_sec=float(train_cfg.get("atlas_training", {}).get("anchor_mma_timeout", 20.0)),
        )

    out_root = repo_root / "benchmark" / "outputs" / args.out_dir_name
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    record_map = {int(r["patch_id"]): r for r in registry.get("patch_records", [])}

    # 统一 master y-grid
    y_grid = np.linspace(-1.0 + args.y_eps, 1.0 - args.y_eps, args.n_y)
    z_grid = 0.5 * (y_grid + 1.0)

    print("=" * 100)
    print(f"[info] registry: {registry_path}")
    print(f"[info] output dir: {out_root}")
    print(f"[info] loww backend: {args.loww_backend}, cut={args.loww_cut}")
    print(f"[info] spectral reference: adaptive (auto z1/z2/N)")
    print("=" * 100)

    try:
        for patch in patch_cover.patches:
            patch_id = int(patch.patch_id)
            rec = record_map.get(patch_id, None)
            if rec is None:
                print(f"[skip] patch {patch_id}: not found in registry")
                continue

            ckpt_key = "best_model_path" if args.ckpt_kind == "best" else "latest_model_path"
            ckpt_path = resolve_maybe_relative(rec[ckpt_key], repo_root, registry_dir)
            if not ckpt_path.exists():
                print(f"[skip] patch {patch_id}: checkpoint not found -> {ckpt_path}")
                continue

            patch_dir = out_root / (
                f"patch_{patch_id:03d}"
                f"_u_{patch.u_center:.3f}"
                f"_v_{patch.v_center:.3f}"
            )
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

            model, dtype = build_model_for_patch(full_cfg, patch, device=device)
            load_checkpoint_into_model(model, ckpt_path, device=device)

            samples = choose_patch_samples(
                patch_aw=patch_aw,
                patch_uv=patch_uv,
                patch=patch,
                n_samples=args.n_samples_per_patch,
            )

            print("-" * 100)
            print(f"[patch {patch_id:03d}] ckpt = {ckpt_path}")
            print(f"[patch {patch_id:03d}] n_safe_points = {len(patch_aw)}")
            print(f"[patch {patch_id:03d}] n_samples    = {len(samples)}")

            for isamp, (a, omega, u, v) in enumerate(samples):
                mode = KerrMode(
                    M=M,
                    a=a,
                    omega=omega,
                    ell=ell,
                    m=m_mode,
                    lam=None,
                    s=s_mode,
                )

                r_from_y, P_grid_y, R_model_y, Psi_model = evaluate_model_on_y_grid(
                    model=model,
                    dtype=dtype,
                    device=device,
                    physics_cfg=physics_cfg,
                    a=a,
                    omega=omega,
                    u=u,
                    v=v,
                    y_grid=y_grid,
                )
                rp = mode.rp
                r_min = rp + args.r_eps
                r_grid = np.linspace(r_min, args.r_max, args.n_r)

                R_model = evaluate_model_on_r_grid(
                    model=model,
                    dtype=dtype,
                    device=device,
                    physics_cfg=physics_cfg,
                    a=a,
                    omega=omega,
                    u=u,
                    v=v,
                    r_grid=r_grid,
                )

                ref_backend, R_ref = build_reference_on_r_grid(
                    physics_cfg=physics_cfg,
                    a=a,
                    omega=omega,
                    r_grid=r_grid,
                    loww_cut=args.loww_cut,
                    loww_backend=args.loww_backend,
                    mma_sampler=mma_sampler,
                    pybhpt_timeout=args.pybhpt_timeout,
                    N_left=args.N_left, N_mid=args.N_mid, N_right=args.N_right,
                    z1=args.z1, z2=args.z2,
                )
                _, Psi_ref = build_reference_on_y_grid(
                    physics_cfg=physics_cfg,
                    a=a,
                    omega=omega,
                    r_grid_from_y=r_from_y,
                    z_grid=z_grid,
                    P_grid=P_grid_y,
                    loww_cut=args.loww_cut,
                    loww_backend=args.loww_backend,
                    mma_sampler=mma_sampler,
                    pybhpt_timeout=args.pybhpt_timeout,
                    N_left=args.N_left, N_mid=args.N_mid, N_right=args.N_right,
                    z1=args.z1, z2=args.z2,
                )

                relR = safe_rel_err(R_model, R_ref)
                relPsi = safe_rel_err(Psi_model, Psi_ref)
                rel_diff = np.abs(relR - relPsi)

                meta_title = (
                    f"patch={patch_id:03d}, sample={isamp:02d}, ckpt={args.ckpt_kind}, ref={ref_backend}\n"
                    f"a={a:.8f}, omega={omega:.8f}, u={u:.8f}, v={v:.8f}\n"
                    f"R on uniform r-grid, Psi on y-grid; loww_cut={args.loww_cut}, loww_backend={args.loww_backend}"
                )

                fig_name = (
                    f"patch_{patch_id:03d}"
                    f"_sample_{isamp:02d}"
                    f"_a_{slug_float(a)}"
                    f"_omega_{slug_float(omega)}"
                    f"_u_{slug_float(u)}"
                    f"_v_{slug_float(v)}"
                    f"_{args.ckpt_kind}"
                    f"_ref_{ref_backend}.png"
                )
                fig_path = patch_dir / fig_name

                make_compare_figure(
                    y_grid=y_grid,
                    r_grid=r_grid,
                    R_model=R_model,
                    R_ref=R_ref,
                    Psi_model=Psi_model,
                    Psi_ref=Psi_ref,
                    relR=relR,
                    relPsi=relPsi,
                    meta_title=meta_title,
                    out_path=fig_path,
                )

                summary_rows.append(
                    {
                        "patch_id": patch_id,
                        "sample_id": isamp,
                        "checkpoint_kind": args.ckpt_kind,
                        "checkpoint_path": str(ckpt_path),
                        "ref_backend": ref_backend,
                        "a": a,
                        "omega": omega,
                        "u": u,
                        "v": v,
                        "loww_cut": args.loww_cut,
                        "N_left": args.N_left,
                        "N_mid": args.N_mid,
                        "N_right": args.N_right,
                        "z1": args.z1,
                        "z2": args.z2,
                        "median_rel_err_R": float(np.median(relR)),
                        "max_rel_err_R": float(np.max(relR)),
                        "median_rel_err_Psi": float(np.median(relPsi)),
                        "max_rel_err_Psi": float(np.max(relPsi)),
                        "median_abs_diff_rel": float(np.median(rel_diff)),
                        "max_abs_diff_rel": float(np.max(rel_diff)),
                        "figure": str(fig_path),
                    }
                )

                print(
                    f"[patch {patch_id:03d}] sample {isamp:02d}: "
                    f"a={a:.6f}, omega={omega:.6e}, ref={ref_backend}, "
                    f"med_rel_R={np.median(relR):.3e}, med_rel_Psi={np.median(relPsi):.3e}, "
                    f"max|Δrel|={np.max(rel_diff):.3e}"
                )

        summary_csv = out_root / f"summary_{args.ckpt_kind}.csv"
        if summary_rows:
            keys = list(summary_rows[0].keys())
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(summary_rows)

        print("=" * 100)
        print(f"[saved] {summary_csv}")
        print(f"[done] figures under {out_root}")
        print("=" * 100)

    finally:
        if mma_sampler is not None:
            mma_sampler.close()


if __name__ == "__main__":
    main()
