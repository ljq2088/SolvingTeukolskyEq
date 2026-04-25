from __future__ import annotations
import sys
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
import argparse
import json
import math
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
from physical_ansatz.mapping import r_from_x
from physical_ansatz.prefactor import Leaver_prefactors
from physical_ansatz.transform_y import (
    compose_reduced_shape_from_f,
    h_factor,
    horizon_regularity_slope,
)
from utils.mode import KerrMode
from utils.amplitude_profile_three_patch import TeukRadAmplitudeIn3PatchWithProfile


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
        # backward compatibility fields
        a_center_local=model_cfg.get("a_center_local", 0.125),
        a_half_range_local=model_cfg.get("a_half_range_local", 0.075),
        omega_min_local=model_cfg.get("omega_min_local", 1.0e-4),
        omega_max_local=model_cfg.get("omega_max_local", 10.0),
        # actual patch-local coords
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

    # 先找离 patch 中心最近的点
    du = patch_uv[:, 0] - float(patch.u_center)
    dv = patch_uv[:, 1] - float(patch.v_center)
    center_idx = int(np.argmin(du * du + dv * dv))

    # 再按 (omega, a) 排序后等间隔取点
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

    # 去重并按 omega 再排序
    idxs = sorted(set(idxs), key=lambda i: (patch_aw[i, 1], patch_aw[i, 0]))

    samples = []
    for i in idxs:
        a = float(patch_aw[i, 0])
        omega = float(patch_aw[i, 1])
        u = float(patch_uv[i, 0])
        v = float(patch_uv[i, 1])
        samples.append((a, omega, u, v))
    return samples


def evaluate_model_rin_on_y(
    model: torch.nn.Module,
    dtype,
    device: torch.device,
    physics_cfg: dict,
    a: float,
    omega: float,
    u: float,
    v: float,
    y_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    返回:
        r_grid_from_y, (R_model / P)(y_grid)
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

    a_t = torch.tensor([a], device=device, dtype=dtype)
    omega_t = torch.tensor([omega], device=device, dtype=dtype)
    u_t = torch.tensor([u], device=device, dtype=dtype)
    v_t = torch.tensor([v], device=device, dtype=dtype)
    y_t = torch.tensor(y_grid, device=device, dtype=dtype)

    with torch.no_grad():
        f = model(a_t, omega_t, y_t, u=u_t, v=v_t)[0]  # (N,), complex

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
        rp = mode.rp
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

        R_model_over_P = S * h2

    return r_t.detach().cpu().numpy(), R_model_over_P.detach().cpu().numpy()


def evaluate_model_rin_on_r(
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


def make_compare_figure(
    r_grid: np.ndarray,
    R_model_r: np.ndarray,
    R_spec_r: np.ndarray,
    y_grid: np.ndarray,
    R_model_y: np.ndarray,
    R_spec_y: np.ndarray,
    meta_title: str,
    out_path: Path,
):
    err_r = safe_rel_err(R_model_r, R_spec_r)
    err_y = safe_rel_err(R_model_y, R_spec_y)

    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex="col")

    # r-space
    axes[0, 0].plot(r_grid, R_spec_r.real, label="spectral")
    axes[0, 0].plot(r_grid, R_model_r.real, ls="--", label="model")
    axes[0, 0].set_ylabel("Re(R_in)")

    axes[1, 0].plot(r_grid, R_spec_r.imag)
    axes[1, 0].plot(r_grid, R_model_r.imag, ls="--")
    axes[1, 0].set_ylabel("Im(R_in)")

    axes[2, 0].plot(r_grid, np.abs(R_spec_r))
    axes[2, 0].plot(r_grid, np.abs(R_model_r), ls="--")
    axes[2, 0].set_ylabel("|R_in|")

    axes[3, 0].plot(r_grid, err_r)
    axes[3, 0].set_yscale("log")
    axes[3, 0].set_ylabel("rel err")
    axes[3, 0].set_xlabel("r")

    # y-space: compare R_in / P
    axes[0, 1].plot(y_grid, R_spec_y.real, label="spectral")
    axes[0, 1].plot(y_grid, R_model_y.real, ls="--", label="model")
    axes[0, 1].set_ylabel("Re(R_in / P)")

    axes[1, 1].plot(y_grid, R_spec_y.imag)
    axes[1, 1].plot(y_grid, R_model_y.imag, ls="--")
    axes[1, 1].set_ylabel("Im(R_in / P)")

    axes[2, 1].plot(y_grid, np.abs(R_spec_y))
    axes[2, 1].plot(y_grid, np.abs(R_model_y), ls="--")
    axes[2, 1].set_ylabel("|R_in / P|")

    axes[3, 1].plot(y_grid, err_y)
    axes[3, 1].set_yscale("log")
    axes[3, 1].set_ylabel("rel err")
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
        description="Compare trained atlas patch PINN models against three-patch spectral R_in."
    )
    parser.add_argument(
        "--registry-json",
        type=str,
        default="..outputs/atlas_multipatch_train_coarse040_retrain_rel1e4/atlas_registry_coarse040_retrain_rel1e4.json",
    )
    parser.add_argument("--cfg", type=str, default="config/pinn_config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ckpt-kind", type=str, default="best", choices=["best", "latest"])
    parser.add_argument("--n-samples-per-patch", type=int, default=6)
    parser.add_argument("--n-y", type=int, default=500)
    parser.add_argument("--n-r", type=int, default=500)
    parser.add_argument("--r-max", type=float, default=1000.0)
    parser.add_argument("--r-eps", type=float, default=1.0e-4)
    parser.add_argument("--N-left", type=int, default=64)
    parser.add_argument("--N-mid", type=int, default=128)
    parser.add_argument("--N-right", type=int, default=64)
    parser.add_argument("--z1", type=float, default=0.1)
    parser.add_argument("--z2", type=float, default=0.9)
    parser.add_argument(
        "--out-dir-name",
        type=str,
        default="atlas_patch_model_vs_threepatch_rin",
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

    out_root = repo_root / "benchmark" / "outputs" / args.out_dir_name
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    # y-grid 避开端点，保证谱参考 profile.R_of_z 合法
    y_grid = np.linspace(-0.99, 0.99, args.n_y)

    # patch records by patch_id
    record_map = {int(r["patch_id"]): r for r in registry.get("patch_records", [])}

    print("=" * 100)
    print(f"[info] registry: {registry_path}")
    print(f"[info] patch count in cover: {patch_cover.n_patches}")
    print(f"[info] patch count in registry: {len(record_map)}")
    print(f"[info] output dir: {out_root}")
    print("=" * 100)

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

        # 当前 patch 覆盖到的 safe 点
        mask = (
            (np.abs(uv_points[:, 0] - patch.u_center) <= patch.h_u)
            & (np.abs(uv_points[:, 1] - patch.v_center) <= patch.h_v)
        )
        patch_uv = uv_points[mask]
        patch_aw = aw_points[mask]

        if len(patch_aw) == 0:
            print(f"[skip] patch {patch_id}: no safe points in patch")
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
            rp = mode.rp
            r_min = rp + args.r_eps
            r_grid = np.linspace(r_min, args.r_max, args.n_r)

            # model
            r_from_y, R_model_y = evaluate_model_rin_on_y(
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
            R_model_r = evaluate_model_rin_on_r(
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

            # spectral 3-patch
            spec = TeukRadAmplitudeIn3PatchWithProfile(
                mode,
                N_left=args.N_left,
                N_mid=args.N_mid,
                N_right=args.N_right,
                z1=args.z1,
                z2=args.z2,
            )
            z_grid = 0.5 * (y_grid + 1.0)
            P_y, _, _ = Leaver_prefactors(
                r=torch.tensor(r_from_y, dtype=dtype, device=device),
                a=torch.tensor(a, dtype=dtype, device=device),
                omega=torch.tensor(omega, dtype=dtype, device=device),
                m=m_mode,
                M=M,
                s=s_mode,
            )
            R_spec_y = spec.profile.R_of_z(z_grid) / P_y.detach().cpu().numpy()
            R_spec_r = spec.profile.R_of_r(r_grid)

            err_y = safe_rel_err(R_model_y, R_spec_y)
            err_r = safe_rel_err(R_model_r, R_spec_r)

            meta_title = (
                f"patch={patch_id:03d}, sample={isamp:02d}, ckpt={args.ckpt_kind}\n"
                f"a={a:.8f}, omega={omega:.8f}, u={u:.8f}, v={v:.8f}\n"
                f"3patch spectral: N_left={args.N_left}, N_mid={args.N_mid}, "
                f"N_right={args.N_right}, z1={args.z1}, z2={args.z2}"
            )

            fig_name = (
                f"patch_{patch_id:03d}"
                f"_sample_{isamp:02d}"
                f"_a_{slug_float(a)}"
                f"_omega_{slug_float(omega)}"
                f"_u_{slug_float(u)}"
                f"_v_{slug_float(v)}"
                f"_{args.ckpt_kind}.png"
            )
            fig_path = patch_dir / fig_name

            make_compare_figure(
                r_grid=r_grid,
                R_model_r=R_model_r,
                R_spec_r=R_spec_r,
                y_grid=y_grid,
                R_model_y=R_model_y,
                R_spec_y=R_spec_y,
                meta_title=meta_title,
                out_path=fig_path,
            )

            summary_rows.append(
                {
                    "patch_id": patch_id,
                    "sample_id": isamp,
                    "checkpoint_kind": args.ckpt_kind,
                    "checkpoint_path": str(ckpt_path),
                    "a": a,
                    "omega": omega,
                    "u": u,
                    "v": v,
                    "N_left": args.N_left,
                    "N_mid": args.N_mid,
                    "N_right": args.N_right,
                    "z1": args.z1,
                    "z2": args.z2,
                    "median_rel_err_y": float(np.median(err_y)),
                    "max_rel_err_y": float(np.max(err_y)),
                    "median_rel_err_r": float(np.median(err_r)),
                    "max_rel_err_r": float(np.max(err_r)),
                    "spectral_B_inc_re": float(spec.B_inc.real),
                    "spectral_B_inc_im": float(spec.B_inc.imag),
                    "spectral_B_ref_re": float(spec.B_ref.real),
                    "spectral_B_ref_im": float(spec.B_ref.imag),
                    "figure": str(fig_path),
                }
            )

            print(
                f"[patch {patch_id:03d}] sample {isamp:02d}: "
                f"a={a:.6f}, omega={omega:.6e}, "
                f"med_err_y={np.median(err_y):.3e}, max_err_y={np.max(err_y):.3e}, "
                f"med_err_r={np.median(err_r):.3e}, max_err_r={np.max(err_r):.3e}"
            )

    # save summary csv
    summary_csv = out_root / f"summary_{args.ckpt_kind}.csv"
    if summary_rows:
        keys = list(summary_rows[0].keys())
        import csv
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_rows)

    print("=" * 100)
    print(f"[saved] {summary_csv}")
    print(f"[done] figures under {out_root}")
    print("=" * 100)


if __name__ == "__main__":
    main()
