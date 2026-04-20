from __future__ import annotations
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
from inference.atlas_predictor import AtlasPredictor
from mma.rin_sampler import MathematicaRinSampler
from physical_ansatz.mapping import r_plus
from physical_ansatz.transform_y import g_factor, h_factor
from physical_ansatz.prefactor import Leaver_prefactors, prefactor_Q, U_prefactor
from physical_ansatz.residual import AuxCache, get_ramp_and_p_from_cfg


def resolve_eval_point_from_registry(registry: dict, patch_id: int | None, a: float | None, omega: float | None):
    """
    优先级：
      1) 如果显式给了 a 和 omega，就直接用
      2) 否则如果给了 patch_id，就从 registry 的 patch_records 里取 patch 中心物理点
    """
    if a is not None and omega is not None:
        return float(a), float(omega), patch_id

    if patch_id is None:
        raise ValueError("Need either (--a and --omega) or --patch-id.")

    for rec in registry.get("patch_records", []):
        if int(rec["patch_id"]) == int(patch_id):
            if "a_center" not in rec or "omega_center" not in rec:
                raise KeyError(
                    f"patch_id={patch_id} record exists, but no a_center / omega_center found in registry."
                )
            return float(rec["a_center"]), float(rec["omega_center"]), int(patch_id)

    raise KeyError(f"patch_id={patch_id} not found in registry.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark atlas predictor against MMA.")
    parser.add_argument("--registry-json", type=str, required=True)

    # 你可以直接给 a, omega；也可以给 patch-id 自动取 patch 中心
    parser.add_argument("--patch-id", type=int, default=None)
    parser.add_argument("--a", type=float, default=None)
    parser.add_argument("--omega", type=float, default=None)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-r", type=int, default=256)
    parser.add_argument("--n-y", type=int, default=None)
    parser.add_argument("--y-grid-mode", choices=["chebyshev", "uniform"], default="chebyshev")
    parser.add_argument("--viz-r-min", type=float, default=2.0)
    parser.add_argument("--viz-r-max", type=float, default=80.0)
    parser.add_argument("--out-dir", type=str, default="outputs/atlas_benchmark")
    args = parser.parse_args()

    registry_path = Path(args.registry_json).resolve()
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    a_eval, omega_eval, patch_id_used = resolve_eval_point_from_registry(
        registry=registry,
        patch_id=args.patch_id,
        a=args.a,
        omega=args.omega,
    )

    out_dir = Path(args.out_dir)
    tag = (
        f"patch_{patch_id_used:03d}" if patch_id_used is not None
        else f"a_{a_eval:.6f}_omega_{omega_eval:.6f}"
    )
    out_dir = out_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = AtlasPredictor(str(registry_path), device=args.device)
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
    r_min = max(float(args.viz_r_min), float(rp.detach().cpu().item()) + 1.0e-4)
    r_max = float(args.viz_r_max)
    n_y = int(args.n_r if args.n_y is None else args.n_y)

    # 左列：r 上均匀采样
    r_grid = torch.linspace(r_min, r_max, args.n_r, device=device, dtype=dtype)
    x_grid = rp / r_grid
    y_grid_from_r = 2.0 * x_grid - 1.0

    # 右列：单独在 y 上采样
    y_min = 2.0 * float(rp.detach().cpu().item()) / r_max - 1.0
    y_max = 2.0 * float(rp.detach().cpu().item()) / r_min - 1.0
    if args.y_grid_mode == "chebyshev":
        y_grid = sample_points_chebyshev_grid(
            n_points=n_y,
            y_min=y_min,
            y_max=y_max,
            device=device,
            dtype=dtype,
        )
    else:
        y_grid = torch.linspace(y_min, y_max, n_y, device=device, dtype=dtype)
    x_grid_from_y = 0.5 * (y_grid + 1.0)
    r_grid_from_y = rp / x_grid_from_y

    # ---------------------------------------------------------
    # 1) atlas stitching 预测 f(y)
    # ---------------------------------------------------------
    f_pred_r = predictor.predict_f(a=float(a_eval), omega=float(omega_eval), y=y_grid_from_r)
    f_pred_y = predictor.predict_f(a=float(a_eval), omega=float(omega_eval), y=y_grid)

    # ---------------------------------------------------------
    # 2) 重构 R' 和 R
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 3) MMA 参考值
    # ---------------------------------------------------------
    mma_cfg = full_cfg["train"]["mathematica"]
    sampler = MathematicaRinSampler(
        kernel_path=mma_cfg["kernel_path"],
        wl_path_win=mma_cfg["wl_path_win"],
    )
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
    finally:
        sampler.close()

    R_mma_t = torch.as_tensor(R_mma, device=device, dtype=torch.complex128)
    Rprime_mma_y = torch.as_tensor(R_mma_y, device=device, dtype=torch.complex128) / U_y

    # ---------------------------------------------------------
    # 4) 误差指标
    # ---------------------------------------------------------
    eps = 1.0e-12
    rel_R_l2 = torch.linalg.norm(R_pred - R_mma_t) / (torch.linalg.norm(R_mma_t) + eps)
    rel_Rprime_l2 = torch.linalg.norm(Rprime_pred_y - Rprime_mma_y) / (torch.linalg.norm(Rprime_mma_y) + eps)

    abs_R_max = torch.max(torch.abs(R_pred - R_mma_t))
    abs_Rprime_max = torch.max(torch.abs(Rprime_pred_y - Rprime_mma_y))

    summary = {
        "registry_json": str(registry_path),
        "patch_id": None if patch_id_used is None else int(patch_id_used),
        "a": float(a_eval),
        "omega": float(omega_eval),
        "n_r": int(args.n_r),
        "n_y": int(n_y),
        "y_grid_mode": str(args.y_grid_mode),
        "r_min": float(r_min),
        "r_max": float(r_max),
        "rel_R_l2": float(rel_R_l2.detach().cpu().item()),
        "rel_Rprime_l2": float(rel_Rprime_l2.detach().cpu().item()),
        "abs_R_max": float(abs_R_max.detach().cpu().item()),
        "abs_Rprime_max": float(abs_Rprime_max.detach().cpu().item()),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    np.savez_compressed(
        out_dir / "benchmark_data.npz",
        r_grid=r_grid.detach().cpu().numpy(),
        y_grid_from_r=y_grid_from_r.detach().cpu().numpy(),
        y_grid=y_grid.detach().cpu().numpy(),
        r_grid_from_y=r_grid_from_y.detach().cpu().numpy(),
        R_pred=R_pred.detach().cpu().numpy(),
        R_mma=np.asarray(R_mma),
        Rprime_pred_y=Rprime_pred_y.detach().cpu().numpy(),
        Rprime_mma_y=Rprime_mma_y.detach().cpu().numpy(),
    )

    # ---------------------------------------------------------
    # 5) 画图
    # ---------------------------------------------------------
    r_np = r_grid.detach().cpu().numpy()
    y_np = y_grid.detach().cpu().numpy()
    R_pred_np = R_pred.detach().cpu().numpy()
    R_mma_np = np.asarray(R_mma)
    Rprime_pred_np = Rprime_pred_y.detach().cpu().numpy()
    Rprime_mma_np = Rprime_mma_y.detach().cpu().numpy()

    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=False)

    axes[0, 0].plot(r_np, np.real(R_pred_np), label="Atlas Re(R)", lw=1.6)
    axes[0, 0].plot(r_np, np.real(R_mma_np), "--", label="MMA Re(R)", lw=1.0)
    axes[0, 0].set_ylabel("Re(R)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[1, 0].plot(r_np, np.imag(R_pred_np), label="Atlas Im(R)", lw=1.6)
    axes[1, 0].plot(r_np, np.imag(R_mma_np), "--", label="MMA Im(R)", lw=1.0)
    axes[1, 0].set_ylabel("Im(R)")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[2, 0].plot(r_np, np.abs(R_pred_np), label="Atlas |R|", lw=1.6)
    axes[2, 0].plot(r_np, np.abs(R_mma_np), "--", label="MMA |R|", lw=1.0)
    axes[2, 0].set_ylabel("|R|")
    axes[2, 0].set_xlabel("r")
    axes[2, 0].legend()
    axes[2, 0].grid(alpha=0.3)

    axes[0, 1].plot(y_np, np.real(Rprime_pred_np), label="Atlas Re(R')", lw=1.6)
    axes[0, 1].plot(y_np, np.real(Rprime_mma_np), "--", label="MMA Re(R')", lw=1.0)
    axes[0, 1].set_ylabel("Re(R')")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 1].plot(y_np, np.imag(Rprime_pred_np), label="Atlas Im(R')", lw=1.6)
    axes[1, 1].plot(y_np, np.imag(Rprime_mma_np), "--", label="MMA Im(R')", lw=1.0)
    axes[1, 1].set_ylabel("Im(R')")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    axes[2, 1].plot(y_np, np.abs(Rprime_pred_np), label="Atlas |R'|", lw=1.6)
    axes[2, 1].plot(y_np, np.abs(Rprime_mma_np), "--", label="MMA |R'|", lw=1.0)
    axes[2, 1].set_ylabel("|R'|")
    axes[2, 1].set_xlabel("y")
    axes[2, 1].legend()
    axes[2, 1].grid(alpha=0.3)

    fig.suptitle(
        f"Atlas vs MMA\n"
        f"a={a_eval:.6f}, omega={omega_eval:.6f}, "
        f"rel_R={summary['rel_R_l2']:.3e}, rel_R'={summary['rel_Rprime_l2']:.3e}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "atlas_vs_mma.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved figure  -> {out_dir / 'atlas_vs_mma.png'}")
    print(f"saved summary -> {out_dir / 'summary.json'}")
    print(f"saved data    -> {out_dir / 'benchmark_data.npz'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
