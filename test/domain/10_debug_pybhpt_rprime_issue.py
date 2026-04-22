from __future__ import annotations
import sys

sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky/pybhpt")
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config.config_loader import load_pinn_full_config
from physical_ansatz.mapping import r_plus
from physical_ansatz.prefactor import (
    Leaver_prefactors,
    prefactor_Q,
    U_prefactor,
    build_prefactor_primitives,
)
from physical_ansatz.residual import AuxCache, get_ramp_and_p_from_cfg
from physical_ansatz.transform_y import h_factor
from compute_solution import compute_pybhpt_solution


def complex_median(z: np.ndarray) -> complex:
    return np.median(z.real) + 1j * np.median(z.imag)


def main():
    parser = argparse.ArgumentParser(description="Diagnose pybhpt -> R' conversion issue.")
    parser.add_argument("--cfg", type=str, default="config/pinn_config.yaml")
    parser.add_argument("--a", type=float, required=True)
    parser.add_argument("--omega", type=float, required=True)
    parser.add_argument("--n-r", type=int, default=256)
    parser.add_argument("--viz-r-min", type=float, default=2.0)
    parser.add_argument("--viz-r-max", type=float, default=80.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out-dir", type=str, default="outputs/debug_pybhpt_rprime_issue")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_cfg = load_pinn_full_config(args.cfg)
    physics_cfg = full_cfg["physics"]
    problem_cfg = physics_cfg["problem"]

    M = float(problem_cfg.get("M", 1.0))
    l = int(problem_cfg.get("l", 2))
    m = int(problem_cfg.get("m", 2))
    s = int(problem_cfg.get("s", -2))

    dtype = torch.float64
    device = torch.device("cpu")

    a_t = torch.tensor(float(args.a), device=device, dtype=dtype)
    omega_t = torch.tensor(float(args.omega), device=device, dtype=dtype)

    rp = r_plus(a_t, M)
    r_min = max(float(args.viz_r_min), float(rp.detach().cpu().item()) + 1.0e-4)
    r_max = float(args.viz_r_max)

    r_grid = torch.linspace(r_min, r_max, args.n_r, device=device, dtype=dtype)
    x_grid = rp / r_grid
    y_grid = 2.0 * x_grid - 1.0

    # 当前 ansatz/prefactor 链路中的 U
    cache = AuxCache()
    p, ramp = get_ramp_and_p_from_cfg(physics_cfg, cache, a_t, omega_t)
    ramp_t = ramp.to(device=device, dtype=torch.complex128)

    rp_v, rm_v, rs_v, rs_r_v, rs_rr_v = build_prefactor_primitives(r_grid, a_t, M=M)
    P, P_r, P_rr = Leaver_prefactors(
        r_grid, a_t, omega_t, m=m, M=M, s=s, rp=rp_v, rm=rm_v
    )
    Q, Q_r, Q_rr = prefactor_Q(
        r_grid, a_t, omega_t,
        p=int(p), R_amp=ramp_t, M=M, s=s,
        rp=rp_v, rs=rs_v, rs_r=rs_r_v, rs_rr=rs_rr_v,
    )
    U, _, _ = U_prefactor(P, P_r, P_rr, Q, Q_r, Q_rr)

    # pybhpt benchmark
    _, R_ref = compute_pybhpt_solution(
        a=float(args.a),
        omega=float(args.omega),
        ell=l,
        m=m,
        r_grid=r_grid.detach().cpu().numpy(),
        timeout=float(args.timeout),
    )
    R_ref_t = torch.as_tensor(R_ref, device=device, dtype=torch.complex128)

    # 直接除 U 得到“raw R'”
    Rprime_raw = R_ref_t / U

    # 近视界理论上 R' -> h(a,omega)
    h = h_factor(a_t, omega_t, m=m, M=M, s=s)
    h_c = complex(h.detach().cpu().item())

    # 用靠近视界但不取最末端的几个点估计归一化常数 C
    # y 越大越接近视界，这里取最后 12 个点，去掉最末端 2 个点
    idx0 = max(0, args.n_r - 14)
    idx1 = max(0, args.n_r - 2)
    sl = slice(idx0, idx1)

    ratio_to_h = Rprime_raw.detach().cpu().numpy()[sl] / h_c
    C_est = complex_median(ratio_to_h)

    # 用这个 C_est 重新缩放 pybhpt 参考
    R_ref_scaled = R_ref_t / C_est
    Rprime_scaled = R_ref_scaled / U

    summary = {
        "a": float(args.a),
        "omega": float(args.omega),
        "n_r": int(args.n_r),
        "r_min": float(r_min),
        "r_max": float(r_max),
        "p": int(p),
        "h": {"real": float(np.real(h_c)), "imag": float(np.imag(h_c))},
        "C_est": {"real": float(np.real(C_est)), "imag": float(np.imag(C_est))},
        "U_abs_min": float(torch.min(torch.abs(U)).detach().cpu().item()),
        "U_abs_max": float(torch.max(torch.abs(U)).detach().cpu().item()),
        "raw_Rprime_abs_max": float(torch.max(torch.abs(Rprime_raw)).detach().cpu().item()),
        "scaled_Rprime_abs_max": float(torch.max(torch.abs(Rprime_scaled)).detach().cpu().item()),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    r_np = r_grid.detach().cpu().numpy()
    y_np = y_grid.detach().cpu().numpy()
    U_np = U.detach().cpu().numpy()
    R_ref_np = R_ref_t.detach().cpu().numpy()
    Rprime_raw_np = Rprime_raw.detach().cpu().numpy()
    Rprime_scaled_np = Rprime_scaled.detach().cpu().numpy()

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=False)

    axes[0, 0].plot(r_np, np.real(R_ref_np), label="pybhpt Re(R)", lw=1.4)
    axes[0, 0].set_ylabel("Re(R)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[1, 0].plot(r_np, np.imag(R_ref_np), label="pybhpt Im(R)", lw=1.4)
    axes[1, 0].set_ylabel("Im(R)")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[2, 0].plot(r_np, np.abs(U_np), label="|U|", lw=1.4)
    axes[2, 0].set_ylabel("|U|")
    axes[2, 0].set_xlabel("r")
    axes[2, 0].legend()
    axes[2, 0].grid(alpha=0.3)

    axes[0, 1].plot(y_np, np.real(Rprime_raw_np), label="raw Re(R') = R/U", lw=1.2)
    axes[0, 1].plot(y_np, np.real(Rprime_scaled_np), label="scaled Re(R')", lw=1.2)
    axes[0, 1].axhline(np.real(h_c), color="k", ls="--", lw=1.0, label="Re(h)")
    axes[0, 1].set_ylabel("Re(R')")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 1].plot(y_np, np.imag(Rprime_raw_np), label="raw Im(R') = R/U", lw=1.2)
    axes[1, 1].plot(y_np, np.imag(Rprime_scaled_np), label="scaled Im(R')", lw=1.2)
    axes[1, 1].axhline(np.imag(h_c), color="k", ls="--", lw=1.0, label="Im(h)")
    axes[1, 1].set_ylabel("Im(R')")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    axes[2, 1].plot(y_np, np.abs(Rprime_raw_np), label="|raw R'|", lw=1.2)
    axes[2, 1].plot(y_np, np.abs(Rprime_scaled_np), label="|scaled R'|", lw=1.2)
    axes[2, 1].set_ylabel("|R'|")
    axes[2, 1].set_xlabel("y")
    axes[2, 1].legend()
    axes[2, 1].grid(alpha=0.3)

    fig.suptitle(
        f"pybhpt -> R' diagnostic\n"
        f"a={args.a:.6f}, omega={args.omega:.6f}, "
        f"C_est={np.real(C_est):.3e}+{np.imag(C_est):.3e}i",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "debug_pybhpt_rprime_issue.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved figure  -> {out_dir / 'debug_pybhpt_rprime_issue.png'}")
    print(f"saved summary -> {out_dir / 'summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()