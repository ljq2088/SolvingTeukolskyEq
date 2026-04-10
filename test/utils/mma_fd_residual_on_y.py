#!/usr/bin/env python3
"""
在 y 网格上调用 Mathematica 采样 Teukolsky 径向解 R(r)，
再用当前项目的 ansatz

    R(r) = U(r) * R'(r)

恢复 regular part R'(y)，并用有限差分计算

    dR'/dy, d²R'/dy²

最后代入 y 坐标下的方程

    B2(y) * R'_{yy} + B1(y) * R'_y + B0(y) * R' = 0

得到残差。

说明：
1. 这份脚本按你上传 zip 的“直接训练 R'”逻辑写，不再走旧的 g(x)f(x)+1 结构。
2. 上传版代码里 physical_ansatz/prefactor.py 的 U_factor / U_factor_r / U_factor_r_r
   对 Inf_prefactor 的位置参数传递有错位；这里在脚本里显式用关键字参数修正，
   否则 U 和系数会错。
3. 这里优先调用 Mathematica 侧的 SampleRinOnX[...]，这样 y/x 采样点和 MMA 返回点一一对应。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr


def find_project_root(start: Path) -> Path:
    """自动寻找项目根目录。"""
    candidates = [start] + list(start.parents)
    for p in candidates:
        if (p / "config" / "teukolsky_radial.yaml").exists() and (p / "physical_ansatz").exists():
            return p
    raise FileNotFoundError(
        "未找到项目根目录；请把脚本放在 SolvingTeukolsky 项目内部，"
        "或用 --project-root 显式指定。"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 MMA + 有限差分检查 R' 方程残差")
    parser.add_argument("--project-root", type=str, default=None, help="SolvingTeukolsky 项目根目录")
    parser.add_argument("--cfg", type=str, default="config/teukolsky_radial.yaml", help="配置文件路径")
    parser.add_argument("--kernel-path", type=str, default="/mnt/f/mma/WolframKernel.exe")
    parser.add_argument("--wl-path-win", type=str, default="F:/EMRI/Radial_flow/Radial_Function.wl")

    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--omega", type=float, default=0.1)
    parser.add_argument("--l", type=int, default=None, help="默认读取 cfg.problem.l")
    parser.add_argument("--m", type=int, default=None, help="默认读取 cfg.problem.m")
    parser.add_argument("--s", type=int, default=None, help="默认读取 cfg.problem.s")
    parser.add_argument("--M", type=float, default=None, help="默认读取 cfg.problem.M")

    parser.add_argument("--y-min", type=float, default=-0.998)
    parser.add_argument("--y-max", type=float, default=0.998)
    parser.add_argument("--npts", type=int, default=801)

    parser.add_argument("--outdir", type=str, default="outputs/mma_fd_residual")
    parser.add_argument("--save-json", action="store_true", help="额外保存 summary.json")
    return parser.parse_args()


# -------------------------
# 有限差分
# -------------------------
def finite_diff_first_second(y: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对复值函数做 y 上的一阶、二阶有限差分。"""
    if y.ndim != 1 or f.ndim != 1:
        raise ValueError("y 和 f 都必须是一维数组")
    if y.size != f.size:
        raise ValueError("y 和 f 长度不一致")
    fy = np.gradient(f, y, edge_order=2)
    fyy = np.gradient(fy, y, edge_order=2)
    return fy, fyy


# -------------------------
# Mathematica 采样
# -------------------------
def sample_rin_on_x(
    kernel_path: str,
    wl_path_win: str,
    s: int,
    ell: int,
    m: int,
    a: float,
    omega: float,
    x_min: float,
    x_max: float,
    npts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """优先按 x 采样，这样与 y=2x-1 的网格严格对应。"""
    session = WolframLanguageSession(kernel=kernel_path)
    try:
        session.evaluate(wlexpr(rf'Get["{wl_path_win}"]'))
        expr = (
            f"SampleRinOnX[{s}, {ell}, {m}, "
            f"{a:.16g}, {omega:.16g}, "
            f"{x_min:.16g}, {x_max:.16g}, {npts}]"
        )
        result = session.evaluate(wlexpr(expr))
        arr = np.array(result, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"Unexpected Mathematica output shape: {arr.shape}")
        r = arr[:, 0]
        R = arr[:, 1] + 1j * arr[:, 2]
        return r, R
    finally:
        session.terminate()


def main() -> None:
    args = parse_args()

    script_path = Path(__file__).resolve()
    if args.project_root is None:
        project_root = find_project_root(script_path.parent)
    else:
        project_root = Path(args.project_root).resolve()

    sys.path.insert(0, str(project_root))

    # 项目内导入放在这里，确保 sys.path 已经就位
    from physical_ansatz.mapping import r_plus, r_from_x, dx_dr_from_x, d2x_dr2_from_x
    from physical_ansatz.prefactor import (
        delta,
        delta_r,
        V_of_r,
        prefactor_Q,
        prefactor_Q_r,
        prefactor_Q_r_r,
        Inf_prefactor,
        Inf_prefactor_r,
        Inf_prefactor_r_r,
    )
    from physical_ansatz.residual import AuxCache, get_lambda_from_cfg, get_ramp_and_p_from_cfg

    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prob = cfg["problem"]
    ell = int(prob["l"] if args.l is None else args.l)
    m = int(prob["m"] if args.m is None else args.m)
    s = int(prob.get("s", -2) if args.s is None else args.s)
    M = float(prob.get("M", 1.0) if args.M is None else args.M)

    if s != -2:
        print("[warning] 这份脚本主要按当前项目的 s=-2 用法检查；其它 s 值请自行再核对 Inf_prefactor_r_r 公式。")

    y = np.linspace(args.y_min, args.y_max, args.npts, dtype=np.float64)
    x = 0.5 * (y + 1.0)

    if np.any(x <= 0.0) or np.any(x >= 1.0):
        raise ValueError("当前实现要求 x∈(0,1)，请把 y_min,y_max 设在 (-1,1) 内部。")

    a_t = torch.tensor(args.a, dtype=torch.float64)
    omega_t = torch.tensor(args.omega, dtype=torch.float64)
    rp = r_plus(a_t, M)
    rp_val = float(rp.item())

    x_min = float(x.min())
    x_max = float(x.max())
    print(f"[info] project_root = {project_root}")
    print(f"[info] cfg          = {cfg_path}")
    print(f"[info] params       = (s,l,m,a,omega)=({s},{ell},{m},{args.a},{args.omega})")
    print(f"[info] y range      = [{args.y_min}, {args.y_max}], npts={args.npts}")
    print(f"[info] x range      = [{x_min}, {x_max}]")

    # ---------- 1. MMA 给出 R ----------
    r_mma, R_mma = sample_rin_on_x(
        kernel_path=args.kernel_path,
        wl_path_win=args.wl_path_win,
        s=s,
        ell=ell,
        m=m,
        a=args.a,
        omega=args.omega,
        x_min=x_min,
        x_max=x_max,
        npts=args.npts,
    )

    r_from_x_np = rp_val / x
    r_alignment_err = float(np.max(np.abs(r_mma - r_from_x_np)))
    print(f"[info] max |r_mma - r_plus/x| = {r_alignment_err:.6e}")

    # ---------- 2. 由 ansatz 恢复 R' ----------
    cache = AuxCache()
    lam_t = get_lambda_from_cfg(cfg, cache, a_t, omega_t)
    p, ramp_t = get_ramp_and_p_from_cfg(cfg, cache, a_t, omega_t)

    r_t = torch.tensor(r_mma, dtype=torch.float64)
    x_t = torch.tensor(x, dtype=torch.float64)

    # 这里显式修正 uploaded 版本中 U_factor 对 Inf_prefactor 的位置参数错位问题
    def U_factor_fixed(
        r: torch.Tensor,
        a: torch.Tensor,
        omega: torch.Tensor,
        p: int,
        R_amp: torch.Tensor,
        m: int,
        s: int = -2,
        M: float = 1.0,
    ) -> torch.Tensor:
        Q = prefactor_Q(r, a, omega, p, R_amp, M=M, s=s)
        I = Inf_prefactor(r, a, omega, M=M, s=s)
        return Q * I

    def U_factor_r_fixed(
        r: torch.Tensor,
        a: torch.Tensor,
        omega: torch.Tensor,
        p: int,
        R_amp: torch.Tensor,
        m: int,
        s: int = -2,
        M: float = 1.0,
    ) -> torch.Tensor:
        Q = prefactor_Q(r, a, omega, p, R_amp, M=M, s=s)
        Q_r = prefactor_Q_r(r, a, omega, p, R_amp, M=M, s=s)
        I = Inf_prefactor(r, a, omega, M=M, s=s)
        I_r = Inf_prefactor_r(r, a, omega, M=M, s=s)
        return Q_r * I + Q * I_r

    def U_factor_rr_fixed(
        r: torch.Tensor,
        a: torch.Tensor,
        omega: torch.Tensor,
        p: int,
        R_amp: torch.Tensor,
        m: int,
        s: int = -2,
        M: float = 1.0,
    ) -> torch.Tensor:
        Q = prefactor_Q(r, a, omega, p, R_amp, M=M, s=s)
        Q_r = prefactor_Q_r(r, a, omega, p, R_amp, M=M, s=s)
        Q_rr = prefactor_Q_r_r(r, a, omega, p, R_amp, M=M, s=s)
        I = Inf_prefactor(r, a, omega,  M=M)
        I_r = Inf_prefactor_r(r, a, omega, M=M)
        I_rr = Inf_prefactor_r_r(r, a, omega, M=M)
        return Q_rr * I + 2.0 * Q_r * I_r + Q * I_rr

    def coeffs_x_fixed(
        x: torch.Tensor,
        a: torch.Tensor,
        omega: torch.Tensor,
        m: int,
        p: int,
        R_amp: torch.Tensor,
        lambda_: torch.Tensor,
        s: int = -2,
        M: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = r_from_x(x, r_plus(a, M))
        Delta = delta(r, a, M)
        Delta_r = delta_r(r, M)
        V = V_of_r(r, a, omega, m, s, lambda_, M)

        U = U_factor_fixed(r, a, omega, p, R_amp, m, s=s, M=M)
        U_r = U_factor_r_fixed(r, a, omega, p, R_amp, m, s=s, M=M)
        U_rr = U_factor_rr_fixed(r, a, omega, p, R_amp, m, s=s, M=M)

        dx_dr = dx_dr_from_x(x, r_plus(a, M))
        d2x_dr2 = d2x_dr2_from_x(x, r_plus(a, M))
        lnU_r = U_r / U

        A2 = Delta * (dx_dr ** 2)
        A1 = Delta * (2.0 * dx_dr * lnU_r + d2x_dr2) + (s + 1) * Delta_r * dx_dr
        A0 = V + (s + 1) * Delta_r * lnU_r + Delta * U_rr / U
        return A2, A1, A0

    U_t = U_factor_fixed(r_t, a_t, omega_t, p, ramp_t, m, s=s, M=M)
    Rprime_t = torch.tensor(R_mma, dtype=torch.complex128) / U_t.to(torch.complex128)
    Rprime = Rprime_t.detach().cpu().numpy()

    # ---------- 3. 有限差分导数 ----------
    Rprime_y, Rprime_yy = finite_diff_first_second(y, Rprime)

    # ---------- 4. 系数 B2,B1,B0 ----------
    A2_t, A1_t, A0_t = coeffs_x_fixed(x_t, a_t, omega_t, m, p, ramp_t, lam_t, s=s, M=M)
    B2 = (4.0 * A2_t).detach().cpu().numpy().astype(np.complex128)
    B1 = (2.0 * A1_t).detach().cpu().numpy().astype(np.complex128)
    B0 = A0_t.detach().cpu().numpy().astype(np.complex128)

    # ---------- 5. 残差 ----------
    residual = B2 * Rprime_yy + B1 * Rprime_y + B0 * Rprime

    # ---------- 6. 统计与保存 ----------
    abs_res = np.abs(residual)
    idx_max = int(np.argmax(abs_res))
    idx_mean_ref = int(abs(abs_res - abs_res.mean()).argmin())

    summary = {
        "project_root": str(project_root),
        "cfg": str(cfg_path),
        "params": {
            "M": M,
            "s": s,
            "l": ell,
            "m": m,
            "a": args.a,
            "omega": args.omega,
            "p": int(p),
            "lambda": [float(lam_t.real.item()), float(lam_t.imag.item())],
            "R_amp": [float(ramp_t.real.item()), float(ramp_t.imag.item())],
        },
        "grid": {
            "y_min": float(y.min()),
            "y_max": float(y.max()),
            "x_min": float(x.min()),
            "x_max": float(x.max()),
            "npts": int(args.npts),
        },
        "r_alignment_max_abs": r_alignment_err,
        "residual": {
            "mean_abs": float(abs_res.mean()),
            "median_abs": float(np.median(abs_res)),
            "max_abs": float(abs_res[idx_max]),
            "y_at_max": float(y[idx_max]),
            "x_at_max": float(x[idx_max]),
            "r_at_max": float(r_mma[idx_max]),
            "sample_mean_like_y": float(y[idx_mean_ref]),
            "sample_mean_like_r": float(r_mma[idx_mean_ref]),
        },
    }

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = project_root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    np.savez(
        outdir / "mma_fd_residual_data.npz",
        y=y,
        x=x,
        r=r_mma,
        R_real=R_mma.real,
        R_imag=R_mma.imag,
        U_real=U_t.detach().cpu().numpy().real,
        U_imag=U_t.detach().cpu().numpy().imag,
        Rprime_real=Rprime.real,
        Rprime_imag=Rprime.imag,
        Rprime_y_real=Rprime_y.real,
        Rprime_y_imag=Rprime_y.imag,
        Rprime_yy_real=Rprime_yy.real,
        Rprime_yy_imag=Rprime_yy.imag,
        B2_real=B2.real,
        B2_imag=B2.imag,
        B1_real=B1.real,
        B1_imag=B1.imag,
        B0_real=B0.real,
        B0_imag=B0.imag,
        residual_real=residual.real,
        residual_imag=residual.imag,
        residual_abs=abs_res,
    )

    if args.save_json:
        with open(outdir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== Summary =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[data] saved npz to: {outdir / 'mma_fd_residual_data.npz'}")
    if args.save_json:
        print(f"[data] saved json to: {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
