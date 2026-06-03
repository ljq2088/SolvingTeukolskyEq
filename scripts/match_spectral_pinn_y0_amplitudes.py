#!/usr/bin/env python3
"""Match Spectral-PINN branch solutions at physical y=0 to get amplitudes.

This intentionally uses the trained coefficient networks, not the direct
spectral teacher.  Physical coordinate convention:
    y = 2 r_+ / r - 1,  horizon y=1, infinity y=-1.

At y=0:
    R_in = B_inc R_down + B_ref R_up
and the same equation is imposed on dR/dr.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_horizon_spectral_coeff import ParamToCoeff as HorizonParamToCoeff  # noqa: E402
from scripts.train_infinity_spectral_coeff import ParamToCoeff as InfinityParamToCoeff  # noqa: E402
from utils.amplitude import A_down, A_in, A_up, dz_dr, q_and_qp, r_of_z  # noqa: E402
from utils.mode import KerrMode  # noqa: E402


RDTYPE = torch.float64


def complex_to_pair(value: complex) -> dict[str, float]:
    return {"re": float(np.real(value)), "im": float(np.imag(value)), "abs": float(abs(value))}


def rel_err(value: complex, ref: complex) -> float:
    return float(abs(value - ref) / max(abs(ref), 1.0e-300))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cheb_endpoint_value_derivative(coeff: np.ndarray, xi: float, half_width: float) -> tuple[complex, complex]:
    """Return f and df/dy at xi=±1 for Chebyshev coefficients."""
    if xi not in (-1.0, 1.0):
        raise ValueError("This helper is only for endpoints xi=±1.")
    k = np.arange(coeff.shape[0], dtype=np.float64)
    if xi == 1.0:
        T = np.ones_like(k)
        dT_dxi = k * k
    else:
        T = (-1.0) ** k
        dT_dxi = np.zeros_like(k)
        dT_dxi[1:] = ((-1.0) ** (k[1:] - 1.0)) * k[1:] * k[1:]
    value = np.dot(T, coeff)
    dy_dxi = half_width
    deriv_y = np.dot(dT_dxi, coeff) / dy_dxi
    return complex(value), complex(deriv_y)


def load_horizon_model(run_dir: Path, device: str):
    ckpt = torch.load(run_dir / "model.pt", map_location=device, weights_only=False)
    cfg = dict(ckpt["args"])
    out_coeff = int(ckpt.get("out_coeff", cfg.get("out_coeff", cfg["n"] + 1)))
    if out_coeff <= 0:
        out_coeff = cfg["n"] + 1
    model = HorizonParamToCoeff(
        cfg["n"] + 1,
        cfg["width"],
        cfg["depth"],
        cfg["a_center"],
        cfg["logw_center"],
        cfg["a_half_width"],
        cfg["logw_half_width"],
        out_coeff=out_coeff,
    ).to(device, dtype=RDTYPE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_infinity_model(run_dir: Path, basis: str, device: str):
    ckpt = torch.load(run_dir / f"model_{basis}.pt", map_location=device, weights_only=False)
    cfg = dict(ckpt["args"])
    out_coeff = int(ckpt.get("out_coeff", cfg.get("out_coeff", cfg["n"] + 1)))
    model = InfinityParamToCoeff(
        cfg["n"] + 1,
        out_coeff,
        cfg["width"],
        cfg["depth"],
        cfg["a_center"],
        cfg["logw_center"],
        cfg["a_half_width"],
        cfg["logw_half_width"],
    ).to(device, dtype=RDTYPE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def predict_coeff(model, a: float, logw: float, device: str) -> np.ndarray:
    with torch.no_grad():
        coeff = model(
            torch.tensor([a], dtype=RDTYPE, device=device),
            torch.tensor([logw], dtype=RDTYPE, device=device),
        ).squeeze(0)
    return coeff.detach().cpu().numpy().astype(np.complex128)


def branch_R_and_Rr(mode: KerrMode, basis: str, u: complex, uy: complex) -> tuple[complex, complex]:
    z = 0.5
    r = r_of_z(z, mode)
    uz = 2.0 * uy
    if basis == "in":
        A = A_in(r, mode)
    elif basis == "down":
        A = A_down(r, mode)
    elif basis == "up":
        A = A_up(r, mode)
    else:
        raise ValueError(basis)
    q, _ = q_and_qp(r, mode, basis)
    R = A * u
    Rr = A * (dz_dr(z, mode) * uz + q * u)
    return complex(R), complex(Rr)


def match_one(
    *,
    a: float,
    omega: float,
    ell: int,
    m: int,
    s: int,
    horizon_model,
    horizon_cfg: dict,
    down_model,
    down_cfg: dict,
    up_model,
    up_cfg: dict,
    device: str,
) -> dict:
    logw = float(np.log10(omega))
    mode = KerrMode(M=1.0, a=a, omega=omega, ell=ell, m=m, s=s)

    c_in = predict_coeff(horizon_model, a, logw, device)
    c_down = predict_coeff(down_model, a, logw, device)
    c_up = predict_coeff(up_model, a, logw, device)

    y_match = float(horizon_cfg["y_match"])
    u_in, uy_in = cheb_endpoint_value_derivative(c_in, -1.0, half_width=0.5 * (1.0 - y_match))
    u_down, uy_down = cheb_endpoint_value_derivative(c_down, 1.0, half_width=0.5 * (float(down_cfg["y_right"]) + 1.0))
    u_up, uy_up = cheb_endpoint_value_derivative(c_up, 1.0, half_width=0.5 * (float(up_cfg["y_right"]) + 1.0))

    R_in, Rr_in = branch_R_and_Rr(mode, "in", u_in, uy_in)
    R_down, Rr_down = branch_R_and_Rr(mode, "down", u_down, uy_down)
    R_up, Rr_up = branch_R_and_Rr(mode, "up", u_up, uy_up)

    mat = np.array([[R_down, R_up], [Rr_down, Rr_up]], dtype=np.complex128)
    rhs = np.array([R_in, Rr_in], dtype=np.complex128)
    B_inc, B_ref = np.linalg.solve(mat, rhs)
    recon = mat @ np.array([B_inc, B_ref])
    return {
        "a": a,
        "omega": omega,
        "logw": logw,
        "lambda": mode.lambda_value,
        "B_inc": complex(B_inc),
        "B_ref": complex(B_ref),
        "B_trans": 1.0 + 0.0j,
        "match_cond": float(np.linalg.cond(mat)),
        "match_abs_res": float(np.linalg.norm(recon - rhs)),
        "match_rel_res": float(np.linalg.norm(recon - rhs) / max(np.linalg.norm(rhs), 1.0e-300)),
        "u_in_y0": u_in,
        "u_down_y0": u_down,
        "u_up_y0": u_up,
    }


def get_mma_kernel_path(raw: str | None) -> str | None:
    if raw:
        return raw if Path(raw).exists() else None
    env_path = os.environ.get("WOLFRAM_KERNEL")
    if env_path and Path(env_path).exists():
        return env_path
    for candidate in [
        "/mnt/f/mma/WolframKernel.exe",
        "/usr/local/Wolfram/Mathematica/14.0/Executables/WolframKernel",
        "/usr/local/Wolfram/Mathematica/13.0/Executables/WolframKernel",
        "/usr/local/bin/WolframKernel",
        "/usr/bin/WolframKernel",
    ]:
        if Path(candidate).exists():
            return candidate
    return None


def run_mma_script(cases: list[tuple[float, float]], *, kernel_path: str, wl_path: str, work_dir: Path, timeout: float):
    work_dir.mkdir(parents=True, exist_ok=True)
    script_path = work_dir / "mma_y0_amp_compare.wls"
    csv_path = work_dir / "mma_y0_amp_compare.csv"
    cases_wl = "{" + ",".join(f"{{{a:.17g},{omega:.17g}}}" for a, omega in cases) + "}"
    script_path.write_text(
        f"""
Get["{wl_path}"];
cases = {cases_wl};
timeout = {timeout:.17g};
headers = {{"a","omega","status","B_inc_re","B_inc_im","B_ref_re","B_ref_im","B_trans_re","B_trans_im"}};
rows = Table[
  a = case[[1]]; omega = case[[2]];
  amp = Quiet[Check[TimeConstrained[ComputeAmplitudes[-2, 2, 2, a, omega], timeout, $Aborted], $Failed]];
  If[AssociationQ[amp],
    {{a, omega, "ok", Re[N[amp["Incidence"]]], Im[N[amp["Incidence"]]], Re[N[amp["Reflection"]]], Im[N[amp["Reflection"]]], Re[N[amp["Transmission"]]], Im[N[amp["Transmission"]]]}},
    {{a, omega, ToString[amp, InputForm], "", "", "", "", "", ""}}
  ],
  {{case, cases}}
];
Export["{csv_path}", Prepend[rows, headers], "CSV"];
Quit[];
"""
    )
    proc = subprocess.run(
        [kernel_path, "-script", str(script_path)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return {}, f"mma-exit-{proc.returncode}: {proc.stderr.strip()[:500]}"
    if not csv_path.exists():
        return {}, f"mma-no-csv: {proc.stderr.strip()[:500]}"
    out = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (float(row["a"]), float(row["omega"]))
            if row["status"] != "ok":
                out[key] = (None, row["status"])
                continue
            out[key] = ({
                "B_inc": complex(float(row["B_inc_re"]), float(row["B_inc_im"])),
                "B_ref": complex(float(row["B_ref_re"]), float(row["B_ref_im"])),
                "B_trans": complex(float(row["B_trans_re"]), float(row["B_trans_im"])),
            }, "ok")
    return out, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-run", default="outputs/horizon_spectral_coeff_k16_test/20260601_013815")
    parser.add_argument("--infinity-run", default="outputs/infinity_spectral_coeff_bias_test/20260601_020702")
    parser.add_argument("--infinity-up-run", default=None)
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--omega", type=float, default=10.0 ** -1.5)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-mma", action="store_true")
    parser.add_argument("--kernel-path", default=None)
    parser.add_argument("--mma-wl", default="mma/Radial_Function.wl")
    parser.add_argument("--mma-wl-win", default="F:/EMRI/Radial_flow/Radial_Function.wl")
    parser.add_argument("--mma-timeout", type=float, default=120.0)
    parser.add_argument("--output-dir", default="outputs/spectral_pinn_y0_amplitude_match")
    args = parser.parse_args()

    torch.set_default_dtype(RDTYPE)
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    horizon_model, horizon_cfg = load_horizon_model(PROJECT_ROOT / args.horizon_run, args.device)
    down_model, down_cfg = load_infinity_model(PROJECT_ROOT / args.infinity_run, "down", args.device)
    up_run = args.infinity_up_run or args.infinity_run
    up_model, up_cfg = load_infinity_model(PROJECT_ROOT / up_run, "up", args.device)

    result = match_one(
        a=args.a,
        omega=args.omega,
        ell=args.ell,
        m=args.m,
        s=args.s,
        horizon_model=horizon_model,
        horizon_cfg=horizon_cfg,
        down_model=down_model,
        down_cfg=down_cfg,
        up_model=up_model,
        up_cfg=up_cfg,
        device=args.device,
    )

    rows = [{
        "a": result["a"],
        "omega": result["omega"],
        "logw": result["logw"],
        "B_inc_re": result["B_inc"].real,
        "B_inc_im": result["B_inc"].imag,
        "B_ref_re": result["B_ref"].real,
        "B_ref_im": result["B_ref"].imag,
        "match_cond": result["match_cond"],
        "match_rel_res": result["match_rel_res"],
    }]

    mma_status = "skipped"
    mma_result = None
    kernel_path = None
    if not args.skip_mma:
        kernel_path = get_mma_kernel_path(args.kernel_path)
        if kernel_path is None:
            mma_status = "kernel-not-found"
        else:
            wl_path = args.mma_wl_win if kernel_path.lower().endswith(".exe") else str((PROJECT_ROOT / args.mma_wl).resolve())
            mma_map, mma_status = run_mma_script(
                [(args.a, args.omega)],
                kernel_path=kernel_path,
                wl_path=wl_path,
                work_dir=run_dir,
                timeout=args.mma_timeout,
            )
            mma_result, case_status = mma_map.get((args.a, args.omega), (None, "missing"))
            mma_status = f"{mma_status}:{case_status}"
            if mma_result is not None:
                rows[0].update({
                    "mma_B_inc_re": mma_result["B_inc"].real,
                    "mma_B_inc_im": mma_result["B_inc"].imag,
                    "mma_B_ref_re": mma_result["B_ref"].real,
                    "mma_B_ref_im": mma_result["B_ref"].imag,
                    "relerr_B_inc": rel_err(result["B_inc"], mma_result["B_inc"]),
                    "relerr_B_ref": rel_err(result["B_ref"], mma_result["B_ref"]),
                })

    write_csv(run_dir / "amplitude_compare.csv", rows)
    summary = {
        "args": vars(args),
        "run_dir": str(run_dir),
        "inputs": {
            "horizon_run": str(PROJECT_ROOT / args.horizon_run),
            "infinity_run_down": str(PROJECT_ROOT / args.infinity_run),
            "infinity_run_up": str(PROJECT_ROOT / up_run),
        },
        "spectral_pinn": {
            "B_inc": complex_to_pair(result["B_inc"]),
            "B_ref": complex_to_pair(result["B_ref"]),
            "B_trans": complex_to_pair(result["B_trans"]),
            "match_cond": result["match_cond"],
            "match_abs_res": result["match_abs_res"],
            "match_rel_res": result["match_rel_res"],
            "u_in_y0": complex_to_pair(result["u_in_y0"]),
            "u_down_y0": complex_to_pair(result["u_down_y0"]),
            "u_up_y0": complex_to_pair(result["u_up_y0"]),
        },
        "mma": {
            "status": mma_status,
            "kernel_path": kernel_path,
            "B_inc": complex_to_pair(mma_result["B_inc"]) if mma_result else None,
            "B_ref": complex_to_pair(mma_result["B_ref"]) if mma_result else None,
            "B_trans": complex_to_pair(mma_result["B_trans"]) if mma_result else None,
            "relerr_B_inc": rel_err(result["B_inc"], mma_result["B_inc"]) if mma_result else None,
            "relerr_B_ref": rel_err(result["B_ref"], mma_result["B_ref"]) if mma_result else None,
        },
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
