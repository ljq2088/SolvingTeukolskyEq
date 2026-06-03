#!/usr/bin/env python3
"""Cross-validate pybhpt, MMA, GSN, and the local spectral solver near omega=0.1.

Benchmark policy for this branch:
  - pybhpt is used for R_in(r), not as trusted amplitude truth.
  - MMA and GSN are used for B_inc/B_ref when available.
  - omega ~= 0.1 is the overlap band where all benchmark conventions should
    be checked before using any source as training data.

All external backends are optional. Missing MMA/GSN is recorded as a status,
not treated as a script failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pybhpt_usage.compute_solution import compute_pybhpt_solution  # noqa: E402
from utils.amplitude import compute_smatrix, compute_smatrix_with_abel  # noqa: E402
from utils.amplitude import adaptive_match_z  # noqa: E402
from utils.mode import KerrMode  # noqa: E402


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def cdict(value: complex | None, prefix: str) -> dict[str, float | None]:
    if value is None:
        return {f"{prefix}_re": None, f"{prefix}_im": None, f"{prefix}_abs": None}
    z = complex(value)
    return {f"{prefix}_re": float(z.real), f"{prefix}_im": float(z.imag), f"{prefix}_abs": float(abs(z))}


def rel_err(value: complex | None, ref: complex | None, floor: float = 1.0e-300) -> float | None:
    if value is None or ref is None:
        return None
    return float(abs(complex(value) - complex(ref)) / max(abs(complex(ref)), floor))


DEFAULT_GSN_PROJECT = Path("/home/ljq/code/GSN/GeneralizedSasakiNakamura.jl")
DEFAULT_MMA_WL_WIN = "F:/EMRI/Radial_flow/Radial_Function.wl"


def run_gsn(
    a: float,
    omega: float,
    timeout: float,
    gsn_project: Path | None,
) -> tuple[dict[str, complex] | None, str]:
    julia = shutil.which("julia")
    if julia is None:
        return None, "julia-not-found"
    script = PROJECT_ROOT / "benchmark/scripts/gsn_ref.jl"
    if not script.exists():
        return None, "gsn-script-not-found"
    project_args = []
    if gsn_project is not None:
        if not (gsn_project / "Project.toml").exists():
            return None, f"gsn-project-not-found: {gsn_project}"
        project_args = [f"--project={gsn_project}"]
    cmd = [
        julia,
        *project_args,
        str(script),
        "--s=-2",
        "--l=2",
        "--m=2",
        f"--a={a:.17g}",
        f"--omega={omega:.17g}",
        "--M=1.0",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if proc.returncode != 0:
        return None, f"exit-{proc.returncode}: {proc.stderr.strip()[:240]}"
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return None, "empty-output"
    try:
        raw = json.loads(lines[-1])
        return {
            "lambda": complex(raw["lambda_re"], raw["lambda_im"]),
            "B_inc": complex(raw["B_inc_re"], raw["B_inc_im"]),
            "B_ref": complex(raw["B_ref_re"], raw["B_ref_im"]),
            "B_trans": complex(raw["B_trans_re"], raw["B_trans_im"]),
        }, "ok"
    except Exception as exc:
        return None, f"parse-error: {exc}"


def _wl_to_complex(value):
    if hasattr(value, "args") and len(value.args) == 2:
        return complex(float(value.args[0]), float(value.args[1]))
    return complex(value)


def _get_mma_kernel_path(kernel_path: str | None) -> str | None:
    if kernel_path:
        return kernel_path if Path(kernel_path).exists() else None
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


def _mma_get_path(kernel_path: str, wl_path: Path, wl_path_win: str | None) -> str | None:
    if kernel_path.lower().endswith(".exe") and wl_path_win:
        return wl_path_win
    if wl_path.exists():
        return str(wl_path.resolve())
    return None


def _run_mma_grid_script(
    cases: list[tuple[float, float]],
    *,
    kernel_path: str,
    mma_get_path: str,
    work_dir: Path,
    case_timeout: float,
) -> tuple[dict[tuple[float, float], tuple[dict[str, complex] | None, str]], str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    script_path = work_dir / "mma_amplitude_grid.wls"
    csv_path = work_dir / "mma_amplitude_grid.csv"
    cases_wl = "{" + ",".join(f"{{{a:.17g},{omega:.17g}}}" for a, omega in cases) + "}"
    script = f"""
Get["{mma_get_path}"];
cases = {cases_wl};
timeout = {case_timeout:.17g};
headers = {{"a", "omega", "status", "B_inc_re", "B_inc_im", "B_ref_re", "B_ref_im", "B_trans_re", "B_trans_im"}};
rows = Table[
  a = case[[1]]; omega = case[[2]];
  amp = Quiet[Check[TimeConstrained[ComputeAmplitudes[-2, 2, 2, a, omega], timeout, $Aborted], $Failed]];
  If[AssociationQ[amp],
    {{a, omega, "ok",
      Re[N[amp["Incidence"]]], Im[N[amp["Incidence"]]],
      Re[N[amp["Reflection"]]], Im[N[amp["Reflection"]]],
      Re[N[amp["Transmission"]]], Im[N[amp["Transmission"]]]}},
    {{a, omega, ToString[amp, InputForm], "", "", "", "", "", ""}}
  ],
  {{case, cases}}
];
Export["{csv_path}", Prepend[rows, headers], "CSV"];
Quit[];
"""
    script_path.write_text(script)
    proc = subprocess.run(
        [kernel_path, "-script", str(script_path)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return {}, f"script-exit-{proc.returncode}: {proc.stderr.strip()[:240]}"
    if not csv_path.exists():
        stderr = proc.stderr.strip()[:240]
        return {}, f"script-no-output: {stderr}"

    results = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = float(row["a"])
            omega = float(row["omega"])
            key = min(cases, key=lambda item: abs(item[0] - a) + abs(item[1] - omega))
            status = row["status"]
            if status != "ok":
                results[key] = (None, f"case-error: {status}")
                continue
            results[key] = ({
                "B_inc": complex(float(row["B_inc_re"]), float(row["B_inc_im"])),
                "B_ref": complex(float(row["B_ref_re"]), float(row["B_ref_im"])),
                "B_trans": complex(float(row["B_trans_re"]), float(row["B_trans_im"])),
            }, "ok")
    return results, f"ok-script:{kernel_path}:{mma_get_path}"


def run_mma_grid(
    cases: list[tuple[float, float]],
    *,
    kernel_path: str | None,
    wl_path: Path,
    wl_path_win: str | None,
    backend: str,
    work_dir: Path,
    case_timeout: float,
    timeout_note: str = "",
):
    kernel_path = _get_mma_kernel_path(kernel_path)
    if kernel_path is None:
        return {}, "kernel-not-found"

    mma_get_path = _mma_get_path(kernel_path, wl_path, wl_path_win)
    if mma_get_path is None:
        return {}, "wl-file-not-found"

    if backend not in {"auto", "wolframclient", "script"}:
        return {}, f"bad-backend:{backend}"
    use_script = backend == "script" or (backend == "auto" and kernel_path.lower().endswith(".exe"))
    if use_script:
        return _run_mma_grid_script(
            cases,
            kernel_path=kernel_path,
            mma_get_path=mma_get_path,
            work_dir=work_dir,
            case_timeout=case_timeout,
        )

    try:
        from wolframclient.evaluation import WolframLanguageSession
        from wolframclient.language import wlexpr
    except Exception as exc:
        return {}, f"wolframclient-unavailable: {exc}"

    results = {}
    session = WolframLanguageSession(kernel_path)
    try:
        session.start()
        session.evaluate(wlexpr(f'Get["{mma_get_path}"]'))
        for a, omega in cases:
            try:
                expr = wlexpr(f"ComputeAmplitudes[-2, 2, 2, {a:.17g}, {omega:.17g}]")
                raw = session.evaluate(expr)
                if raw is None:
                    results[(a, omega)] = (None, "returned-none")
                else:
                    results[(a, omega)] = ({
                        "B_inc": _wl_to_complex(raw["Incidence"]),
                        "B_ref": _wl_to_complex(raw["Reflection"]),
                        "B_trans": _wl_to_complex(raw["Transmission"]),
                    }, "ok")
            except Exception as exc:
                results[(a, omega)] = (None, f"case-error: {exc}")
    except Exception as exc:
        return {}, f"session-error: {exc}"
    finally:
        try:
            session.terminate()
        except Exception:
            pass
    return results, f"ok:{kernel_path}:{mma_get_path}{timeout_note}"


def parse_optional_float(raw: str) -> float | None:
    return None if str(raw).strip().lower() == "auto" else float(raw)


def spectral_profile_and_amplitudes(
    mode: KerrMode,
    N: int,
    z_m: float | None,
    grid_kind: str,
    omega_mp_cut: float,
    mp_dps_loww: int,
):
    if grid_kind == "scan":
        sm_best, score_best = None, None
        base_z = adaptive_match_z(mode) if z_m is None else float(z_m)
        z_candidates = sorted({
            float(np.clip(base_z * fac, 1.0e-5, 0.85))
            for fac in [0.4, 0.6, 0.8, 1.0, 1.4, 2.0, 3.0]
        })
        n_candidates = sorted({max(24, N - 40), max(24, N - 20), N, N + 20})
        resolved_grid = "anmr" if abs(mode.omega) < 1.0e-1 else "linear"
        for n_try in n_candidates:
            for z_try in z_candidates:
                try:
                    sm_try = compute_smatrix_with_abel(
                        mode,
                        N_in=n_try,
                        N_out=n_try,
                        z_m=z_try,
                        grid_kind=resolved_grid,
                        omega_mp_cut=omega_mp_cut,
                        mp_dps_loww=mp_dps_loww,
                    )
                except Exception:
                    continue
                score = max(
                    float(sm_try["outer_abel_residual"]),
                    float(sm_try["inner_abel_residual"]),
                    float(sm_try["detS_residual"]),
                )
                if score_best is None or score < score_best:
                    sm_best = sm_try
                    score_best = score
                    sm_best["selected_N"] = int(n_try)
                    sm_best["scan_score"] = float(score)
        if sm_best is None:
            raise RuntimeError("spectral scan found no valid candidate")
        prof = compute_smatrix(
            mode,
            N_in=int(sm_best["selected_N"]),
            N_out=int(sm_best["selected_N"]),
            z_m=float(sm_best["z_m"]),
            return_profile=True,
            grid_kind=str(sm_best["grid_kind"]),
            omega_mp_cut=omega_mp_cut,
            mp_dps_loww=mp_dps_loww,
        )["profile"]
        return sm_best, prof

    sm = compute_smatrix_with_abel(
        mode,
        N_in=N,
        N_out=N,
        z_m=z_m,
        grid_kind=grid_kind,
        omega_mp_cut=omega_mp_cut,
        mp_dps_loww=mp_dps_loww,
    )
    prof = compute_smatrix(
        mode,
        N_in=N,
        N_out=N,
        z_m=z_m,
        return_profile=True,
        grid_kind=grid_kind,
        omega_mp_cut=omega_mp_cut,
        mp_dps_loww=mp_dps_loww,
    )["profile"]
    return sm, prof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-list", default="0.0,0.2,0.4,0.6,0.8,0.95,0.99")
    parser.add_argument("--omega-list", default="0.08,0.1,0.12")
    parser.add_argument("--n-r", type=int, default=256)
    parser.add_argument("--r-max", type=float, default=1000.0)
    parser.add_argument("--spectral-N", type=int, default=120)
    parser.add_argument("--z-m", default="0.3")
    parser.add_argument("--spectral-grid", choices=["linear", "anmr", "auto", "scan"], default="linear")
    parser.add_argument("--omega-mp-cut", type=float, default=1.0e-2)
    parser.add_argument("--mp-dps-loww", type=int, default=100)
    parser.add_argument("--pybhpt-timeout", type=float, default=60.0)
    parser.add_argument("--gsn-timeout", type=float, default=120.0)
    parser.add_argument("--gsn-project", default=str(DEFAULT_GSN_PROJECT))
    parser.add_argument("--mma-kernel", default=None)
    parser.add_argument("--mma-wl", default=str(PROJECT_ROOT / "mma/Radial_Function.wl"))
    parser.add_argument("--mma-wl-win", default=DEFAULT_MMA_WL_WIN)
    parser.add_argument("--mma-backend", choices=["auto", "wolframclient", "script"], default="auto")
    parser.add_argument("--mma-case-timeout", type=float, default=120.0)
    parser.add_argument("--skip-pybhpt", action="store_true")
    parser.add_argument("--skip-mma", action="store_true")
    parser.add_argument("--skip-gsn", action="store_true")
    parser.add_argument("--output-dir", default="outputs/benchmark_cross_validation/omega01")
    args = parser.parse_args()

    a_values = parse_float_list(args.a_list)
    omega_values = parse_float_list(args.omega_list)
    z_m = parse_optional_float(args.z_m)
    gsn_project = Path(args.gsn_project).expanduser() if args.gsn_project else None
    cases = [(a, omega) for a in a_values for omega in omega_values]
    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    mma_results = {}
    mma_global_status = "skipped"
    if not args.skip_mma:
        mma_results, mma_global_status = run_mma_grid(
            cases,
            kernel_path=args.mma_kernel,
            wl_path=Path(args.mma_wl),
            wl_path_win=args.mma_wl_win,
            backend=args.mma_backend,
            work_dir=out_dir,
            case_timeout=args.mma_case_timeout,
        )

    rows = []
    failures = []
    for idx, (a, omega) in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] a={a:g} omega={omega:g}", flush=True)
        row: dict[str, object] = {"a": a, "omega": omega}
        mode = KerrMode(M=1.0, a=a, omega=omega, ell=2, m=2, s=-2)
        row["lambda_re"] = float(mode.lambda_value.real)
        row["lambda_im"] = float(mode.lambda_value.imag)
        row["k_hor"] = float(mode.k_hor)
        row["r_plus"] = float(mode.rp)

        r_min = max(float(mode.rp) + 1.0e-4, 2.0)
        r_grid = np.linspace(r_min, args.r_max, args.n_r)
        np.save(out_dir / f"r_grid_a{a:.4f}_w{omega:.4f}.npy", r_grid)

        try:
            sm, profile = spectral_profile_and_amplitudes(
                mode,
                N=args.spectral_N,
                z_m=z_m,
                grid_kind=args.spectral_grid,
                omega_mp_cut=args.omega_mp_cut,
                mp_dps_loww=args.mp_dps_loww,
            )
            row["spectral_status"] = "ok"
            row.update(cdict(sm["B_inc"], "spectral_B_inc"))
            row.update(cdict(sm["B_ref"], "spectral_B_ref"))
            row["spectral_outer_abel_residual"] = float(sm["outer_abel_residual"])
            row["spectral_inner_abel_residual"] = float(sm["inner_abel_residual"])
            row["spectral_detS_residual"] = float(sm["detS_residual"])
            row["spectral_match_cond_in_raw"] = float(sm["solve_in_cond_raw"])
            row["spectral_match_cond_in_scaled"] = float(sm["solve_in_cond_scaled"])
            row["spectral_z_m"] = float(sm["z_m"])
            row["spectral_grid_kind"] = str(sm["grid_kind"])
            if "selected_N" in sm:
                row["spectral_selected_N"] = int(sm["selected_N"])
                row["spectral_scan_score"] = float(sm["scan_score"])
            R_spec = profile.R_of_r(r_grid)
        except Exception as exc:
            row["spectral_status"] = str(exc)
            R_spec = None
            failures.append({"a": a, "omega": omega, "backend": "spectral", "error": str(exc)})

        if args.skip_pybhpt:
            row["pybhpt_status"] = "skipped"
        else:
            try:
                _, R_py = compute_pybhpt_solution(
                    a,
                    omega,
                    ell=2,
                    m=2,
                    r_grid=r_grid,
                    timeout=args.pybhpt_timeout,
                )
                row["pybhpt_status"] = "ok"
                if R_spec is not None:
                    rel = np.abs(R_spec - R_py) / (np.abs(R_py) + 1.0e-14)
                    row["pybhpt_vs_spectral_R_median_rel"] = float(np.median(rel))
                    row["pybhpt_vs_spectral_R_p90_rel"] = float(np.percentile(rel, 90))
                    row["pybhpt_vs_spectral_R_max_rel"] = float(np.max(rel))
                    np.savez_compressed(
                        out_dir / f"R_compare_a{a:.4f}_w{omega:.4f}.npz",
                        r=r_grid,
                        R_pybhpt=R_py,
                        R_spectral=R_spec,
                        rel_err=rel,
                    )
            except Exception as exc:
                row["pybhpt_status"] = str(exc)
                failures.append({"a": a, "omega": omega, "backend": "pybhpt", "error": str(exc)})

        if args.skip_gsn:
            row["gsn_status"] = "skipped"
            gsn = None
        else:
            gsn, gsn_status = run_gsn(a, omega, timeout=args.gsn_timeout, gsn_project=gsn_project)
            row["gsn_status"] = gsn_status
            if gsn is not None:
                row.update(cdict(gsn["B_inc"], "gsn_B_inc"))
                row.update(cdict(gsn["B_ref"], "gsn_B_ref"))
                row["spectral_vs_gsn_B_inc_rel"] = rel_err(sm["B_inc"] if R_spec is not None else None, gsn["B_inc"])
                row["spectral_vs_gsn_B_ref_rel"] = rel_err(sm["B_ref"] if R_spec is not None else None, gsn["B_ref"])

        if args.skip_mma:
            row["mma_status"] = "skipped"
            mma = None
        else:
            mma, mma_status = mma_results.get((a, omega), (None, mma_global_status))
            row["mma_status"] = mma_status
            if mma is not None:
                row.update(cdict(mma["B_inc"], "mma_B_inc"))
                row.update(cdict(mma["B_ref"], "mma_B_ref"))
                row["spectral_vs_mma_B_inc_rel"] = rel_err(sm["B_inc"] if R_spec is not None else None, mma["B_inc"])
                row["spectral_vs_mma_B_ref_rel"] = rel_err(sm["B_ref"] if R_spec is not None else None, mma["B_ref"])
                if gsn is not None:
                    row["gsn_vs_mma_B_inc_rel"] = rel_err(gsn["B_inc"], mma["B_inc"])
                    row["gsn_vs_mma_B_ref_rel"] = rel_err(gsn["B_ref"], mma["B_ref"])

        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(out_dir / "benchmark_cross_validation_cases.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "args": vars(args),
        "n_cases": len(rows),
        "failures": failures,
        "backend_status": {
            "mma_global": mma_global_status,
            "julia_available": shutil.which("julia") is not None,
            "gsn_project": str(gsn_project) if gsn_project is not None else None,
            "gsn_project_available": (
                gsn_project is not None and (gsn_project / "Project.toml").exists()
            ),
        },
    }
    numeric_keys = [
        "pybhpt_vs_spectral_R_median_rel",
        "pybhpt_vs_spectral_R_max_rel",
        "spectral_vs_mma_B_inc_rel",
        "spectral_vs_mma_B_ref_rel",
        "spectral_vs_gsn_B_inc_rel",
        "spectral_vs_gsn_B_ref_rel",
        "gsn_vs_mma_B_inc_rel",
        "gsn_vs_mma_B_ref_rel",
    ]
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if values:
            summary[key] = {
                "median": float(np.median(values)),
                "max": float(np.max(values)),
                "n": len(values),
            }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "README.md", "w") as f:
        f.write("# Omega ~ 0.1 benchmark cross-validation\n\n")
        f.write(f"- Cases: {len(rows)}\n")
        f.write(f"- MMA status: `{mma_global_status}`\n")
        f.write(f"- Julia/GSN available: `{shutil.which('julia') is not None}`\n")
        f.write(f"- GSN project: `{gsn_project}`\n")
        for key in numeric_keys:
            if key in summary:
                f.write(f"- {key}: median `{summary[key]['median']:.3e}`, max `{summary[key]['max']:.3e}`\n")
        f.write("\nSee `benchmark_cross_validation_cases.csv` and `summary.json`.\n")

    print(f"Saved cross-validation to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
