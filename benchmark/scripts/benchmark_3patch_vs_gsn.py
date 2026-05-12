"""
Benchmark 3-patch spectral solver against GSN reference for low ω (1e-4 to 3.0).

Compares B_inc, B_ref for s=-2, l=2, m=2, a=0.1 (and other a values).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.mode import KerrMode
from utils.amplitude_three_patch_modified import TeukRadAmplitudeIn3PatchWithAbelChecks

GSN_SCRIPT = Path(__file__).resolve().parent / "gsn_ref.jl"
JULIA_BIN = "/home/ljq/julia-1.10.7/bin/julia"


def gsn_ref(s: int, l: int, m: int, a: float, omega: float, M: float = 1.0) -> dict:
    """Call GSN.jl to get reference B_inc, B_ref."""
    cmd = [
        JULIA_BIN, "--project=/home/ljq/code/GSN/GeneralizedSasakiNakamura.jl",
        str(GSN_SCRIPT),
        f"--s={s}", f"--l={l}", f"--m={m}",
        f"--a={a}", f"--omega={omega}", f"--M={M}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"GSN failed: {proc.stderr}")
    # Parse JSON from stdout (may have Julia startup messages before JSON)
    for line in proc.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"No JSON found in GSN output:\n{proc.stdout}")


def gsn_ref_batch(
    s: int,
    l: int,
    m: int,
    a: float,
    omega_list: list[float],
    M: float = 1.0,
) -> dict[float, dict]:
    """Call GSN.jl once for a batch of omegas and return a map omega -> record."""
    omega_arg = ",".join(f"{float(omega):.17g}" for omega in omega_list)
    cmd = [
        JULIA_BIN, "--project=/home/ljq/code/GSN/GeneralizedSasakiNakamura.jl",
        str(GSN_SCRIPT),
        f"--s={s}", f"--l={l}", f"--m={m}",
        f"--a={a}", f"--omega-list={omega_arg}", f"--M={M}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"GSN batch failed: {proc.stderr}")

    out: dict[float, dict] = {}
    for line in proc.stdout.strip().split("\n"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        rec = json.loads(line)
        out[float(rec["omega"])] = rec
    if not out:
        raise RuntimeError(f"No JSON found in GSN batch output:\n{proc.stdout}")
    return out


def rel_err(x: complex, xref: complex, floor: float = 1e-30) -> float:
    return float(abs(x - xref) / max(abs(xref), floor))


def run_benchmark(
    a: float = 0.1,
    s: int = -2,
    ll: int = 2,
    m: int = 2,
    omega_list: list[float] | None = None,
):
    if omega_list is None:
        omega_list = [10.0**p for p in np.linspace(-4, np.log10(3.0), 20)]

    print(f"{'omega':>10s}  {'B_inc_3p':>32s}  {'B_inc_gsn':>32s}  {'err_B_inc':>10s}  {'err_B_ref':>10s}  {'abel_out':>10s}  {'abel_in':>10s}  {'cond_L':>10s}")
    print("-" * 160)
    print(f"[gsn] launching batch for {len(omega_list)} omegas...", file=sys.stderr, flush=True)

    try:
        gsn_batch = gsn_ref_batch(s=s, l=ll, m=m, a=a, omega_list=omega_list)
        print(f"[gsn] batch done: {len(gsn_batch)} records", file=sys.stderr, flush=True)
    except Exception:
        gsn_batch = {}
        print("[gsn] batch failed, will fallback to per-omega calls", file=sys.stderr, flush=True)

    results = []
    for idx, omega in enumerate(omega_list, start=1):
        print(f"[{idx}/{len(omega_list)}] omega={omega:.6e}: solving 3-patch", file=sys.stderr, flush=True)
        mode = KerrMode(M=1.0, a=float(a), omega=float(omega), ell=ll, m=m, lam=None, s=s)

        # 3-patch solve
        try:
            amp = TeukRadAmplitudeIn3PatchWithAbelChecks(mode)
            B_inc_3p = amp.B_inc
            B_ref_3p = amp.B_ref
            outer_abel = amp.outer_abel_residual
            inner_abel = amp.inner_abel_residual
            cond_left = float(amp.smatrix.get("cond_M_left", np.nan))
            cond_tmid = float(amp.smatrix.get("cond_T_mid", np.nan))
            ok_3p = True
        except Exception as e:
            B_inc_3p = np.nan
            B_ref_3p = np.nan
            outer_abel = np.nan
            inner_abel = np.nan
            cond_left = np.nan
            cond_tmid = np.nan
            ok_3p = False

        # GSN reference
        try:
            print(f"[{idx}/{len(omega_list)}] omega={omega:.6e}: fetching GSN", file=sys.stderr, flush=True)
            ref = gsn_batch.get(float(omega))
            if ref is None:
                ref = gsn_ref(s=s, l=ll, m=m, a=a, omega=omega)
            B_inc_gsn = complex(ref["B_inc_re"], ref["B_inc_im"])
            B_ref_gsn = complex(ref["B_ref_re"], ref["B_ref_im"])
            lam_gsn = complex(ref["lambda_re"], ref["lambda_im"])
            ok_gsn = True
        except Exception as e:
            B_inc_gsn = np.nan
            B_ref_gsn = np.nan
            lam_gsn = np.nan
            ok_gsn = False

        if ok_3p and ok_gsn:
            e_inc = rel_err(B_inc_3p, B_inc_gsn)
            e_ref = rel_err(B_ref_3p, B_ref_gsn)
            print(f"{omega:10.2e}  {str(B_inc_3p):>32s}  {str(B_inc_gsn):>32s}  {e_inc:10.2e}  {e_ref:10.2e}  {outer_abel:10.2e}  {inner_abel:10.2e}  {cond_left:10.2e}")
        elif ok_3p:
            print(f"{omega:10.2e}  {str(B_inc_3p):>32s}  {'GSN_FAILED':>32s}")
        else:
            print(f"{omega:10.2e}  {'3P_FAILED':>32s}")

        results.append({
            "omega": omega,
            "B_inc_3p": B_inc_3p,
            "B_ref_3p": B_ref_3p,
            "B_inc_gsn": B_inc_gsn,
            "B_ref_gsn": B_ref_gsn,
            "err_B_inc": e_inc if (ok_3p and ok_gsn) else np.nan,
            "err_B_ref": e_ref if (ok_3p and ok_gsn) else np.nan,
            "outer_abel_residual": outer_abel,
            "inner_abel_residual": inner_abel,
            "cond_M_left": cond_left,
            "cond_T_mid": cond_tmid,
            "lambda_gsn": lam_gsn,
        })

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--l", type=int, default=2, dest="ll")
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--omega-min", type=float, default=1e-4)
    parser.add_argument("--omega-max", type=float, default=3.0)
    parser.add_argument("--n-omega", type=int, default=20)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    omegas = [10.0**p for p in np.linspace(np.log10(args.omega_min), np.log10(args.omega_max), args.n_omega)]
    results = run_benchmark(a=args.a, s=args.s, ll=args.ll, m=args.m, omega_list=omegas)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert complex to dict for JSON
        serializable = []
        for r in results:
            d = {}
            for k, v in r.items():
                if isinstance(v, complex):
                    d[k + "_re"] = v.real
                    d[k + "_im"] = v.imag
                    d[k + "_abs"] = abs(v)
                else:
                    d[k] = v
            serializable.append(d)
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved to {out_path}")
