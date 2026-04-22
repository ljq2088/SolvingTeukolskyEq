from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mode import KerrMode
from utils.amplitude import TeukRadAmplitudeInWithAbelChecks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print TeukRadAmplitudeInWithAbelChecks diagnostics to stdout."
    )
    parser.add_argument("--M", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--omega", type=float, default=0.1)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--N-in", dest="N_in", type=int, default=64)
    parser.add_argument("--N-out", dest="N_out", type=int, default=64)
    parser.add_argument("--z-m", dest="z_m", type=float, default=0.3)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        print("[dbg] start")
        print(
            f"[dbg] args: M={args.M}, a={args.a}, omega={args.omega}, "
            f"ell={args.ell}, m={args.m}, s={args.s}, lam={args.lam}, "
            f"N_in={args.N_in}, N_out={args.N_out}, z_m={args.z_m}"
        )

    mode = KerrMode(
        M=args.M,
        a=args.a,
        omega=args.omega,
        ell=args.ell,
        m=args.m,
        lam=args.lam,
        s=args.s,
    )
    if args.debug:
        print("[dbg] mode constructed")
        print(f"[dbg] rp={mode.rp}, rm={mode.rm}, k_hor={mode.k_hor}")

    amp = TeukRadAmplitudeInWithAbelChecks(
        mode,
        N_in=args.N_in,
        N_out=args.N_out,
        z_m=args.z_m,
    )
    if args.debug:
        print("[dbg] amplitude object constructed")
        print("[dbg] about to access amp.B_inc (this triggers smatrix solve)")

    print("B_inc =", amp.B_inc)
    if args.debug:
        print("[dbg] amp.B_inc done")
    print("B_ref =", amp.B_ref)
    if args.debug:
        print("[dbg] amp.B_ref done")
    print("B_trans =", amp.B_trans)
    if args.debug:
        print("[dbg] amp.B_trans done")

    if args.debug:
        print("[dbg] about to access outer abel diagnostics")
    print("outer_abel_num =", amp.smatrix["outer_abel_num"])
    print("outer_abel_th  =", amp.smatrix["outer_abel_th"])
    print("outer_abel_residual =", amp.outer_abel_residual)

    if args.debug:
        print("[dbg] about to access inner abel diagnostics")
    print("inner_abel_num =", amp.smatrix["inner_abel_num"])
    print("inner_abel_th  =", amp.smatrix["inner_abel_th"])
    print("inner_abel_residual =", amp.inner_abel_residual)

    if args.debug:
        print("[dbg] about to access detS diagnostics")
    print("detS_num =", amp.smatrix["detS_num"])
    print("detS_th  =", amp.smatrix["detS_th"])
    print("detS_residual =", amp.detS_residual)

    if args.debug:
        print("[dbg] about to access coefficient entries")
    print("Cin_down =", amp.smatrix["Cin_down"])
    print("Cin_up   =", amp.smatrix["Cin_up"])
    print("Cout_down =", amp.smatrix["Cout_down"])
    print("Cout_up   =", amp.smatrix["Cout_up"])
    if args.debug:
        print("[dbg] done")


if __name__ == "__main__":
    main()
