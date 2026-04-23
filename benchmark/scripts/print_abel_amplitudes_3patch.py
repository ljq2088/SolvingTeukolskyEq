from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mode import KerrMode
from utils.amplitude_three_patch import TeukRadAmplitudeIn3PatchWithAbelChecks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print three-patch Abel diagnostics for TeukRadAmplitudeIn3PatchWithAbelChecks."
    )
    parser.add_argument("--M", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--omega", type=float, default=0.1)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--lam", type=float, default=None)

    parser.add_argument("--N-left", dest="N_left", type=int, default=64)
    parser.add_argument("--N-mid", dest="N_mid", type=int, default=96)
    parser.add_argument("--N-right", dest="N_right", type=int, default=64)

    parser.add_argument("--z1", type=float, default=0.1)
    parser.add_argument("--z2", type=float, default=0.9)

    args = parser.parse_args()

    mode = KerrMode(
        M=args.M,
        a=args.a,
        omega=args.omega,
        ell=args.ell,
        m=args.m,
        lam=args.lam,
        s=args.s,
    )

    amp = TeukRadAmplitudeIn3PatchWithAbelChecks(
        mode,
        N_left=args.N_left,
        N_mid=args.N_mid,
        N_right=args.N_right,
        z1=args.z1,
        z2=args.z2,
    )

    print("B_inc =", amp.B_inc)
    print("B_ref =", amp.B_ref)
    print("B_trans =", amp.B_trans)

    print("outer_abel_num =", amp.smatrix["outer_abel_num"])
    print("outer_abel_th  =", amp.smatrix["outer_abel_th"])
    print("outer_abel_residual =", amp.outer_abel_residual)

    print("inner_abel_num =", amp.smatrix["inner_abel_num"])
    print("inner_abel_th  =", amp.smatrix["inner_abel_th"])
    print("inner_abel_residual =", amp.inner_abel_residual)

    print("detS_num =", amp.smatrix["detS_num"])
    print("detS_th  =", amp.smatrix["detS_th"])
    print("detS_residual =", amp.detS_residual)

    print("Cin_down =", amp.smatrix["Cin_down"])
    print("Cin_up   =", amp.smatrix["Cin_up"])
    print("Cout_down =", amp.smatrix["Cout_down"])
    print("Cout_up   =", amp.smatrix["Cout_up"])


if __name__ == "__main__":
    main()