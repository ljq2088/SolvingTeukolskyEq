from __future__ import annotations

import argparse
import sys
from pathlib import Path
import sys
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mode import KerrMode
from utils.amplitude_three_patch import TeukRadAmplitudeIn3PatchWithAbelChecks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute three-patch Teukolsky amplitudes and print them in bash."
    )
    parser.add_argument("--M", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--omega", type=float, default=0.1)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--lam", type=float, default=None)

    parser.add_argument("--N-left", dest="N_left", type=int, default=64)
    parser.add_argument("--N-mid", dest="N_mid", type=int, default=128)
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

    print("=== Three-patch amplitudes ===")
    print(f"B_inc   = {amp.B_inc}")
    print(f"B_ref   = {amp.B_ref}")
    print(f"B_trans = {amp.B_trans}")
    print(f"B_ref/B_inc = {amp.ratio_ref_over_inc}")
    print(f"B_inc/B_ref = {amp.ratio_inc_over_ref}")

    print("\n=== Abel diagnostics ===")
    print(f"outer_abel_num      = {amp.smatrix['outer_abel_num']}")
    print(f"outer_abel_th       = {amp.smatrix['outer_abel_th']}")
    print(f"outer_abel_residual = {amp.outer_abel_residual:.6e}")

    print(f"inner_abel_num      = {amp.smatrix['inner_abel_num']}")
    print(f"inner_abel_th       = {amp.smatrix['inner_abel_th']}")
    print(f"inner_abel_residual = {amp.inner_abel_residual:.6e}")

    print(f"detS_num            = {amp.smatrix['detS_num']}")
    print(f"detS_th             = {amp.smatrix['detS_th']}")
    print(f"detS_residual       = {amp.detS_residual:.6e}")

    print("\n=== Transfer-matrix entries ===")
    print(f"Cin_down  = {amp.smatrix['Cin_down']}")
    print(f"Cin_up    = {amp.smatrix['Cin_up']}")
    print(f"Cout_down = {amp.smatrix['Cout_down']}")
    print(f"Cout_up   = {amp.smatrix['Cout_up']}")
    print("\n=== Conditioning diagnostics ===")
    print(f"cond(M_left)         = {amp.smatrix['cond_M_left']:.6e}")
    print(f"cond(T_mid)          = {amp.smatrix['cond_T_mid']:.6e}")
    print(f"cond(M_outer_at_z2)  = {amp.smatrix['cond_M_outer_at_z2']:.6e}")

    print(f"solve_in_relres      = {amp.smatrix['solve_in_relres']:.6e}")
    print(f"solve_out_relres     = {amp.smatrix['solve_out_relres']:.6e}")
    print(f"solve_in_cond_raw    = {amp.smatrix['solve_in_cond_raw']:.6e}")
    print(f"solve_out_cond_raw   = {amp.smatrix['solve_out_cond_raw']:.6e}")
    print(f"solve_in_cond_scaled = {amp.smatrix['solve_in_cond_scaled']:.6e}")
    print(f"solve_out_cond_scaled= {amp.smatrix['solve_out_cond_scaled']:.6e}")

    print(f"solve_in_smax        = {amp.smatrix['solve_in_smax']:.6e}")
    print(f"solve_in_smin        = {amp.smatrix['solve_in_smin']:.6e}")
    print(f"solve_out_smax       = {amp.smatrix['solve_out_smax']:.6e}")
    print(f"solve_out_smin       = {amp.smatrix['solve_out_smin']:.6e}")

    print(f"state_in_z1_relerr   = {amp.smatrix['state_in_z1_relerr']:.6e}")
    print(f"state_out_z1_relerr  = {amp.smatrix['state_out_z1_relerr']:.6e}")


if __name__ == "__main__":
    main()