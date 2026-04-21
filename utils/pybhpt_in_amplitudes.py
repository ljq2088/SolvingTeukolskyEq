"""Thin CLI wrapper around the unified KerrMode / TeukRadAmplitudeIn chain."""

from __future__ import annotations

import argparse
import json

from utils.amplitude import TeukRadAmplitudeIn
from utils.mode import InAmplitudesResult, KerrMode


def _complex_to_dict(z: complex | None) -> dict[str, float | None]:
    if z is None:
        return {"re": None, "im": None, "abs": None}
    return {"re": float(z.real), "im": float(z.imag), "abs": float(abs(z))}


class InAmplitudeSolver:
    def __init__(
        self,
        *,
        m_bh: float = 1.0,
        s: int = -2,
        lambda_sep: float | None = None,
        n_in: int = 80,
        n_out: int = 80,
        z_m: float = 0.3,
    ) -> None:
        self.m_bh = float(m_bh)
        self.s = int(s)
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.z_m = float(z_m)

    def solve(self, *, l: int, m: int, a: float, omega: float, lambda_sep: float | None = None) -> InAmplitudesResult:
        mode = KerrMode(M=self.m_bh, a=a, omega=omega, ell=l, m=m, lam=lambda_sep, s=self.s)
        return TeukRadAmplitudeIn(mode=mode, N_in=self.n_in, N_out=self.n_out, z_m=self.z_m).result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute B_inc, B_ref, B_trans of the In solution.")
    parser.add_argument("--l", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--a", type=float, required=True)
    parser.add_argument("--omega", type=float, required=True)
    parser.add_argument("--lambda-sep", type=float, default=None)
    parser.add_argument("--M", type=float, default=1.0)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--n-in", type=int, default=80)
    parser.add_argument("--n-out", type=int, default=80)
    parser.add_argument("--z-m", type=float, default=0.3)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    solver = InAmplitudeSolver(
        m_bh=args.M,
        s=args.s,
        n_in=args.n_in,
        n_out=args.n_out,
        z_m=args.z_m,
    )
    result = solver.solve(l=args.l, m=args.m, a=args.a, omega=args.omega, lambda_sep=args.lambda_sep)

    if args.json:
        print(json.dumps({
            "l": result.l,
            "m": result.m,
            "s": result.s,
            "a": result.a,
            "omega": result.omega,
            "lambda": _complex_to_dict(result.lam),
            "B_inc": _complex_to_dict(result.B_inc),
            "B_ref": _complex_to_dict(result.B_ref),
            "B_trans": _complex_to_dict(result.B_trans),
            "ratio_ref_over_inc": _complex_to_dict(result.ratio_ref_over_inc),
            "ratio_inc_over_ref": _complex_to_dict(result.ratio_inc_over_ref),
            "N_in": result.N_in,
            "N_out": result.N_out,
            "z_m": result.z_m,
        }, indent=2, ensure_ascii=False))
        return

    print(f"l={result.l}, m={result.m}, s={result.s}, a={result.a}, omega={result.omega}")
    print(f"lambda = {result.lam}")
    print(f"N_in = {result.N_in}, N_out = {result.N_out}, z_m = {result.z_m}")
    print(f"B_inc = {result.B_inc}")
    print(f"B_ref = {result.B_ref}")
    print(f"B_trans = {result.B_trans}")
    print(f"B_ref / B_inc = {result.ratio_ref_over_inc}")
    print(f"B_inc / B_ref = {result.ratio_inc_over_ref}")


if __name__ == "__main__":
    main()
