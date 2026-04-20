"""Wrapper around kerr_teuk_null_cheb_smatrix_fast for In-solution amplitudes.

Despite the filename kept for continuity, this utility does not use pybhpt.
It directly calls the fast spectral S-matrix routine from
`kerr_matcher.kerr_teuk_null_cheb_smatrix_fast`, which returns the amplitudes
of the In solution:

    B_inc, B_ref, B_trans

with the convention documented in that module.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any

try:
    from .compute_lambda import compute_lambda
except ImportError:  # direct script execution
    from compute_lambda import compute_lambda

_KERR_MATCHER_PATH = "/home/ljq/code/radial_flow/spec_flow_method_Kerr/kerr_matcher_project/src"
if _KERR_MATCHER_PATH not in sys.path:
    sys.path.insert(0, _KERR_MATCHER_PATH)

from kerr_matcher.kerr_teuk_null_cheb_smatrix_fast import KerrMode, compute_smatrix


def _complex_to_dict(z: complex) -> dict[str, float]:
    return {"re": float(z.real), "im": float(z.imag), "abs": float(abs(z))}


@dataclass(slots=True)
class InAmplitudesResult:
    l: int
    m: int
    s: int
    a: float
    omega: float
    lambda_sep: float
    B_inc: complex
    B_ref: complex
    B_trans: complex
    N_in: int
    N_out: int
    z_m: float

    @property
    def ref_over_inc(self) -> complex:
        return self.B_ref / self.B_inc

    @property
    def trans_over_inc(self) -> complex:
        return self.B_trans / self.B_inc

    def as_dict(self) -> dict[str, Any]:
        return {
            "l": self.l,
            "m": self.m,
            "s": self.s,
            "a": self.a,
            "omega": self.omega,
            "lambda": self.lambda_sep,
            "B_inc": _complex_to_dict(self.B_inc),
            "B_ref": _complex_to_dict(self.B_ref),
            "B_trans": _complex_to_dict(self.B_trans),
            "B_ref_over_B_inc": _complex_to_dict(self.ref_over_inc),
            "B_trans_over_B_inc": _complex_to_dict(self.trans_over_inc),
            "N_in": self.N_in,
            "N_out": self.N_out,
            "z_m": self.z_m,
        }


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
        self.lambda_sep = lambda_sep
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.z_m = float(z_m)

    def _resolve_lambda(self, a: float, omega: float, l: int, m: int, lambda_sep: float | None) -> float:
        if lambda_sep is not None:
            return float(lambda_sep)
        if self.lambda_sep is not None:
            return float(self.lambda_sep)
        return float(compute_lambda(a=a, omega=omega, l=l, m=m, s=self.s))

    def solve(self, *, l: int, m: int, a: float, omega: float, lambda_sep: float | None = None) -> InAmplitudesResult:
        lam = self._resolve_lambda(a, omega, l, m, lambda_sep)
        mode = KerrMode(M=self.m_bh, a=a, omega=omega, ell=l, m=m, lam=lam, s=self.s)
        out = compute_smatrix(mode, N_in=self.n_in, N_out=self.n_out, z_m=self.z_m)
        return InAmplitudesResult(
            l=l,
            m=m,
            s=self.s,
            a=a,
            omega=omega,
            lambda_sep=lam,
            B_inc=complex(out["B_inc"]),
            B_ref=complex(out["B_ref"]),
            B_trans=complex(out["B_trans"]),
            N_in=self.n_in,
            N_out=self.n_out,
            z_m=self.z_m,
        )


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
        lambda_sep=args.lambda_sep,
        n_in=args.n_in,
        n_out=args.n_out,
        z_m=args.z_m,
    )
    result = solver.solve(l=args.l, m=args.m, a=args.a, omega=args.omega, lambda_sep=args.lambda_sep)

    if args.json:
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
        return

    print(f"l={result.l}, m={result.m}, s={result.s}, a={result.a}, omega={result.omega}")
    print(f"lambda = {result.lambda_sep:.16g}")
    print(f"N_in = {result.N_in}, N_out = {result.N_out}, z_m = {result.z_m}")
    print(f"B_inc = {result.B_inc}")
    print(f"B_ref = {result.B_ref}")
    print(f"B_trans = {result.B_trans}")
    print(f"B_ref / B_inc = {result.ref_over_inc}")
    print(f"B_trans / B_inc = {result.trans_over_inc}")


if __name__ == "__main__":
    main()
