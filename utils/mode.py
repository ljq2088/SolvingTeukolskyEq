from __future__ import annotations

from dataclasses import dataclass
import math

try:
    from .compute_lambda import compute_lambda
except ImportError:  # direct script execution
    from compute_lambda import compute_lambda


def _safe_complex_ratio(numer: complex, denom: complex, *, tol: float = 1.0e-30) -> complex | None:
    if abs(denom) <= tol:
        return None
    return numer / denom


@dataclass(frozen=True)
class KerrMode:
    M: float
    a: float
    omega: float
    ell: int
    m: int
    lam: complex | None = None
    s: int = -2

    @property
    def rp(self) -> float:
        return self.M + math.sqrt(self.M * self.M - self.a * self.a)

    @property
    def rm(self) -> float:
        return self.M - math.sqrt(self.M * self.M - self.a * self.a)

    @property
    def delta_h(self) -> float:
        return self.rp - self.rm

    @property
    def Omega_H(self) -> float:
        return self.a / (self.rp * self.rp + self.a * self.a)

    @property
    def k_hor(self) -> float:
        return self.omega - self.m * self.Omega_H

    @property
    def lambda_value(self) -> complex:
        if self.lam is None:
            value = compute_lambda(self.a, self.omega, self.ell, self.m, self.s)
            object.__setattr__(self, "lam", complex(value))
        return complex(self.lam)


@dataclass(slots=True)
class InAmplitudesResult:
    l: int
    m: int
    s: int
    a: float
    omega: float
    lam: complex
    B_inc: complex
    B_ref: complex
    B_trans: complex
    N_in: int
    N_out: int
    z_m: float
    ratio_ref_over_inc: complex | None = None
    ratio_inc_over_ref: complex | None = None

    def __post_init__(self) -> None:
        if self.ratio_ref_over_inc is None:
            self.ratio_ref_over_inc = _safe_complex_ratio(self.B_ref, self.B_inc)
        if self.ratio_inc_over_ref is None:
            self.ratio_inc_over_ref = _safe_complex_ratio(self.B_inc, self.B_ref)
