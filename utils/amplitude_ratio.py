"""Backward-compatible amplitude-ratio wrapper built on KerrMode / TeukRadAmplitudeIn."""

from __future__ import annotations

import math

from .amplitude import TeukRadAmplitudeIn
from .mode import KerrMode


def _ratio_arg(z: complex) -> float:
    denom = abs(z)
    if denom == 0.0:
        return 0.0
    return z.real / denom


def _build_ratio_dict(result) -> dict:
    ratio = result.ratio_inc_over_ref
    if ratio is None:
        raise FloatingPointError(
            f"Cannot form incidence/reflection ratio because B_ref is too small: "
            f"B_inc={result.B_inc}, B_ref={result.B_ref}"
        )

    if not (math.isfinite(ratio.real) and math.isfinite(ratio.imag)):
        raise FloatingPointError(
            f"Non-finite incidence/reflection ratio: {ratio}"
        )

    return {
        "ratio": ratio,
        "lambda": result.lam,
        "ratio_abs": abs(ratio),
        "ratio_arg": _ratio_arg(ratio),
        "B_inc": result.B_inc,
        "B_ref": result.B_ref,
        "B_trans": result.B_trans,
        "ratio_ref_over_inc": result.ratio_ref_over_inc,
        "ratio_inc_over_ref": result.ratio_inc_over_ref,
    }


def compute_amplitude_ratio(
    a=None,
    omega=None,
    l=None,
    m=None,
    lambda_sep=None,
    r_match: float = 8.0,
    n_cheb: int = 32,
    s: int = -2,
    mode: KerrMode | None = None,
):
    """Compute B_inc/B_ref-style amplitude information.

    Preferred usage:
        compute_amplitude_ratio(mode=my_mode, r_match=..., n_cheb=...)

    Backward-compatible usage:
        compute_amplitude_ratio(a, omega, l, m, lambda_sep=None, r_match=..., n_cheb=..., s=-2)
    """
    if mode is None and isinstance(a, KerrMode):
        mode = a

    if mode is None:
        if None in (a, omega, l, m):
            raise ValueError("Either provide mode=KerrMode or provide a, omega, l, m.")
        mode = KerrMode(
            M=1.0,
            a=float(a),
            omega=float(omega),
            ell=int(l),
            m=int(m),
            lam=None if lambda_sep is None else complex(lambda_sep),
            s=int(s),
        )

    z_m = mode.rp / float(r_match)
    amp = TeukRadAmplitudeIn(mode=mode, N_in=int(n_cheb), N_out=int(n_cheb), z_m=z_m)
    return _build_ratio_dict(amp.result)
