from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import math
import traceback

from utils.compute_lambda import compute_lambda
from utils.amplitude_ratio import compute_amplitude_ratio


@dataclass
class SolverStatus:
    ok: bool
    code: str
    message: str = ""
    value: Optional[complex] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PointStatus:
    valid: bool
    code: str
    message: str = ""
    a: float = 0.0
    omega: float = 0.0
    l: int = 2
    m: int = 2
    s: int = -2
    k_h: Optional[float] = None
    lambda_status: Optional[SolverStatus] = None
    ramp_status: Optional[SolverStatus] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def _is_finite_complex(z: complex) -> bool:
    return math.isfinite(z.real) and math.isfinite(z.imag)


def r_plus_of_a(a: float, M: float = 1.0) -> float:
    return M + math.sqrt(max(M * M - a * a, 0.0))


def omega_horizon(a: float, M: float = 1.0) -> float:
    rp = r_plus_of_a(a, M=M)
    return a / (2.0 * M * rp)


def try_compute_lambda_status(
    a: float,
    omega: float,
    l: int,
    m: int,
    s: int = -2,
) -> SolverStatus:
    try:
        lam = compute_lambda(a, omega, l, m, s)
        lam_c = complex(lam)
        if not _is_finite_complex(lam_c):
            return SolverStatus(
                ok=False,
                code="lambda_nonfinite",
                message=f"lambda is non-finite: {lam_c}",
                value=None,
            )
        return SolverStatus(
            ok=True,
            code="ok",
            message="lambda success",
            value=lam_c,
            meta={"lambda_real": lam_c.real, "lambda_imag": lam_c.imag},
        )
    except ImportError as e:
        return SolverStatus(
            ok=False,
            code="lambda_import_error",
            message=str(e),
            value=None,
        )
    except Exception as e:
        return SolverStatus(
            ok=False,
            code="lambda_exception",
            message=f"{type(e).__name__}: {e}",
            value=None,
            meta={"traceback": traceback.format_exc(limit=3)},
        )


def try_compute_ramp_status(
    a: float,
    omega: float,
    l: int,
    m: int,
    s: int = -2,
    lambda_sep: Optional[complex] = None,
    r_match: float = 8.0,
    n_cheb: int = 32,
) -> SolverStatus:
    try:
        lam_use = None if lambda_sep is None else complex(lambda_sep).real
        out = compute_amplitude_ratio(
            a=a,
            omega=omega,
            l=l,
            m=m,
            lambda_sep=lam_use,
            r_match=r_match,
            n_cheb=n_cheb,
            s=s,
        )
        ratio = complex(out["ratio"])
        if not _is_finite_complex(ratio):
            return SolverStatus(
                ok=False,
                code="ramp_nonfinite",
                message=f"ratio is non-finite: {ratio}",
                value=None,
                meta=out,
            )
        if abs(ratio) == 0.0:
            return SolverStatus(
                ok=False,
                code="ramp_zero",
                message="ratio is exactly zero",
                value=None,
                meta=out,
            )
        return SolverStatus(
            ok=True,
            code="ok",
            message="R_amp success",
            value=ratio,
            meta=out,
        )
    except FloatingPointError as e:
        return SolverStatus(
            ok=False,
            code="ramp_floating_point",
            message=str(e),
            value=None,
        )
    except ImportError as e:
        return SolverStatus(
            ok=False,
            code="ramp_import_error",
            message=str(e),
            value=None,
        )
    except Exception as e:
        return SolverStatus(
            ok=False,
            code="ramp_exception",
            message=f"{type(e).__name__}: {e}",
            value=None,
            meta={"traceback": traceback.format_exc(limit=3)},
        )


def evaluate_param_point(
    a: float,
    omega: float,
    l: int,
    m: int,
    s: int = -2,
    M: float = 1.0,
    k_horizon_margin: float = 1.0e-2,
    require_ramp: bool = True,
    r_match: float = 8.0,
    n_cheb: int = 32,
) -> PointStatus:
    """
    固定 (l,m,s) 下，对单个 (a, omega) 进行安全性评估。

    valid = True 当且仅当：
    1. 不落在 |k_H| < margin 的危险带
    2. lambda 成功
    3. 若 require_ramp=True，则 R_amp 也成功
    """
    omg_h = omega_horizon(a, M=M)
    k_h = omega - m * omg_h

    if abs(k_h) < k_horizon_margin:
        return PointStatus(
            valid=False,
            code="k_horizon_margin",
            message=f"|omega - m*Omega_H| < {k_horizon_margin}",
            a=a,
            omega=omega,
            l=l,
            m=m,
            s=s,
            k_h=k_h,
        )

    lam_status = try_compute_lambda_status(a=a, omega=omega, l=l, m=m, s=s)
    if not lam_status.ok:
        return PointStatus(
            valid=False,
            code=lam_status.code,
            message=lam_status.message,
            a=a,
            omega=omega,
            l=l,
            m=m,
            s=s,
            k_h=k_h,
            lambda_status=lam_status,
            ramp_status=None,
        )

    ramp_status = None
    if require_ramp:
        ramp_status = try_compute_ramp_status(
            a=a,
            omega=omega,
            l=l,
            m=m,
            s=s,
            lambda_sep=lam_status.value,
            r_match=r_match,
            n_cheb=n_cheb,
        )
        if not ramp_status.ok:
            return PointStatus(
                valid=False,
                code=ramp_status.code,
                message=ramp_status.message,
                a=a,
                omega=omega,
                l=l,
                m=m,
                s=s,
                k_h=k_h,
                lambda_status=lam_status,
                ramp_status=ramp_status,
            )

    return PointStatus(
        valid=True,
        code="ok",
        message="safe",
        a=a,
        omega=omega,
        l=l,
        m=m,
        s=s,
        k_h=k_h,
        lambda_status=lam_status,
        ramp_status=ramp_status,
        meta={
            "Omega_H": omg_h,
            "lambda": None if lam_status.value is None else complex(lam_status.value),
            "ramp": None if ramp_status is None or ramp_status.value is None else complex(ramp_status.value),
        },
    )