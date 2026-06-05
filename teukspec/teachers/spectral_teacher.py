from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from teukspec.core.chebyshev import DomainSpec, cheb_tail_ratio
from utils.amplitude import _domain_D, coeffs_numeric, solve_basis_domain
from utils.matlcheb import real_to_cheb
from utils.mode import KerrMode


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def branch_specs(y_match: float, match_width: float) -> dict[str, DomainSpec]:
    z_match = 0.5 * (y_match + 1.0)
    half = 0.5 * match_width
    z_left = max(1.0e-8, z_match - half)
    z_right = min(1.0 - 1.0e-8, z_match + half)
    return {
        "in": DomainSpec("in", z_left, 1.0, "right", "inner"),
        "down": DomainSpec("down", 0.0, z_right, "left", "outer"),
        "up": DomainSpec("up", 0.0, z_right, "left", "outer"),
    }


def solve_teacher(mode: KerrMode, spec: DomainSpec, n: int, grid_kind: str) -> tuple[np.ndarray, dict]:
    sol = solve_basis_domain(mode, spec.name, n, spec.z_left, spec.z_right, spec.bc_side, domain=spec.domain, grid_kind=grid_kind)
    coeff = real_to_cheb(sol["u"])
    D, _ = _domain_D(mode, n, spec.z_left, spec.z_right, grid_kind=grid_kind, domain=spec.domain)
    uz = D @ sol["u"]
    uzz = D @ uz
    idx = np.arange(2, n - 1)
    B2, B1, B0 = coeffs_numeric(sol["z"][idx], mode, spec.name)
    res = B2 * uzz[idx] + B1 * uz[idx] + B0 * sol["u"][idx]
    den = np.maximum.reduce([np.abs(B2 * uzz[idx]), np.abs(B1 * uz[idx]), np.abs(B0 * sol["u"][idx])])
    rel = np.abs(res) / np.maximum(den, 1.0e-300)
    return coeff, {
        "res_med": float(np.median(rel)),
        "res_max": float(np.max(rel)),
        "tail_rel": cheb_tail_ratio(coeff),
        **{f"spec_{key}": value for key, value in asdict(spec).items() if key != "name"},
    }


def build_teacher_grid(
    *,
    basis_names: list[str],
    specs: dict[str, DomainSpec],
    a_vals: np.ndarray,
    logw_vals: np.ndarray,
    n: int,
    grid_kind: str,
    ell: int,
    m: int,
    s: int,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    coeffs = {basis: np.zeros((len(a_vals), len(logw_vals), n + 1), dtype=np.complex128) for basis in basis_names}
    rows: list[dict] = []
    for ia, av in enumerate(a_vals):
        for iw, lw in enumerate(logw_vals):
            mode = KerrMode(M=1.0, a=float(av), omega=10.0 ** float(lw), ell=ell, m=m, s=s)
            lam = complex(mode.lambda_value)
            row = {
                "a": float(av),
                "logw": float(lw),
                "omega": float(mode.omega),
                "lambda_re": float(lam.real),
                "lambda_im": float(lam.imag),
            }
            for basis in basis_names:
                coeff, metrics = solve_teacher(mode, specs[basis], n, grid_kind)
                coeffs[basis][ia, iw] = coeff
                for key, value in metrics.items():
                    row[f"{basis}_{key}"] = value
            rows.append(row)
    return coeffs, rows
