from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.amplitude import _domain_D
from utils.matlcheb import cheb
from utils.mode import KerrMode


def cheb_lobatto_physical(center: float, half_width: float, n_side: int) -> tuple[np.ndarray, np.ndarray]:
    _, xi = cheb(n_side - 1)
    return center + half_width * xi, xi


def cheb_vandermonde(xi: np.ndarray, degree: int | None = None) -> np.ndarray:
    xi = np.asarray(xi, dtype=float)
    if degree is None:
        degree = xi.size - 1
    theta = np.arccos(np.clip(xi, -1.0, 1.0))
    return np.cos(np.outer(theta, np.arange(degree + 1)))


def tensor_vandermonde(xi_a: np.ndarray, xi_w: np.ndarray, deg_a: int, deg_w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Va = cheb_vandermonde(xi_a, deg_a)
    Vw = cheb_vandermonde(xi_w, deg_w)
    rows = [np.kron(Va[ia], Vw[iw]) for ia in range(len(xi_a)) for iw in range(len(xi_w))]
    return np.asarray(rows), Va, Vw


def fit_tensor_decoder(coeffs: np.ndarray, xi_a: np.ndarray, xi_w: np.ndarray, deg_a: int, deg_w: int) -> np.ndarray:
    n_a, n_w, n_coeff = coeffs.shape
    A, _, _ = tensor_vandermonde(xi_a, xi_w, deg_a, deg_w)
    y = coeffs.reshape(n_a * n_w, n_coeff)
    scale = np.linalg.norm(A, axis=0)
    scale = np.where(scale > 1.0e-300, scale, 1.0)
    x, *_ = np.linalg.lstsq(A / scale[None, :], y, rcond=None)
    x = x / scale[:, None]
    return x.reshape(deg_a + 1, deg_w + 1, n_coeff)


def eval_tensor_decoder(decoder: np.ndarray, xi_a: np.ndarray, xi_w: np.ndarray) -> np.ndarray:
    decoder = np.asarray(decoder)
    deg_a = decoder.shape[0] - 1
    deg_w = decoder.shape[1] - 1
    A, _, _ = tensor_vandermonde(np.asarray(xi_a), np.asarray(xi_w), deg_a, deg_w)
    return (A @ decoder.reshape((deg_a + 1) * (deg_w + 1), decoder.shape[2])).reshape(
        len(xi_a), len(xi_w), decoder.shape[2]
    )


def clenshaw_eval(coeff: np.ndarray, xi: np.ndarray | float) -> np.ndarray:
    coeff = np.asarray(coeff)
    scalar = np.ndim(xi) == 0
    x = np.asarray(xi, dtype=float)
    flat = np.ravel(np.clip(x, -1.0, 1.0))
    if coeff.ndim == 1:
        V = cheb_vandermonde(flat, coeff.shape[0] - 1)
        out = V @ coeff
        out = out.reshape(x.shape)
        return out.item() if scalar else out
    V = cheb_vandermonde(flat, coeff.shape[-1] - 1)
    out = coeff @ V.T
    return out.reshape(coeff.shape[:-1] + x.shape)


def cheb_eval_matrix(n: int, xi: np.ndarray) -> np.ndarray:
    return cheb_vandermonde(np.asarray(xi, dtype=float), n)


def z_to_xi(z: np.ndarray | float, z_left: float, z_right: float) -> np.ndarray:
    return 1.0 - 2.0 * (np.asarray(z, dtype=float) - z_left) / (z_right - z_left)


def anmr_kappa(mode: KerrMode, domain: str) -> float:
    kappa = abs(np.log(max(abs(mode.omega) * mode.rp, 1.0e-300)))
    if domain == "inner":
        kappa *= 0.5
    return float(kappa)


def z_to_comp_xi(
    z: np.ndarray | float,
    z_left: float,
    z_right: float,
    mode: KerrMode,
    domain: str,
    grid_kind: str,
) -> np.ndarray:
    if grid_kind == "linear":
        return z_to_xi(z, z_left, z_right)
    if grid_kind == "auto":
        resolved = "anmr" if abs(mode.omega) < 1.0e-1 else "linear"
        return z_to_comp_xi(z, z_left, z_right, mode, domain, resolved)
    if grid_kind != "anmr":
        raise ValueError(grid_kind)
    kappa = anmr_kappa(mode, domain)
    if kappa <= 1.0e-12:
        return z_to_xi(z, z_left, z_right)
    z_arr = np.asarray(z, dtype=float)
    t = np.arcsinh((z_arr - z_left) / (z_right - z_left) * np.sinh(kappa)) / kappa
    return 1.0 - 2.0 * t


def xi_to_z(xi: np.ndarray | float, z_left: float, z_right: float) -> np.ndarray:
    return z_left + (z_right - z_left) * (1.0 - np.asarray(xi, dtype=float)) / 2.0


def cheb_derivative_matrices(n: int, z_left: float, z_right: float, grid_kind: str, mode: KerrMode, domain: str):
    D, z = _domain_D(mode, n, z_left, z_right, grid_kind=grid_kind, domain=domain)
    return D, D @ D, z


def coeff_l2_value_errors(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = target.shape[-1] - 1
    _, xi = cheb(n)
    V = cheb_vandermonde(xi, n).astype(np.complex128)
    pred_flat = pred.reshape(-1, n + 1)
    target_flat = target.reshape(-1, n + 1)
    pred_u = pred_flat @ V.T
    target_u = target_flat @ V.T
    coeff_rel = np.linalg.norm(pred_flat - target_flat, axis=1) / np.maximum(np.linalg.norm(target_flat, axis=1), 1.0e-300)
    value_rel = np.linalg.norm(pred_u - target_u, axis=1) / np.maximum(np.linalg.norm(target_u, axis=1), 1.0e-300)
    return coeff_rel, value_rel


def cheb_tail_ratio(coeff: np.ndarray, tail: int = 8) -> float:
    coeff = np.asarray(coeff)
    denom = np.max(np.abs(coeff))
    if denom <= 1.0e-300:
        return 0.0
    return float(np.max(np.abs(coeff[-tail:])) / denom)


@dataclass(frozen=True)
class DomainSpec:
    name: str
    z_left: float
    z_right: float
    bc_side: str
    domain: str
