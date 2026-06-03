#!/usr/bin/env python3
"""Patch-local spectral surrogate plus amplitude decoder.

This experiment uses one fixed matching window.  For each (a, log10 omega)
sample it first solves direct spectral teachers for three regular factors:

  in:   R_in   = A_in   u_in,   right-normalized at the horizon
  down: R_down = A_down u_down, left-normalized at infinity
  up:   R_up   = A_up   u_up,   left-normalized at infinity

The coefficient networks learn Chebyshev coefficients for the three u's on
overlapping domains around one matching point.  After optional residual
refinement the coefficient networks are frozen and an amplitude head is
trained only from value-only multi-point matching:

  R_in(y_j) = B_inc R_down(y_j) + B_ref R_up(y_j).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.amplitude import (  # noqa: E402
    A_down,
    A_in,
    A_up,
    _domain_D,
    coeffs_numeric,
    r_of_z,
    solve_basis_domain,
)
from utils.matlcheb import cheb, real_to_cheb  # noqa: E402
from utils.mode import KerrMode  # noqa: E402


RDTYPE = torch.float64
CDTYPE = torch.complex128


@dataclass(frozen=True)
class BasisSpec:
    name: str
    z_left: float
    z_right: float
    bc_side: str
    domain: str


class ParamToCoeff(nn.Module):
    def __init__(
        self,
        n_coeff: int,
        out_coeff: int,
        width: int,
        depth: int,
        a_center: float,
        logw_center: float,
        a_scale: float,
        logw_scale: float,
        feature_kind: str = "raw",
        fourier_bands: int = 4,
    ):
        super().__init__()
        self.n_coeff = n_coeff
        self.out_coeff = out_coeff
        self.a_center = a_center
        self.logw_center = logw_center
        self.a_scale = a_scale
        self.logw_scale = logw_scale
        self.feature_kind = feature_kind
        self.fourier_bands = fourier_bands
        layers: list[nn.Module] = []
        in_dim = self.feature_dim()
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), nn.SiLU()]
            in_dim = width
        layers.append(nn.Linear(in_dim, 2 * out_coeff))
        self.net = nn.Sequential(*layers)

    def feature_dim(self) -> int:
        if self.feature_kind == "raw":
            return 2
        if self.feature_kind == "poly_fourier":
            return 5 + 4 * self.fourier_bands
        raise ValueError(self.feature_kind)

    def features(self, a: torch.Tensor, logw: torch.Tensor) -> torch.Tensor:
        base = torch.stack(
            [(a - self.a_center) / self.a_scale, (logw - self.logw_center) / self.logw_scale],
            dim=-1,
        )
        if self.feature_kind == "raw":
            return base
        xa = base[:, 0]
        xw = base[:, 1]
        feats = [xa, xw, xa * xw, xa * xa, xw * xw]
        for band in range(self.fourier_bands):
            freq = float(2**band) * torch.pi
            feats += [torch.sin(freq * xa), torch.cos(freq * xa), torch.sin(freq * xw), torch.cos(freq * xw)]
        return torch.stack(feats, dim=-1)

    def forward(self, a: torch.Tensor, logw: torch.Tensor) -> torch.Tensor:
        x = self.features(a, logw)
        out = self.net(x)
        low = out[:, 0::2].to(CDTYPE) + 1j * out[:, 1::2].to(CDTYPE)
        if self.out_coeff == self.n_coeff:
            return low
        full = torch.zeros((a.shape[0], self.n_coeff), dtype=CDTYPE, device=a.device)
        full[:, : self.out_coeff] = low
        return full


class AmplitudeHead(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        a_center: float,
        logw_center: float,
        a_scale: float,
        logw_scale: float,
        feature_kind: str = "raw",
        fourier_bands: int = 4,
    ):
        super().__init__()
        self.a_center = a_center
        self.logw_center = logw_center
        self.a_scale = a_scale
        self.logw_scale = logw_scale
        self.feature_kind = feature_kind
        self.fourier_bands = fourier_bands
        layers: list[nn.Module] = []
        in_dim = self.feature_dim()
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), nn.SiLU()]
            in_dim = width
        layers.append(nn.Linear(in_dim, 4))
        self.net = nn.Sequential(*layers)
        self.register_buffer("b_inc_scale", torch.tensor(1.0, dtype=RDTYPE))
        self.register_buffer("b_ref_scale", torch.tensor(1.0, dtype=RDTYPE))

    def feature_dim(self) -> int:
        if self.feature_kind == "raw":
            return 2
        if self.feature_kind == "poly_fourier":
            return 5 + 4 * self.fourier_bands
        raise ValueError(self.feature_kind)

    def features(self, a: torch.Tensor, logw: torch.Tensor) -> torch.Tensor:
        base = torch.stack(
            [(a - self.a_center) / self.a_scale, (logw - self.logw_center) / self.logw_scale],
            dim=-1,
        )
        if self.feature_kind == "raw":
            return base
        xa = base[:, 0]
        xw = base[:, 1]
        feats = [xa, xw, xa * xw, xa * xa, xw * xw]
        for band in range(self.fourier_bands):
            freq = float(2**band) * torch.pi
            feats += [torch.sin(freq * xa), torch.cos(freq * xa), torch.sin(freq * xw), torch.cos(freq * xw)]
        return torch.stack(feats, dim=-1)

    def forward(self, a: torch.Tensor, logw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(a, logw)
        out = self.net(x)
        b_inc = self.b_inc_scale.to(CDTYPE) * (out[:, 0].to(CDTYPE) + 1j * out[:, 1].to(CDTYPE))
        b_ref = self.b_ref_scale.to(CDTYPE) * (out[:, 2].to(CDTYPE) + 1j * out[:, 3].to(CDTYPE))
        return b_inc, b_ref


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cheb_eval_matrix(n: int, x: np.ndarray, device: str) -> torch.Tensor:
    k = np.arange(n + 1)
    V = np.cos(np.outer(np.arccos(np.clip(x, -1.0, 1.0)), k))
    return torch.tensor(V, dtype=RDTYPE, device=device)


def domain_nodes_and_mats(n: int, z_left: float, z_right: float, device: str):
    D, z = cheb_Domain(n, z_left, z_right)
    return (
        z,
        torch.tensor(z, dtype=RDTYPE, device=device),
        torch.tensor(D, dtype=RDTYPE, device=device),
        torch.tensor(D @ D, dtype=RDTYPE, device=device),
    )


def cheb_Domain(n: int, z_left: float, z_right: float):
    Dxi, xi = cheb(n)
    z = z_left + (z_right - z_left) * (1.0 - xi) / 2.0
    Dz = (-2.0 / (z_right - z_left)) * Dxi
    return Dz, z


def z_to_xi(z: np.ndarray, z_left: float, z_right: float) -> np.ndarray:
    return 1.0 - 2.0 * (z - z_left) / (z_right - z_left)


def anmr_kappa(mode: KerrMode, domain: str) -> float:
    kappa = abs(np.log(max(abs(mode.omega) * mode.rp, 1.0e-300)))
    if domain == "inner":
        kappa *= 0.5
    return float(kappa)


def z_to_comp_xi(z: np.ndarray, z_left: float, z_right: float, mode: KerrMode, domain: str, grid_kind: str) -> np.ndarray:
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
    t = np.arcsinh((z - z_left) / (z_right - z_left) * np.sinh(kappa)) / kappa
    return 1.0 - 2.0 * t


def basis_factor_np(basis: str, z: np.ndarray, mode: KerrMode) -> np.ndarray:
    r = r_of_z(z, mode)
    if basis == "in":
        return A_in(r, mode)
    if basis == "down":
        return A_down(r, mode)
    if basis == "up":
        return A_up(r, mode)
    raise ValueError(basis)


def branch_specs(args) -> dict[str, BasisSpec]:
    z_match = 0.5 * (args.y_match + 1.0)
    z_fit = 0.5 * args.match_width
    z_left = max(1.0e-8, z_match - z_fit)
    z_right = min(1.0 - 1.0e-8, z_match + z_fit)
    return {
        "in": BasisSpec("in", z_left, 1.0, "right", "inner"),
        "down": BasisSpec("down", 0.0, z_right, "left", "outer"),
        "up": BasisSpec("up", 0.0, z_right, "left", "outer"),
    }


def solve_teacher(mode: KerrMode, spec: BasisSpec, n: int, grid_kind: str):
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
        "tail_rel": tail_ratio(coeff),
    }


def tail_ratio(coeff: np.ndarray, tail: int = 8) -> float:
    denom = np.max(np.abs(coeff))
    if denom <= 1.0e-300:
        return 0.0
    return float(np.max(np.abs(coeff[-tail:])) / denom)


def auto_truncation(coeffs: np.ndarray, tol: float, min_n: int, tail: int = 6) -> int:
    max_abs = np.max(np.abs(coeffs), axis=(0, 1))
    denom = max(np.max(max_abs), 1.0e-300)
    for n_cut in range(max(min_n, tail), coeffs.shape[-1]):
        if np.max(max_abs[n_cut - tail + 1 : n_cut + 1]) / denom < tol:
            return n_cut + 1
    return coeffs.shape[-1]


def build_dataset(args, run_dir: Path, specs: dict[str, BasisSpec]):
    a_vals = np.linspace(args.a_center - args.a_half_width, args.a_center + args.a_half_width, args.n_param_side)
    logw_vals = np.linspace(args.logw_center - args.logw_half_width, args.logw_center + args.logw_half_width, args.n_param_side)
    rows = []
    coeffs = {basis: [] for basis in specs}
    for a in a_vals:
        for logw in logw_vals:
            mode = KerrMode(M=1.0, a=float(a), omega=10.0 ** float(logw), ell=args.ell, m=args.m, s=args.s)
            row = {"a": float(a), "logw": float(logw), "omega": mode.omega}
            for basis, spec in specs.items():
                coeff, metrics = solve_teacher(mode, spec, args.n, args.grid_kind)
                coeffs[basis].append(coeff)
                for key, value in metrics.items():
                    row[f"{basis}_{key}"] = value
            rows.append(row)
    packed = {f"coeff_{basis}": np.stack(values).astype(np.complex128) for basis, values in coeffs.items()}
    packed["a"] = np.array([row["a"] for row in rows], dtype=np.float64)
    packed["logw"] = np.array([row["logw"] for row in rows], dtype=np.float64)
    np.savez(run_dir / "teacher_coeffs.npz", **packed)
    write_csv(run_dir / "teacher_metrics.csv", rows)
    return rows, packed


def coeff_training_loss(pred: torch.Tensor, target: torch.Tensor, V: torch.Tensor, out_coeff: int, rel_floor: float, worst_weight: float):
    coeff_scale = torch.max(torch.abs(target[:, :out_coeff]).detach(), dim=1, keepdim=True).values.clamp_min(rel_floor)
    coeff_loss_i = torch.mean((torch.abs(pred[:, :out_coeff] - target[:, :out_coeff]) / coeff_scale) ** 2, dim=1)
    target_value = target @ V.to(CDTYPE).T
    pred_value = pred @ V.to(CDTYPE).T
    value_scale = torch.max(torch.abs(target_value).detach(), dim=1, keepdim=True).values.clamp_min(rel_floor)
    value_loss_i = torch.mean((torch.abs(pred_value - target_value) / value_scale) ** 2, dim=1)
    sample_loss = coeff_loss_i + value_loss_i
    return sample_loss.mean() + worst_weight * sample_loss.max()


def residual_loss(model: ParamToCoeff, basis: str, spec: BasisSpec, a: torch.Tensor, logw: torch.Tensor, args, device: str):
    coeff = model(a, logw)
    interior = slice(2, -2)
    _, xi_nodes = cheb(args.n)
    V = cheb_eval_matrix(args.n, xi_nodes, device)
    u = coeff @ V.to(CDTYPE).T
    losses = []
    for i in range(a.shape[0]):
        mode = KerrMode(M=1.0, a=float(a[i].detach().cpu()), omega=10.0 ** float(logw[i].detach().cpu()), ell=args.ell, m=args.m, s=args.s)
        D_np, z_np = _domain_D(mode, args.n, spec.z_left, spec.z_right, grid_kind=args.grid_kind, domain=spec.domain)
        D1 = torch.tensor(D_np, dtype=RDTYPE, device=device)
        D2 = D1 @ D1
        uz_i_all = u[i] @ D1.to(CDTYPE).T
        uzz_i_all = u[i] @ D2.to(CDTYPE).T
        z_int = z_np[interior]
        B2, B1, B0 = coeffs_numeric(z_int, mode, basis)
        B2t = torch.tensor(B2, dtype=CDTYPE, device=device)
        B1t = torch.tensor(B1, dtype=CDTYPE, device=device)
        B0t = torch.tensor(B0, dtype=CDTYPE, device=device)
        ui = u[i, interior]
        uzi = uz_i_all[interior]
        uzzi = uzz_i_all[interior]
        res = B2t * uzzi + B1t * uzi + B0t * ui
        den = torch.maximum(torch.maximum(torch.abs(B2t * uzzi), torch.abs(B1t * uzi)), torch.abs(B0t * ui)).clamp_min(1e-30)
        rel = torch.abs(res) / den
        losses.append(torch.mean(rel * rel))
    return torch.stack(losses).mean()


def init_center_bias(model: ParamToCoeff, center_coeff: np.ndarray):
    last_linear = None
    for module in reversed(model.net):
        if isinstance(module, nn.Linear):
            last_linear = module
            break
    if last_linear is None:
        return
    with torch.no_grad():
        last_linear.weight.zero_()
        bias = np.empty(2 * model.out_coeff, dtype=np.float64)
        bias[0::2] = center_coeff[: model.out_coeff].real
        bias[1::2] = center_coeff[: model.out_coeff].imag
        last_linear.bias.copy_(torch.tensor(bias, dtype=RDTYPE, device=last_linear.bias.device))


def train_coeff_nets(args, run_dir: Path, specs: dict[str, BasisSpec], rows: list[dict], packed: dict):
    device = args.device
    a = torch.tensor(packed["a"], dtype=RDTYPE, device=device)
    logw = torch.tensor(packed["logw"], dtype=RDTYPE, device=device)
    models = {}
    histories = {}
    trunc_rows = []
    rng = np.random.default_rng(args.seed)
    full_indices = torch.arange(len(rows), dtype=torch.long, device=device)
    center_idx = int(np.argmin(np.abs(packed["a"] - args.a_center) + np.abs(packed["logw"] - args.logw_center)))
    for basis, spec in specs.items():
        coeff_np = packed[f"coeff_{basis}"]
        n_auto = auto_truncation(coeff_np[:, None, :], args.trunc_tol, args.min_out_coeff)
        out_coeff = min(args.out_coeff or n_auto, args.n + 1)
        trunc_rows.append({"basis": basis, "n_auto": n_auto, "out_coeff": out_coeff, "tail_tol": args.trunc_tol})
        target = torch.tensor(coeff_np, dtype=CDTYPE, device=device)
        z_np, _, _, _ = domain_nodes_and_mats(args.n, spec.z_left, spec.z_right, device)
        V = cheb_eval_matrix(args.n, z_to_xi(z_np, spec.z_left, spec.z_right), device)
        model = ParamToCoeff(
            args.n + 1,
            out_coeff,
            args.width,
            args.depth,
            args.a_center,
            args.logw_center,
            args.a_half_width,
            args.logw_half_width,
            feature_kind=args.feature_kind,
            fourier_bands=args.fourier_bands,
        ).to(device, dtype=RDTYPE)
        init_center_bias(model, coeff_np[center_idx])
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-8)
        history = []
        for step in range(1, args.coeff_steps + 1):
            if args.batch <= 0 or args.batch >= len(rows):
                idx = full_indices
            else:
                idx = torch.tensor(rng.choice(len(rows), min(args.batch, len(rows)), replace=False), dtype=torch.long, device=device)
            pred = model(a[idx], logw[idx])
            loss = coeff_training_loss(pred, target[idx], V, out_coeff, args.coeff_rel_floor, args.worst_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % args.log_every == 0 or step == 1 or step == args.coeff_steps:
                item = {"basis": basis, "stage": "coeff", "step": step, "loss": float(loss.detach().cpu())}
                history.append(item)
                print(json.dumps(item), flush=True)
        for group in opt.param_groups:
            group["lr"] = args.res_lr
        for step in range(1, args.res_steps + 1):
            if args.batch <= 0 or args.batch >= len(rows):
                idx = full_indices
            else:
                idx = torch.tensor(rng.choice(len(rows), min(args.batch, len(rows)), replace=False), dtype=torch.long, device=device)
            loss = residual_loss(model, basis, spec, a[idx], logw[idx], args, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % args.log_every == 0 or step == 1 or step == args.res_steps:
                item = {"basis": basis, "stage": "residual", "step": step, "loss": float(loss.detach().cpu())}
                history.append(item)
                print(json.dumps(item), flush=True)
        models[basis] = model
        histories[basis] = history
        torch.save(model.state_dict(), run_dir / f"{basis}_coeff_net.pt")
    write_csv(run_dir / "truncation.csv", trunc_rows)
    write_csv(run_dir / "coeff_train_history.csv", [item for history in histories.values() for item in history])
    evaluate_coeff_models(args, run_dir, rows, packed, models)
    return models, trunc_rows


def evaluate_coeff_models(args, run_dir: Path, rows: list[dict], packed: dict, models: dict[str, ParamToCoeff]):
    device = args.device
    _, xi = cheb(args.n)
    V = cheb_eval_matrix(args.n, xi, device).to(CDTYPE)
    a_all = torch.tensor(packed["a"], dtype=RDTYPE, device=device)
    logw_all = torch.tensor(packed["logw"], dtype=RDTYPE, device=device)
    eval_rows = []
    with torch.no_grad():
        for basis, model in models.items():
            target = torch.tensor(packed[f"coeff_{basis}"], dtype=CDTYPE, device=device)
            pred = model(a_all, logw_all)
            target_value = target @ V.T
            pred_value = pred @ V.T
            coeff_scale = torch.max(torch.abs(target), dim=1).values.clamp_min(1e-30)
            value_scale = torch.max(torch.abs(target_value), dim=1).values.clamp_min(1e-30)
            coeff_rel = torch.linalg.vector_norm(pred - target, dim=1) / (torch.sqrt(torch.tensor(target.shape[1], dtype=RDTYPE, device=device)) * coeff_scale)
            value_rel = torch.linalg.vector_norm(pred_value - target_value, dim=1) / (torch.sqrt(torch.tensor(target_value.shape[1], dtype=RDTYPE, device=device)) * value_scale)
            for i, row in enumerate(rows):
                eval_rows.append({
                    "basis": basis,
                    "a": row["a"],
                    "logw": row["logw"],
                    "omega": row["omega"],
                    "coeff_rel_rms": float(coeff_rel[i].cpu()),
                    "value_rel_rms": float(value_rel[i].cpu()),
                })
    write_csv(run_dir / "coeff_eval.csv", eval_rows)


def eval_branch_values(model: ParamToCoeff, spec: BasisSpec, z_values: np.ndarray, a: torch.Tensor, logw: torch.Tensor, args, device: str):
    mode = KerrMode(M=1.0, a=float(a.item()), omega=10.0 ** float(logw.item()), ell=args.ell, m=args.m, s=args.s)
    xi = z_to_comp_xi(z_values, spec.z_left, spec.z_right, mode, spec.domain, args.grid_kind)
    V = cheb_eval_matrix(model.n_coeff - 1, xi, device)
    coeff = model(a, logw)
    return coeff @ V.to(CDTYPE).T


def factor_torch(basis: str, z_values: np.ndarray, mode: KerrMode, device: str):
    factor = basis_factor_np(basis, z_values, mode)
    return torch.tensor(factor, dtype=CDTYPE, device=device)


def teacher_amplitude_label(args, specs: dict[str, BasisSpec], row: dict):
    mode = KerrMode(M=1.0, a=row["a"], omega=row["omega"], ell=args.ell, m=args.m, s=args.s)
    z_vals = np.linspace(0.5 * (args.y_match + 1.0) - 0.5 * args.fit_width, 0.5 * (args.y_match + 1.0) + 0.5 * args.fit_width, args.n_match)
    cols = []
    rhs = []
    solved = {}
    for basis, spec in specs.items():
        solved[basis] = solve_basis_domain(mode, basis, args.n, spec.z_left, spec.z_right, spec.bc_side, domain=spec.domain, grid_kind=args.grid_kind)
        coeff = real_to_cheb(solved[basis]["u"])
        xi = z_to_comp_xi(z_vals, spec.z_left, spec.z_right, mode, spec.domain, args.grid_kind)
        V = np.cos(np.outer(np.arccos(np.clip(xi, -1.0, 1.0)), np.arange(args.n + 1)))
        solved[basis]["R"] = (V @ coeff) * basis_factor_np(basis, z_vals, mode)
    rd = solved["down"]["R"]
    ru = solved["up"]["R"]
    ri = solved["in"]["R"]
    A = np.stack([rd, ru], axis=1)
    scale = np.linalg.norm(A, axis=0)
    scale = np.where(scale > 1.0e-300, scale, 1.0)
    x, *_ = np.linalg.lstsq(A / scale[None, :], ri, rcond=None)
    x = x / scale
    residual = np.linalg.norm(A @ x - ri) / max(np.linalg.norm(ri), 1.0e-300)
    return complex(x[0]), complex(x[1]), float(residual)


def solve_amplitudes_from_network(args, specs: dict[str, BasisSpec], row: dict, coeff_models: dict[str, ParamToCoeff]):
    device = args.device
    mode = KerrMode(M=1.0, a=row["a"], omega=row["omega"], ell=args.ell, m=args.m, s=args.s)
    z_vals = np.linspace(0.5 * (args.y_match + 1.0) - 0.5 * args.fit_width, 0.5 * (args.y_match + 1.0) + 0.5 * args.fit_width, args.n_match)
    a = torch.tensor([row["a"]], dtype=RDTYPE, device=device)
    logw = torch.tensor([row["logw"]], dtype=RDTYPE, device=device)
    with torch.no_grad():
        u_in = eval_branch_values(coeff_models["in"], specs["in"], z_vals, a, logw, args, device).squeeze(0).cpu().numpy()
        u_down = eval_branch_values(coeff_models["down"], specs["down"], z_vals, a, logw, args, device).squeeze(0).cpu().numpy()
        u_up = eval_branch_values(coeff_models["up"], specs["up"], z_vals, a, logw, args, device).squeeze(0).cpu().numpy()
    ri = basis_factor_np("in", z_vals, mode) * u_in
    rd = basis_factor_np("down", z_vals, mode) * u_down
    ru = basis_factor_np("up", z_vals, mode) * u_up
    rhs = ri / rd
    col = ru / rd
    A = np.stack([np.ones_like(col), col], axis=1)
    scale = np.linalg.norm(A, axis=0)
    scale = np.where(scale > 1.0e-300, scale, 1.0)
    x, *_ = np.linalg.lstsq(A / scale[None, :], rhs, rcond=None)
    x = x / scale
    residual = np.linalg.norm(A @ x - rhs) / max(np.linalg.norm(rhs), 1.0e-300)
    return complex(x[0]), complex(x[1]), float(residual)


def evaluate_analytic_amplitudes(args, run_dir: Path, specs: dict[str, BasisSpec], rows: list[dict], coeff_models: dict[str, ParamToCoeff]):
    out_rows = []
    for row in rows:
        b_inc_t, b_ref_t, fit_t = teacher_amplitude_label(args, specs, row)
        b_inc_n, b_ref_n, fit_n = solve_amplitudes_from_network(args, specs, row, coeff_models)
        out_rows.append({
            "a": row["a"],
            "logw": row["logw"],
            "omega": row["omega"],
            "teacher_B_inc_re": b_inc_t.real,
            "teacher_B_inc_im": b_inc_t.imag,
            "teacher_B_ref_re": b_ref_t.real,
            "teacher_B_ref_im": b_ref_t.imag,
            "network_B_inc_re": b_inc_n.real,
            "network_B_inc_im": b_inc_n.imag,
            "network_B_ref_re": b_ref_n.real,
            "network_B_ref_im": b_ref_n.imag,
            "teacher_fit_res": fit_t,
            "network_fit_res": fit_n,
            "relerr_B_inc_teacher": abs(b_inc_n - b_inc_t) / max(abs(b_inc_t), 1.0e-300),
            "relerr_B_ref_teacher": abs(b_ref_n - b_ref_t) / max(abs(b_ref_t), 1.0e-300),
        })
    write_csv(run_dir / "analytic_amplitude_eval.csv", out_rows)


def train_amplitude_head(args, run_dir: Path, specs: dict[str, BasisSpec], rows: list[dict], packed: dict, coeff_models: dict[str, ParamToCoeff]):
    device = args.device
    for model in coeff_models.values():
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    a_all = torch.tensor(packed["a"], dtype=RDTYPE, device=device)
    logw_all = torch.tensor(packed["logw"], dtype=RDTYPE, device=device)
    z_values = np.linspace(0.5 * (args.y_match + 1.0) - 0.5 * args.fit_width, 0.5 * (args.y_match + 1.0) + 0.5 * args.fit_width, args.n_match)
    label_rows = []
    with torch.no_grad():
        labels = []
        for row in rows:
            b_inc, b_ref, fit_res = teacher_amplitude_label(args, specs, row)
            labels.append((b_inc, b_ref))
            label_rows.append({
                "a": row["a"],
                "logw": row["logw"],
                "omega": row["omega"],
                "B_inc_re": b_inc.real,
                "B_inc_im": b_inc.imag,
                "B_ref_re": b_ref.real,
                "B_ref_im": b_ref.imag,
                "teacher_fit_res": fit_res,
            })
        center = labels[int(np.argmin(np.abs(packed["a"] - args.a_center) + np.abs(packed["logw"] - args.logw_center)))]
    head = AmplitudeHead(
        args.amp_width,
        args.amp_depth,
        args.a_center,
        args.logw_center,
        args.a_half_width,
        args.logw_half_width,
        feature_kind=args.feature_kind,
        fourier_bands=args.fourier_bands,
    ).to(device, dtype=RDTYPE)
    inc_scale = float(np.median([abs(item[0]) for item in labels])) or 1.0
    ref_scale = float(np.median([abs(item[1]) for item in labels])) or 1.0
    with torch.no_grad():
        head.b_inc_scale.copy_(torch.tensor(inc_scale, dtype=RDTYPE, device=device))
        head.b_ref_scale.copy_(torch.tensor(ref_scale, dtype=RDTYPE, device=device))
        last = [module for module in head.net if isinstance(module, nn.Linear)][-1]
        last.weight.zero_()
        last.bias.copy_(
            torch.tensor(
                [center[0].real / inc_scale, center[0].imag / inc_scale, center[1].real / ref_scale, center[1].imag / ref_scale],
                dtype=RDTYPE,
                device=device,
            )
        )
    write_csv(run_dir / "teacher_amplitudes.csv", label_rows)
    opt = torch.optim.AdamW(head.parameters(), lr=args.amp_lr, weight_decay=1e-8)
    rng = np.random.default_rng(args.seed + 17)
    full_indices = torch.arange(len(rows), dtype=torch.long, device=device)
    history = []
    for step in range(1, args.amp_steps + 1):
        if args.batch <= 0 or args.batch >= len(rows):
            idx_np = np.arange(len(rows))
            idx = full_indices
        else:
            idx_np = rng.choice(len(rows), min(args.batch, len(rows)), replace=False)
            idx = torch.tensor(idx_np, dtype=torch.long, device=device)
        a = a_all[idx]
        logw = logw_all[idx]
        b_inc, b_ref = head(a, logw)
        losses = []
        for local_i, row_i in enumerate(idx_np):
            mode = KerrMode(M=1.0, a=rows[row_i]["a"], omega=rows[row_i]["omega"], ell=args.ell, m=args.m, s=args.s)
            u_in = eval_branch_values(coeff_models["in"], specs["in"], z_values, a[local_i : local_i + 1], logw[local_i : local_i + 1], args, device).squeeze(0)
            u_down = eval_branch_values(coeff_models["down"], specs["down"], z_values, a[local_i : local_i + 1], logw[local_i : local_i + 1], args, device).squeeze(0)
            u_up = eval_branch_values(coeff_models["up"], specs["up"], z_values, a[local_i : local_i + 1], logw[local_i : local_i + 1], args, device).squeeze(0)
            R_in = factor_torch("in", z_values, mode, device) * u_in
            R_down = factor_torch("down", z_values, mode, device) * u_down
            R_up = factor_torch("up", z_values, mode, device) * u_up
            if args.amp_fit_space == "raw":
                pred = b_inc[local_i] * R_down + b_ref[local_i] * R_up
                losses.append(torch.sum(torch.abs(pred - R_in) ** 2) / torch.sum(torch.abs(R_in).clamp_min(1e-30) ** 2))
            else:
                rhs = R_in / R_down
                ratio = R_up / R_down
                pred = b_inc[local_i] + b_ref[local_i] * ratio
                losses.append(torch.sum(torch.abs(pred - rhs) ** 2) / torch.sum(torch.abs(rhs).clamp_min(1e-30) ** 2))
        sample_losses = torch.stack(losses)
        loss = sample_losses.mean() + args.worst_weight * sample_losses.max()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.log_every == 0 or step == 1 or step == args.amp_steps:
            item = {"stage": "amplitude", "step": step, "loss": float(loss.detach().cpu())}
            history.append(item)
            print(json.dumps(item), flush=True)
    torch.save(head.state_dict(), run_dir / "amplitude_head.pt")
    write_csv(run_dir / "amplitude_train_history.csv", history)
    evaluate_amplitudes(args, run_dir, specs, rows, packed, coeff_models, head, label_rows)
    return head


def evaluate_amplitudes(args, run_dir: Path, specs: dict[str, BasisSpec], rows: list[dict], packed: dict, coeff_models, head, label_rows):
    device = args.device
    z_values = np.linspace(0.5 * (args.y_match + 1.0) - 0.5 * args.fit_width, 0.5 * (args.y_match + 1.0) + 0.5 * args.fit_width, args.n_match)
    out_rows = []
    with torch.no_grad():
        for row, label in zip(rows, label_rows):
            a = torch.tensor([row["a"]], dtype=RDTYPE, device=device)
            logw = torch.tensor([row["logw"]], dtype=RDTYPE, device=device)
            b_inc, b_ref = head(a, logw)
            b_inc_c = complex(b_inc.item())
            b_ref_c = complex(b_ref.item())
            b_inc_t = complex(label["B_inc_re"], label["B_inc_im"])
            b_ref_t = complex(label["B_ref_re"], label["B_ref_im"])
            out_rows.append({
                "a": row["a"],
                "logw": row["logw"],
                "omega": row["omega"],
                "B_inc_re": b_inc_c.real,
                "B_inc_im": b_inc_c.imag,
                "B_ref_re": b_ref_c.real,
                "B_ref_im": b_ref_c.imag,
                "relerr_B_inc_teacher": abs(b_inc_c - b_inc_t) / max(abs(b_inc_t), 1.0e-300),
                "relerr_B_ref_teacher": abs(b_ref_c - b_ref_t) / max(abs(b_ref_t), 1.0e-300),
            })
    write_csv(run_dir / "amplitude_eval.csv", out_rows)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    logw = np.array([row["logw"] for row in out_rows])
    err_inc = np.array([row["relerr_B_inc_teacher"] for row in out_rows])
    err_ref = np.array([row["relerr_B_ref_teacher"] for row in out_rows])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(logw, err_inc, "o", label="B_inc")
    ax.semilogy(logw, err_ref, "s", label="B_ref")
    ax.set_xlabel("log10 omega")
    ax.set_ylabel("relative error vs spectral teacher")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "amplitude_eval_errors.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-center", type=float, default=0.5)
    parser.add_argument("--logw-center", type=float, default=-1.5)
    parser.add_argument("--a-half-width", type=float, default=0.12)
    parser.add_argument("--logw-half-width", type=float, default=0.25)
    parser.add_argument("--n-param-side", type=int, default=7)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--y-match", type=float, default=-0.25)
    parser.add_argument("--match-width", type=float, default=0.12)
    parser.add_argument("--fit-width", type=float, default=0.06)
    parser.add_argument("--n", type=int, default=56)
    parser.add_argument("--grid-kind", choices=["linear", "anmr", "auto"], default="anmr")
    parser.add_argument("--out-coeff", type=int, default=0)
    parser.add_argument("--min-out-coeff", type=int, default=18)
    parser.add_argument("--trunc-tol", type=float, default=1.0e-10)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--feature-kind", choices=["raw", "poly_fourier"], default="raw")
    parser.add_argument("--fourier-bands", type=int, default=4)
    parser.add_argument("--amp-width", type=int, default=96)
    parser.add_argument("--amp-depth", type=int, default=3)
    parser.add_argument("--coeff-steps", type=int, default=800)
    parser.add_argument("--res-steps", type=int, default=100)
    parser.add_argument("--amp-steps", type=int, default=800)
    parser.add_argument("--train-amplitude-head", action="store_true")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--worst-weight", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--res-lr", type=float, default=1.0e-5)
    parser.add_argument("--amp-lr", type=float, default=1.0e-3)
    parser.add_argument("--amp-fit-space", choices=["raw", "down-ratio"], default="down-ratio")
    parser.add_argument("--coeff-rel-floor", type=float, default=1.0e-10)
    parser.add_argument("--n-match", type=int, default=41)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/patch_spectral_decoder")
    args = parser.parse_args()

    torch.set_default_dtype(RDTYPE)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    specs = branch_specs(args)
    config = vars(args).copy()
    config["branch_specs"] = {name: spec.__dict__ for name, spec in specs.items()}
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(json.dumps({"run_dir": str(run_dir), "branch_specs": config["branch_specs"]}, indent=2), flush=True)

    rows, packed = build_dataset(args, run_dir, specs)
    coeff_models, trunc_rows = train_coeff_nets(args, run_dir, specs, rows, packed)
    evaluate_analytic_amplitudes(args, run_dir, specs, rows, coeff_models)
    if args.train_amplitude_head:
        train_amplitude_head(args, run_dir, specs, rows, packed, coeff_models)
    print(json.dumps({"run_dir": str(run_dir), "truncation": trunc_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
