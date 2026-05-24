#!/usr/bin/env python3
"""Test: add infinity Robin BC to single-point training.
The Robin BC constrains S_y/S at y=-1, providing the missing second boundary condition.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json, numpy as np, torch
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.transform_y import (
    compose_reduced_shape_from_f, horizon_regularity_slope,
    h_factor, h1_factor, g_factor,
)
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives
from physical_ansatz.mapping import r_plus
from utils.compute_lambda_usage import compute_lambda
from domain.atlas_builder import load_atlas, map_to_chart
from physical_ansatz.residual_pinn import (
    pinn_residual_loss, compute_f_derivatives_autograd,
)
from physical_ansatz.infinity_robin import (
    analytic_c_inf, infinity_robin_loss, compute_S_and_Sy_at_infinity,
)


def benchmark(model, a, omega, u, v, lam, r_grid, M, m, s, device):
    dtype = torch.float64
    a_t = torch.tensor([a], device=device, dtype=dtype)
    omega_t = torch.tensor([omega], device=device, dtype=dtype)
    u_t = torch.tensor([u], device=device, dtype=dtype)
    v_t = torch.tensor([v], device=device, dtype=dtype)
    r_grid_t = torch.tensor(r_grid, device=device, dtype=dtype)
    rp = r_plus(a_t, M)
    x_grid = rp / r_grid_t
    y_grid = 2.0 * x_grid - 1.0
    with torch.no_grad():
        valid = (y_grid >= -1) & (y_grid < 1)
        y_v = y_grid[valid]
        f_p = model(a_t, omega_t, y_v, u=u_t, v=v_t)
        slope = horizon_regularity_slope(a=a_t, omega=omega_t,
            lambda_=torch.tensor([lam], device=device, dtype=torch.complex128),
            m=m, M=M, s=s)
        shape = compose_reduced_shape_from_f(f=f_p.squeeze(0), y=y_v.squeeze(0), slope=slope.squeeze(0))
        h2 = h_factor(a_t, omega_t, m=m, M=M, s=s)
        r_eff = r_grid_t[valid]
        _, rm_v, _, _, _ = build_prefactor_primitives(r_eff, a_t, M=M, need_rs=False)
        P_v, _, _ = Leaver_prefactors(r_eff, a_t, omega_t, m=m, M=M, s=s, rp=rp, rm=rm_v)
        R_pred = P_v.squeeze(0) * h2 * shape
    try:
        from pybhpt_usage.compute_solution import compute_pybhpt_solution
        r_mod = r_eff.cpu().numpy()
        R_mod = R_pred.detach().cpu().numpy()
        r_ref, R_ref = compute_pybhpt_solution(a, omega, ell=2, m=m, r_grid=r_mod, timeout=30.0)
        R_ref = np.asarray(R_ref, dtype=np.complex128)
        R_mod_i = np.interp(r_ref, r_mod, np.abs(R_mod)) * np.exp(1j*np.interp(r_ref, r_mod, np.angle(R_mod)))
        mask = np.abs(R_ref) > 1e-15
        rel_err = np.abs(np.abs(R_mod_i)[mask] - np.abs(R_ref)[mask]) / np.abs(R_ref)[mask]
        return float(np.median(rel_err))
    except Exception as e:
        return None


def main():
    device = torch.device("cuda")
    M, m_mode, s_val = 1.0, 2, -2
    dtype = torch.float64
    cdtype = torch.complex128

    atlas = load_atlas("outputs/domain/atlas_l2_m2_logw.json")
    comp = atlas.components[0]
    patch_file = json.load(open("outputs/domain/patch_cover_l2_m2_logw.json"))
    p0 = [p for p in patch_file["patches"] if p["patch_id"] == 0][0]
    a_c = float(p0["a_center"])
    omega_c = float(p0["omega_center"])
    omega_chart = patch_file["meta"]["omega_chart_mode"]
    u_c, v_c = map_to_chart(comp, a_c, omega_c, omega_chart_mode=omega_chart)
    lam_c = compute_lambda(a_c, omega_c, 2, m_mode, s=s_val)

    rp = float(M + np.sqrt(M**2 - a_c**2))
    r_grid = np.logspace(np.log10(rp + 0.001), 2.0, 128)

    a_t = torch.tensor([a_c], device=device, dtype=dtype)
    omega_t = torch.tensor([omega_c], device=device, dtype=dtype)
    u_t = torch.tensor([u_c], device=device, dtype=dtype)
    v_t = torch.tensor([v_c], device=device, dtype=dtype)
    lam_t = torch.tensor([lam_c], device=device, dtype=cdtype)

    r_grid_t = torch.tensor(r_grid, device=device, dtype=dtype)
    x_grid = r_plus(a_t, M) / r_grid_t
    y_grid = 2.0 * x_grid - 1.0
    valid = (y_grid >= -1) & (y_grid < 1)
    y_valid = y_grid[valid].unsqueeze(0)
    y_bd = torch.empty(0, device=device, dtype=dtype)
    y_inf = torch.tensor([[-1.0]], device=device, dtype=dtype)

    cfg = {"problem": {"M": M, "s": s_val, "m": m_mode}}

    n_steps = 4000
    configs = [
        ("PDE only (baseline)", 0.0),
        ("PDE + Robin(0.01)", 0.01),
        ("PDE + Robin(0.1)", 0.1),
        ("PDE + Robin(1.0)", 1.0),
    ]

    for label, robin_weight in configs:
        print(f"\n{'='*60}")
        print(f"  {label} ({n_steps} steps)")
        print(f"{'='*60}")

        model = AutoencoderTeukolskyPINN().to(device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

        for step in range(n_steps):
            opt.zero_grad()

            loss, info = pinn_residual_loss(
                model=model, cfg=cfg,
                a_batch=a_t, omega_batch=omega_t, lambda_batch=lam_t,
                y_interior=y_valid, y_boundary=y_bd,
                weight_interior=1.0, weight_boundary=0.0,
                normalize_residual=False, u_batch=u_t, v_batch=v_t,
            )

            if robin_weight > 0:
                f_inf, fy_inf, _ = compute_f_derivatives_autograd(
                    model, a_t, omega_t, y_inf, u_batch=u_t, v_batch=v_t,
                )
                f_inf = f_inf.squeeze(-1)
                fy_inf = fy_inf.squeeze(-1)
                slope_inf = horizon_regularity_slope(
                    a=a_t, omega=omega_t, lambda_=lam_t, m=m_mode, M=M, s=s_val,
                )
                S_inf, Sy_inf = compute_S_and_Sy_at_infinity(f_inf, fy_inf, slope_inf)
                if not torch.is_complex(S_inf):
                    S_inf = S_inf.to(dtype=cdtype)
                if not torch.is_complex(Sy_inf):
                    Sy_inf = Sy_inf.to(dtype=cdtype)
                c_inf = analytic_c_inf(a_t, omega_t, lam_t, m=m_mode, M=M, s=s_val)
                loss_robin = infinity_robin_loss(S_inf, Sy_inf, c_inf).mean()
                if torch.isfinite(loss_robin):
                    loss = loss + robin_weight * loss_robin

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 400 == 0:
                model.eval()
                rel_err = benchmark(model, a_c, omega_c, u_c, v_c, lam_c,
                                    r_grid, M, m_mode, s_val, device)
                model.train()
                r_str = f" robin={loss_robin.item():.4e}" if robin_weight > 0 else ""
                e_str = f" rel_err={rel_err:.4e}" if rel_err else ""
                print(f"  step {step:>5d}: PDE={loss.item():.4f}{r_str}{e_str}")

        model.eval()
        final_rel_err = benchmark(model, a_c, omega_c, u_c, v_c, lam_c,
                                   r_grid, M, m_mode, s_val, device)
        print(f"  Final rel_err: {final_rel_err}")


if __name__ == "__main__":
    main()
