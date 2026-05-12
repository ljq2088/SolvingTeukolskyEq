"""
Evaluate v3 patch 0 model vs pybhpt benchmark.

Selects a-ω pairs within patch 0 and compares:
- Model R_in(r) = P(r) * h2 * S(y)  against
- pybhpt R_in(r) reference
"""
import json
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/ljq/code/PINN/SolvingTeukolsky")

from model.pinn_mlp import PINN_MLP
from physical_ansatz.transform_y import compose_reduced_shape_from_f, horizon_regularity_slope, h_factor
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives
from physical_ansatz.mapping import r_plus
from physical_ansatz.residual import get_lambda_from_cfg, AuxCache


def get_ab_from_uv(u, v, ps_cfg):
    """Convert (u,v) atlas coords back to (a, omega)."""
    a_c = ps_cfg["a"]["center"]
    a_r = ps_cfg["a"]["range"]
    w_c = ps_cfg["omega"]["center"]
    w_r = ps_cfg["omega"]["range"]
    a = a_c + (2*u - 1) * a_r
    omega = w_c + (2*v - 1) * w_r
    return a, omega


def predict_R(model, a, omega, u, v, lam, r_grid, physics_cfg):
    """Model prediction of full R_in(r)."""
    M = float(physics_cfg["problem"].get("M", 1.0))
    m = int(physics_cfg["problem"].get("m", 2))
    s = int(physics_cfg["problem"].get("s", -2))

    a_t = torch.tensor([a], dtype=torch.float64)
    omega_t = torch.tensor([omega], dtype=torch.float64)
    u_t = torch.tensor([u], dtype=torch.float64)
    v_t = torch.tensor([v], dtype=torch.float64)
    lam_t = lam.clone().detach().to(torch.float64)

    r_grid_t = torch.tensor(r_grid, dtype=torch.float64)
    rp = r_plus(a_t, M)
    x_grid = rp / r_grid_t
    y_grid = 2.0 * x_grid - 1.0

    with torch.no_grad():
        f_pred = model(a_t, omega_t, y_grid, u=u_t, v=v_t).squeeze(0)

        slope = horizon_regularity_slope(
            a=a_t, omega=omega_t, lambda_=lam_t.unsqueeze(0), m=m, M=M, s=s,
        ).squeeze(0)

        shape = compose_reduced_shape_from_f(f=f_pred, y=y_grid, slope=slope)

        h2 = h_factor(a_t, omega_t, m=m, M=M, s=s)
        _, rm, _, _, _ = build_prefactor_primitives(r_grid_t, a_t, M=M, need_rs=False)
        P, _, _ = Leaver_prefactors(r_grid_t, a_t, omega_t, m=m, M=M, s=s, rp=rp, rm=rm)

        R_pred = P * h2 * shape

    return R_pred.detach().cpu().numpy()


def main():
    ckpt_path = "outputs/atlas_multipatch_phase4_integral/patch_runs/20260510_202716_patch_000_comp_0_u_0.500_v_0.513/checkpoints/best_model.pt"
    cfg_path = "config/pinn_config_phase4_integral.yaml"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    full_cfg = ckpt["full_cfg"]
    model_cfg = full_cfg.get("model", {})
    step = ckpt["step"]
    best_val = ckpt["best_val_mean"]
    print(f"Loaded checkpoint: step={step}, best_val={best_val:.6e}")

    model = PINN_MLP(
        hidden_dims=model_cfg.get("hidden_dims", [128, 128, 128, 128]),
        activation=model_cfg.get("activation", "silu"),
        fourier_num_freqs=model_cfg.get("fourier_num_freqs", 2),
        fourier_scale=model_cfg.get("fourier_scale", 1.0),
        param_embed_dim=model_cfg.get("param_embed_dim", 64),
        use_film=model_cfg.get("use_film", True),
        use_residual=model_cfg.get("use_residual", True),
    ).to(torch.float64)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Physics config
    physics_cfg = full_cfg["physics"]
    train_cfg = full_cfg["train"]
    ps_cfg = train_cfg["parameter_space"]
    M = float(physics_cfg["problem"].get("M", 1.0))
    m = int(physics_cfg["problem"].get("m", 2))
    s = int(physics_cfg["problem"].get("s", -2))

    # Cache for lambda lookup — preload from probe file
    cache = AuxCache()
    probe_file = "outputs/domain/probe_l2_m2.json"
    with open(probe_file) as f:
        probe_data = json.load(f)
    for item in probe_data.get("lambda_cache", {}).values():
        # item is a dict with keys: a, omega, lambda_re, lambda_im, l, m, s
        a_key = round(float(item["a"]), 12)
        omega_key = round(float(item["omega"]), 12)
        l_val = int(item.get("l", 2))
        m_val = int(item.get("m", 2))
        s_val = int(item.get("s", -2))
        key = ("lambda", a_key, omega_key, l_val, m_val, s_val)
        cache.lambda_cache[key] = complex(float(item["lambda_re"]), float(item["lambda_im"]))
    print(f"Preloaded {len(cache.lambda_cache)} lambda entries")

    # Patch info
    with open("outputs/domain/patch_cover_l2_m2.json") as f:
        patch_data = json.load(f)
    p0 = [p for p in patch_data["patches"] if p["patch_id"] == 0][0]
    uc, vc = p0["u_center"], p0["v_center"]

    # Test cases
    test_uv = [
        (uc, vc, "patch center"),
        (uc, vc - 0.2, "lower omega"),
        (uc, vc + 0.2, "higher omega"),
        (uc - 0.2, vc, "lower a"),
        (uc + 0.2, vc, "higher a"),
    ]

    # r-grid for comparison: from just outside horizon to r=100
    a_center = ps_cfg["a"]["center"]
    rp_center = float(M + np.sqrt(M**2 - a_center**2))
    r_grid = np.logspace(np.log10(rp_center + 0.001), np.log10(100), 300)

    from pathlib import Path

    print(f"\n{'='*80}")
    print(f"Evaluating {len(test_uv)} test cases")
    print(f"{'='*80}")

    for u, v, label in test_uv:
        a, omega = get_ab_from_uv(u, v, ps_cfg)
        print(f"\n--- {label}: a={a:.4f}, ω={omega:.4f} (u={u:.4f}, v={v:.4f}) ---")

        # Get lambda
        a_t = torch.tensor([a], dtype=torch.float64)
        omega_t = torch.tensor([omega], dtype=torch.float64)
        lam = get_lambda_from_cfg(physics_cfg, cache, a_t, omega_t)
        lam_c = complex(lam.item()) if lam.numel() == 1 else complex(lam[0].item())
        print(f"  λ = {lam_c.real:.8f} + {lam_c.imag:.8f}i")

        # pybhpt benchmark
        try:
            from pybhpt_usage.compute_solution import compute_pybhpt_solution

            r_ref, R_ref = compute_pybhpt_solution(a, omega, ell=2, m=2, r_grid=r_grid, timeout=15.0)
            R_ref = np.asarray(R_ref, dtype=np.complex128)
            print(f"  pybhpt: {len(r_ref)} r-points, r∈[{r_ref[0]:.3f}, {r_ref[-1]:.3f}]")
        except Exception as e:
            print(f"  pybhpt failed: {e}")
            continue

        # Model prediction on pybhpt r-grid
        try:
            R_pred = predict_R(model, a, omega, u, v, lam, r_ref, physics_cfg)
            R_pred = np.asarray(R_pred, dtype=np.complex128).ravel()
        except Exception as e:
            print(f"  Model prediction failed: {e}")
            import traceback; traceback.print_exc()
            continue

        # Relative errors
        R_ref_abs = np.abs(R_ref)
        R_pred_abs = np.abs(R_pred)

        # Avoid division by zero near nodes
        mask = R_ref_abs > 1e-15
        rel_err = np.abs(R_pred_abs[mask] - R_ref_abs[mask]) / R_ref_abs[mask]
        mean_rel = np.mean(rel_err)
        max_rel = np.max(rel_err)

        # Phase error
        phase_ref = np.angle(R_ref[mask])
        phase_pred = np.angle(R_pred[mask])
        phase_diff = np.mod(phase_pred - phase_ref + np.pi, 2*np.pi) - np.pi
        mean_phase = np.mean(np.abs(phase_diff))

        print(f"  |R_ref| range: [{R_ref_abs.min():.4e}, {R_ref_abs.max():.4e}]")
        print(f"  |R_pred| range: [{R_pred_abs.min():.4e}, {R_pred_abs.max():.4e}]")
        print(f"  Mean relative error: {mean_rel:.4e}")
        print(f"  Max relative error:  {max_rel:.4e}")
        print(f"  Mean phase error:    {mean_phase:.4e} rad")

    print(f"\n{'='*80}")
    print("Done.")


if __name__ == "__main__":
    main()
