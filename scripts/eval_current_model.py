"""Evaluate current training model (PINN_MLP) against spectral benchmark."""
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/ljq/code/PINN/SolvingTeukolsky")

from model.pinn_mlp import PINN_MLP
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives, U_prefactor, prefactor_Q
from physical_ansatz.transform_y import (
    h_factor, horizon_regularity_slope, compose_reduced_shape_from_f,
    g_factor, h1_factor,
)
from physical_ansatz.mapping import r_plus, r_from_x
from physical_ansatz.teukolsky_coeffs import coeffs_x
from physical_ansatz.residual import AuxCache, get_ramp_and_p_from_cfg
from utils.amplitude import TeukRadAmplitudeInWithInterpolant
from utils.mode import KerrMode
from domain.atlas_builder import map_to_chart, load_atlas
from domain.patch_cover import load_patch_cover


def load_pinn_mlp(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PINN_MLP(
        hidden_dims=[128, 256, 256, 128],
        activation="silu",
        fourier_num_freqs=2,
        param_embed_dim=128,
        use_film=True,
        use_residual=True,
        local_coord_mode="chart_uv",
        u_center_local=0.5,
        v_center_local=0.5,
        u_half_range_local=0.12,
        v_half_range_local=0.12,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model


def compute_benchmark(a, omega, M=1.0, s=-2, l=2, m=2):
    mode = KerrMode(M=M, a=a, ell=l, m=m, s=s, omega=omega)
    spectral = TeukRadAmplitudeInWithInterpolant(mode, N_in=64, N_out=64, z_m=0.3)
    return spectral.profile, spectral


def evaluate_pinn(model, a_val, omega_val, atlas, M=1.0, s=-2, l=2, m=2, n_y=400, device="cuda"):
    """Evaluate PINN_MLP f(y) and convert to R(r) on a fine y grid."""
    y = torch.linspace(-0.999, 0.999, n_y, device=device, dtype=torch.float64)
    a_t = torch.tensor([a_val], device=device, dtype=torch.float64)
    omega_t = torch.tensor([omega_val], device=device, dtype=torch.float64)

    # Compute u,v from a,omega for chart_uv mode
    comp = atlas.components[0]
    omega_scale = atlas.meta.get("omega_scale", "linear")
    u_val, v_val = map_to_chart(comp, a_val, omega_val, omega_scale=omega_scale)
    u_t = torch.tensor([u_val], device=device, dtype=torch.float64)
    v_t = torch.tensor([v_val], device=device, dtype=torch.float64)

    with torch.no_grad():
        f_pred = model(a_t, omega_t, y, u=u_t, v=v_t)  # (1, n_y) complex -> f(y)
    f_pred = f_pred.squeeze(0).cpu().numpy()

    # Compute benchmark
    profile, spectral = compute_benchmark(a_val, omega_val, M=M, s=s, l=l, m=m)

    # Convert y -> x -> r
    y_np = y.cpu().numpy()
    x_np = (y_np + 1.0) / 2.0
    rp = float(r_plus(torch.tensor([a_val]), M).item())
    r_np = rp / x_np

    # Model to full R: f(y) -> S(x) -> R(r) = P(r) * h2 * S(x)
    # S(x) = g(x) * (h1(x) * f(y) + 1) + 1
    # Where g(x) = slope * (exp(x-1) - 1), slope = -A0/A1 at horizon
    # R = P * h2 * S

    # Compute lambda for slope
    from utils.mode import KerrMode as KM
    mode = KM(M=M, a=a_val, ell=l, m=m, s=s, omega=omega_val)
    lambda_val = mode.lambda_value

    a_batch = torch.tensor([a_val])
    omega_batch = torch.tensor([omega_val])
    lambda_batch = torch.tensor([lambda_val])

    slope = horizon_regularity_slope(
        a=a_batch, omega=omega_batch, lambda_=lambda_batch, m=m, M=M, s=s,
    )

    x_t = torch.from_numpy(x_np)
    h2 = h_factor(a_batch, omega_batch, m, M, s)

    # P(r) Leaver prefactor
    P, _, _ = Leaver_prefactors(torch.from_numpy(r_np), a_batch, omega_batch, m, M, s)
    P_np = P.squeeze(0).numpy() if P.ndim > 1 else P.numpy()
    h2_np = h2.squeeze(0).numpy() if h2.ndim > 0 else complex(h2.item().real, h2.item().imag) if hasattr(h2, 'item') else complex(h2)

    # S(x) = g(x)*(h1(x)*f(y) + 1) + 1
    g, _, _ = g_factor(x_t, slope)
    h1, _, _ = h1_factor(x_t)
    g_np = g.squeeze(0).numpy() if g.ndim > 1 else g.numpy()
    h1_np = h1.squeeze(0).numpy() if h1.ndim > 1 else h1.numpy()

    S_pred = g_np * (h1_np * f_pred + 1.0) + 1.0

    # Full R
    R_model = P_np * h2_np * S_pred

    # Benchmark R on same r grid
    R_bench = profile.R_of_r(r_np)

    # Relative error
    rel_err = np.abs(R_model - R_bench) / (np.abs(R_bench) + 1e-30)

    return {
        "r": r_np,
        "y": y_np,
        "f_pred": f_pred,
        "S_pred": S_pred,
        "R_model": R_model,
        "R_bench": R_bench,
        "rel_err": rel_err,
        "profile": profile,
        "spectral": spectral,
        "a": a_val,
        "omega": omega_val,
    }


def plot_comparison(results, save_path):
    r = results["r"]
    f_pred = results["f_pred"]
    S_pred = results["S_pred"]
    R_model = results["R_model"]
    R_bench = results["R_bench"]
    rel_err = results["rel_err"]
    a = results["a"]
    omega = results["omega"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # f(y) amplitude
    ax = axes[0, 0]
    ax.plot(results["y"], np.abs(f_pred), "b-", label="|f_pred|")
    ax.set_xlabel("y")
    ax.set_ylabel("|f(y)|")
    ax.set_title(f"f(y) amplitude (a={a:.3f}, ω={omega:.6f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # S(y) amplitude
    ax = axes[0, 1]
    ax.plot(results["y"], np.abs(S_pred), "b-", label="|S_pred|", alpha=0.7)
    ax.set_xlabel("y")
    ax.set_ylabel("|S(y)|")
    ax.set_title("S(y) amplitude (reconstructed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # |R| comparison
    ax = axes[0, 2]
    ax.semilogy(r, np.abs(R_model), "b-", label="|R_model|", alpha=0.7)
    ax.semilogy(r, np.abs(R_bench), "r--", label="|R_bench|", alpha=0.7)
    ax.set_xlabel("r/M")
    ax.set_ylabel("|R(r)|")
    ax.set_title("|R(r)| comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Re(R) comparison
    ax = axes[1, 0]
    ax.plot(r, np.real(R_model), "b-", label="Re(R_model)", alpha=0.7)
    ax.plot(r, np.real(R_bench), "r--", label="Re(R_bench)", alpha=0.7)
    ax.set_xlabel("r/M")
    ax.set_ylabel("Re(R)")
    ax.set_title("Re(R) comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Im(R) comparison
    ax = axes[1, 1]
    ax.plot(r, np.imag(R_model), "b-", label="Im(R_model)", alpha=0.7)
    ax.plot(r, np.imag(R_bench), "r--", label="Im(R_bench)", alpha=0.7)
    ax.set_xlabel("r/M")
    ax.set_ylabel("Im(R)")
    ax.set_title("Im(R) comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Relative error
    ax = axes[1, 2]
    ax.semilogy(r, rel_err, "k-")
    ax.set_xlabel("r/M")
    ax.set_ylabel("relative error")
    ax.set_title(f"Relative error (mean={np.mean(rel_err):.2e}, max={np.max(rel_err):.2e})")
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"PINN_MLP Evaluation: a={a:.3f}, ω={omega:.6f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ckpt_path = "/home/ljq/code/PINN/SolvingTeukolsky/outputs/atlas_multipatch_phase4_cheb/20260511_031308_patch_000_comp_0_u_0.500_v_0.513/checkpoints/step_013200.pt"
    print(f"Loading: {ckpt_path}")
    model = load_pinn_mlp(ckpt_path, device)

    # Load atlas for u,v computation
    atlas = load_atlas("outputs/domain/atlas_l2_m2.json")

    # Test parameters: center of patch 0
    test_points = [
        (0.5, 0.5128205),     # patch 0 center
        (0.5, 0.3),
        (0.5, 0.8),
    ]

    for a, omega in test_points:
        print(f"\n{'='*60}")
        print(f"Evaluating a={a:.4f}, ω={omega:.6f}")
        try:
            results = evaluate_pinn(model, a, omega, atlas, device=device)
            max_err = float(np.max(results["rel_err"]))
            mean_err = float(np.mean(results["rel_err"]))
            print(f"  Max relative error: {max_err:.4e}")
            print(f"  Mean relative error: {mean_err:.4e}")

            save_path = f"/home/ljq/code/PINN/SolvingTeukolsky/outputs/pinn_eval_a{a:.3f}_w{omega:.6f}.png"
            plot_comparison(results, save_path)
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
