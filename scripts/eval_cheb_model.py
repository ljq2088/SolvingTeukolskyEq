"""Evaluate ChebDeepONet model against spectral benchmark."""
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/ljq/code/PINN/SolvingTeukolsky")

from model.cheb_deeponet import ChebDeepONet
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives
from physical_ansatz.transform_y import h_factor, horizon_regularity_slope, compose_reduced_shape_from_f
from physical_ansatz.mapping import r_plus, r_from_x
from physical_ansatz.teukolsky_coeffs import coeffs_x
from physical_ansatz.residual import AuxCache, get_ramp_and_p_from_cfg, get_lambda_from_cfg
from utils.amplitude import TeukRadAmplitudeInWithInterpolant
from utils.mode import KerrMode


def load_model(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ChebDeepONet(
        N=48,
        hidden_dims=[128, 256, 256, 128],
        activation="silu",
        param_embed_dim=128,
        local_coord_mode="chart_uv",
        u_center_local=0.5,
        v_center_local=0.5,
        u_half_range_local=0.12,
        v_half_range_local=0.12,
        use_taylor=True,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def compute_benchmark(a, omega, M=1.0, s=-2, l=2, m=2):
    """Compute spectral benchmark R_in(r) profile."""
    mode = KerrMode(M=M, a=a, ell=l, m=m, s=s, omega=omega)
    spectral = TeukRadAmplitudeInWithInterpolant(mode, N_in=64, N_out=64, z_m=0.3)
    profile = spectral.profile
    return profile, spectral


def evaluate(model, cfg, a_val, omega_val, M=1.0, s=-2, l=2, m=2, n_y=400, device="cuda"):
    """Evaluate model S(y) and convert to R(r) on a fine y grid."""
    # y grid: [-1, 1]
    y = torch.linspace(-0.999, 0.999, n_y, device=device, dtype=torch.float64)
    a_t = torch.tensor([a_val], device=device, dtype=torch.float64)
    omega_t = torch.tensor([omega_val], device=device, dtype=torch.float64)

    with torch.no_grad():
        S_pred = model(a_t, omega_t, y)  # (1, n_y) complex
    S_pred = S_pred.squeeze(0).cpu()

    # Compute benchmark
    profile, spectral = compute_benchmark(a_val, omega_val, M=M, s=s, l=l, m=m)

    # Convert y -> x -> r
    y_np = y.cpu().numpy()
    x_np = (y_np + 1.0) / 2.0
    rp = float(r_plus(torch.tensor([a_val]), M).item())
    r_np = rp / x_np

    # Compute benchmark R on same r grid
    R_bench = profile.R_of_r(r_np)  # complex numpy array

    # Compute model R: R = P * h2 * S
    h2 = float(h_factor(torch.tensor([a_val]), torch.tensor([omega_val]), m, M, s).real.item()) + \
         1j * float(h_factor(torch.tensor([a_val]), torch.tensor([omega_val]), m, M, s).imag.item())

    r_t = torch.from_numpy(r_np)
    P, _, _ = Leaver_prefactors(r_t, torch.tensor([a_val]), torch.tensor([omega_val]), m, M, s)
    P_np = P.squeeze(0).numpy() if P.ndim > 1 else P.numpy()
    R_model = P_np * h2 * S_pred.numpy()

    # Relative error
    rel_err = np.abs(R_model - R_bench) / (np.abs(R_bench) + 1e-30)

    return {
        "r": r_np,
        "y": y_np,
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
    """Plot S(y), R(r) amplitude, phase, and relative error."""
    r = results["r"]
    S_pred = results["S_pred"]
    R_model = results["R_model"]
    R_bench = results["R_bench"]
    rel_err = results["rel_err"]
    a = results["a"]
    omega = results["omega"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: S(y) - the learned function
    ax = axes[0, 0]
    ax.plot(results["y"], np.abs(S_pred), "b-", label="|S_pred|")
    ax.set_xlabel("y")
    ax.set_ylabel("|S(y)|")
    ax.set_title(f"S(y) amplitude (a={a:.3f}, ω={omega:.6f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(results["y"], np.real(S_pred), "b-", label="Re(S)", alpha=0.7)
    ax.plot(results["y"], np.imag(S_pred), "r-", label="Im(S)", alpha=0.7)
    ax.set_xlabel("y")
    ax.set_title("S(y) real/imag")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 1, col 2: |R| comparison
    ax = axes[0, 2]
    ax.semilogy(r, np.abs(R_model), "b-", label="|R_model|", alpha=0.7)
    ax.semilogy(r, np.abs(R_bench), "r--", label="|R_bench|", alpha=0.7)
    ax.set_xlabel("r/M")
    ax.set_ylabel("|R(r)|")
    ax.set_title("|R(r)| comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2: Re(R), Im(R) comparison
    ax = axes[1, 0]
    ax.plot(r, np.real(R_model), "b-", label="Re(R_model)", alpha=0.7)
    ax.plot(r, np.real(R_bench), "r--", label="Re(R_bench)", alpha=0.7)
    ax.set_xlabel("r/M")
    ax.set_ylabel("Re(R)")
    ax.set_title("Re(R) comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(r, np.imag(R_model), "b-", label="Im(R_model)", alpha=0.7)
    ax.plot(r, np.imag(R_bench), "r--", label="Im(R_bench)", alpha=0.7)
    ax.set_xlabel("r/M")
    ax.set_ylabel("Im(R)")
    ax.set_title("Im(R) comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2, col 2: relative error
    ax = axes[1, 2]
    ax.semilogy(r, rel_err, "k-")
    ax.set_xlabel("r/M")
    ax.set_ylabel("relative error |R_pred - R_bench| / |R_bench|")
    ax.set_title(f"Relative error (mean={np.mean(rel_err):.2e})")
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"ChebDeepONet Evaluation: a={a:.3f}, ω={omega:.6f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ckpt_path = "/home/ljq/code/PINN/SolvingTeukolsky/outputs/atlas_multipatch_phase4_cheb/20260511_031308_patch_000_comp_0_u_0.500_v_0.513/checkpoints/step_013200.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    model = load_model(ckpt_path, device)

    # Test parameter points
    test_points = [
        (0.5, 0.51),          # center of patch 0
        (0.5, 0.1),
        (0.5, 1.0),
        (0.1, 0.51),
        (0.9, 0.51),
    ]

    cfg = {"problem": {"M": 1.0, "s": -2, "l": 2, "m": 2}}

    for a, omega in test_points:
        print(f"\n{'='*60}")
        print(f"Evaluating a={a:.3f}, ω={omega:.6f}")
        try:
            results = evaluate(model, cfg, a, omega, device=device)
            max_err = float(np.max(results["rel_err"]))
            mean_err = float(np.mean(results["rel_err"]))
            print(f"  Max relative error: {max_err:.4e}")
            print(f"  Mean relative error: {mean_err:.4e}")

            save_path = f"/home/ljq/code/PINN/SolvingTeukolsky/outputs/cheb_eval_a{a:.3f}_w{omega:.6f}.png"
            plot_comparison(results, save_path)
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
