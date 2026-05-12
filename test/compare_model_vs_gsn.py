"""
Compare PINN_MLP and ChebDeepONet against GSN benchmark.
"""
import json, sys, subprocess, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/ljq/code/PINN/SolvingTeukolsky")

from model.pinn_mlp import PINN_MLP
from model.cheb_deeponet import ChebDeepONet
from physical_ansatz.transform_y import compose_reduced_shape_from_f, horizon_regularity_slope, h_factor, h1_factor
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives
from physical_ansatz.mapping import r_plus
from physical_ansatz.residual import get_lambda_from_cfg, AuxCache


def get_gsn_benchmark(a, omega):
    """Call GSN.jl to get R_in(r)."""
    code = f"""
using GeneralizedSasakiNakamura
s = -2; l = 2; m = 2; a = {a}; omega = {omega}
R = Teukolsky_radial(s, l, m, a, omega, IN, -60.0, 40.0; method="Riccati")
rp = 1.0 + sqrt(1 - a^2)
r_grid = exp.(range(log(rp + 1e-4), log(100.0), length=500))
Rvals = [R.Teukolsky_solution(r)[1] for r in r_grid]
for (r, Rv) in zip(r_grid, Rvals)
    println("$r $(real(Rv)) $(imag(Rv))")
end
println("B_INC $(real(R.incidence_amplitude)) $(imag(R.incidence_amplitude))")
println("B_REF $(real(R.reflection_amplitude)) $(imag(R.reflection_amplitude))")
println("LAMBDA $(real(R.mode.lambda)) $(imag(R.mode.lambda))")
"""
    proc = subprocess.run(
        ["julia", "--project=.", "-e", code],
        cwd="/home/ljq/code/GSN/GeneralizedSasakiNakamura.jl",
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"GSN failed:\n{proc.stderr}")
    lines = proc.stdout.strip().split("\n")
    r_list, R_re, R_im = [], [], []
    B_inc = B_ref = lam = None
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "B_INC":
            B_inc = complex(float(parts[1]), float(parts[2]))
        elif parts[0] == "B_REF":
            B_ref = complex(float(parts[1]), float(parts[2]))
        elif parts[0] == "LAMBDA":
            lam = complex(float(parts[1]), float(parts[2]))
        else:
            r_list.append(float(parts[0]))
            R_re.append(float(parts[1]))
            R_im.append(float(parts[2]))
    r_grid = np.array(r_list)
    R_in = np.array(R_re) + 1j * np.array(R_im)
    return r_grid, R_in, B_inc, B_ref, lam


def predict_R_and_S(model, a, omega, u, v, lam, r_grid, physics_cfg):
    M = 1.0; m = 2; s = -2
    a_t = torch.tensor([a], dtype=torch.float64)
    omega_t = torch.tensor([omega], dtype=torch.float64)
    u_t = torch.tensor([u], dtype=torch.float64)
    v_t = torch.tensor([v], dtype=torch.float64)
    lam_t = lam.clone().detach().to(torch.float64)

    r_grid_t = torch.tensor(r_grid, dtype=torch.float64)
    rp_val = float(r_plus(a_t, M))
    x_grid = rp_val / r_grid_t
    y_grid = 2.0 * x_grid - 1.0

    with torch.no_grad():
        f_pred = model(a_t, omega_t, y_grid, u=u_t, v=v_t)
        if f_pred.dim() == 3:
            f_pred = f_pred.squeeze(0)
        elif f_pred.dim() == 2 and f_pred.shape[0] == 1:
            f_pred = f_pred.squeeze(0)

        slope = horizon_regularity_slope(
            a=a_t, omega=omega_t, lambda_=lam_t.unsqueeze(0), m=m, M=M, s=s
        ).squeeze(0)

        S_pred = compose_reduced_shape_from_f(f=f_pred, y=y_grid, slope=slope)

        h2 = h_factor(a_t, omega_t, m=m, M=M, s=s)
        rp_t, rm_t, _, _, _ = build_prefactor_primitives(r_grid_t, a_t, M=M, need_rs=False)
        P, _, _ = Leaver_prefactors(r_grid_t, a_t, omega_t, m=m, M=M, s=s, rp=rp_t, rm=rm_t)
        R_pred = P * h2 * S_pred

    return (
        R_pred.detach().cpu().numpy().ravel(),
        S_pred.detach().cpu().numpy().ravel(),
        y_grid.detach().cpu().numpy().ravel(),
        f_pred.detach().cpu().numpy().ravel(),
    )


def main():
    a = 0.5; omega = 5.128253846153846; u = 0.5; v = 0.5128205128205128
    M = 1.0; m = 2; s = -2

    # GSN benchmark
    print("Computing GSN benchmark...")
    r_grid, R_gsn, B_inc, B_ref, lam = get_gsn_benchmark(a, omega)
    print(f"  r range: [{r_grid[0]:.3f}, {r_grid[-1]:.1f}], N={len(r_grid)}")
    print(f"  |R_gsn| range: [{np.abs(R_gsn).min():.4e}, {np.abs(R_gsn).max():.4e}]")
    print(f"  B_inc = {B_inc:.6f}, |B_inc| = {abs(B_inc):.4f}")
    print(f"  B_ref = {B_ref:.6e}, |B_ref| = {abs(B_ref):.4e}")
    print(f"  lambda = {lam}")

    # GSN-derived S(y): S = R / (P * h2)
    a_t = torch.tensor([a], dtype=torch.float64)
    omega_t = torch.tensor([omega], dtype=torch.float64)
    lam_t = torch.tensor(lam, dtype=torch.complex128)
    r_grid_t = torch.tensor(r_grid, dtype=torch.float64)
    rp_val = float(r_plus(a_t, M))
    x_grid = rp_val / r_grid_t
    y_grid = 2.0 * x_grid - 1.0
    y_grid_np = y_grid.numpy()

    h2 = h_factor(a_t, omega_t, m=m, M=M, s=s)
    rp_t, rm_t, _, _, _ = build_prefactor_primitives(r_grid_t, a_t, M=M, need_rs=False)
    P, _, _ = Leaver_prefactors(r_grid_t, a_t, omega_t, m=m, M=M, s=s, rp=rp_t, rm=rm_t)
    P_np = P.detach().cpu().numpy().ravel()
    h2_c = complex(h2.detach().cpu().item())
    S_gsn = R_gsn / (P_np * h2_c)

    # Also compute S via compose_reduced_shape_from_f in reverse
    # S = g * (h1 * f + 1) + 1, so f = (S - 1 - g) / (g * h1)
    # g(x) = slope * h1(x)
    # But we don't have slope directly... Let me compute f_gsn = (S_gsn - 1) / (slope * h1^2) - 1/h1
    lam_for_model = lam_t.unsqueeze(0)
    slope = horizon_regularity_slope(a=a_t, omega=omega_t, lambda_=lam_for_model, m=m, M=M, s=s).squeeze(0)
    slope_c = complex(slope.detach().cpu().item())
    h1_vals, _, _ = h1_factor(x_grid)
    h1_np = h1_vals.detach().cpu().numpy().ravel()
    # S = slope*h1^2*f + slope*h1 + 1
    # f = (S - 1 - slope*h1) / (slope*h1^2)
    f_gsn = (S_gsn - 1.0 - slope_c * h1_np) / (slope_c * h1_np**2 + 1e-300)

    # Build PINN_MLP (same config as v3 training)
    print("\nBuilding PINN_MLP (240K)...")
    model_mlp = PINN_MLP(
        hidden_dims=[128, 128, 128, 128],
        activation="silu",
        fourier_num_freqs=2,
        fourier_scale=1.0,
        param_embed_dim=64,
        use_film=True,
        use_residual=True,
    ).to(torch.float64)
    model_mlp.eval()
    n_mlp = sum(p.numel() for p in model_mlp.parameters())
    print(f"  Params: {n_mlp:,}")

    # Build ChebDeepONet
    print("Building ChebDeepONet (N=48, 200K)...")
    model_cheb = ChebDeepONet(
        N=48,
        hidden_dims=[128, 256, 256, 128],
        activation="silu",
        param_embed_dim=128,
        use_taylor=True,
    ).to(torch.float64)
    model_cheb.eval()
    n_cheb = sum(p.numel() for p in model_cheb.parameters())
    print(f"  Params: {n_cheb:,}")

    # Compute model predictions
    print("Computing PINN_MLP predictions...")
    R_mlp, S_mlp, y_mlp, f_mlp = predict_R_and_S(model_mlp, a, omega, u, v, lam_for_model, r_grid, {})
    print("Computing ChebDeepONet predictions...")
    R_cheb, S_cheb, y_cheb, f_cheb = predict_R_and_S(model_cheb, a, omega, u, v, lam_for_model, r_grid, {})

    # ============================================================
    # Plot: 3 rows × 3 cols
    # Row 0: |R(r)| — MLP, Cheb, GSN asymptotics
    # Row 1: |S(y)| — MLP, Cheb, GSN real/imag
    # Row 2: |f(y)| — MLP, Cheb, GSN
    # ============================================================
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    model_data = [
        ("PINN_MLP (untrained)", R_mlp, S_mlp, f_mlp, n_mlp),
        ("ChebDeepONet (untrained)", R_cheb, S_cheb, f_cheb, n_cheb),
    ]

    for col, (label, R_pred, S_pred, f_pred, n_param) in enumerate(model_data[:2]):
        # Row 0: |R(r)|
        ax = axes[0, col]
        ax.plot(r_grid, np.abs(R_gsn), "k-", lw=0.8, label="GSN")
        ax.plot(r_grid, np.abs(R_pred), "r--", lw=1.2, label=label)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("r"); ax.set_ylabel("|R_in(r)|")
        ax.set_title(f"|R_in(r)| — {label}")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # Row 1: |S(y)|
        ax = axes[1, col]
        ax.plot(y_grid_np, np.abs(S_gsn), "k-", lw=0.8, label="GSN S(y)")
        ax.plot(y_grid_np, np.abs(S_pred), "r--", lw=1.2, label=label)
        ax.set_xlabel("y"); ax.set_ylabel("|S(y)|")
        ax.set_title(f"|S(y)| — {label}")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # Row 2: |f(y)|
        ax = axes[2, col]
        ax.plot(y_grid_np, np.abs(f_gsn), "k-", lw=0.8, label="GSN f(y)")
        ax.plot(y_grid_np, np.abs(f_pred), "r--", lw=1.2, label=label)
        ax.set_xlabel("y"); ax.set_ylabel("|f(y)|")
        ax.set_title(f"|f(y)| — {label}")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Column 2: GSN reference
    # Row 0, Col 2: Asymptotic components
    ax = axes[0, 2]
    ax.plot(r_grid, np.abs(R_gsn), "k-", lw=1.0, label="GSN R_in")
    ax.plot(r_grid, np.abs(B_inc) / r_grid, "b:", lw=0.8, label=r"$|B_{inc}|/r$")
    ax.plot(r_grid, np.abs(B_ref) * r_grid**3, "g:", lw=0.8, label=r"$|B_{ref}|\cdot r^3$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("r")
    ax.set_title("GSN asymptotic components")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Row 1, Col 2: S(y) real & imag
    ax = axes[1, 2]
    ax.plot(y_grid_np, np.real(S_gsn), "b-", lw=0.7, label="Re(S)")
    ax.plot(y_grid_np, np.imag(S_gsn), "r-", lw=0.7, label="Im(S)")
    ax.set_xlabel("y"); ax.set_ylabel("S(y)")
    ax.set_title("GSN S(y) — real & imag")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Row 2, Col 2: f(y) real & imag
    ax = axes[2, 2]
    ax.plot(y_grid_np, np.real(f_gsn), "b-", lw=0.7, label="Re(f)")
    ax.plot(y_grid_np, np.imag(f_gsn), "r-", lw=0.7, label="Im(f)")
    ax.set_xlabel("y"); ax.set_ylabel("f(y)")
    ax.set_title("GSN f(y) — real & imag")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle(
        f"Model vs GSN: a={a}, ω={omega:.4f}  "
        f"(B_inc={abs(B_inc):.2f}, B_ref={abs(B_ref):.2e})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out_path = "outputs/model_vs_gsn_comparison.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")

    # Quick summary
    print(f"\n=== Summary ===")
    print(f"GSN:  |R|∈[{np.abs(R_gsn).min():.2e},{np.abs(R_gsn).max():.2f}]  |S|∈[{np.abs(S_gsn).min():.2e},{np.abs(S_gsn).max():.2f}]  |f|∈[{np.abs(f_gsn).min():.2e},{np.abs(f_gsn).max():.2e}]")
    print(f"MLP:  |R|∈[{np.abs(R_mlp).min():.2e},{np.abs(R_mlp).max():.2e}]  |S|∈[{np.abs(S_mlp).min():.2f},{np.abs(S_mlp).max():.2f}]  |f|∈[{np.abs(f_mlp).min():.2e},{np.abs(f_mlp).max():.2f}]")
    print(f"Cheb: |R|∈[{np.abs(R_cheb).min():.2e},{np.abs(R_cheb).max():.2e}]  |S|∈[{np.abs(S_cheb).min():.2f},{np.abs(S_cheb).max():.2f}]  |f|∈[{np.abs(f_cheb).min():.2e},{np.abs(f_cheb).max():.2f}]")


if __name__ == "__main__":
    main()
