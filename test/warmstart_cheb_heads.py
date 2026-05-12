"""
Warm-start ChebDeepONet output heads via spectral coefficients from GSN.

Strategy:
  1. Pick K (a,ω) points around patch center
  2. For each point, use GSN to get true f(y), project to Chebyshev coeffs c_target
  3. Decompose c_target into base/a/omega/nl components via Taylor structure
  4. Run (random) branch net encoder to get h_i for each point
  5. For each head k, solve: W_k @ h_i + b_k ≈ c_k_target_i  via least squares
  6. Replace head weights with optimized values
"""

import json, sys, subprocess
import numpy as np
import torch

sys.path.insert(0, "/home/ljq/code/PINN/SolvingTeukolsky")

from model.cheb_deeponet import ChebDeepONet, clenshaw_evaluate


# ===========================================================================
# Step 1: Get true f(y) from GSN
# ===========================================================================

def get_gsn_f_of_y(a, omega, N_y=200):
    """Call GSN, compute R_in(r), strip prefactors to get f(y)."""
    code = f"""
using GeneralizedSasakiNakamura
s = -2; l = 2; m = 2; a = {a}; omega = {omega}
R = Teukolsky_radial(s, l, m, a, omega, IN, -60.0, 40.0; method="Riccati")
rp = 1.0 + sqrt(1 - a^2)
r_grid = exp.(range(log(rp + 1e-4), log(100.0), length={N_y}))
for r in r_grid
    Rv = R.Teukolsky_solution(r)[1]
    println("$r $(real(Rv)) $(imag(Rv))")
end
println("LAMBDA $(real(R.mode.lambda)) $(imag(R.mode.lambda))")
println("B_INC $(real(R.incidence_amplitude)) $(imag(R.incidence_amplitude))")
println("B_REF $(real(R.reflection_amplitude)) $(imag(R.reflection_amplitude))")
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
    lam = None
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "LAMBDA":
            lam = complex(float(parts[1]), float(parts[2]))
        elif parts[0] not in ("B_INC", "B_REF"):
            r_list.append(float(parts[0]))
            R_re.append(float(parts[1]))
            R_im.append(float(parts[2]))

    r_grid = np.array(r_list)
    R_in = np.array(R_re) + 1j * np.array(R_im)
    return r_grid, R_in, lam


def strip_prefactors_to_f(r_grid, R_in, a, omega):
    """Invert R_in = P * h2 * S  →  f(y)."""
    from physical_ansatz.transform_y import (
        compose_reduced_shape_from_f, horizon_regularity_slope,
        h_factor, h1_factor,
    )
    from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives
    from physical_ansatz.mapping import r_plus

    M = 1.0; m = 2; s = -2

    a_t = torch.tensor([a], dtype=torch.float64)
    omega_t = torch.tensor([omega], dtype=torch.float64)
    r_t = torch.tensor(r_grid, dtype=torch.float64)
    rp_val = float(r_plus(a_t, M))
    x_grid = rp_val / r_t
    y_grid = 2.0 * x_grid - 1.0

    # S = R_in / (P * h2)
    h2 = h_factor(a_t, omega_t, m=m, M=M, s=s)
    h2_c = complex(h2.item())
    rp_t, rm_t, _, _, _ = build_prefactor_primitives(r_t, a_t, M=M, need_rs=False)
    P, _, _ = Leaver_prefactors(r_t, a_t, omega_t, m=m, M=M, s=s, rp=rp_t, rm=rm_t)
    P_np = P.detach().cpu().numpy().ravel()
    S_gsn = R_in / (P_np * h2_c)

    # S = slope*h1^2*f + slope*h1 + 1
    # f = (S - 1 - slope*h1) / (slope*h1^2)
    lam_t = torch.tensor(lam, dtype=torch.complex128).unsqueeze(0)
    slope = horizon_regularity_slope(a=a_t, omega=omega_t, lambda_=lam_t, m=m, M=M, s=s).squeeze(0)
    slope_c = complex(slope.item())
    h1_vals, _, _ = h1_factor(x_grid)
    h1_np = h1_vals.detach().cpu().numpy().ravel()
    f_gsn = (S_gsn - 1.0 - slope_c * h1_np) / (slope_c * h1_np**2 + 1e-300)

    return y_grid.numpy(), f_gsn


# ===========================================================================
# Step 2: Project f(y) onto Chebyshev coefficients
# ===========================================================================

def chebyshev_project(f_vals, y_vals, N):
    """Project f(y) onto first N Chebyshev polynomials using Clenshaw-Curtis quadrature.

    f_vals: (Ny,) complex
    y_vals: (Ny,) real in [-1, 1]
    Returns: (N,) complex coefficients
    """
    Ny = len(y_vals)
    # Clenshaw-Curtis: use Chebyshev nodes of second kind for quadrature
    # But our y_vals are arbitrary (from GSN r-grid). Use least-squares instead.
    # Build Vandermonde matrix: T_n(y_j)
    T = np.zeros((Ny, N), dtype=np.float64)
    for n in range(N):
        T[:, n] = np.cos(n * np.arccos(y_vals))
    # Least squares: c = (T^T T)^{-1} T^T f
    c_re, _, _, _ = np.linalg.lstsq(T, np.real(f_vals), rcond=None)
    c_im, _, _, _ = np.linalg.lstsq(T, np.imag(f_vals), rcond=None)
    return c_re + 1j * c_im


# ===========================================================================
# Step 3+4: Collect target coefficients + encode, then solve for head weights
# ===========================================================================

def decompose_taylor_coeffs(alpha_xi_list, c_targets):
    """Given (alpha_i, xi_i) and c_target_i for K points, solve for c_base, c_a, c_omega, c_nl.

    c_total = c_base + alpha*c_a + xi*c_omega + (alpha^2+xi^2)*c_nl

    Returns: c_base, c_a, c_omega, c_nl each (N,) complex
    """
    K = len(alpha_xi_list)
    N = len(c_targets[0])
    A = np.zeros((K, 4), dtype=np.float64)
    for i, (alpha, xi) in enumerate(alpha_xi_list):
        A[i, 0] = 1.0
        A[i, 1] = alpha
        A[i, 2] = xi
        A[i, 3] = alpha**2 + xi**2

    components = []
    for comp_idx in range(4):
        b_re = np.array([np.real(c_targets[i]) for i in range(K)])
        b_im = np.array([np.imag(c_targets[i]) for i in range(K)])
        sol_re, _, _, _ = np.linalg.lstsq(A, b_re, rcond=None)
        sol_im, _, _, _ = np.linalg.lstsq(A, b_im, rcond=None)
        # sol_re[j] is the contribution of component j to all N coefficients... wait no.
        # Actually A is K×4, b_re is K×N  →  we need to solve for each n separately
        ...

    # Wait, the above is wrong. A is K×4, but c_targets are K×N.
    # We need to solve N independent 4-unknown problems.
    c_base = np.zeros(N, dtype=np.complex128)
    c_a = np.zeros(N, dtype=np.complex128)
    c_omega = np.zeros(N, dtype=np.complex128)
    c_nl = np.zeros(N, dtype=np.complex128)

    for n in range(N):
        b_n_re = np.array([np.real(c_targets[i][n]) for i in range(K)])
        b_n_im = np.array([np.imag(c_targets[i][n]) for i in range(K)])
        sol_re, _, _, _ = np.linalg.lstsq(A, b_n_re, rcond=None)
        sol_im, _, _, _ = np.linalg.lstsq(A, b_n_im, rcond=None)
        c_base[n] = sol_re[0] + 1j * sol_im[0]
        c_a[n] = sol_re[1] + 1j * sol_im[1]
        c_omega[n] = sol_re[2] + 1j * sol_im[2]
        c_nl[n] = sol_re[3] + 1j * sol_im[3]

    return c_base, c_a, c_omega, c_nl


def main():
    N = 48
    patch_u_center = 0.5
    patch_v_center = 0.5128205128205128
    patch_u_half = 0.12
    patch_v_half = 0.12

    # ---- Pick K points for warm-start ----
    # Center + offsets in u,v (atlas coords for patch 0)
    # alpha = (u - uc)/u_half, xi = (v - vc)/v_half
    uv_points = [
        (patch_u_center, patch_v_center, "center"),
        (patch_u_center + 0.06, patch_v_center, "+alpha"),
        (patch_u_center - 0.06, patch_v_center, "-alpha"),
        (patch_u_center, patch_v_center + 0.06, "+xi"),
        (patch_u_center, patch_v_center - 0.06, "-xi"),
        (patch_u_center + 0.04, patch_v_center + 0.04, "+alpha+xi"),
        (patch_u_center - 0.04, patch_v_center + 0.04, "-alpha+xi"),
        (patch_u_center + 0.04, patch_v_center - 0.04, "+alpha-xi"),
    ]

    # Convert (u,v) to (a,omega) using patch mapping
    a_c = 0.5; a_r = 0.499
    w_c = 5.128253846153846; w_r = 0.8950253846153846

    print("=== Step 1: Collecting GSN targets ===")
    alpha_xi_list = []
    c_targets = []
    y_ref = None

    for u, v, label in uv_points:
        a = a_c + (2*u - 1) * a_r
        omega = w_c + (2*v - 1) * w_r
        alpha = (u - patch_u_center) / patch_u_half
        xi = (v - patch_v_center) / patch_v_half

        print(f"  {label}: a={a:.4f}, ω={omega:.6f}  (α={alpha:.3f}, ξ={xi:.3f})")
        try:
            r_grid, R_in, lam = get_gsn_f_of_y(a, omega)
            y_grid, f_gsn = strip_prefactors_to_f(r_grid, R_in, a, omega)
            if y_ref is None:
                y_ref = y_grid
            c = chebyshev_project(f_gsn, y_grid, N)
            c_targets.append(c)
            alpha_xi_list.append((alpha, xi))
            print(f"    |f|∈[{np.abs(f_gsn).min():.1e}, {np.abs(f_gsn).max():.1e}], "
                  f"|c|∈[{np.abs(c).min():.2e}, {np.abs(c).max():.2e}]")
        except Exception as e:
            print(f"    FAILED: {e}")
            continue

    K = len(c_targets)
    print(f"  Collected {K} valid points")

    # ---- Step 2: Decompose into base/a/omega/nl ----
    print("\n=== Step 2: Taylor decomposition ===")
    c_base, c_a, c_omega, c_nl = decompose_taylor_coeffs(alpha_xi_list, c_targets)

    # Check reconstruction error
    max_err = 0.0
    for i in range(K):
        alpha, xi = alpha_xi_list[i]
        c_recon = c_base + alpha*c_a + xi*c_omega + (alpha**2+xi**2)*c_nl
        err = np.max(np.abs(c_recon - c_targets[i]))
        max_err = max(max_err, err)
    print(f"  Max decomposition error: {max_err:.2e}")

    # ---- Step 3: Build model, run encoder, solve for head weights ----
    print("\n=== Step 3: Solving for head weights ===")
    model = ChebDeepONet(
        N=N, hidden_dims=[128, 256, 256, 128],
        activation="silu", param_embed_dim=128, use_taylor=True,
    ).to(torch.float64)
    model.eval()

    # Encode all points
    a_t = torch.tensor([a_c + (2*u - 1)*a_r for u, v, _ in uv_points[:K]], dtype=torch.float64)
    omega_t = torch.tensor([w_c + (2*v - 1)*w_r for u, v, _ in uv_points[:K]], dtype=torch.float64)
    u_t = torch.tensor([u for u, v, _ in uv_points[:K]], dtype=torch.float64)
    v_t = torch.tensor([v for u, v, _ in uv_points[:K]], dtype=torch.float64)

    with torch.no_grad():
        feats, _, _ = model._build_param_features(a_t, omega_t, u_t, v_t)
        h = model.param_encoder(feats)  # (K, enc_dim)
        h_np = h.detach().cpu().numpy()

    # Target component coefficients
    target_components = [
        np.column_stack([np.real(c_base), np.imag(c_base)]),  # (N, 2)
        np.column_stack([np.real(c_a), np.imag(c_a)]),
        np.column_stack([np.real(c_omega), np.imag(c_omega)]),
        np.column_stack([np.real(c_nl), np.imag(c_nl)]),
    ]

    # For each head, solve: W @ h_i + b = target_i
    # Augment h_i with 1 for bias
    h_aug = np.column_stack([h_np, np.ones(K)])  # (K, enc_dim+1)

    for head_idx in range(4):
        head = getattr(model, f"coeff_head_{head_idx}")
        # Target: (N, 2N_out) where 2N_out = real + imag interleaved
        # Actually head outputs (2*N,) = [real_0,...,real_{N-1}, imag_0,...,imag_{N-1}]
        target_comp = target_components[head_idx]  # (N, 2): [real, imag]
        target_flat = target_comp.T.ravel()  # (2*N,)

        # We need K targets for each of the 2*N outputs
        # But we only have ONE target per output dimension (the Taylor component).
        # The Taylor component is a SINGLE target c_base (N complex = 2N real).
        # Each head outputs 2N values that should equal this single target
        # REGARDLESS of which point we're at (since it's the component, not the full coeff).
        #
        # So target for point i is always the same: c_component.
        # We have K equations, each: W @ h_i + b = c_component
        # This means we're fitting a constant function of h → trivial: W=0, b=c_component

        if head_idx == 0:
            # For base head: c_base is the target at ALL points
            # W should be ~0, bias should be ~c_base
            # But we can also just solve it properly with K repeated targets
            target_matrix = np.tile(target_flat, (K, 1))  # (K, 2*N)
        else:
            # For a/omega/nl heads: the TARGET is the COMPONENT, not the full coefficient
            # At the decomposition stage, c_a is the coefficient MULTIPLIER for alpha.
            # So target for head k is always c_k (constant across points).
            target_matrix = np.tile(target_flat, (K, 1))

        # Solve: W_aug @ h_aug = target_matrix
        # h_aug: (K, D+1), target_matrix: (K, 2N)
        W_aug, residuals, rank, s = np.linalg.lstsq(h_aug, target_matrix, rcond=None)
        # W_aug: (D+1, 2N)
        W_new = W_aug[:-1, :].T  # (2N, D)
        b_new = W_aug[-1, :]     # (2N,)

        # Update head weights
        with torch.no_grad():
            head.weight.copy_(torch.tensor(W_new, dtype=torch.float64))
            head.bias.copy_(torch.tensor(b_new, dtype=torch.float64))

        err = np.max(np.abs(h_np @ W_new.T + b_new - target_flat))
        print(f"  Head {head_idx}: max fit error = {err:.2e}")

    # ---- Step 4: Verify ----
    print("\n=== Step 4: Verification ===")
    y_test = torch.tensor(y_ref, dtype=torch.float64)
    with torch.no_grad():
        f_warm = model(a_t[:1], omega_t[:1], y_test, u=u_t[:1], v=v_t[:1])
        f_warm_np = f_warm.squeeze(0).detach().cpu().numpy()

    # Compare with GSN target at center point
    _, f_gsn_center = strip_prefactors_to_f(r_grid, R_in, a_c,
        w_c + (2*patch_v_center - 1)*w_r)
    print(f"  |f_warm| range: [{np.abs(f_warm_np).min():.2e}, {np.abs(f_warm_np).max():.2e}]")
    print(f"  |f_gsn| range:  [{np.abs(f_gsn_center).min():.2e}, {np.abs(f_gsn_center).max():.2e}]")

    rel_err = np.abs(f_warm_np - f_gsn_center) / (np.abs(f_gsn_center) + 1e-15)
    print(f"  Max relative error: {rel_err.max():.4e}")
    print(f"  Mean relative error: {rel_err.mean():.4e}")

    # Save
    out = {
        "warmstart_state_dict": {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()},
        "head_weights": {f"head_{i}": getattr(model, f"coeff_head_{i}").weight.detach().cpu().numpy().tolist()
                         for i in range(4)},
        "head_biases": {f"head_{i}": getattr(model, f"coeff_head_{i}").bias.detach().cpu().numpy().tolist()
                        for i in range(4)},
        "target_coeffs": {
            "base": [c_base.real.tolist(), c_base.imag.tolist()],
            "a": [c_a.real.tolist(), c_a.imag.tolist()],
            "omega": [c_omega.real.tolist(), c_omega.imag.tolist()],
            "nl": [c_nl.real.tolist(), c_nl.imag.tolist()],
        },
    }
    torch.save({"model_state_dict": model.state_dict()}, "outputs/cheb_warmstart_heads.pt")
    print("\nSaved: outputs/cheb_warmstart_heads.pt")


if __name__ == "__main__":
    main()
