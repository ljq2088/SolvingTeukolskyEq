"""
Debug script to reproduce the visualization issue where pybhpt ref curves don't show.
"""
import sys
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky/pybhpt")

import numpy as np
import matplotlib.pyplot as plt
from compute_solution import compute_pybhpt_solution

# Test parameters (from image title: u=0.500, v=1.000, patch_006)
# These map to specific (a, omega) values
a = 0.5  # example
omega = 0.3  # example

# Create r grid similar to trainer
r_min, r_max = 2.0, 80.0
r_grid = np.linspace(r_min, r_max, 100)

print(f"Testing pybhpt with a={a}, omega={omega}")
print(f"r_grid: {len(r_grid)} points, range [{r_grid.min():.2f}, {r_grid.max():.2f}]")

try:
    r_vals, R_ref = compute_pybhpt_solution(
        a=a, omega=omega, ell=2, m=2, r_grid=r_grid, timeout=10.0
    )

    print(f"\nSuccess! Computed {len(R_ref)} points")
    print(f"|R_ref| range: [{np.min(np.abs(R_ref)):.4e}, {np.max(np.abs(R_ref)):.4e}]")
    print(f"Re(R_ref) range: [{np.min(R_ref.real):.4e}, {np.max(R_ref.real):.4e}]")
    print(f"Im(R_ref) range: [{np.min(R_ref.imag):.4e}, {np.max(R_ref.imag):.4e}]")
    print(f"Has NaN: {np.any(np.isnan(R_ref))}")
    print(f"Has Inf: {np.any(np.isinf(R_ref))}")
    print(f"\nFirst 5 values:")
    for i in range(min(5, len(R_ref))):
        print(f"  r={r_vals[i]:.2f}: R={R_ref[i]:.4e}")

    # Create visualization similar to trainer
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    # Simulate pred values (just for comparison)
    R_pred = R_ref * (1.0 + 0.1 * np.random.randn(len(R_ref)))

    axes[0].plot(r_vals, np.real(R_pred), label="Pred Re(R)", lw=1.6)
    axes[0].plot(r_vals, np.real(R_ref), "--", label="ref Re(R)", lw=1.0, color='orange')
    axes[0].set_ylabel("Re(R)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(r_vals, np.imag(R_pred), label="Pred Im(R)", lw=1.6)
    axes[1].plot(r_vals, np.imag(R_ref), "--", label="ref Im(R)", lw=1.0, color='orange')
    axes[1].set_ylabel("Im(R)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(r_vals, np.abs(R_pred), label="Pred |R|", lw=1.6)
    axes[2].plot(r_vals, np.abs(R_ref), "--", label="ref |R|", lw=1.0, color='orange')
    axes[2].set_ylabel("|R|")
    axes[2].set_xlabel("r")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Debug: a={a}, omega={omega}, benchmark=pybhpt-ok")
    fig.tight_layout()
    fig.savefig("pybhpt/debug_viz_test.png", dpi=150)
    print(f"\nPlot saved to pybhpt/debug_viz_test.png")

except Exception as e:
    print(f"\nFailed: {e}")
    import traceback
    traceback.print_exc()
