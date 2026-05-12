"""Plot GSN benchmark R_in(r) and S(y) at a=0.1, ω=0.1."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/ljq/code/PINN/SolvingTeukolsky")

from utils.amplitude import TeukRadAmplitudeInWithInterpolant
from utils.mode import KerrMode
from physical_ansatz.prefactor import Leaver_prefactors
from physical_ansatz.transform_y import h_factor
from physical_ansatz.mapping import r_plus

a, omega = 0.1, 0.1
M, s, l, m = 1.0, -2, 2, 2

# Compute spectral benchmark
mode = KerrMode(M=M, a=a, ell=l, m=m, s=s, omega=omega)
spectral = TeukRadAmplitudeInWithInterpolant(mode, N_in=64, N_out=64, z_m=0.3)
profile = spectral.profile

# r grid
r_np = np.logspace(np.log10(2.0), np.log10(500.0), 800)
R_bench = profile.R_of_r(r_np)

# Compute S(y): S = R / (P * h2)
import torch
rp = float(r_plus(torch.tensor([a]), M).item())
x_np = rp / r_np
y_np = 2.0 * x_np - 1.0

a_t = torch.tensor([a])
omega_t = torch.tensor([omega])
h2_t = h_factor(a_t, omega_t, m, M, s)
h2_val = complex(h2_t.item().real, h2_t.item().imag) if h2_t.ndim == 0 else h2_t[0].item()

P_t, _, _ = Leaver_prefactors(torch.from_numpy(r_np).double(), a_t, omega_t, m, M, s)
P_np = P_t.squeeze(0).numpy() if P_t.ndim > 1 else P_t.numpy()

S_bench = R_bench / (P_np * h2_val)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.semilogy(r_np, np.abs(R_bench), "b-")
ax.set_xlabel("r/M")
ax.set_ylabel("|R_in(r)|")
ax.set_title(f"|R_in(r)| — GSN (a={a}, ω={omega})")
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(r_np, np.real(R_bench), "b-", label="Re", alpha=0.7)
ax.plot(r_np, np.imag(R_bench), "r-", label="Im", alpha=0.7)
ax.set_xlabel("r/M")
ax.set_title("R_in(r) real/imag")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(y_np, np.abs(S_bench), "g-")
ax.set_xlabel("y = 2r₊/r - 1")
ax.set_ylabel("|S(y)|")
ax.set_title(f"|S(y)| = |R_in| / |P·h₂| (a={a}, ω={omega})")
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(y_np, np.real(S_bench), "b-", label="Re", alpha=0.7)
ax.plot(y_np, np.imag(S_bench), "r-", label="Im", alpha=0.7)
ax.set_xlabel("y = 2r₊/r - 1")
ax.set_title("S(y) real/imag")
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle(f"GSN Benchmark: a={a}, ω={omega}", fontsize=14)
plt.tight_layout()
plt.savefig("/home/ljq/code/PINN/SolvingTeukolsky/outputs/gsn_benchmark_a0.1_w0.1.png", dpi=150)
plt.close()

print(f"R_in range: |R| ∈ [{np.min(np.abs(R_bench)):.3e}, {np.max(np.abs(R_bench)):.3e}]")
print(f"S(y) range: |S| ∈ [{np.min(np.abs(S_bench)):.3e}, {np.max(np.abs(S_bench)):.3e}]")
print(f"S(y) real range: [{np.min(np.real(S_bench)):.3f}, {np.max(np.real(S_bench)):.3f}]")
print(f"S(y) imag range: [{np.min(np.imag(S_bench)):.3f}, {np.max(np.imag(S_bench)):.3f}]")
print(f"S(1) = {S_bench[-1]} (at horizon y=1)")
print(f"S(-1) = {S_bench[0]} (at infinity y=-1)")
print("Saved to outputs/gsn_benchmark_a0.1_w0.1.png")
