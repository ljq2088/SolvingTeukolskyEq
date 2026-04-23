import numpy as np
from scipy.optimize import minimize_scalar
import sys
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
from utils.compute_lambda_usage import compute_lambda



def r_plus(a, M=1.0):
    """
    Outer horizon radius for Kerr black hole.
    """
    kappa = np.sqrt(1.0 - a**2)
    return M * (1.0 + kappa)
def Delta(r, a, M=1.0):
    """
    Δ(r) = r^2 - 2Mr + a^2
    """
    return r**2 - 2.0 * M * r + a**2

def Sigma_r(r, a):
    """
    Σ(r) = r^2 + a^2
    """
    return r**2 + a**2

def P_s(r, a, s=-2, M=1.0):
    D = Delta(r, a, M)
    S = Sigma_r(r, a)
    return 2.0 * r * D / S**2 + 2.0 * s * (r - M) / S

def dP_drstar(r, a, s=-2, M=1.0, h=1e-5):
    D = Delta(r, a, M)
    S = Sigma_r(r, a)
    A = D / S  # d/dr* = A d/dr
    return A * (P_s(r + h, a, s=s, M=M) - P_s(r - h, a, s=s, M=M)) / (2.0 * h)

def ReOmega_s2(r, ell, m, a, omega, M=1.0):
    """
    Real part of Omega(r) for s=-2, assuming omega and lambda are real.
    """
    D = Delta(r, a, M)
    S = Sigma_r(r, a)
    K = S * omega - a * m
    lam = compute_lambda(a, omega,ell, m,  s=-2)

    P = P_s(r, a, s=-2, M=M)
    dPst = dP_drstar(r, a, s=-2, M=M)

    return (K**2 - lam * D) / S**2 - 0.5 * dPst - 0.25 * P**2


def ImOmega_s2(r, ell, m, a, omega, M=1.0):
    """
    Imag part of Omega(r) for s=-2, assuming omega and lambda are real.
    """
    D = Delta(r, a, M)
    S = Sigma_r(r, a)
    K = S * omega - a * m

    return (4.0 * (r - M) * K - 8.0 * omega * r * D) / S**2

def barrier_peak_r(ell, m, a, omega, M=1.0, rmax=20.0):
    """
    Return the barrier-top location for Re(V_can),
    equivalently the minimum of Re(Omega).
    """
    rp = r_plus(a, M=M)
    eps = 1.0e-5

    obj = lambda r: ReOmega_s2(r, ell, m, a, omega, M=M)

    res = minimize_scalar(
        obj,
        bounds=(rp + eps, rmax),
        method="bounded"
    )

    return res.x, res.fun

#使用示例
if __name__ == "__main__":
    rpk, omin = barrier_peak_r(2, 2, a=0.7, omega=0.35, M=1.0, rmax=15.0)
    print(f"(l,m)=(2,2), a=0.7, omega=0.35 --> r_peak/M = {rpk:.6f}")





# omega_real = 0.35
# omega = float(omega_real)

# omega_list = [0.2, 0.3, 0.4, 0.5]
# for omega_real in omega_list:
#     rpk, omin = barrier_peak_r(2, 2, a=0.7, omega=omega_real, M=1.0, rmax=15.0)
#     print(f"(l,m)=(2,2), a=0.7, omega={omega_real:.3f} --> r_peak/M = {rpk:.6f}")



# import matplotlib.pyplot as plt

# M = 1.0
# a = 0.7
# omega_real = 0.35

# r0 = r_plus(a, M=M) + 1e-3
# r_grid = np.linspace(r0, 12.0, 3000)

# fig, ax = plt.subplots(1, 1, figsize=(7, 5))

# for ell, m in [(2, 2), (3, 3)]:
#     ReOm = np.array([ReOmega_s2(r, ell, m, a, omega_real, M=M) for r in r_grid])
#     ImOm = np.array([ImOmega_s2(r, ell, m, a, omega_real, M=M) for r in r_grid])
#     ReV  = omega_real**2 - ReOm

#     rpk, omin = barrier_peak_r(ell, m, a, omega_real, M=M, rmax=12.0)

#     ax.plot(r_grid, ReV, lw=2, label=fr"$\Re V_{{can}}$ ({ell},{m})")
#     ax.plot(r_grid, ReOm, lw=1.5, ls="--", label=fr"$\Re\Omega$ ({ell},{m})")
#     ax.plot(r_grid, ImOm, lw=1.2, ls=":", label=fr"$\Im\Omega$ ({ell},{m})")
#     ax.axvline(rpk, ls="dashdot", alpha=0.7)

#     print(f"(l,m)=({ell},{m}), a={a}, omega={omega_real} --> r_peak/M = {rpk/M:.6f}")

# ax.set_xlabel(r"$r/M$")
# ax.set_ylabel("value")
# ax.set_title(rf"Real-$\omega$ canonical potential, $a/M={a}$, $\omega M={omega_real}$")
# ax.legend(fontsize=9, ncol=2)
# plt.tight_layout()
# plt.show()

# omega_scan = np.linspace(0.1, 0.8, 60)
# rpeak_22 = []
# rpeak_33 = []

# for om in omega_scan:
#     rpk22, _ = barrier_peak_r(2, 2, a=0.7, omega=om, M=1.0, rmax=15.0)
#     rpk33, _ = barrier_peak_r(3, 3, a=0.7, omega=om, M=1.0, rmax=15.0)
#     rpeak_22.append(rpk22)
#     rpeak_33.append(rpk33)

# plt.figure(figsize=(7,5))
# plt.plot(omega_scan, rpeak_22, label="(2,2)")
# plt.plot(omega_scan, rpeak_33, label="(3,3)")
# plt.xlabel(r"$\omega M$")
# plt.ylabel(r"$r_{\rm peak}/M$")
# plt.title(r"Barrier-top location vs real frequency")
# plt.legend()
# plt.tight_layout()
# plt.show()