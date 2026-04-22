"""
pybhpt 使用示例
https://github.com/znasipak/pybhpt
"""
import numpy as np

# ============================================================
# 1. 轨道参数 (geo)
# ============================================================
from pybhpt.geo import (
    KerrGeodesic,
    kerr_orbital_constants,
    kerr_mino_frequencies,
    kerr_fundamental_frequencies,
)

a, p, e, x = 0.9, 10.0, 0.5, 1.0  # spin, semi-latus, eccentricity, inclination

# 守恒量 [E, L, Q]
ELQ = kerr_orbital_constants(a, p, e, x)
print(f"E={ELQ[0]:.6f}, L={ELQ[1]:.6f}, Q={ELQ[2]:.6f}")

# Mino 时间频率 [Lambda, Upsilon_r, Upsilon_theta, Upsilon_phi]
freqs_mino = kerr_mino_frequencies(a, p, e, x)
print(f"Mino freqs: {freqs_mino}")

# 坐标时间频率 [Omega_r, Omega_theta, Omega_phi]
freqs_fund = kerr_fundamental_frequencies(a, p, e, x)
print(f"Fundamental freqs: {freqs_fund}")

# 完整轨道对象
geo = KerrGeodesic(a, p, e, x, nsamples=256)
r1, r2, r3, r4 = geo.radialroots          # 四个径向转折点
print(f"r_max={r1:.4f}, r_min={r2:.4f}")
print(f"E={geo.orbitalenergy:.6f}, L={geo.orbitalangularmomentum:.6f}")

# 轨道位置采样（Mino 时间 lambda）
r_at_0 = geo.radial_position(0.0)
print(f"r(lambda=0) = {r_at_0:.4f}")

# ============================================================
# 2. 球谐函数 (swsh)
# ============================================================
from pybhpt.swsh import SpinWeightedSpheroidalHarmonic, swsh_eigenvalue, Yslm_eigenvalue

s, l, m = -2, 2, 2
omega = 0.5

# 本征值
lam = swsh_eigenvalue(s, l, m, a * omega)
print(f"\nSWSH eigenvalue lambda={lam:.6f}")
print(f"Yslm eigenvalue (c=0): {Yslm_eigenvalue(s, l, m):.6f}")

# 球谐函数对象
sph = SpinWeightedSpheroidalHarmonic(s=s, j=l, m=m, g=a*omega)
print(f"SWSH eigenvalue (object): {sph.eigenvalue:.6f}")

# ============================================================
# 3. 径向 Teukolsky 方程 (radial) — R_in 和 R_up
# ============================================================
from pybhpt.radial import RadialTeukolsky

r_horizon = 1 + np.sqrt(1 - a**2)
r_vals = np.linspace(r_horizon + 0.1, 30.0, 10)

rad = RadialTeukolsky(s=s, j=l, m=m, a=a, omega=omega, r=r_vals)
rad.solve()

# 所有采样点的 R_in（horizon 边界条件）和 R_up（infinity 边界条件）
R_in = rad.radialsolutions("In")
R_up = rad.radialsolutions("Up")
print(f"\nR_in (first 3): {R_in[:3]}")
print(f"R_up (first 3): {R_up[:3]}")

# 单点求值
print(f"R_in at r[5]={r_vals[5]:.2f}: {rad.radialsolution('In', 5):.6f}")
print(f"R_up at r[5]={r_vals[5]:.2f}: {rad.radialsolution('Up', 5):.6f}")

# 导数
print(f"dR_in/dr at r[5]: {rad.radialderivative('In', 5):.6f}")
print(f"d2R_in/dr2 at r[5]: {rad.radialderivative2('In', 5):.6f}")

# 边界值
print(f"R_in boundary (horizon): {rad.boundarysolution('In'):.6f}")
print(f"R_up boundary (infinity): {rad.boundarysolution('Up'):.6f}")

# ============================================================
# 4. Teukolsky 模式 (teuk)
# ============================================================
from pybhpt.teuk import TeukolskyMode

# k=1 prograde, n=0 fundamental radial mode
mode = TeukolskyMode(s=s, j=l, m=m, k=1, n=0, geo=geo, auto_solve=True)
print(f"\nTeukolsky mode omega={mode.omega:.6f}")
print(f"eigenvalue={mode.eigenvalue:.6f}")

# 径向解（通过 TeukolskyMode）
R_in_mode = mode.homogeneousradialsolutions["In"]
print(f"R_in via TeukolskyMode (first 3): {R_in_mode[:3]}")

# 极角解
theta_vals = mode.polarpoints
S_vals = mode.polarsolutions
print(f"polar points shape: {theta_vals.shape}")

# ============================================================
# 5. 引力波通量 (flux)
# ============================================================
from pybhpt.flux import FluxMode

fl = FluxMode(geo=geo, teuk=mode)
print(f"\nEdot (infinity): {fl.Edot['I']:.6e}")
print(f"Edot (horizon):  {fl.Edot['H']:.6e}")
print(f"Ldot: {fl.Ldot}")
