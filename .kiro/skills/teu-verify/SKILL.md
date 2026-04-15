---
name: teu-verify
description: 当用户需要校验计算结果、与Mathematica对比、检查残差、验证prefactor或系数正确性时触发。
---

# Teukolsky 关键步骤校验与 Benchmark

## 1. 单参数谱方法基准

```bash
# 运行单参数高精度谱解（基准）
python test/test_single_case_spectral.py

# 对比输出图
ls outputs/compare_*.png
```

基准参数：a=0.1, ω=0.1, l=2, m=2, s=-2

## 2. 残差校验

```python
from physical_ansatz.residual import residual_from_nodes, complex_mse
from dataset.grids import chebyshev_grid_bundle

grid = chebyshev_grid_bundle(64)
# 计算残差
res = residual_from_nodes(y_nodes, f, fy, fyy, a_batch, omega_batch, lambda_batch, p, ramp_batch, cfg)
loss = complex_mse(res)
print(f"residual MSE: {loss:.2e}")  # 目标 < 1e-10
```

## 3. Prefactor 校验

```python
from physical_ansatz.prefactor import Leaver_prefactors, prefactor_Q

# 检查 P(r) 在视界附近的渐近行为
r_near_horizon = r_plus + 1e-3
P, P_r, P_rr = Leaver_prefactors(r_near_horizon, a, omega, m)
# 期望：P 有限，P_r/P 匹配解析值

# 检查 Q(r=r_plus) = 1（无反射项时）
Q, Q_r, Q_rr = prefactor_Q(r, a, omega, p=0, R_amp=0.0)
assert abs(Q - 1.0) < 1e-10
```

## 4. λ 和 R_amp 校验

```bash
python test/test_compute_lambda.py
python test/test_amplitude_ratio.py
```

期望值（a=0.1, ω=0.1, l=2, m=2）：
- λ ≈ 4 + O(aω)（Schwarzschild 极限）
- R_amp：与 Mathematica kerr_matcher 对比

## 5. Chebyshev 网格校验

```bash
python test/test_grids.py
python test/test_chebyshev.py
```

检查项：
- D @ T_n = n * U_{n-1}（微分矩阵精度）
- D2 = D @ D（二阶矩阵一致性）
- 插值误差 < 1e-12

## 6. 与 Mathematica 对比

```bash
# 生成对比图
python mma/plot_divide_by_P.py
python mma/plot_regular_part.py
```

对比量：
- `|f(y)|` vs Mathematica 的 regular part
- `|R(r)|` vs Mathematica 的 R_in

## 通过标准

| 检验项 | 通过标准 |
|--------|---------|
| PDE residual | MSE < 1e-8 |
| 边界条件 | \|f(±1)\| < 1e-6 |
| λ 误差 | < 1e-6（与 Mathematica） |
| R_amp 误差 | < 1e-4（相对误差） |
| 谱系数衰减 | \|c_n\| < 1e-8 for n > N/2 |
