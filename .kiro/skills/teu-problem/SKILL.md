---
name: teu-problem
description: 当用户问"这个物理问题是怎么描述的"、"方程是怎么化归的"、"任务是怎么拆分的"、或需要理解Teukolsky项目的物理框架时触发。
---

# Teukolsky 物理问题描述与任务拆分

## 物理问题

求解 Kerr 黑洞背景下的 Teukolsky 径向方程（旋量场 s=-2，模式 l=2, m=2）：

```
A2(x) R''(x) + A1(x) R'(x) + A0(x) R(x) = 0
```

参数空间：a ∈ [0.01, 0.99]，ω ∈ [0.01, 1.50]（实频或复频）

## 化归链

```
R(r)  →  分解为 R = U(r) · f(x)
         U(r) = P(r) · Q(r)   （物理渐近因子）
         P(r): Leaver prefactor（视界/无穷远渐近行为）
         Q(r): 反射项（含 R_amp）

r     →  x = r_+/r ∈ (0,1]   （紧化坐标，视界→1，无穷→0）
x     →  y = 2x-1 ∈ [-1,1]   （Chebyshev 域）

R(r)  →  f(y) 满足：B2·f_yy + B1·f_y + B0·f = 0
         边界条件：f(y=±1) = 0
```

关键量：
- λ：角向分离常数，由 `utils/compute_lambda.py` 计算
- R_amp：反射振幅，由 `utils/amplitude_ratio.py` 计算
- Δ(r) = r²-2Mr+a²，K(r) = (r²+a²)ω - am

## 任务拆分

| 子任务 | 文件 | 状态 |
|--------|------|------|
| 坐标变换 | `physical_ansatz/mapping.py` | ✅ |
| 渐近因子 | `physical_ansatz/prefactor.py` | ✅ |
| 方程系数 | `physical_ansatz/teukolsky_coeffs.py` | ✅ |
| y坐标变换 | `physical_ansatz/transform_y.py` | ✅ |
| 谱网格 | `dataset/grids.py` | ✅ |
| 残差计算 | `physical_ansatz/residual.py` | ✅ |
| Branch网络 | `model/branch_mlp.py` | 🔧 stub |
| Trunk网络 | `model/chebyshev_trunk.py` | ✅ |
| 算子模型 | `model/operator_model.py` | 🔧 stub |
| PINN训练 | `trainer/pinn_trainer.py` | ✅ |

## 两条技术路线

```
谱方法（单参数高精度）：
  (a,ω) → λ,R_amp → 系数矩阵 → Chebyshev 展开 → 系数 c_n

PINN（参数范围泛化）：
  MLP(a,ω,y) → f(y)，autograd 计算导数，PDE residual 作 loss
```

## 快速定位

```bash
# 查看方程系数定义
cat physical_ansatz/teukolsky_coeffs.py

# 查看坐标变换
cat physical_ansatz/mapping.py

# 查看配置
cat config/teukolsky_radial.yaml
```
