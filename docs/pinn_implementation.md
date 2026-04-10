# PINN实现说明

## 概述

将谱方法改为点值型PINN（Physics-Informed Neural Network），使用自动微分计算导数。

## 核心改动

### 1. 保持不变的部分

- **物理分解**: R = U(r)R'(r)，其中 U(r) = Q(r) × I(r)
  - Q(r): prefactor_Q（包含 R_amp 项）
  - I(r): Inf_prefactor = r³ exp(iωr_*)
- **坐标变换**: R'(r(x)) = g(x)f(x) + 1
  - x = r_+/r ∈ (0, 1]
  - y = 2x - 1 ∈ [-1, 1]
- **方程形式**: A2(y)f_yy + A1(y)f_y + A0(y)f = 0
- **边界条件**: f(y=±1) = 0（因为边界上 g=0）

### 2. 改变的部分

#### 从谱方法到PINN

**谱方法**（之前）:
- f(y) = Σ c_n T_n(y)（Chebyshev展开）
- 训练系数 c_n
- 使用谱微分矩阵计算导数

**PINN**（现在）:
- f(y) = MLP(a, ω, y)
- 训练神经网络权重
- 使用自动微分计算 f_y, f_yy

#### 网络结构

```
输入: (a, ω, y) ∈ R³
  ↓
MLP: [64, 64, 64, 64] with Tanh
  ↓
输出: (Re[f], Im[f]) ∈ R²
```

参数范围:
- a ∈ [0.09, 0.11] (a=0.1±0.01)
- ω ∈ [0.09, 0.11] (ω=0.1±0.01)
- y ∈ [-1, 1]

#### 采样策略（类似Luna）

**边界密集采样**:
- 内点: n_interior=100，在 y∈[-1,1] 均匀随机采样
- 边界点: n_boundary=20×2，在 y=±1 附近密集采样
- 边界层宽度: 0.1（相对于[-1,1]的范围）

**参数采样**:
- 每个batch随机采样 (a, ω)
- batch_size=4

#### Loss函数

```
Loss = w_int × L_interior + w_bd × L_boundary
```

- **L_interior**: 内点residual的MSE
  - residual = A2·f_yy + A1·f_y + A0·f
- **L_boundary**: 边界条件的MSE
  - f(y=±1) = 0
- **权重**: w_int=1.0, w_bd=10.0（边界权重更大）

## 文件结构

```
model/
  pinn_mlp.py              # PINN MLP网络

physical_ansatz/
  residual_pinn.py         # PINN residual计算（自动微分）

dataset/
  sampling.py              # 采样策略

trainer/
  pinn_trainer.py          # PINN训练器

config/
  pinn_config.yaml         # PINN配置

test/
  test_pinn_training.py    # 测试脚本
```

## 使用方法

### 训练

```bash
python test/test_pinn_training.py
```

### 配置

编辑 `config/pinn_config.yaml`:

```yaml
# 参数范围
a_center: 0.1
a_range: 0.01

# 网络结构
hidden_dims: [64, 64, 64, 64]

# 采样
n_interior: 100
n_boundary: 20
batch_size: 4

# 训练
n_steps: 50000
lr: 0.001
weight_boundary: 10.0
```

## 优化策略

### 1. 采样优化
- **边界密集采样**: 边界附近采点更多，确保边界条件满足
- **自适应采样**: 可根据residual大小动态调整采样密度
- **参数空间采样**: 每步随机采样(a,ω)，增强泛化能力

### 2. Loss优化
- **加权loss**: 边界权重 > 内点权重
- **多尺度loss**: 可添加不同y区域的分段权重
- **正则化**: 可添加网络权重正则化

### 3. 网络优化
- **激活函数**: Tanh（光滑，适合高阶导数）
- **初始化**: Xavier初始化
- **深度**: 4层隐藏层，每层64个神经元

### 4. 训练优化
- **优化器**: Adam（自适应学习率）
- **梯度裁剪**: max_norm=1.0（防止梯度爆炸）
- **学习率调度**: 可添加学习率衰减

## 与谱方法对比

| 特性 | 谱方法 | PINN |
|------|--------|------|
| 表示 | Chebyshev系数 | 神经网络权重 |
| 导数 | 谱微分矩阵 | 自动微分 |
| 采样 | 固定Chebyshev节点 | 随机采样 |
| 泛化 | 单参数点 | 参数范围 |
| 边界 | 自然满足 | 需要强制 |
| 精度 | 指数收敛 | 依赖网络容量 |

## 下一步优化

1. **RAR（Residual-based Adaptive Refinement）**: 根据residual大小自适应采样
2. **FiLM/LoRA conditioning**: 更高效的参数条件化
3. **Multi-fidelity**: 结合低精度和高精度数据
4. **Transfer learning**: 从简单参数迁移到复杂参数
5. **Curriculum learning**: 从简单区域到复杂区域
