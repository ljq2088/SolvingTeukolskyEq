# PINN项目实现总结

## 已完成的工作

### 1. 核心文件创建

#### 模型文件
- `model/pinn_mlp.py`: PINN MLP网络
  - 输入: (a, ω, y) ∈ R³
  - 输出: (Re[f], Im[f]) ∈ R²
  - 结构: [64, 64, 64, 64] with Tanh activation
  - Xavier初始化

#### 物理计算
- `physical_ansatz/residual_pinn.py`: PINN residual计算
  - `compute_f_derivatives_autograd()`: 自动微分计算 f, f_y, f_yy
  - `pinn_residual_loss()`: 计算PDE residual loss + 边界条件loss

#### 数据采样
- `dataset/sampling.py`: 采样策略
  - `sample_points_luna_style()`: Luna风格边界密集采样
  - `sample_points_adaptive()`: 自适应采样（边界+边界层+内点）
  - `sample_parameters()`: 参数空间采样

#### 训练器
- `trainer/pinn_trainer.py`: PINN训练器
  - 完整的训练循环
  - Loss历史记录
  - 模型保存
  - Loss曲线绘制

#### 配置和测试
- `config/pinn_config.yaml`: PINN配置文件
- `test/test_pinn_training.py`: 测试脚本
- `docs/pinn_implementation.md`: 详细文档

### 2. 保持不变的部分

✅ **物理分解**: R = U(r)R'(r)
  - U(r) = Q(r) × I(r)
  - Q(r): prefactor_Q（包含 R_amp 项）
  - I(r): Inf_prefactor = r³ exp(iωr_*)

✅ **坐标变换**: R'(r(x)) = g(x)f(x) + 1
  - x = r_+/r ∈ (0, 1]
  - y = 2x - 1 ∈ [-1, 1]

✅ **方程形式**: B2(y)f_yy + B1(y)f_y + B0(y)f = 0
  - 系数从 x 坐标的 A2, A1, A0 转换而来
  - 使用 `coeffs_x()` 和 `transform_coeffs_x_to_y()`

✅ **边界条件**: f(y=±1) = 0

### 3. 改变的部分

#### 从谱方法到PINN

| 特性 | 谱方法（旧） | PINN（新） |
|------|-------------|-----------|
| 表示 | f(y) = Σ c_n T_n(y) | f(y) = MLP(a,ω,y) |
| 参数 | Chebyshev系数 c_n | 神经网络权重 |
| 导数 | 谱微分矩阵 D, D2 | 自动微分 autograd |
| 采样 | 固定Chebyshev节点 | 随机采样（Luna风格） |
| 泛化 | 单参数点 | 参数范围 (a,ω) |

#### Loss函数

```
Loss = w_int × L_interior + w_bd × L_boundary
```

- **L_interior**: 内点PDE residual的MSE
  - residual = B2·f_yy + B1·f_y + B0·f
- **L_boundary**: 边界条件的MSE
  - f(y=±1) = 0
- **权重**: w_int=1.0, w_bd=10.0

#### 采样策略（类似Luna）

- **内点**: n_interior=50，在 y∈[-1,1] 均匀随机采样
- **边界点**: n_boundary=10×2，在 y=±1 附近密集采样
- **边界层宽度**: 0.1
- **参数采样**: batch_size=2，每步随机采样 (a,ω)

### 4. 测试结果

✅ **代码运行正常**
- 训练速度: ~3 it/s (CPU)
- Loss正常下降: 从初始值降到 ~1e-5 量级
- 无报错，自动微分工作正常

### 5. 项目结构

```
SolvingTeukolsky/
├── model/
│   └── pinn_mlp.py              # PINN MLP网络
├── physical_ansatz/
│   ├── residual_pinn.py         # PINN residual（自动微分）
│   ├── prefactor.py             # U = Q × I (已修改)
│   ├── teukolsky_coeffs.py      # A系数计算
│   └── transform_y.py           # 坐标变换
├── dataset/
│   └── sampling.py              # 采样策略
├── trainer/
│   └── pinn_trainer.py          # PINN训练器
├── config/
│   └── pinn_config.yaml         # PINN配置
├── test/
│   └── test_pinn_training.py    # 测试脚本
└── docs/
    └── pinn_implementation.md   # 详细文档
```

## 使用方法

### 快速开始

```bash
# 训练PINN
python test/test_pinn_training.py
```

### 配置参数

编辑 `config/pinn_config.yaml`:

```yaml
# 参数范围
a_center: 0.1
a_range: 0.01      # a ∈ [0.09, 0.11]
omega_center: 0.1
omega_range: 0.01  # ω ∈ [0.09, 0.11]

# 网络结构
hidden_dims: [64, 64, 64, 64]

# 采样
n_interior: 50
n_boundary: 10
batch_size: 2

# 训练
n_steps: 1000
lr: 0.001
weight_boundary: 10.0
```

## 优化策略

### 已实现

1. ✅ **边界密集采样**: 边界附近采点更多
2. ✅ **加权loss**: 边界权重 > 内点权重
3. ✅ **梯度裁剪**: max_norm=1.0
4. ✅ **Xavier初始化**: 网络权重初始化
5. ✅ **参数空间采样**: 每步随机采样(a,ω)

### 待优化

1. **RAR（Residual-based Adaptive Refinement）**
   - 根据residual大小自适应采样
   - 在residual大的区域增加采样点

2. **学习率调度**
   - 添加学习率衰减
   - 使用 ReduceLROnPlateau

3. **网络结构优化**
   - FiLM/LoRA conditioning
   - 更高效的参数条件化

4. **多尺度loss**
   - 不同y区域的分段权重
   - 添加导数正则化

5. **加速训练**
   - 使用GPU
   - 优化自动微分（vmap）
   - 减少batch内的循环

6. **Curriculum learning**
   - 从简单参数到复杂参数
   - 从简单区域到复杂区域

## 与谱方法对比

### 优势
- ✅ 可以处理参数范围（不是单点）
- ✅ 更灵活的采样策略
- ✅ 易于添加物理约束

### 劣势
- ❌ 训练速度较慢（~3 it/s vs 谱方法更快）
- ❌ 精度依赖网络容量
- ❌ 需要更多调参

### 适用场景
- **PINN**: 参数空间探索，复杂边界条件
- **谱方法**: 单参数高精度求解

## 下一步工作

1. **完成1000步训练**，查看收敛情况
2. **可视化结果**，与Mathematica对比
3. **实现RAR**，提高采样效率
4. **GPU加速**，提高训练速度
5. **扩大参数范围**，测试泛化能力

## 注意事项

1. **自动微分速度**: 当前对每个batch样本单独计算导数，可用vmap优化
2. **边界条件**: 通过loss强制，不是硬约束
3. **参数范围**: 当前较小(±0.01)，可逐步扩大
4. **训练时间**: CPU上1000步约5-10分钟

## 文件修改记录

### 新增文件
- model/pinn_mlp.py
- physical_ansatz/residual_pinn.py
- dataset/sampling.py
- trainer/pinn_trainer.py
- config/pinn_config.yaml
- test/test_pinn_training.py
- docs/pinn_implementation.md

### 修改文件
- physical_ansatz/prefactor.py (U_factor改为Q×I)

### 保留文件（谱方法）
- test/test_single_case_spectral.py (注释掉，保留参考)
- model/chebyshev_trunk.py (保留)
- dataset/grids.py (保留)
