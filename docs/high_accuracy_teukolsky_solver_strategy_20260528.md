# 高精度高速 Teukolsky `R_in` 与振幅求解器路线图

日期：2026-05-28  
工作分支：`research/high-accuracy-teukolsky-plan-20260528`  
目标仓库：`/home/ljq/code/PINN/SolvingTeukolskyEq_autoencoder`，远端 `https://github.com/ljq2088/SolvingTeukolskyEq.git`

## 目标

构筑一个面向固定 `s=-2, l=m=2` 的高速 surrogate / PINN-hybrid 求解器：

- 输入：`a ∈ [0, 1)`, `ω ∈ (1e-4, 10.0)`, `r ∈ [r_+(a), +∞)`。
- 输出：
  - 视界入射齐次解 `R_in(r; a, ω)`。
  - 无穷远渐近系数 `B^ref(a,ω)` 与 `B^inc(a,ω)`，对应项目约定中的 `r^3 exp(+iωr*)` 与 `r^{-1} exp(-iωr*)` 分量。
- 精度目标：
  - 近中期：patch 内 `R_in` 相对误差 `1e-4 ~ 1e-6`，振幅相对误差 `1e-4`。
  - 长期：主域达到 `1e-8` 级，困难区域单独标记并降级到可靠基准后端。
- 速度目标：
  - 单点 `R_in(a,ω,r)` GPU batch 推理 `O(μs)`。
  - 一组 `(a,ω)` 上 1000 个 `r` 点 `O(ms)`。

## 当前项目状态评估

### 已有可靠资产

- `physical_ansatz/residual.py` 中已有可用的 `λ(a,ω)` 调用路径。
- `pybhpt_usage/compute_solution.py` 可调用 `pybhpt.radial.RadialTeukolsky` 作为小/中频 `R_in` benchmark；本项目不把 pybhpt 当作可信振幅来源。
- `mma/` 与 Mathematica 相关脚本可作为小/中频 `B_inc/B_ref` benchmark。
- `benchmark/scripts/gsn_ref.jl` 与 GSN 相关脚本可作为较高频 `B_inc/B_ref` benchmark。
- `utils/amplitude.py` 中已有两域 Chebyshev 谱方法，可作为数值实践和待诊断的 teacher 候选，但极低频/极高频不能默认可信。
- `physical_ansatz/transform_y.py` 已经把视界约束编码到 reduced shape：
  - `S(x)=g(x)(h1(x)f(y)+1)+1`
  - `h1(1)=0`
  - `g(1)=0`
  - `g_x(1)=c_H=-A0/A1`
  - 因此 `S(1)=1` 且 `S_x(1)=c_H`。
- `model/autoencoder_pinn.py` 已经将 `R_in`、`u_up/u_down`、`B_inc/B_ref` 放进一个共享 encoder 的多头结构。

### 主要失败模式

1. **纯 residual 训练不够约束全域高精度。**  
   即使视界处 `S` 与 `S_y` 理论上唯一确定齐次解，PINN 在有限配点、有限网络容量和不均衡残差尺度下仍会走到数值上低残差但物理误差大的函数。

2. **全参数域跨度过大。**  
   `ω ∈ (1e-4,10)` 横跨 5 个数量级，低频 MST 结构、高频快速振荡、近极端自旋 `a→1` 的边界层行为完全不同。单一 MLP/PINN 很难在这个域内同时保持高精度。

3. **当前训练存在采样/目标缺陷。**  
   近期 `lbfgs_refine_stage1.py` 的随机 `y` 采样实际只覆盖 `[-1,0]` 左半区间，导致微调不能约束靠近视界的大段区域；历史 run 显示相对 residual 降低不等价于 `R_in` 相对误差降低。

4. **`R_in` 与振幅不应拆成弱耦合 stage 后再“希望一致”。**  
   当前 Stage 1/2/3/4 的方向是合理探索，但如果没有统一的渐近匹配损失、Wronskian/Abel 检查和 benchmark 校准，`R_in` 与 `B_inc/B_ref` 会各自拟合，最终不一致。

5. **PINN 自动微分成本和噪声较大。**  
   对 `y` 做二阶/三阶 autograd 在训练速度和数值稳定性上都不如谱/递推导数，尤其在高频和边界层区域。

## 文献与方法判断

### PINN 相关结论

- PINN 常见训练失败来自 loss 多目标刚性、梯度病态和不同误差项收敛速度不一致；NTK 与梯度病态文献均建议使用自适应权重、硬约束、预条件或分域训练。
- Fourier features 与 SIREN 能缓解高频谱偏置，但不能单独解决齐次 ODE 的幅值/相位唯一性和极端参数域泛化问题。
- XPINN/cPINN、FBPINN 等 domain decomposition 思路更适合多尺度/高频 PDE，尤其当不同子域可用不同网络容量与损失权重。

### Teukolsky 求解相关结论

- pybhpt 的 `RadialTeukolsky` 已经直接求解齐次径向 Teukolsky 方程，是本项目最直接的 Python `R_in` benchmark；但它不输出本项目所需 convention 下的可信 `B_inc/B_ref`。
- Mathematica/MMA 和 GSN 是振幅 benchmark 主来源：MMA 偏小/中频，GSN 偏高频。
- `ω≈0.1` 是 pybhpt、MMA、GSN 的强制交叉验证带；只有通过 convention 检查后，相关 benchmark 才进入训练数据。
- 对 `R_in`，视界正则条件可以确定解；对 `B_inc/B_ref`，最稳定的做法是从同一个全局解的远场渐近匹配中提取，而不是独立训练一个完全无物理耦合的参数网络。

## 推荐总体路线：Spectral Operator + Physics Correction

不建议继续把核心目标定义为“纯 PINN 从 residual 独立求全域”。最佳方案是：

1. **用可信求解器生成高保真离线数据。**
2. **训练一个谱系数神经算子作为主求解器。**
3. **用物理 residual、边界条件、渐近匹配和 Wronskian 做 fine-tune / consistency regularization。**
4. **用主动学习补齐误差峰值区域。**

这仍然是 physics-informed solver，但不是让 PINN 从零独立承担所有困难。

## 坐标与函数表示

### 参数坐标

使用全局参数变换，而不是裸 `a,ω`：

- 自旋：
  - `q = sqrt(1-a^2)` 或 `κ = sqrt(1-a^2)/(1+sqrt(1-a^2))`，用于解析近极端行为。
  - `χ = atanh(a clipped)` 或分 patch 使用局部 `u_a`。
- 频率：
  - `ζ = log10(ω)` 作为主坐标。
  - 附加 `ω`, `aω`, `k=ω-mΩ_H`, `sign/log|k|`，因为超辐射边界 `k≈0` 是困难面。
- 分域：
  - `a`：`[0,0.5]`, `[0.5,0.9]`, `[0.9,0.98]`, `[0.98,0.999]`, `[0.999,1)`。
  - `ω`：log 分段，例如 `[1e-4,1e-3]`, `[1e-3,1e-2]`, `[1e-2,1e-1]`, `[1e-1,1]`, `[1,10]`。
  - 额外沿 `k=0` 建窄带 patch。

### 径向坐标

推荐并行支持两套坐标：

1. **紧致坐标**：`x = r_+/r`, `y=2x-1`，适合全域。
2. **远场相位坐标**：`ρ = log(r/r_+)` 或 `τ = ω r`，适合 `ω r` 大的高频远场。

训练时不要直接拟合剧烈振荡的 `R`，而是拟合剥离主相位/幂律后的慢变 envelope。

## 解的表示：从 MLP 值函数改为 Chebyshev coefficient operator

### 主网络输出

每个 patch 内训练一个 operator：

```text
G_theta: (a, logω, k, λ, patch_features) -> Chebyshev coefficients
```

输出不再是 `R(r)` 的点值，而是多个基函数展开系数：

- 视界基底 `S_H(x)`：保证 `S(1)=1`, `S_x(1)=c_H`。
- 无穷远上行/下行基底 envelope：
  - `U_ref(x)` 对应 `r^3 exp(+iωr*)`
  - `U_inc(x)` 对应 `r^{-1} exp(-iωr*)`
- 振幅：
  - `log|B_ref|`, `phase_unwrapped(B_ref)`
  - `log|B_inc|`, `phase_unwrapped(B_inc)`

### 推荐形式

令

```text
R_in(r) = B_ref(a,ω) * A_ref(r,a,ω) * U_ref(x;a,ω)
        + B_inc(a,ω) * A_inc(r,a,ω) * U_inc(x;a,ω)
```

其中 `A_ref/A_inc` 是解析渐近因子，`U_ref/U_inc -> 1`。同时在近视界用 `P_H(r,a,ω) * S_H(x)` 表示，训练时强制两种表示在重叠区一致。

最终推理可有两种模式：

- **fast mode**：直接用全局组合公式输出 `R_in` 与振幅。
- **certified mode**：额外计算 residual/Wronskian 估计误差，必要时回退 pybhpt/GSN。

### 为什么不用单一点值 PINN

- Chebyshev 系数天然给出高阶导数，避免 `autograd` 噪声。
- 系数空间更适合监督学习和主动学习。
- 对任意 `r` 的推理是 Clenshaw 递推，速度极快。
- 系数可直接用于远场拟合与误差估计。

## 数据生成与基准层

### 参考求解器分工

- `pybhpt`：小/中频 `R_in(r)` 主参考；不作为 `B_inc/B_ref` 真值。
- Mathematica/MMA：小/中频 `B_inc/B_ref` 与 convention 审计。
- GSN/MST：高频 `B_inc/B_ref` 与高频解行为审计。
- `ω≈0.1`：三套基准交叉验证区域，默认 `a` 方向 7–9 点、`ω={0.08,0.1,0.12}`。
- 谱方法：作为可控数值 teacher 候选，必须先通过条件数、Abel/detS、相对方程残差诊断。

### 数据内容

每个 `(a,ω)` 样本保存：

- `λ(a,ω)`。
- Chebyshev `x` 网格上的 `R_in`。
- 剥离 prefactor 后的 `S` 或 envelope。
- `R_x`, `R_xx` 或通过谱微分矩阵得到的 residual。
- 远场拟合得到的 `B_inc/B_ref`。
- Wronskian/Abel 检查量。
- 参考来源和精度标记。

### 采样策略

- 初始：Sobol / Latin hypercube in `(a, logω)`，每 patch 至少 1k–10k 样本。
- 重点加密：
  - `ω≈1e-4`。
  - `ω≈10`。
  - `a→1`。
  - `k=ω-mΩ_H≈0`。
  - 误差热图中的高误差岛。
- 主动学习：
  1. 训练初版。
  2. 在大候选池上预测并计算 physics residual 与不确定性。
  3. 抽取 worst cases 调用 pybhpt/GSN/MMA。
  4. 增量训练。

## 训练目标

### 监督项

- `L_R`: 剥离 prefactor 后的 complex envelope 相对误差。
- `L_B`: `B_inc/B_ref` 的 log-magnitude 与 unwrapped phase 误差。
- `L_coeff`: Chebyshev 系数误差，优先约束低阶系数，再逐渐开放高阶。

### 物理项与最终验收

- `L_res`: 相对 residual：

```text
|D2 S_yy + D1 S_y + D0 S|^2 / max(|D2 S_yy|, |D1 S_y|, |D0 S|, eps)^2
```

最终验收采用未平方形式：

```text
|D2 S_yy + D1 S_y + D0 S| / max(|D2 S_yy|, |D1 S_y|, |D0 S|)
```

分母中的二阶、一阶、零阶项必须带上当前解本身，不除掉外部归一化常数。benchmark/anchor loss 只用于校正解和选择正确分支，不作为最终真值标准。

- `L_H`: 视界 hard constraint 已解析保证，训练只监控。
- `L_inf`: 远场渐近 Robin/匹配条件。
- `L_match`: 近视界表示与远场组合表示在重叠区一致。
- `L_W`: Wronskian/Abel 守恒检查。

### 优化策略

- 第一阶段：纯监督训练，快速学到正确分支和尺度。
- 第二阶段：加入 physics consistency，权重从小到大。
- 第三阶段：per-patch L-BFGS 或 Shampoo/KFAC 类二阶微调，但只在系数网络或低秩 adapter 上做。
- 第四阶段：主动学习补点。

## 模型架构建议

### Patch ensemble

全域不要用一个模型硬扛。使用 mixture-of-experts：

- 每个专家负责一个 `(a, logω, k)` patch。
- gating 网络只做平滑权重，不直接生成物理解。
- patch 间重叠区加入 consistency loss。

### 专家结构

推荐默认专家：

```text
ParameterEncoder
  inputs: [a, q, logω, ω, aω, Ω_H, k, log|k|, λ_re, λ_im]
  backbone: residual MLP / SIREN-hybrid
  outputs:
    Cheb coefficients for S_H / U_ref / U_inc
    B_inc logmag + phase
    B_ref logmag + phase
```

径向方向固定 Chebyshev trunk，不再让 MLP 直接学习 `r` 点值。

### 相位处理

振幅相位必须 unwrap，并按 patch 存储相位 gauge。直接训练 `Re/Im` 在跨越零点或相位快速变化时会不稳定。

## 验证指标

每次训练必须输出：

- `(a,ω,r)` 三维误差数据集。
- 按 `a`、`logω`、`k`、`r` 的误差热图。
- `B_inc/B_ref` 与 pybhpt/GSN/MMA 的相对误差。
- 最终验收用的相对方程 residual heatmap。
- Wronskian/Abel 误差。
- 推理速度 benchmark。

验收阈值应按区域分级：

| 区域 | `R_in` median | `R_in` p99 | 振幅 median | 备注 |
|---|---:|---:|---:|---|
| 常规域 | `1e-6` | `1e-4` | `1e-5` | 主力目标 |
| 低频 `ω<1e-3` | `1e-5` | `1e-3` | `1e-4` | MST 特征明显 |
| 高频 `ω>1` | `1e-5` | `1e-3` | `1e-4` | 需相位剥离 |
| 近极端 `a>0.99` | `1e-4` | `1e-2` | `1e-3` | 单独专家/回退 |

## 当前代码的近期改造建议

### 立即修复

1. 使用 `scripts/cross_validate_benchmarks_omega01.py` 在 `ω≈0.1` 做 pybhpt/MMA/GSN 交叉验证。
2. 使用 `scripts/diagnose_spectral_conditioning.py` 扫描谱方法在低/中/高频的条件数、Abel/detS 和振幅动态范围。
3. 修正 `scripts/lbfgs_refine_stage1.py` 的 `y` 随机采样区间。
4. 将最终验收 residual 的 max-term 定义写入正式 API、训练配置和评估脚本。
5. 给 `pybhpt`、MMA、GSN 参考解加缓存，避免反复调用。

### 新增模块

建议新增：

```text
model/cheb_operator_teukolsky.py
physical_ansatz/asymptotic_basis.py
dataset/reference_cache.py
scripts/build_reference_dataset.py
scripts/train_cheb_operator.py
scripts/evaluate_aw_r_grid.py
docs/reference_conventions.md
```

### 保留/重用

- 重用 `λ` 求解接口。
- 重用 `Leaver_prefactors`、`h_factor`、`horizon_regularity_slope`。
- 重用 pybhpt/MMA/GSN 作为 benchmark。
- 当前 autoencoder 多头结构可作为过渡 baseline，但不建议作为最终主架构。

## 分阶段实施计划

### Phase 0：约定与基准冻结

- 写清 `R_in`、`B_inc`、`B_ref`、`r*`、prefactor convention。
- 用 `ω={0.08,0.1,0.12}` 与 7–9 个 `a` 点比较 pybhpt、GSN、MMA，确定一致性和归一化转换。
- 建立参考缓存格式。

### Phase 0.5：谱方法失败机制诊断

- 在 `ω∈{1e-4,1e-3,1e-2,0.1,1,10}` 与代表 `a` 上扫描 `N_in/N_out/z_m`。
- 记录全谱矩阵条件数、行列预处理后条件数、匹配矩阵条件数、Abel/detS、`|B_inc|/|B_ref|`。
- 分析极低频/极高频失败是否来自渐近形式不匹配、振幅量级差异、矩阵条件数过大、分域不足或浮点精度不足。
- 2026-05-29 诊断结论：
  - 低频 `ω≈1e-4` 的主要改进来自 `GF_adaptive_match.m` 风格的 `r_match≈3M+ω^{-1/2}` 与 sinh-Cheb 映射；旧 linear 网格的振幅误差为 `O(1)`，自动扫描后可降到 `~1e-3~1e-2`。
  - 高频 `ω≈10` 的 `B_inc` 已可由当前 Teukolsky 谱匹配稳定给出，和 GSN 相对差约 `4e-11`；但直接匹配得到的 `B_ref` 落到 `1e-16`，而 GSN 为 `~1e-8`。
  - 对高频 `B_ref`，2x2 匹配矩阵列缩放后条件数只有 `O(10)`，谱域矩阵行列均衡和 mpmath 高精度求解均不能恢复 `B_ref`；瓶颈不是普通线性代数预处理。
  - 有效变量是 `R/A_down = B_inc + B_ref(A_up/A_down)`。固定可靠 `B_inc` 后，在远场窗口上拟合 `B_ref`，`ω=10,a=0.5` 可把 `B_ref` 相对误差降到 `~3e-3`。
  - 因此高频振幅 teacher 应优先采用 GSN/SN 远场采样 + 固定 `B_inc` 的放大变量拟合；当前 Teukolsky 两域谱匹配只用于 `B_inc` 和 residual/Abel gate。
  - 进一步扫描 `a∈{0,0.5,0.9}`, `ω∈{3,10}` 后，使用 spectral `B_inc` 固定、窗口 `r∈[500,5000]`、只拟合 leading `B_ref` 的选择器可稳定达到 `B_ref` 相对误差 median `1.1e-4`、max `1.7e-4`。该规则比残差最小化更可靠，因为高阶远场项容易用高条件数过拟合。
  - 已将 `utils/MatlCheb-main.zip` 与 `GF_adaptive_match.m` 的 Schwarzschild/Bondi 双域匹配原型移植到 Python：`utils/matlcheb.py` 与 `scripts/schwarzschild_gf_adaptive_match.py`。该短程/正则变量的纯谱匹配在 `ω=1,3,10` 下可保留很小反射比，例如 `ω=1` 的 `|Ciu/Cid|≈9.6e-6`，`ω=3,10` 达 `1e-14` 级，说明“纯谱法”可行的关键不是继续直接匹配 Teukolsky `R`，而是换成短程/正则 scattering 变量。

### Phase 1：Patch-0 Cheb operator proof-of-concept

- 只做 `a∈[0.206,0.794]`, `ω∈[0.008523,0.0777]`。
- 训练 `S_H` Chebyshev coefficient operator。
- 目标：`R_in` median `<1e-4`，p90 `<1e-3`。
- 与当前 L-BFGS final 模型的三维误差数据直接对比。

### Phase 2：加入远场振幅

- 从同一参考解拟合 `B_inc/B_ref`。
- 训练共享 encoder + coefficient heads + amplitude heads。
- 加入远场匹配和 Wronskian loss。

### Phase 3：全域分 patch 扩展

- 按 `(a,logω,k)` 建专家。
- 对每个 patch 训练并主动学习补点。
- 重叠区做 consistency。

### Phase 4：部署与回退

- 导出 TorchScript/ONNX。
- 推理时计算 cheap residual/error proxy。
- 超出可信域自动回退 pybhpt/GSN。

## 关键决策

- 不再以“纯 PINN residual 最小化”为主路线。
- 不使用单一全域 MLP。
- 不把 `B_inc/B_ref` 作为完全独立的 amplitude net；它必须与同一个 `R_in` 解的远场匹配耦合。
- 主表示从点值网络改为谱系数 operator。
- 基准程序不是可选验证，而是训练数据与主动学习闭环的一部分；但最终验收只看相对方程残差。
- pybhpt 只作为 `R_in` benchmark，MMA/GSN 才作为振幅 benchmark。

## 参考资料

- pybhpt radial documentation: https://pybhpt.readthedocs.io/en/latest/pybhpt.radial.html
- Black Hole Perturbation Toolkit Teukolsky package: https://github.com/BlackHolePerturbationToolkit/Teukolsky
- Mano, Suzuki, Takasugi, analytic Teukolsky solutions: https://arxiv.org/abs/gr-qc/9603020 and https://arxiv.org/abs/gr-qc/9611014
- Fourier features and spectral bias: https://arxiv.org/abs/2006.10739
- SIREN periodic implicit representations: https://proceedings.neurips.cc/paper/2020/hash/53c04118df112c13a8c34b38343b9c10-Abstract.html
- PINN gradient pathologies: https://arxiv.org/abs/2001.04536
- PINN failure through NTK: https://arxiv.org/abs/2007.14527
- XPINN/cPINN domain decomposition: https://arxiv.org/abs/2104.10013
- FBPINNs multilevel domain decomposition: https://arxiv.org/abs/2306.05486
- DeepONet operator learning: https://www.nature.com/articles/s42256-021-00302-5
- Fourier Neural Operator: https://arxiv.org/abs/2010.08895
