# Agent 指令：在 Stage-1 训练中加入 \(F_y=0\) derivative residual，先关闭 causal weighting

当前仓库：

`ljq2088/SolvingTeukolskyEq`

当前分支必须是：

`feature/autoencoder-asymptotic-decoder2`

不要切换分支，不要新建分支。

当前 Stage-1 retrain 主入口是：

```bash
python scripts/retrain_stage1_patch0.py --patch-id 0 --device cuda --steps 50000 \
    --resume-checkpoint outputs/stage1_retrain/saved_best_v3.pt
```

本轮任务不是继续 Stage-4，也不是加入 pybhpt/MMA/spectral anchor，也不是训练 learned \(u_{\uparrow},u_{\downarrow}\) decoder。  
本轮只测试一个新的无监督 PINN 约束：

**在原 Stage-1 方程 residual \(F=0\) 的基础上，加入 near-infinity 区域的 derivative residual \(F_y=0\)，用于补偿无穷远端附近二阶导系数 \(A_2\to0\) 导致的曲率约束不足。**

本轮先不要使用 causal weighting。也就是说：  
在本次实验配置中必须设置：

```yaml
atlas_training:
  causal:
    enabled: false
```

原因：当前目标是专门研究 near-infinity shift，而原来的 horizon-first causal weighting 可能会降低或延迟无穷远端配点的训练权重，从而混淆 \(F_y\) residual 的效果。

---

## 1. 当前实验基准

本轮从之前记录的 step 20000 附近最佳模型继续训练。优先使用：

```bash
outputs/stage1_retrain/saved_best_v3.pt
```

如果该文件不是 step 20000 的最佳模型，或仓库/日志中另有明确的 step 20000 best checkpoint，请使用真正的 step 20000 best checkpoint，并在报告中写清楚：

- 使用的 checkpoint 路径；
- checkpoint 对应 step；
- 对应 best/val 指标；
- 是否从 `saved_best_v3.pt` 加载。

不要从头训练，不要换 patch。当前 patch 是 patch 0。

---

## 2. 数学目标

当前 reduced 方程写成：

\[
F(y)
=
A_2(y)u_{yy}
+
A_1(y)u_y
+
A_0(y)u
=0.
\]

这里 \(u\) 可以理解为已经提取 \(P\) 后的变量：

\[
u=\frac{R}{P}.
\]

在当前 Stage-1 代码里，模型实际输出可能不是 \(u\) 或 \(S\)，而是自由函数 \(f_R(y)\)。因此实现时必须注意：

1. 如果模型输出的是 \(f_R(y)\)，不能直接把 \(f_R\) 当作 \(u\)；
2. 必须通过当前 Stage-1 ansatz 构造 reduced shape \(S(y)\)；
3. 因为 \(R/P=h_2S(y)\)，且 \(h_2\) 与 \(y\) 无关，所以可在 \(S\)-space 中实现：
   \[
   F(y)=D_2(y)S_{yy}+D_1(y)S_y+D_0(y)S.
   \]
4. \(D_2,D_1,D_0\) 必须是提取 \(P\) 后的 reduced coefficients，不能误用 raw \(R\)-space coefficients。

对 \(F=0\) 关于 \(y\) 求导得到：

\[
F_y
=
D_2 S_{yyy}
+
(D_{2,y}+D_1)S_{yy}
+
(D_{1,y}+D_0)S_y
+
D_{0,y}S
=0.
\]

因此 derivative residual 的系数是：

\[
E_3=D_2,
\]

\[
E_2=D_{2,y}+D_1,
\]

\[
E_1=D_{1,y}+D_0,
\]

\[
E_0=D_{0,y}.
\]

即：

\[
F_y=E_3S_{yyy}+E_2S_{yy}+E_1S_y+E_0S.
\]

---

## 3. 为什么 \(F_y=0\) 有意义

普通方程在无穷远端附近有：

\[
D_2(y)\to0,
\]

因此 \(S_{yy}\) 的约束会变弱。  
但对 \(F_y=0\) 来说，\(S_{yy}\) 的系数是：

\[
D_{2,y}+D_1.
\]

在 \(y\to -1\) 附近，虽然 \(D_2\to0\)，但 \(D_1\) 通常趋于有限值，因此 \(F_y\) 对 \(S_{yy}\) 的约束不会完全消失。

这正是本轮实验的核心：  
**用 \(F_y=0\) 在 near-infinity 区域恢复对曲率的约束，测试它是否能改善 Stage-1 在无穷远端的 shift / 截距 / 斜率问题。**

---

## 4. 重要边界原则

不要在精确 \(y=-1\) 处计算 \(F_y\) loss。

本轮：

- \(F_y\) 只在 \(y>-1\) 的内点上计算；
- 精确 \(y=-1\) 的点如果使用，只能用于 infinity Robin 诊断；
- 本轮主实验先不把 infinity Robin loss 加进总 loss；
- 本轮目标是单独测试 \(F_y\) residual 的效果。

---

## 5. 配点策略

本轮 \(F_y\) loss 的配点先少量加入，观察趋势，不要一开始就大规模增加高阶导数点。

使用和当前 Stage-1 类似的 `chebyshev_half` 配点。配点定义为：

\[
k=0,1,\ldots,N-1,
\]

\[
y_{\rm ref}(k)=3-2\cos\left(\frac{\pi k}{2N}\right),
\]

\[
y(k)=y_{\min}+(y_{\max}-y_{\min})\frac{y_{\rm ref}(k)-1}{2}.
\]

建议第一轮：

```yaml
fy_residual:
  enabled: true
  n_points: 32
  y_min: -0.9999
  y_max: 0.9999
  strategy: chebyshev_half
```

如果显存压力较大，先用：

```yaml
n_points: 16
```

如果趋势有效，再增加到：

```yaml
n_points: 64
```

注意：

1. \(F_y\) 点数先少一点；
2. 先沿用当前全域 `chebyshev_half`；
3. 因为 `chebyshev_half` 本身会偏向 \(y_{\min}\)，若 \(y_{\min}=-0.9999\)，它自然会更关注 near-infinity；
4. 不要直接用随机点作为第一轮 \(F_y\) 点；
5. 后续若需要，可以再做 near-infinity-only band，但本轮先用当前 chebyshev_half 策略做对照。

---

## 6. Loss 形式

本轮总 loss 保持简单，不要叠加过多机制。

基本形式：

\[
L=L_F+w_{F_y}L_{F_y}.
\]

其中：

\[
L_F=\langle |F|^2\rangle,
\]

\[
L_{F_y}=\langle |F_y|^2\rangle.
\]

也可以写成类似 penalty / Lagrange multiplier 的形式：

\[
L = L_F + \lambda_y L_{F_y}.
\]

其中 \(\lambda_y=w_{F_y}\)。

第一轮权重建议做小扫描：

```yaml
fy_residual:
  weight: 1.0e-4
```

然后如果稳定，再测试：

```yaml
weight: 1.0e-3
```

再测试：

```yaml
weight: 1.0e-2
```

不要一开始用太大权重。  
因为 \(F_y\) 涉及三阶导数，其数值尺度可能比 \(F\) 更大，也更容易引入高阶 autograd 噪声。

本轮不加入：

- pybhpt anchor；
- MMA anchor；
- spectral anchor；
- Stage-4 consistency；
- learned \(u\)-decoder；
- additional Robin training loss；
- causal weighting。

---

## 7. 配置文件要求

新增配置文件：

```bash
config/autoencoder_stage1_retrain_fy.yaml
```

可以从当前：

```bash
config/autoencoder_stage1_retrain.yaml
```

复制，然后修改：

```yaml
atlas_training:
  normalize_residual: false

  causal:
    enabled: false

  fy_residual:
    enabled: true
    weight: 1.0e-4
    n_points: 32
    strategy: chebyshev_half
    y_min: -0.9999
    y_max: 0.9999
    implementation: coefficient_autograd
    log_components: true

  infinity_robin_weight: 0.0
  f_anchor_weight: 0.0
```

其他训练配置先尽量保持与当前 retrain 一致，包括：

- batch_size；
- n_interior；
- learning rate；
- validation 设置；
- visualization 设置；
- output_root。

输出目录建议：

```yaml
output_root: outputs/stage1_retrain_fy
```

---

## 8. 代码实现位置

优先新增：

```bash
physical_ansatz/residual_derivative_y.py
```

实现内容：

### 8.1 `chebyshev_half_points`

```python
def chebyshev_half_points(n_points, y_min, y_max, device, dtype):
    ...
```

必须使用：

\[
y_{\rm ref}(k)=3-2\cos\left(\frac{\pi k}{2N}\right)
\]

\[
y(k)=y_{\min}+(y_{\max}-y_{\min})\frac{y_{\rm ref}(k)-1}{2}.
\]

### 8.2 `compute_S_derivatives_order3`

实现：

```python
def compute_S_derivatives_order3(model, a_batch, omega_batch, y_batch, u_batch, v_batch, lambda_batch, ...):
    ...
```

返回：

- `S`
- `S_y`
- `S_yy`
- `S_yyy`

要求：

1. 如果模型输出 \(f_R\)，必须先 compose 成 \(S\)；
2. 不允许把 \(f_R\) 当成 \(S\)；
3. 使用 autograd；
4. 只在少量 \(F_y\) 点上计算；
5. dtype 保持 float64 / complex128；
6. 如果 OOM，应减少 \(F_y\) 点数，不要改数学定义。

### 8.3 `compute_S_reduced_coeffs_y`

实现：

```python
def compute_S_reduced_coeffs_y(a_batch, omega_batch, lambda_batch, y_batch, ...):
    ...
```

返回：

- `D2`
- `D1`
- `D0`

这些必须对应：

\[
F=D_2S_{yy}+D_1S_y+D_0S.
\]

应复用当前已有的：

```python
transform_coeffs_x_to_y_S
```

不要重写一套不一致的系数。

### 8.4 `compute_coeff_derivatives_y`

实现：

```python
def compute_coeff_derivatives_y(D2, D1, D0, y_batch):
    ...
```

返回：

- `D2_y`
- `D1_y`
- `D0_y`

如果 batch 维度导致 autograd 不方便，可以逐 batch sample 计算。  
本轮 \(F_y\) 点数较少，允许牺牲速度换正确性。

### 8.5 `compute_Fy_residual`

实现：

```python
def compute_Fy_residual(S, S_y, S_yy, S_yyy, D2, D1, D0, D2_y, D1_y, D0_y):
    Fy = D2*S_yyy + (D2_y + D1)*S_yy + (D1_y + D0)*S_y + D0_y*S
    return Fy
```

### 8.6 `compute_Fy_loss`

实现：

```python
def compute_Fy_loss(...):
    Fy = compute_Fy_residual(...)
    return mean(abs(Fy)**2)
```

第一轮先用 raw mean loss。  
同时在日志中额外打印一个 normalized diagnostic，但不要一开始把 normalized loss 接入训练，避免同时引入太多变化。

---

## 9. 系数验证脚本

新增：

```bash
scripts/debug_fy_residual_coeffs.py
```

该脚本必须先验证 \(F_y=0\) 系数推导是否正确，然后才能训练。

验证内容：

### 9.1 代数一致性验证

对少量随机参数点和少量 \(y\) 点：

1. 计算：
   \[
   F=D_2S_{yy}+D_1S_y+D_0S.
   \]

2. 用 autograd 直接计算：
   \[
   \frac{dF}{dy}.
   \]

3. 用系数公式计算：
   \[
   F_y^{\rm coeff}
   =
   D_2S_{yyy}
   +(D_{2,y}+D_1)S_{yy}
   +(D_{1,y}+D_0)S_y
   +D_{0,y}S.
   \]

4. 比较：
   \[
   |F_y^{\rm autograd}-F_y^{\rm coeff}|.
   \]

要求最大误差尽量达到 float64 下合理范围，例如：

```text
max_abs_error < 1e-6
relative_error < 1e-5
```

如果达不到，先修系数，不允许训练。

### 9.2 Near-infinity 系数行为验证

在 \(y\to -1\) 附近打印：

- \(D_2\)
- \(D_1\)
- \(D_{2,y}+D_1\)

重点确认：

\[
D_2\to0,
\]

但：

\[
D_{2,y}+D_1
\]

不是同步趋零，而是保持有限量级。  
这说明 \(F_y\) 对 \(S_{yy}\) 确实恢复了 near-infinity 约束。

### 9.3 形状和 dtype 检查

检查：

- shape 是否为 `(B, N_fy)`；
- complex dtype 是否保持；
- 无 NaN；
- 无 Inf；
- \(S,S_y,S_{yy},S_{yyy}\) 都 finite；
- \(D_i,D_{i,y}\) 都 finite。

运行：

```bash
python scripts/debug_fy_residual_coeffs.py \
  --config config/autoencoder_stage1_retrain_fy.yaml \
  --checkpoint outputs/stage1_retrain/saved_best_v3.pt \
  --patch-id 0 \
  --device cuda
```

如果 OOM，先改：

```yaml
fy_residual:
  n_points: 16
```

然后重试。

---

## 10. 接入 trainer

修改：

```bash
trainer/atlas_patch_trainer.py
```

在 `train_one_step()` 中，在普通 PDE loss 计算完成后：

1. 读取：

```python
fy_cfg = self.atlas_train_cfg.get("fy_residual", {})
```

2. 如果：

```python
fy_cfg.get("enabled", False)
```

为 True，则：

- 构造 \(F_y\) 配点；
- 计算 \(L_{F_y}\)；
- 加入总 loss：

\[
L=L_F+w_{F_y}L_{F_y}.
\]

3. 日志中必须记录：

- `loss_fy`
- `weight_fy`
- `loss_pde`
- `loss_total`
- `fy_n_points`
- `fy_max_abs`
- `fy_mean_abs`

本轮总 loss 应尽量保持：

\[
L = L_F + w_{F_y} L_{F_y}.
\]

不要在同一个实验里同时打开 causal、anchor、Stage-4 consistency 或 Robin loss。

---

## 11. 训练实验顺序

### 11.1 Git 状态

运行：

```bash
git branch --show-current
git pull
git status
git log --oneline -5
```

确认当前分支：

```bash
feature/autoencoder-asymptotic-decoder2
```

### 11.2 debug 系数

运行：

```bash
python scripts/debug_fy_residual_coeffs.py \
  --config config/autoencoder_stage1_retrain_fy.yaml \
  --checkpoint outputs/stage1_retrain/saved_best_v3.pt \
  --patch-id 0 \
  --device cuda
```

debug 不通过，不允许训练。

### 11.3 200-step smoke

运行：

```bash
python scripts/retrain_stage1_patch0.py \
  --config config/autoencoder_stage1_retrain_fy.yaml \
  --patch-id 0 \
  --device cuda \
  --steps 200 \
  --resume-checkpoint outputs/stage1_retrain/saved_best_v3.pt
```

报告：

- 初始 \(L_F\)
- 初始 \(L_{F_y}\)
- 200 step 后 \(L_F\)
- 200 step 后 \(L_{F_y}\)
- total loss
- grad norm
- 是否 NaN/Inf
- visualization 是否正常
- near-infinity 曲线是否有变化趋势

### 11.4 2000-step 趋势测试

如果 200-step smoke 稳定，继续：

```bash
python scripts/retrain_stage1_patch0.py \
  --config config/autoencoder_stage1_retrain_fy.yaml \
  --patch-id 0 \
  --device cuda \
  --steps 2000 \
  --resume-checkpoint <200-step-best-checkpoint>
```

报告：

- \(L_F\) 趋势；
- \(L_{F_y}\) 趋势；
- validation 趋势；
- near-infinity 可视化；
- 是否改善 shift；
- 是否出现 PDE loss 爆炸；
- 是否需要调整 \(w_{F_y}\)。

### 11.5 权重扫描

如果 \(w_{F_y}=10^{-4}\) 稳定但效果弱，测试：

```yaml
fy_residual:
  weight: 1.0e-3
```

再测试：

```yaml
fy_residual:
  weight: 1.0e-2
```

每个权重至少跑 200-step smoke。  
不要直接长训。

---

## 12. Causal 策略分析与本轮关闭原因

当前配置中使用了类似 Wang et al. causal PINN 的空间 causal weighting：

```yaml
causal:
  enabled: true
  n_chunks: 8
  epsilon: 0.1
  direction: horizon_first
  y_protect_above: -0.75
```

Wang et al. 的 causal PINN 思路是：对于时间演化问题，后续时间片的 residual 权重依赖于前面时间片 residual 的累计误差，避免模型在早期时间还没学好时去拟合后期。

当前代码把这个思想移植到径向方向。`horizon_first` 的含义是先保护视界/内区，再逐步影响 near-infinity 区域。

这个策略对稳定全局训练有一定合理性，但本轮要专门研究 near-infinity shift。若继续开启 horizon-first causal weighting，near-infinity 的 \(F_y\) residual 可能被削弱，无法判断 \(F_y=0\) 是否有效。

因此，本轮必须关闭 causal：

```yaml
causal:
  enabled: false
```

后续如果 \(F_y\) loss 有效，再考虑重新加入 causal 或设计双端 causal 策略。

---

## 13. 成功与失败判断标准

### 成功信号

1. `debug_fy_residual_coeffs.py` 验证通过；
2. \(L_{F_y}\) 可以下降；
3. \(L_F\) 不爆炸；
4. grad norm 不失控；
5. near-infinity 可视化中 shift / slope 有改善趋势；
6. validation 不明显恶化；
7. 对比原 step20000 best，远端 \(S(y)\) 曲线更稳定。

### 失败信号

1. \(F_y\) 系数验证不通过；
2. 训练立即 OOM；
3. \(S_{yyy}\) 造成严重数值噪声；
4. \(L_F\) 爆炸；
5. \(L_{F_y}\) 不降；
6. near-infinity 曲线更差；
7. 梯度范数异常大；
8. 需要极小 \(w_{F_y}\) 才能稳定，且效果不可见。

如果失败，不要继续长训；先报告失败原因。

---

## 14. 输出与文档

新增实验报告：

```bash
docs/experiments/stage1_fy_residual_experiment.md
```

必须包含：

1. 实验动机；
2. \(F_y=0\) 推导；
3. 系数验证结果；
4. 配点策略；
5. loss 形式；
6. 关闭 causal 的原因；
7. 200-step smoke；
8. 2000-step trend；
9. 权重扫描结果；
10. 可视化路径；
11. 是否建议继续使用 \(F_y\) residual。

---

## 15. 禁止事项

本轮不要做以下事情：

1. 不要开启 causal weighting；
2. 不要加入 pybhpt anchor；
3. 不要加入 MMA anchor；
4. 不要加入 spectral anchor；
5. 不要训练 Stage-2 amplitude；
6. 不要训练 UpDecoder/DownDecoder；
7. 不要开启 Stage-4 consistency；
8. 不要把 exact \(y=-1\) 点用于 \(F_y\) loss；
9. 不要一开始使用大量 \(F_y\) 点；
10. 不要一开始使用大 \(F_y\) 权重；
11. 不要在 debug 系数未通过前训练；
12. 不要新建分支。

---

## 16. 提交要求

提交代码，不提交大型 outputs。

建议提交：

```bash
git add \
  physical_ansatz/residual_derivative_y.py \
  scripts/debug_fy_residual_coeffs.py \
  config/autoencoder_stage1_retrain_fy.yaml \
  trainer/atlas_patch_trainer.py \
  docs/experiments/stage1_fy_residual_experiment.md
```

提交信息：

```bash
git commit -m "Add derivative residual Fy loss for Stage-1 near-infinity refinement"
git push
```

最终报告必须包含：

1. 当前 commit hash；
2. 使用的 checkpoint；
3. 是否关闭 causal；
4. \(F_y\) 系数验证结果；
5. \(F_y\) 配点数量；
6. \(F_y\) 权重；
7. 200-step smoke 结果；
8. 2000-step trend 结果；
9. near-infinity 可视化结论；
10. 是否建议继续该方向。
