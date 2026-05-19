# Stage-1 后期无穷远校正训练：Exact Infinity Robin + Full-batch / L-BFGS Refinement

当前仓库：

`ljq2088/SolvingTeukolskyEq`

当前分支必须是：

`feature/autoencoder-asymptotic-decoder2`

不要切换分支，不要新建分支。

本轮任务是在当前 Stage-1 主模型基础上，测试一种新的纯无监督 PINN 后期校正范式：

1. 先保留/加载已经训练好的 Stage-1 主模型；
2. 加入精确无穷远端 \(y=-1\) 的解析 Robin loss；
3. 用 Adam/AdamW 做短程适应性 fine-tune；
4. 再用固定配点的 full-batch / L-BFGS 做 near-infinity shift 校正；
5. 重点观察无穷远端的 slope / shift 是否改善。

本轮不继续 Stage-3 learned \(u_{\uparrow},u_{\downarrow}\) decoder，不做 Stage-4 spectral consistency polishing，不加入 pybhpt/MMA/spectral anchor。

---

## 0. 重要前提：必须读取解析系数文档

用户已经把无穷远端 reduced coefficient 的解析推导放在：

`docs/instructions/inf_coeff.md`

本轮必须先阅读这个文件。

如果该文件在本地不存在，请立即停止并报告：

```bash
ls docs/instructions/inf_coeff.md
```

不要再使用此前数值外推得到的发散 \(c_{\infty}\)。此前 \(c_{\infty}\sim x^{-2}\) 的结果是错误实现导致的，原因是漏掉了 \(P_x/P\)、\(P_{xx}/P\) 抵消项或误用了 raw \(R\)-space 的 \(A_0\)。

---

## 1. 当前 Git 状态检查

先运行：

```bash
git branch --show-current
git pull
git status
git log --oneline -5
```

必须确认当前分支是：

```bash
feature/autoencoder-asymptotic-decoder2
```

如果工作区不干净，先报告，不要覆盖已有改动。

---

## 2. 本轮核心数学约束

Stage-1 主模型采用：

\[
R_{\rm in}=P h_2 S(y).
\]

因此：

\[
U(y)=\frac{R_{\rm in}}{P}=h_2 S(y).
\]

因为 \(h_2\) 与 \(y\) 无关，所以 \(U\) 与 \(S\) 满足相同的无穷远 Robin 条件。

根据 `docs/instructions/inf_coeff.md`，对于当前主要情形 \(s=-2\)、\(M=1\)、固定非零实频 \(\omega\)，提取 \(R=P u\) 后的 reduced equation 在 \(y=-1\) 处满足：

\[
A_2(-1)=0,
\]

\[
A_1(-1)=-4i\omega r_+,
\]

\[
A_0(-1)=
8r_+\omega^2
-4am\omega
-\lambda
+i\omega(2r_+ +4).
\]

因此：

\[
u_y(-1)=c_{\infty}u(-1),
\]

其中：

\[
c_{\infty}
=
\frac12+\frac{1}{r_+}
-
i\frac{
8r_+\omega^2
-4am\omega
-\lambda
}{
4\omega r_+
}.
\]

因为 \(U=h_2S\)，所以 Stage-1 的 reduced shape 满足：

\[
S_y(-1)=c_{\infty}S(-1).
\]

本轮加入的 endpoint loss 是：

\[
\mathcal{L}_{\infty}
=
\frac{
|S_y(-1)-c_{\infty}S(-1)|^2
}{
|S_y(-1)|^2+|c_{\infty}S(-1)|^2+\epsilon
}.
\]

注意：

1. 这是无监督物理约束；
2. 不依赖 pybhpt；
3. 不依赖 MMA；
4. 不依赖谱方法；
5. 不要把 \(S(-1)\) 固定成 1；
6. 约束的是导数-函数值关系；
7. 必须精确在 \(y=-1\) 计算；
8. 不要用 \(y=-1+\epsilon\) 代替 endpoint Robin loss；
9. 不要在 \(y=-1\) 计算完整二阶 residual；
10. 完整 PDE residual 只在 \(y>-1\) 的内点计算。

---

## 3. 先检查当前实现是否已有 infinity Robin 相关文件

运行：

```bash
find . -path '*infinity*' -o -path '*robin*'
grep -R "infinity_robin" -n physical_ansatz scripts config trainer model docs || true
grep -R "c_inf" -n physical_ansatz scripts config trainer model docs || true
```

如果当前仓库已经存在以下文件，请先阅读并复用，不要重复造轮子：

- `physical_ansatz/infinity_robin.py`
- `physical_ansatz/stage1_endpoint.py`
- `scripts/diagnose_infinity_robin_stage1.py`
- `scripts/debug_stage1_infinity_robin.py`
- `scripts/train_stage1_with_infinity_robin.py`
- `config/autoencoder_stage1_infinity_robin.yaml`

如果这些文件不存在，再按本文档新增。

---

## 4. 必须阅读的项目文件

阅读：

```bash
docs/instructions/inf_coeff.md
physical_ansatz/transform_y.py
physical_ansatz/prefactor.py
physical_ansatz/residual.py
physical_ansatz/teukolsky_coeffs.py
physical_ansatz/stage1_reconstruction.py
model/autoencoder_pinn.py
trainer/atlas_patch_trainer.py
```

重点确认：

1. RinDecoder 输出的是自由函数 \(f_R(y)\)，不是 \(S(y)\)，更不是 \(R/P\)；
2. `compose_reduced_shape_from_f(f, y, slope)` 如何从 \(f_R(y)\) 构造 \(S(y)\)；
3. `horizon_regularity_slope` 如何处理视界端 hard boundary；
4. Stage-1 residual 当前如何计算；
5. `transform_coeffs_x_to_y_S` 是否包含提取 \(P\) 后的 \(P_x/P\)、\(P_{xx}/P\) 项；
6. 当前 checkpoint 中 encoder / RinDecoder / AmplitudeNet / UpDecoder / DownDecoder 的命名；
7. 当前训练脚本如何采样参数与 \(y\)。

---

## 5. 实现或修正解析 infinity Robin 模块

新增或修正：

```bash
physical_ansatz/infinity_robin.py
```

实现：

```python
def infinity_robin_slope_sminus2(
    a,
    omega,
    lambda_,
    m: int = 2,
    M: float = 1.0,
):
    ...
```

返回：

\[
c_{\infty}
=
\frac12+\frac{1}{r_+}
-
i\frac{
8r_+\omega^2
-4am\omega
-\lambda
}{
4\omega r_+
},
\]

其中：

\[
r_+=M+\sqrt{M^2-a^2}.
\]

要求：

1. 当前只支持 \(s=-2\)，不要默默推广到其他 \(s\)；
2. 当前默认 \(M=1\)；
3. 必须检查 \(\omega\neq0\)；
4. 若 \(|\omega|\) 太小，给出 warning 或 raise error；
5. 支持 torch tensor batch；
6. 支持 complex128；
7. 不要丢掉虚部；
8. 注释中明确写：公式来自 `docs/instructions/inf_coeff.md`。

还要实现：

```python
def infinity_robin_residual(S, Sy, c_inf):
    return Sy - c_inf * S
```

以及：

```python
def infinity_robin_loss(S, Sy, c_inf, eps=1e-12):
    res = Sy - c_inf * S
    denom = abs(Sy)**2 + abs(c_inf * S)**2 + eps
    return mean(abs(res)**2 / denom)
```

---

## 6. 实现精确 \(y=-1\) 的 Stage-1 endpoint helper

新增或修正：

```bash
physical_ansatz/stage1_endpoint.py
```

实现：

```python
def compute_stage1_S_and_Sy_at_infinity(
    model,
    a,
    omega,
    u,
    v,
    lambda_,
    m: int = 2,
    M: float = 1.0,
    ...
):
    ...
```

逻辑：

1. 构造精确点：

   \[
   y_0=-1.
   \]

2. 设置：

   ```python
   y0.requires_grad_(True)
   ```

3. 调用模型 encoder + RinDecoder，得到 \(f_R(y_0)\)。

4. 使用当前 Stage-1 ansatz 构造 \(S(y_0)\)。

   注意：不要把 `predict_Rin()` 直接当成 \(S\)。  
   如果 `predict_Rin()` 返回的是 \(f_R(y)\)，必须继续调用 `compose_reduced_shape_from_f`。

5. 用 autograd 计算：

   \[
   S_y(-1).
   \]

6. 计算：

   \[
   c_{\infty}.
   \]

7. 返回：

   - `S_inf`
   - `Sy_inf`
   - `c_inf`
   - `robin_residual`
   - `robin_rel`

要求：

- 必须精确在 \(y=-1\)；
- 不要用 \(y=-1+\epsilon\)；
- 不要计算完整二阶 residual；
- 所有输出必须 finite；
- 不允许 silent cast 到 real dtype；
- shape 不匹配时直接报错。

---

## 7. 诊断脚本：先不训练

新增或修正：

```bash
scripts/diagnose_infinity_robin_stage1.py
```

功能：

1. 加载当前 Stage-1 best checkpoint；
2. 读取 patch 0 的参数点；
3. 对每个参数点精确计算：

   - \(S(-1)\)
   - \(S_y(-1)\)
   - \(c_{\infty}\)
   - \(B_{\infty}=S_y(-1)-c_{\infty}S(-1)\)
   - relative Robin residual

4. 在 \(y=-1+\epsilon\) 附近计算普通 PDE residual 作为参考；
5. 输出表格和 summary。

运行示例：

```bash
python scripts/diagnose_infinity_robin_stage1.py \
  --checkpoint outputs/stage1_artifacts/patch_000_logw_v2/stage1_best_model.pt \
  --device cuda
```

如果 checkpoint 路径不存在，自动寻找最新 Stage-1 best checkpoint，但必须在报告中说明实际使用的路径。

输出目录：

```bash
outputs/infinity_robin_diagnostics/<timestamp>/
```

保存：

```bash
diagnostics.json
diagnostics_summary.md
robin_residual_table.csv
```

报告必须包含：

- median robin_rel；
- max robin_rel；
- per-point robin_rel；
- \(c_\infty\) 的数值范围；
- 是否出现 NaN/Inf；
- \(S(-1)\)、\(S_y(-1)\) 的量级；
- near-infinity PDE residual 参考值。

---

## 8. 新增训练配置：Adam + full-batch / L-BFGS 两阶段 refinement

新增或修正：

```bash
config/autoencoder_stage1_infinity_robin.yaml
```

建议配置：

```yaml
stage1_infinity_robin:
  base_checkpoint: outputs/stage1_artifacts/patch_000_logw_v2/stage1_best_model.pt
  output_root: outputs/stage1_infinity_robin

  train_encoder: true
  train_rin_decoder: true
  train_amplitude_net: false
  train_up_decoder: false
  train_down_decoder: false

  adam:
    enabled: true
    lr_encoder: 2.0e-7
    lr_rin_decoder: 1.0e-6
    weight_decay: 0.0
    epochs: 50
    batch_size: 8
    grad_clip: 0.1

  lbfgs:
    enabled: true
    max_epochs: 50
    max_iter_per_epoch: 20
    history_size: 50
    line_search_fn: strong_wolfe
    lr: 0.5
    fixed_batch_size: 16
    restart_every: 10
    grad_clip: 0.1

  loss:
    weight_pde: 1.0
    weight_inf_robin: 1.0
    weight_near_pde: 2.0
    weight_drift: 0.1
    eps: 1.0e-12

  sampling:
    near_infinity:
      y_min: -0.999
      y_max: -0.95
      n_y: 96
    transition:
      y_min: -0.95
      y_max: 0.0
      n_y: 48
    horizon_side:
      y_min: 0.0
      y_max: 0.999
      n_y: 32

runtime:
  dtype: float64
```

解释：

1. 本轮允许 Stage-1 主体全解冻，即 encoder + RinDecoder；
2. encoder 学习率必须比 RinDecoder 小；
3. AmplitudeNet 不训练；
4. UpDecoder / DownDecoder 不训练；
5. Adam 阶段允许随机配点；
6. L-BFGS 阶段必须使用固定 collocation 点集；
7. L-BFGS closure 内禁止重新随机采样；
8. 若要随机配点，只能在 L-BFGS restart 之间重新生成固定点集。

---

## 9. 训练脚本

新增或修正：

```bash
scripts/train_stage1_with_infinity_robin.py
```

训练总 loss：

\[
\mathcal{L}_{\rm total}
=
w_{\rm pde}\mathcal{L}_{\rm pde}
+
w_{\infty}\mathcal{L}_{\infty}
+
w_{\rm near}\mathcal{L}_{\rm near}
+
w_{\rm drift}\mathcal{L}_{\rm drift}.
\]

### 9.1 PDE loss

复用原 Stage-1 的 Teukolsky residual，不要重写方程。

采样区域：

- near-infinity；
- transition；
- horizon-side。

### 9.2 Infinity Robin loss

精确在 \(y=-1\) 计算：

\[
\mathcal{L}_{\infty}
=
\frac{
|S_y(-1)-c_{\infty}S(-1)|^2
}{
|S_y(-1)|^2+|c_{\infty}S(-1)|^2+\epsilon
}.
\]

### 9.3 Near-infinity PDE loss

从 PDE loss 中单独统计 near-infinity 区域：

\[
y\in[-0.999,-0.95].
\]

这个区域可额外加权：

\[
w_{\rm near}\mathcal{L}_{\rm near}.
\]

注意：near-infinity PDE residual 仍然只能在 \(y>-1\) 的内点计算，不要在 \(y=-1\) 计算完整二阶 residual。

### 9.4 Drift loss

使用原 Stage-1 frozen copy 作为参考：

\[
\mathcal{L}_{\rm drift}
=
\frac{
\langle |S_{\rm new}-S_{\rm old}|^2\rangle
}{
\langle |S_{\rm old}|^2\rangle+\epsilon
}.
\]

也可以使用 \(R/P=h_2S\)，但 before/after 必须一致。

---

## 10. 优化策略

### 10.1 Adam fine-tune

目的：让模型适应新的 exact infinity Robin loss。

特点：

- 可以随机配点；
- 每个 epoch 可以重新采样；
- 学习率小；
- 先跑 20 epoch smoke；
- 若有效，再跑到 50 epoch。

### 10.2 Full-batch / L-BFGS refinement

目的：校正后期低频全局 shift / 截距偏移 / 慢收敛模。

关键要求：

1. L-BFGS closure 内必须使用固定点集；
2. closure 内不能随机配点；
3. 每个 L-BFGS restart 之间可以重新生成一次固定点集；
4. 使用 float64；
5. optimizer 只包含 encoder 和 RinDecoder；
6. AmplitudeNet / UpDecoder / DownDecoder 必须保持不变；
7. 若全解冻不稳定，先退回只训练 RinDecoder；
8. 每个 restart 后记录 loss 分量和 drift。

L-BFGS 不应作为替代物，而是后期 refinement：

\[
\text{Stage-1 checkpoint}
\rightarrow
\text{Adam + infinity Robin}
\rightarrow
\text{fixed-grid L-BFGS}
\rightarrow
\text{optional restart with new fixed grid}.
\]

---

## 11. Debug 脚本

新增或修正：

```bash
scripts/debug_stage1_infinity_robin.py
```

运行：

```bash
python scripts/debug_stage1_infinity_robin.py \
  --config config/autoencoder_stage1_infinity_robin.yaml \
  --device cuda
```

必须检查：

1. checkpoint 可加载；
2. `docs/instructions/inf_coeff.md` 存在；
3. \(c_\infty\) finite；
4. \(y=-1\) 精确点 forward 正常；
5. \(S(-1)\) finite；
6. \(S_y(-1)\) finite；
7. Robin loss finite；
8. PDE loss finite；
9. near-infinity PDE loss finite；
10. drift loss finite；
11. backward 后 encoder 和 RinDecoder 有梯度；
12. AmplitudeNet 无梯度；
13. UpDecoder / DownDecoder 无梯度；
14. optimizer.step 后只有 encoder / RinDecoder 参数改变；
15. L-BFGS closure 在固定点集上可重复评估；
16. closure 连续两次评估不应因重新采样而改变采样点；
17. dtype 为 float64/complex128；
18. 无 NaN/Inf。

---

## 12. 训练实验顺序

### Step 1: Preflight

```bash
python scripts/diagnose_infinity_robin_stage1.py \
  --checkpoint outputs/stage1_artifacts/patch_000_logw_v2/stage1_best_model.pt \
  --device cuda
```

记录初始 Robin residual。

### Step 2: Debug

```bash
python scripts/debug_stage1_infinity_robin.py \
  --config config/autoencoder_stage1_infinity_robin.yaml \
  --device cuda
```

debug 不通过，不允许训练。

### Step 3: 20 epoch Adam smoke

```bash
python scripts/train_stage1_with_infinity_robin.py \
  --config config/autoencoder_stage1_infinity_robin.yaml \
  --device cuda \
  --phase adam \
  --epochs 20 \
  --verbose
```

报告：

- initial \(L_{\rm pde}\)
- initial \(L_{\infty}\)
- initial \(L_{\rm near}\)
- initial \(L_{\rm drift}\)
- epoch 20 \(L_{\rm pde}\)
- epoch 20 \(L_{\infty}\)
- epoch 20 \(L_{\rm near}\)
- epoch 20 \(L_{\rm drift}\)
- Robin residual before/after
- near-infinity PDE residual before/after
- \(S(-1)\) before/after
- \(S_y(-1)\) before/after
- parameter drift
- NaN/Inf 检查
- best checkpoint path

若 \(L_{\infty}\) 完全不降或 PDE loss 爆炸，不要继续。

### Step 4: 50 epoch Adam fine-tune

如果 20 epoch 有效：

```bash
python scripts/train_stage1_with_infinity_robin.py \
  --config config/autoencoder_stage1_infinity_robin.yaml \
  --device cuda \
  --phase adam \
  --epochs 50 \
  --resume-checkpoint <20epoch_best> \
  --verbose
```

### Step 5: L-BFGS refinement smoke

Adam fine-tune 有效后，运行短 L-BFGS：

```bash
python scripts/train_stage1_with_infinity_robin.py \
  --config config/autoencoder_stage1_infinity_robin.yaml \
  --device cuda \
  --phase lbfgs \
  --epochs 5 \
  --resume-checkpoint <adam_best> \
  --verbose
```

要求：

- 固定点集；
- closure 内不重新采样；
- 每个 epoch 报告 loss 分量；
- 若 NaN/Inf，立即停止；
- 若 drift 过大，立即停止；
- 若 loss 不降，停止并报告。

### Step 6: L-BFGS restart refinement

如果 5 epoch L-BFGS 有效，再运行：

```bash
python scripts/train_stage1_with_infinity_robin.py \
  --config config/autoencoder_stage1_infinity_robin.yaml \
  --device cuda \
  --phase lbfgs \
  --epochs 50 \
  --resume-checkpoint <lbfgs_smoke_best> \
  --verbose
```

每 `restart_every=10` epoch 允许重新生成一次固定点集；但每个 closure 内仍然必须固定。

---

## 13. 可视化脚本

新增或修正：

```bash
scripts/plot_stage1_infinity_robin_effect.py
```

画同一批参数点 before/after：

1. \(\mathrm{Re}(R/P)\)
2. \(\mathrm{Im}(R/P)\)
3. \(|R/P|\)
4. \(\arg(R/P)\)
5. Robin residual per point
6. near-infinity PDE residual per \(y\)
7. \(S_y(-1)\) vs \(c_{\infty}S(-1)\)

对比对象：

- 原 Stage-1 checkpoint；
- Adam fine-tune 后 checkpoint；
- L-BFGS 后 checkpoint；
- pybhpt/MMA 曲线只能作为 optional diagnostic，不参与 loss。

输出目录：

```bash
outputs/stage1_infinity_robin_compare/<timestamp>/
```

保存：

```bash
comparison_summary.md
metrics.json
plots/*.png
```

评价：

1. Robin 条件是否改善；
2. 近无穷远端斜率是否改善；
3. 原本“趋势一致但截距偏移”的问题是否改善；
4. PDE residual 是否保持可控；
5. 是否破坏视界端行为；
6. Adam 与 L-BFGS 哪个阶段带来主要改善。

---

## 14. 如果有效，记录新范式

如果 exact infinity Robin + Adam/L-BFGS 后期 refinement 明显改善远端 shift，请新增或更新：

```bash
docs/pinn_with_infinity_robin_paradigm.md
```

写明新范式：

1. 主体是纯无监督 PINN：
   - horizon hard boundary；
   - exact infinity Robin endpoint constraint；
   - interior PDE residual；
   - near-infinity weighted residual；
   - optional full-batch / L-BFGS refinement；
   - optional radial curriculum。

2. AmplitudeNet 只是辅助模块：
   - 可以保留共享参数特征；
   - 后续冻结主 PINN 后训练 AmplitudeNet；
   - 振幅网络不再驱动主模型训练。

3. learned \(u_{\uparrow},u_{\downarrow}\) decoder 暂时不是主线。

4. Stage-4 spectral consistency polishing 暂时不是主线。

---

## 15. 禁止事项

本轮不要做以下事情：

1. 不要继续 Stage-4 spectral consistency polishing；
2. 不要训练 learned \(u_{\uparrow},u_{\downarrow}\) decoders；
3. 不要加入 pybhpt anchor loss；
4. 不要加入 MMA anchor loss；
5. 不要加入 spectral anchor loss；
6. 不要把 \(S(-1)\) 固定成 1；
7. 不要把 \(y=-1+\epsilon\) 当成 infinity Robin；
8. 不要在 \(y=-1\) 计算完整二阶 residual；
9. 不要把 AmplitudeNet 加入 optimizer；
10. 不要把 UpDecoder / DownDecoder 加入 optimizer；
11. 不要在 L-BFGS closure 内随机配点；
12. 不要新建分支；
13. 不要重构整个模型。

---

## 16. 提交要求

提交代码，不提交大型 outputs。

建议提交：

```bash
git add \
  physical_ansatz/infinity_robin.py \
  physical_ansatz/stage1_endpoint.py \
  scripts/diagnose_infinity_robin_stage1.py \
  scripts/debug_stage1_infinity_robin.py \
  scripts/train_stage1_with_infinity_robin.py \
  scripts/plot_stage1_infinity_robin_effect.py \
  config/autoencoder_stage1_infinity_robin.yaml
```

如果实验有效，再提交：

```bash
git add docs/pinn_with_infinity_robin_paradigm.md
```

提交信息：

```bash
git commit -m "Add infinity Robin Adam and LBFGS refinement for Stage-1 PINN"
git push
```

最终报告必须包含：

1. 当前 commit hash；
2. 是否读取并使用 `docs/instructions/inf_coeff.md`；
3. 当前分支确认；
4. \(c_{\infty}\) 实现位置；
5. preflight Robin residual；
6. debug 结果；
7. 20 epoch Adam smoke 结果；
8. 50 epoch Adam fine-tune 结果，如果运行；
9. L-BFGS smoke 结果，如果运行；
10. L-BFGS restart 结果，如果运行；
11. before/after 可视化结论；
12. 是否改善 near-infinity shift；
13. 是否支持新范式：
    纯无监督 PINN + exact infinity Robin + full-batch/L-BFGS 后期校正 + AmplitudeNet 辅助训练。
