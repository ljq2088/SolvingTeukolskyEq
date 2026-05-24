# Agent 指令：修正 \(F_y\) derivative residual 的系数链条，排查 double-prefactor extraction

当前仓库：

`ljq2088/SolvingTeukolskyEq`

当前分支必须是：

`feature/autoencoder-asymptotic-decoder2`

不要切换分支，不要新建分支。

本轮任务不是继续训练，也不是调权重，也不是只给 \(F_y\) 做归一化。  
本轮任务是先严格检查并修正 \(F_y\) derivative residual 的系数链条，确认当前实现是否错误地对 \(P\) 因子提取了两次。

目前诊断显示：

\[
D_2S_{yyy}\sim 0.3,
\]

\[
(D_{2,y}+D_1)S_{yy}\sim 3\times 10^4,
\]

\[
(D_{1,y}+D_0)S_y\sim 10^{10},
\]

\[
D_{0,y}S\sim 10^{13}.
\]

agent 的初步判断是：S ansatz 与三阶导数链条无误，\(D_{0,y}\) 发散来自 \(y\to -1\) 的坐标压缩奇性，因此需要归一化。

现在需要重新审查这个判断。  
当前更可疑的是：**`coeffs_x` 已经返回 \(R=P S\) 后的 \(S(x)\)-方程系数，但 `residual_derivative_y.py` 又调用 `transform_coeffs_x_to_y_S`，再次加入 \(P_x/P\)、\(P_{xx}/P\)，导致 \(P\) 因子被提取了两次。**

如果这个判断成立，那么当前看到的 \(D_0\sim x^{-2}\)、\(D_{0,y}\sim x^{-3}\) 不是物理或坐标奇性的必然后果，而是实现错误导致的伪发散。

---

## 1. 必须先检查的代码位置

重点检查：

```bash
physical_ansatz/teukolsky_coeffs.py
physical_ansatz/transform_y.py
physical_ansatz/residual_derivative_y.py
scripts/debug_fy_residual_coeffs.py
trainer/atlas_patch_trainer.py
```

尤其检查：

```python
physical_ansatz.teukolsky_coeffs.coeffs_x
```

和：

```python
physical_ansatz.transform_y.transform_coeffs_x_to_y_S
```

的输入输出约定是否被混用。

---

## 2. 关键事实：`coeffs_x` 已经是 \(R=P S\) 后的系数

当前 `physical_ansatz/teukolsky_coeffs.py` 中，`coeffs_x` 的文档明确写的是：

\[
R(r)=P(r)S(x).
\]

它返回的是：

\[
A_2S_{xx}+A_1S_x+A_0S=0
\]

中的 \(A_2,A_1,A_0\)。

其公式为：

\[
A_2=\Delta\left(\frac{dx}{dr}\right)^2,
\]

\[
A_1=
\Delta
\left[
2\frac{P_r}{P}\frac{dx}{dr}
+
\frac{d^2x}{dr^2}
\right]
+
(s+1)\Delta_r\frac{dx}{dr},
\]

\[
A_0
=
V
+
(s+1)\Delta_r\frac{P_r}{P}
+
\Delta\frac{P_{rr}}{P}.
\]

这说明：

**`coeffs_x` 已经包含 \(P_r/P\)、\(P_{rr}/P\) 的贡献。**

所以它不是 raw \(R\)-space 系数。

---

## 3. 当前可疑错误：再次调用 `transform_coeffs_x_to_y_S`

当前 `physical_ansatz/residual_derivative_y.py` 中的实现大概率是：

```python
A2, A1, A0 = coeffs_x(...)
D2, D1, D0 = transform_coeffs_x_to_y_S(A2, A1, A0, r, a, omega, ...)
```

这个链条很可疑。

因为 `transform_coeffs_x_to_y_S` 的文档写的是：从

\[
R=P h_2S
\]

的角度，把 \(P\)-factor 继续加入：

\[
D_2=4A_2,
\]

\[
D_1=4A_2\frac{P_x}{P}+2A_1,
\]

\[
D_0=A_2\frac{P_{xx}}{P}+A_1\frac{P_x}{P}+A_0.
\]

如果输入的 \(A_2,A_1,A_0\) 已经来自 `coeffs_x`，也就是已经是 \(R=P S\) 后的 \(S(x)\)-space 系数，那么这里再调用 `transform_coeffs_x_to_y_S` 就会把 \(P\) 提取做第二次。

也就是说，当前实现可能实际对应：

\[
R=P(P S),
\]

而不是：

\[
R=PS.
\]

这会人为制造无穷远处的伪发散。

---

## 4. 正确的 \(x\to y\) 变换应该是什么？

如果已经调用了：

```python
A2, A1, A0 = coeffs_x(...)
```

那么我们已经有：

\[
A_2S_{xx}+A_1S_x+A_0S=0.
\]

现在只需要做坐标变换：

\[
x=\frac{y+1}{2}.
\]

因此：

\[
S_x=2S_y,
\]

\[
S_{xx}=4S_{yy}.
\]

所以正确的 \(y\)-space 方程是：

\[
D_2S_{yy}+D_1S_y+D_0S=0,
\]

其中：

\[
D_2=4A_2,
\]

\[
D_1=2A_1,
\]

\[
D_0=A_0.
\]

没有额外的 \(P_x/P\)，也没有额外的 \(P_{xx}/P\)。

因此本轮需要把 `residual_derivative_y.py` 中的 \(D_i\) 构造修正为：

```python
D2 = 4.0 * A2
D1 = 2.0 * A1
D0 = A0
```

不要在这个链条里调用：

```python
transform_coeffs_x_to_y_S
```

除非你能证明输入的 \(A_2,A_1,A_0\) 是 raw \(R\)-space 系数，而不是 `coeffs_x` 的输出。

---

## 5. 为什么当前 \(D_{0,y}\sim 10^{11}\) 很可能是伪发散？

在无穷远处：

\[
x=\frac{r_+}{r}\to0.
\]

有：

\[
\frac{P_r}{P}\sim i\omega+\frac{n}{r},
\]

而：

\[
\frac{dr}{dx}=-\frac{r_+}{x^2}.
\]

所以：

\[
\frac{P_x}{P}
=
\frac{P_r}{P}\frac{dr}{dx}
\sim
-\frac{i\omega r_+}{x^2}
+O(x^{-1}).
\]

如果错误地在已经提取 \(P\) 的系数上再次加入：

\[
A_1\frac{P_x}{P},
\]

就会人为得到：

\[
D_0^{\rm wrong}\sim x^{-2}.
\]

再对 \(y\) 求导，因为：

\[
x=\frac{y+1}{2},
\]

会得到：

\[
D_{0,y}^{\rm wrong}\sim x^{-3}.
\]

这正好解释当前诊断中：

\[
D_0 \text{ 暴增},
\]

\[
D_{0,y}\sim 10^{11},
\]

\[
D_{0,y}S\sim 10^{13}.
\]

因此，当前诊断结果不能立即解释为“\(F_y=dF/dy\) 原始形式不可用”。  
更可能是当前 \(D_i\) 构造链条有 double \(P\)-factor extraction。

---

## 6. 正确无穷远行为应该是什么？

对于正确的：

\[
u=\frac{R}{P},
\]

或：

\[
S=\frac{R}{P h_2},
\]

reduced 方程在无穷远端应满足：

\[
D_2\to0,
\]

\[
D_1\to -4i\omega r_+,
\]

\[
D_0\to
8r_+\omega^2
-4am\omega
-\lambda
+i\omega(2r_+ +4).
\]

这里 \(D_0\) 是有限复数。

因此：

\[
D_0(y)=D_0^\infty+O(x),
\]

其中：

\[
x=\frac{y+1}{2}.
\]

所以：

\[
D_{0,y}=O(1),
\]

不应该达到 \(10^{11}\)。

如果修正 \(D_i\) 链条后 \(D_{0,y}\) 仍然巨大，再考虑数值 cancellation 或 loss 归一化问题。  
但在修正前，不要把当前 \(D_{0,y}\) 发散当成真实物理结论。

---

## 7. \(S\) ansatz 与三阶导数链条的解析确认

当前 Stage-1 ansatz 是：

\[
S(x)=g(x)\left[h_1(x)f(y)+1\right]+1.
\]

其中：

\[
h_1(x)=e^{x-1}-1,
\]

\[
g(x)=c_H(e^{x-1}-1).
\]

所以：

\[
S(x)=W(x)f(y)+G(x),
\]

其中：

\[
W(x)=g(x)h_1(x),
\]

\[
G(x)=g(x)+1.
\]

由于：

\[
x=\frac{y+1}{2},
\]

有：

\[
S_y
=
\frac12 W_x f
+
W f_y
+
\frac12 G_x,
\]

\[
S_{yy}
=
\frac14 W_{xx}f
+
W_x f_y
+
W f_{yy}
+
\frac14G_{xx},
\]

\[
S_{yyy}
=
\frac18 W_{xxx}f
+
\frac34 W_{xx}f_y
+
\frac32 W_xf_{yy}
+
Wf_{yyy}
+
\frac18G_{xxx}.
\]

这里 \(W,G\) 由指数函数构成，在 \(y=-1\) 和 \(y=1\) 都没有解析奇异性。  
所以：

1. \(g(y)\) 在 \(y\to -1\) 没有奇异性；
2. \(h_1(y)\) 在 \(y\to -1\) 没有奇异性；
3. \(S_{yyy}\) 的固定 ansatz 部分没有解析奇异性；
4. 若 \(S_{yyy}\) 数值很大，来源应是网络 \(f\) 的高阶导、训练状态或错误系数导致的 loss pressure，而不是 \(g,h_1\) 的端点奇性。

因此，本轮重点不是推翻 \(S\) ansatz，而是修正 reduced coefficient 链条。

---

## 8. 需要修改的实现逻辑

### 8.1 修改 `physical_ansatz/residual_derivative_y.py`

在以下函数中检查并修正：

```python
compute_S_reduced_coeffs_y
compute_coeff_derivatives_y
```

如果内部逻辑是：

```python
A2, A1, A0 = coeffs_x(...)
D2, D1, D0 = transform_coeffs_x_to_y_S(...)
```

则改为：

```python
A2, A1, A0 = coeffs_x(...)
D2 = 4.0 * A2
D1 = 2.0 * A1
D0 = A0
```

并加注释：

```python
# coeffs_x already returns the P-extracted S(x)-equation:
#   A2 S_xx + A1 S_x + A0 S = 0
# Therefore x=(y+1)/2 only gives:
#   D2=4*A2, D1=2*A1, D0=A0
# Do NOT call transform_coeffs_x_to_y_S here, otherwise P is extracted twice.
```

### 8.2 保留 `transform_coeffs_x_to_y_S`，但不要在这里用

不要删除 `transform_coeffs_x_to_y_S`。  
它可能在其他地方有用。

但在本轮 \(F_y\) derivative residual 里，只要输入来自 `coeffs_x`，就不能再调用它。

---

## 9. 重新验证 \(F_y\) 系数

修改后重新运行：

```bash
python scripts/debug_fy_residual_coeffs.py \
  --config config/autoencoder_stage1_retrain_fy.yaml \
  --checkpoint outputs/stage1_retrain/saved_best_v3.pt \
  --patch-id 0 \
  --device cuda
```

但 debug 脚本也需要修正：  
验证 \(F_y^{\rm autograd}=dF/dy\) 时，必须保证 \(S_{yy}\) 对 \(y\) 的计算图没有被截断。

也就是说 debug 中计算 \(S_{yy}\) 时应使用：

```python
create_graph=True
```

否则直接对：

\[
F=D_2S_{yy}+D_1S_y+D_0S
\]

求导时，autograd 不会包含完整的 \(D_2S_{yyy}\) 项。

修正后验证：

\[
F_y^{\rm coeff}
=
D_2S_{yyy}
+
(D_{2,y}+D_1)S_{yy}
+
(D_{1,y}+D_0)S_y
+
D_{0,y}S
\]

与：

\[
F_y^{\rm autograd}
=
\frac{d}{dy}
\left(
D_2S_{yy}+D_1S_y+D_0S
\right)
\]

一致。

要求：

```text
max_abs_error < 1e-6
relative_error < 1e-5
```

如果因为尺度较大无法达到该阈值，应报告 max/relative error，并说明是否由浮点尺度导致。

---

## 10. 重新打印 near-infinity 四项量级

修正 \(D_i\) 后，在 \(y=-0.9999\) 附近重新打印：

\[
D_2S_{yyy},
\]

\[
(D_{2,y}+D_1)S_{yy},
\]

\[
(D_{1,y}+D_0)S_y,
\]

\[
D_{0,y}S.
\]

同时打印：

\[
D_2,\quad D_1,\quad D_0,\quad D_{2,y},\quad D_{1,y},\quad D_{0,y}.
\]

重点检查：

1. \(D_2\to0\)；
2. \(D_1\) 有限；
3. \(D_0\) 有限；
4. \(D_{0,y}\) 不再出现 \(10^{11}\) 这类伪发散；
5. \((D_{2,y}+D_1)S_{yy}\) 是否仍能提供目标曲率约束；
6. \(F_y\) 四项是否仍有巨大尺度不平衡。

如果修正后 \(D_{0,y}S\) 仍然主导，再讨论 term-wise normalization。  
在修正前，不要继续调 \(F_y\) 权重。

---

## 11. 训练图问题：先诊断，暂不强行训练

当前 `compute_S_derivatives_order3` 中如果使用：

```python
Syyy = torch.autograd.grad(..., create_graph=False)
```

那么：

- 可以用于数值诊断；
- 但不能代表严格的 \(F_y\)-PINN 训练；
- 因为 \(S_{yyy}\) 对网络参数的梯度被截断。

本轮先完成系数链条修正和 debug 验证。  
修正前不要继续正式训练。

修正后如果要训练，需要决定：

1. 是否把 \(S_{yyy}\) 的 `create_graph=True` 打开；
2. 是否改用有限差分 \(D_yF\)，避免显式三阶 autograd；
3. 是否只用 near-infinity 主导项做近似实验。

但这些是下一步。  
本轮重点是修正 double \(P\)-factor extraction。

---

## 12. 可选：更一致的 \(f\)-space 方案

当前主训练 residual 实际可能是 \(f\)-space：

\[
F_f=B_2f_{yy}+B_1f_y+B_0f-rhs=0.
\]

如果后续发现 \(S\)-space 的 \(F_y\) 仍然难以稳定训练，更推荐直接做：

\[
(F_f)_y=0.
\]

即：

\[
(F_f)_y
=
B_2f_{yyy}
+
(B_{2,y}+B_1)f_{yy}
+
(B_{1,y}+B_0)f_y
+
B_{0,y}f
-
rhs_y
=0.
\]

这个版本与当前主 loss 完全一致，也能避免在 \(S\)-space 中重复转换或混淆 \(P\)-factor。

但本轮先不要实现这个方案。  
先修正当前 \(S\)-space \(D_i\) 构造并验证。

---

## 13. 成功判断标准

修正后需要满足：

1. `compute_S_reduced_coeffs_y` 不再 double extract \(P\)；
2. `compute_coeff_derivatives_y` 使用同一套正确 \(D_i\)；
3. debug 中 \(F_y^{\rm coeff}\) 与 \(F_y^{\rm autograd}\) 一致；
4. near-infinity 处 \(D_0\) 有限；
5. near-infinity 处 \(D_{0,y}\) 不再出现 \(10^{11}\) 级别伪发散；
6. 重新打印四项后，再判断是否需要归一化；
7. 不在修正前继续训练。

---

## 14. 提交要求

提交代码，不提交大型 outputs。

建议提交：

```bash
git add \
  physical_ansatz/residual_derivative_y.py \
  scripts/debug_fy_residual_coeffs.py \
  docs/experiments/stage1_fy_residual_experiment.md
```

提交信息：

```bash
git commit -m "Fix Fy residual coefficient chain to avoid double prefactor extraction"
git push
```

最终报告必须包含：

1. 当前 commit hash；
2. 是否确认 `coeffs_x` 已经是 \(R=P S\) 后的 \(S(x)\)-方程；
3. 是否移除了 `transform_coeffs_x_to_y_S` 在 \(F_y\) 链条中的误用；
4. 修正后的 \(D_2,D_1,D_0,D_{2,y},D_{1,y},D_{0,y}\) 量级；
5. 修正后的四项量级：
   - \(D_2S_{yyy}\)
   - \((D_{2,y}+D_1)S_{yy}\)
   - \((D_{1,y}+D_0)S_y\)
   - \(D_{0,y}S\)
6. \(F_y^{\rm coeff}\) 与 \(dF/dy\) autograd 验证结果；
7. 是否仍需要归一化；
8. 是否建议继续训练。
