# Stage-1 主模型继续训练：加入精确无穷远端 \(y=-1\) Robin 约束

当前仓库：

`ljq2088/SolvingTeukolskyEq`

当前分支必须是：

`feature/autoencoder-asymptotic-decoder2`

不要切换分支，不要新建分支。

本轮任务不是继续 Stage-3 learned \(u_{\uparrow},u_{\downarrow}\) decoder，也不是 Stage-4 spectral consistency polishing，也不是加入 pybhpt/MMA/spectral anchor。当前任务是：

**在 Stage-1 主 PINN 模型上，加入精确无穷远端 \(y=-1\) 的解析 Robin 正则性 loss，然后继续训练 Stage-1 主模型，使训练后期更加关注无穷远端的斜率/截距校正。**

本轮必须优先阅读并使用：

`docs/instructions/inf_coeff.md`

该文档给出了 \(R=P u\) 后 reduced equation 在无穷远端的解析系数和 Robin 条件。不要再使用之前数值外推得到的发散 \(c_{\infty}\)。之前 \(c_{\infty}\sim x^{-2}\) 的诊断是错误实现导致的，原因是漏掉了 \(P_x/P\)、\(P_{xx}/P\) 的抵消项或误用了 raw \(R\)-space \(A_0\)。

---

## 1. 核心数学约束

Stage-1 使用的基本形式是：

\[
R_{\rm in}=P h_2 S(y).
\]

因此：

\[
U(y)=\frac{R_{\rm in}}{P}=h_2S(y).
\]

因为 \(h_2\) 与 \(y\) 无关，所以 \(U\) 与 \(S\) 满足相同的无穷远 Robin 比例条件。

根据 `docs/instructions/inf_coeff.md`，对于当前主要情形 \(s=-2\)、\(M=1\)、固定非零实频率 \(\omega\)，提取 \(R=P u\) 后的 reduced equation 在 \(y=-1\) 处满足：

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

因此无穷远端 Robin 条件为：

\[
u_y(-1)=c_{\infty}u(-1),
\]

其中

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

由于 \(U=h_2S\)，所以对 Stage-1 的 reduced shape 也有：

\[
S_y(-1)=c_{\infty}S(-1).
\]

本轮要加入的 loss 是：

\[
\mathcal{L}_{\infty}
=
\frac{
\left|S_y(-1)-c_{\infty}S(-1)\right|^2
}{
|S_y(-1)|^2+|c_{\infty}S(-1)|^2+\epsilon
}.
\]

注意：

1. 这个约束是无监督物理约束；
2. 不依赖 pybhpt；
3. 不依赖 Mathematica；
4. 不依赖谱方法；
5. 不要把 \(S(-1)\) 固定成 1；
6. 约束的是导数-函数值关系，而不是函数值本身；
7. 必须精确在 \(y=-1\) 计算；
8. 不要用 \(y=-1+\epsilon\) 替代该端点 loss；
9. 不要在 \(y=-1\) 直接计算完整二阶 residual；
10. 完整二阶 residual 仍然只在内点 \(y>-1\) 计算。

---

## 2. 当前训练范式

如果本轮有效，项目主线将从旧的四阶段方案转向：

### 主体：纯无监督 PINN

- 视界端 hard boundary；
- 内点 Teukolsky residual；
- 精确无穷远端 \(y=-1\) Robin loss；
- near-infinity weighted residual；
- 可选 radial curriculum。

### 辅助：振幅网络

- AmplitudeNet 可保留；
- 后续可冻结主 PINN，仅训练振幅网络；
- 振幅网络不参与本轮无穷远 Robin 训练；
- 不再依赖 learned \(u_{\uparrow},u_{\downarrow}\) decoder 作为主线。

因此本轮训练的是 **Stage-1 主体**，也就是：

- shared encoder；
- RinDecoder；

不要训练：

- AmplitudeNet；
- UpDecoder；
- DownDecoder。

如果当前代码中 Stage-1 checkpoint 只包含原始 PINN，没有这些模块，就按原 Stage-1 全模型训练即可。若使用 `AutoencoderTeukolskyPINN`，则本轮 optimizer 只应包含 `encoder` 和 `rin_decoder`。

用户说“全都解冻”的含义是：**Stage-1 主体全部解冻**。不要把与 Stage-1 PDE 无关的 amplitude/up/down heads 加入 optimizer。

---

## 3. Git 状态检查

先运行：

```bash
git branch --show-current
git pull
git status
git log --oneline -5