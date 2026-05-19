# \(R=P u\) 后的 Teukolsky 方程与无穷远端 Robin 系数推导

本文档用于澄清一个关键问题：  
把

\[
R=P u
\]

代入 Teukolsky 径向方程后，\(u=R/P\) 所满足的 reduced equation 在无穷远端 \(y=-1\) 是否给出有限的 Robin 条件。

结论先行：

**对固定非零 \(\omega\)，正确提取 \(P\) 因子后的 \(u\)-方程在无穷远端有有限的 \(A_0/A_1\)。如果数值诊断得到 \(A_0/A_1\sim x^{-2}\) 发散，大概率说明代码没有完整加入 \(P_x/P\)、\(P_{xx}/P\) 的抵消项，或者误用了未提取 \(P\) 前的 raw \(R\)-space 零阶系数。**

---

## 1. 当前问题背景

agent 此前诊断得到：

\[
D_2\sim x^2\to 0,
\]

\[
D_1\to \text{finite},
\]

\[
D_0\sim x^{-2},
\]

因此

\[
c_{\infty}=-D_0/D_1\sim x^{-2}.
\]

这个结果被解释为：Leaver \(P\)-factor 无法同时捕获入射波和反射波，因此 \(S=R/P\) 的无穷远端 Robin 条件退化成 \(S(-1)=0\)。

这个解释需要重新检查。

如果我们讨论的是 clean reduced variable

\[
u=\frac{R}{P},
\]

并且 \(P\) 的 log-derivative 项完整进入方程，那么 \(D_0\) 的 leading \(x^{-2}\) 发散应被抵消，不应该保留。

---

## 2. 原始径向 Teukolsky 方程

当前代码使用的 homogeneous radial Teukolsky equation 可写成：

\[
\Delta R_{rr}+(s+1)\Delta_r R_r+V R=0.
\]

其中

\[
\Delta=r^2-2Mr+a^2,
\]

\[
\Delta_r=2r-2M,
\]

\[
K=(r^2+a^2)\omega-am,
\]

\[
V=\frac{K^2-2is(r-M)K}{\Delta}+4is\omega r-\lambda.
\]

在当前主要情形 \(s=-2\) 下：

\[
\Delta R_{rr}-\Delta_r R_r+V R=0,
\]

并且

\[
V=\frac{K^2+4i(r-M)K}{\Delta}-8i\omega r-\lambda.
\]

---

## 3. 令 \(R=P u\)

令

\[
R=P u,
\]

并定义

\[
q=\frac{P_r}{P}.
\]

则

\[
R_r=P(u_r+q u),
\]

\[
R_{rr}=P[u_{rr}+2q u_r+(q_r+q^2)u].
\]

代入原方程并除以 \(P\)，得到 \(u\) 的 \(r\)-坐标方程：

\[
\Delta u_{rr}
+
[2\Delta q-\Delta_r]u_r
+
[\Delta(q_r+q^2)-\Delta_r q+V]u
=0.
\]

因此在 \(r\)-坐标下：

\[
B_2^{(r)}=\Delta,
\]

\[
B_1^{(r)}=2\Delta q-\Delta_r,
\]

\[
B_0^{(r)}=\Delta(q_r+q^2)-\Delta_r q+V.
\]

关键点：

**\(B_0^{(r)}\) 不是原始 potential \(V\)，而是**

\[
B_0^{(r)}
=
\Delta(q_r+q^2)-\Delta_r q+V.
\]

如果漏掉前两项，\(V\sim \omega^2 r^2\) 的发散不会被抵消。

---

## 4. 当前 Leaver \(P\) 因子

当前代码中的 Leaver prefactor 形式为：

\[
P=(r-r_+)^{p_+}(r-r_-)^{p_-}e^{i\omega r}.
\]

对 \(s=-2\)，有：

\[
p_+=2-i\sigma_+,
\]

\[
p_-=1+2i\omega+i\sigma_+,
\]

其中

\[
\sigma_+=\frac{2\omega r_+-am}{r_+-r_-}.
\]

因此无穷远处：

\[
P\sim r^{p_++p_-}e^{i\omega r}.
\]

因为

\[
p_++p_-=3+2i\omega,
\]

所以

\[
P\sim r^{3+2i\omega}e^{i\omega r}.
\]

这正好对应 outgoing branch 的主导渐近行为：

\[
r^3 e^{i\omega r_*},
\]

因为

\[
r_*\sim r+2\ln r+\text{const}
\]

在 \(M=1\) 单位下成立。

---

## 5. 转换到 \(y\) 坐标

定义

\[
x=\frac{r_+}{r},
\]

\[
y=2x-1.
\]

于是

\[
y_r=-\frac{2r_+}{r^2},
\]

\[
y_{rr}=\frac{4r_+}{r^3}.
\]

由于

\[
u_r=y_r u_y,
\]

\[
u_{rr}=y_r^2u_{yy}+y_{rr}u_y,
\]

所以 \(u\) 的 \(y\)-坐标方程为：

\[
A_2(y)u_{yy}+A_1(y)u_y+A_0(y)u=0.
\]

其中

\[
A_2=\Delta y_r^2,
\]

\[
A_1=\Delta y_{rr}+[2\Delta q-\Delta_r]y_r,
\]

\[
A_0=\Delta(q_r+q^2)-\Delta_r q+V.
\]

注意：

**这里的 \(A_0\) 已经是提取 \(P\) 后的 reduced coefficient，不是 raw Teukolsky potential \(V\)。**

---

## 6. 无穷远展开

设 \(M=1\)，这是当前项目默认单位。

无穷远处有：

\[
q=\frac{P_r}{P}
=
i\omega+\frac{n}{r}+\frac{d}{r^2}+O(r^{-3}),
\]

其中

\[
n=p_++p_-=3+2i\omega,
\]

\[
d=p_+r_+ + p_-r_-.
\]

### 6.1 \(A_2\) 的极限

\[
A_2=\Delta y_r^2
=
(r^2-2r+a^2)\frac{4r_+^2}{r^4}.
\]

因此

\[
A_2=4\frac{r_+^2}{r^2}+O(r^{-3}).
\]

由于

\[
x=\frac{r_+}{r},
\]

所以

\[
A_2=4x^2+O(x^3).
\]

因此

\[
A_2(-1)=0.
\]

这与数值诊断中的 \(D_2\propto x^2\) 一致。

---

### 6.2 \(A_1\) 的极限

\[
A_1=\Delta y_{rr}+[2\Delta q-\Delta_r]y_r.
\]

使用

\[
q=i\omega+\frac{3+2i\omega}{r}+O(r^{-2}),
\]

可得

\[
2\Delta q-\Delta_r
=
2i\omega r^2+4r+O(1).
\]

同时

\[
y_r=-\frac{2r_+}{r^2},
\]

\[
y_{rr}=\frac{4r_+}{r^3}.
\]

于是

\[
\Delta y_{rr}
=
\frac{4r_+}{r}+O(r^{-2}),
\]

\[
[2\Delta q-\Delta_r]y_r
=
-4i\omega r_+ -\frac{8r_+}{r}+O(r^{-2}).
\]

因此

\[
A_1
=
-4i\omega r_+
+O(r^{-1}).
\]

所以无穷远端极限为：

\[
A_1^\infty=-4i\omega r_+.
\]

如果数值输出的是模长，则

\[
|A_1^\infty|=4|\omega|r_+.
\]

例如 \(\omega=0.2\)、\(r_+\approx1.866\)，则

\[
4\omega r_+\approx1.49,
\]

这正好解释此前诊断中的 \(D_1\approx1.49\)。

---

### 6.3 \(A_0\) 的极限

\[
A_0=\Delta(q_r+q^2)-\Delta_r q+V.
\]

展开：

\[
q=i\omega+\frac{n}{r}+\frac{d}{r^2}+O(r^{-3}),
\]

\[
q_r=-\frac{n}{r^2}-\frac{2d}{r^3}+O(r^{-4}),
\]

\[
q^2
=
-\omega^2+\frac{2i\omega n}{r}
+
\frac{n^2+2i\omega d}{r^2}
+O(r^{-3}).
\]

因此

\[
q_r+q^2
=
-\omega^2+\frac{2i\omega n}{r}
+
\frac{n^2+2i\omega d-n}{r^2}
+O(r^{-3}).
\]

另一方面，原始 \(V\) 在 \(s=-2\) 时的无穷远展开为：

\[
V
=
\omega^2r^2
+
(2\omega^2-4i\omega)r
+
[\omega^2(a^2+4)-2am\omega-\lambda+4i\omega]
+
O(r^{-1}).
\]

把

\[
\Delta(q_r+q^2),
\]

\[
-\Delta_r q,
\]

\[
V
\]

三部分相加后，\(r^2\) 项和 \(r\) 项完全抵消。最终得到有限常数：

\[
A_0^\infty
=
8r_+\omega^2
-4am\omega
-\lambda
+i\omega(2r_+ +4).
\]

因此：

\[
A_0
=
8r_+\omega^2
-4am\omega
-\lambda
+i\omega(2r_+ +4)
+
O(r^{-1}).
\]

---

## 7. 无穷远端 \(A_0/A_1\) 与 Robin 条件

对固定非零 \(\omega\)：

\[
\left.\frac{A_0}{A_1}\right|_{y=-1}
=
\frac{
8r_+\omega^2
-4am\omega
-\lambda
+i\omega(2r_+ +4)
}{
-4i\omega r_+
}.
\]

端点方程在 \(y=-1\) 处退化为：

\[
A_1^\infty u_y(-1)+A_0^\infty u(-1)=0.
\]

因此：

\[
u_y(-1)
=
-\frac{A_0^\infty}{A_1^\infty}u(-1).
\]

也就是：

\[
u_y(-1)
=
\left[
\frac12+\frac{1}{r_+}
-
i\frac{
8r_+\omega^2
-4am\omega
-\lambda
}{
4\omega r_+
}
\right]
u(-1).
\]

所以 infinity Robin slope 是：

\[
c_\infty
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

由于

\[
U=\frac{R}{P}=h_2S,
\]

且 \(h_2\) 与 \(y\) 无关，因此 \(U\) 和 \(S\) 满足相同的 Robin 比例条件：

\[
S_y(-1)=c_\infty S(-1).
\]

---

## 8. 特殊情况：\(\omega=0\)

上面的结果假设固定非零 \(\omega\)。

当

\[
\omega=0
\]

时，

\[
A_1^\infty=-4i\omega r_+=0.
\]

此时端点方程的退化阶数改变，不能直接使用

\[
c_\infty=-A_0^\infty/A_1^\infty.
\]

因此 \(\omega=0\) 必须单独处理。当前 patch 若包含极低频但非零频率，可以使用该公式；但数值上当 \(\omega\) 很小时，\(c_\infty\) 会变大，训练权重需要小心归一化。

---

## 9. 对 agent 诊断结果的解释

如果代码诊断得到：

\[
D_0\sim x^{-2},
\]

则这不是 clean \(u=R/P\) 方程的解析结果。

最可能的原因是：

1. 实际计算的 \(D_0\) 仍然包含 raw potential 的主导项：

   \[
   V\sim \omega^2r^2\sim x^{-2}.
   \]

2. 缺少了 \(P\) log-derivative 的抵消项：

   \[
   \Delta(q_r+q^2)-\Delta_r q.
   \]

3. 或者在代码中把 raw \(R\)-space 的 \(A_0\) 当成 reduced \(u\)-space 的 \(A_0\)。

4. 或者在 \(x\)-space 到 \(y\)-space 转换时只转换了导数系数，却没有同步转换 \(P_x/P\)、\(P_{xx}/P\)。

正确的 reduced coefficient 必须是：

\[
A_0=\Delta(q_r+q^2)-\Delta_r q+V.
\]

或者在 \(x\)-space 代码中等价地包含：

\[
A_2 \frac{P_{xx}}{P}
+
A_1 \frac{P_x}{P}
+
A_0.
\]

当前仓库中的 `transform_coeffs_x_to_y_S` 已经写出了这一结构：

\[
D_2=4A_2,
\]

\[
D_1=4A_2P_x/P+2A_1,
\]

\[
D_0=A_2P_{xx}/P+A_1P_x/P+A_0.
\]

因此后续排查应重点检查实际诊断脚本是否真的调用了这个 clean S-space transformation，而不是误用了 raw coefficients。

---

## 10. 需要 agent 重点检查的地方

请检查所有 infinity Robin 诊断脚本中：

1. 是否使用的是 \(R=P u\) 后的 reduced equation；
2. \(D_0\) 是否包含 \(P_x/P\)、\(P_{xx}/P\)；
3. 是否误用了 raw \(V\) 或 raw \(A_0\)；
4. 是否混淆了 \(x\)-space 和 \(y\)-space 系数；
5. 是否在 \(y=-1+\epsilon\) 处用 raw \(r\)-formula 导致数值上漏掉 cancellation；
6. 是否使用 complex coefficient 的模长打印，从而隐藏了相位；
7. 是否在 \(M=1\)、\(s=-2\) 之外误用上述公式。

正确的 benchmark 应该是：

\[
A_2\sim 4x^2,
\]

\[
A_1\to -4i\omega r_+,
\]

\[
A_0\to
8r_+\omega^2
-4am\omega
-\lambda
+i\omega(2r_+ +4).
\]

如果数值脚本正确，它应该看到 \(A_0\) 收敛到有限复数，而不是 \(x^{-2}\) 发散。

---

## 11. 最终结论

对当前 \(R=P u\) 的 reduced Teukolsky 方程，在固定非零 \(\omega\) 下，无穷远端的 reduced coefficients 是：

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
\left.\frac{A_0}{A_1}\right|_{y=-1}
=
\frac{
8r_+\omega^2
-4am\omega
-\lambda
+i\omega(2r_+ +4)
}{
-4i\omega r_+
}.
\]

对应无穷远端 Robin 条件为：

\[
u_y(-1)
=
\left[
\frac12+\frac{1}{r_+}
-
i\frac{
8r_+\omega^2
-4am\omega
-\lambda
}{
4\omega r_+
}
\right]
u(-1).
\]

同样：

\[
S_y(-1)=c_\infty S(-1).
\]

如果当前代码得到 \(c_\infty\sim x^{-2}\)，应先修正 infinity coefficient 诊断实现，而不是放弃 infinity Robin 约束。