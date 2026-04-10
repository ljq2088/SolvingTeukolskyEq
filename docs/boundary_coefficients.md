# 边界上的方程系数完整表达式

## 坐标与函数变换

### 坐标变换
- **原始坐标**: $r \in [r_+, \infty)$
- **紧化坐标**: $x = \frac{r_+}{r} \in (0, 1]$
- **Chebyshev 坐标**: $y = 2x - 1 \in [-1, 1]$，即 $x = \frac{y+1}{2}$

### 函数变换
- **完整波函数**: $R(r) = R'(x) \cdot U(r)$
- **Regular 部分**: $R'(x) = [e^{x-1} - 1] \cdot f(x) + 1$
- **Prefactor**: $U(r) = Q(r) \cdot P(r)$

其中：
- $g(x) = e^{x-1} - 1$
- $g'(x) = e^{x-1}$
- $g''(x) = e^{x-1}$

## 原始 Teukolsky 方程（r 坐标）

$$\Delta R_{rr} + (s+1)\Delta_r R_r + V R = 0$$

其中：
- $\Delta(r) = r^2 - 2Mr + a^2 = (r-r_+)(r-r_-)$
- $\Delta_r = 2r - 2M$
- $V(r) = \frac{K^2 - 2is(r-M)K}{\Delta} + 4is\omega r - \lambda$
- $K(r) = (r^2 + a^2)\omega - am$

## R'(x) 满足的方程（x 坐标）

$$A_2(x) R'_{xx} + A_1(x) R'_x + A_0(x) R' = 0$$

### 系数表达式

$$A_2 = \Delta \left(\frac{dx}{dr}\right)^2 = \Delta \cdot \frac{x^4}{r_+^2}$$

$$A_1 = \Delta \left[2\frac{U_r}{U}\frac{dx}{dr} + \frac{d^2x}{dr^2}\right] + (s+1)\Delta_r\frac{dx}{dr}$$

$$A_0 = V + (s+1)\Delta_r\frac{U_r}{U} + \Delta\frac{U_{rr}}{U}$$

其中：
- $\frac{dx}{dr} = -\frac{x^2}{r_+}$
- $\frac{d^2x}{dr^2} = \frac{2x^3}{r_+^2}$

## f(y) 满足的方程（y 坐标）

$$B_2(y) f_{yy} + B_1(y) f_y + B_0(y) f = \text{rhs}(y)$$

### 变换系数

$$B_2 = 4 A_2 \cdot g$$

$$B_1 = 4 A_2 \cdot g' + 2 A_1 \cdot g$$

$$B_0 = A_2 \cdot g'' + A_1 \cdot g' + A_0 \cdot g$$

$$\text{rhs} = -A_0$$

