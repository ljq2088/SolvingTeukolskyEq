当前分支：feature/autoencoder-asymptotic-decoder2。

你已经完成 Stage-3 v2：使用 u-equation residual 替代 direct R=A*u residual。当前 y_eps=1e-3 是用于 PDE collocation 的数值截断，不代表边界 ansatz 不能在 y=-1 精确成立。

现在发现 diagnose_stage3_ubasis_solution.py 中的边界检查是误报：因为诊断用了训练网格或 clipped grid，没有真正取 y=-1。compose_u_from_f 在数学上保证：

u(-1)=1
du/dy(-1)=c_inf

所以请不要因为这个边界误报停止当前 5000 epoch 训练。如果训练 loss finite、grad finite、max|u| 合理，就继续监控训练。

但需要立刻修正诊断逻辑。

------------------------------------------------------------
任务 1：不要停止 5000 epoch，先检查训练是否健康
------------------------------------------------------------

检查当前 5000 epoch 训练：

- 是否 NaN/Inf；
- loss_up 是否整体下降；
- loss_down 是否整体下降；
- grad_norm 是否 finite；
- max|u_up| 是否仍在合理范围，例如 10^2 以内；
- max|u_down| 是否仍约 O(1)；
- best_model.pt 是否正常保存。

如果这些都正常，不要停止训练。

------------------------------------------------------------
任务 2：修正边界诊断误报
------------------------------------------------------------

修改 scripts/diagnose_stage3_ubasis_solution.py 或对应诊断脚本。

请明确区分两类 y-grid：

A. boundary grid:
   只用于检查 boundary ansatz。
   必须精确使用：
   y_boundary = -1.0

B. residual/collocation grid:
   只用于计算 PDE residual。
   继续使用：
   y >= -1 + y_eps
   例如 y_eps=1e-3

不要用 clipped collocation grid 来判断 u(-1) 和 u_y(-1)。

边界检查应写成：

1. 构造 y0 = torch.tensor([[-1.0]], requires_grad=True)
2. 调用 model.predict_u_up / predict_u_down 得到 f_up/f_down
3. 用 infinity_slopes_y 得到 c_up/c_down
4. 用 compose_u_from_f(f, y0, c_inf) 得到 u
5. 检查：
   u_up(y0) = 1
   u_down(y0) = 1

对于导数，有两种方式：

方式 A：直接解析检查
因为：
u(y) = 1 + c_inf*(y+1) + (y+1)^2*f(y)

所以：
u_y(-1) = c_inf

这可以直接报告 analytic pass。

方式 B：autograd 检查
使用 y0.requires_grad=True，计算 u.real 和 u.imag 对 y 的导数，分别检查：

Re[u_y(-1)] = Re[c_inf]
Im[u_y(-1)] = Im[c_inf]

注意：
必须检查复数导数的实部和虚部，不要只检查 real part。

------------------------------------------------------------
任务 3：不要在当前 residual 函数里直接取 y=-1
------------------------------------------------------------

当前 compute_u_equation_residual 仍然通过：

x = (y+1)/2
r = r_plus / x

来构造系数。这个形式在 y=-1 时 x=0, r=∞，不能直接数值代入。

所以：
- PDE residual 采样继续避开 y=-1；
- 保持 y_eps=1e-3；
- 不要把 y=-1 加入 residual collocation grid；
- boundary 条件已经由 ansatz 精确满足，不需要用 PDE residual 在 y=-1 再约束。

------------------------------------------------------------
任务 4：可选实现 endpoint-limit 诊断，但不要用于训练
------------------------------------------------------------

理论上，提取 A_basis 后，u-equation 在无穷远处可以通过乘以适当因子或做 x=(y+1)/2 的级数展开，得到有限 endpoint-limit 方程。

但这需要单独实现，不要直接把 y=-1 代入当前 C2/C1/C0 计算。

如果要做，请新增：

compute_u_equation_endpoint_limit(...)

或者：

compute_u_equation_coeffs_regularized(...)

要求：
- 在 x->0 的极限下返回有限系数；
- 明确说明是否对方程整体乘了 regularization factor；
- 只用于诊断，不用于当前训练；
- 与 infinity_slopes_y 给出的 c_inf 做一致性检查。

当前阶段不要为了 endpoint limit 阻塞 5000 epoch 训练。

------------------------------------------------------------
任务 5：继续监控 5000 epoch 训练
------------------------------------------------------------

当前 5000 epoch 训练结束后，请报告：

- run_dir；
- best_epoch；
- best_val_loss；
- best loss_up；
- best loss_down；
- final loss_up；
- final loss_down；
- 是否发生 loss_up 后期反弹；
- 是否 NaN/Inf；
- grad_norm 是否 finite；
- max|u_up| range；
- max|u_down| range；
- boundary ansatz exact check 是否通过；
- near-infinity residual 是否可控；
- best checkpoint 路径。

如果 loss_up 最优值明显低于 500 epoch pilot 的 best 值约 2e7，说明长训练有效。

------------------------------------------------------------
任务 6：提交诊断修正
------------------------------------------------------------

如果只修正诊断脚本，不需要停止训练。训练可以继续跑。

完成后：

git add scripts/diagnose_stage3_ubasis_solution.py
git commit -m "Fix exact boundary diagnostics for Stage-3 u-basis"
git push

大型 outputs 不要提交。

最终报告中请明确区分：

1. Boundary ansatz exact check:
   y=-1, exact, should pass.

2. PDE residual grid:
   y >= -1 + y_eps, not exact boundary.

3. Endpoint-limit residual:
   analytic optional diagnostic, not currently used in training.