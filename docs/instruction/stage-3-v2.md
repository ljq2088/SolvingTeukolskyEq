当前分支：feature/autoencoder-asymptotic-decoder2。

你已经完成 Stage-3 初始 pipeline：
- physical_ansatz/u_residual.py
- config/autoencoder_stage3_ubasis.yaml
- scripts/train_autoencoder_stage3_ubasis.py
- scripts/debug_autoencoder_stage3_ubasis.py
并且 debug 37/37 通过。

但是当前 500 epoch pilot 显示：
- loss_down: 4.6e8 -> 2.5e7，可以下降；
- loss_up: 卡在 ~1e29，不下降。

不要进入 Stage-4，不要继续盲目训练 Stage-3 5000 epoch。现在先修正 Stage-3 residual 的数学形式和数值条件。

核心原则：
Stage-3 不是直接训练巨大函数 R_up=A_up*u_up。
Stage-3 应训练光滑归一化函数 u_up/u_down。
我们要把 A_up/A_down 从方程中解析提取出来，得到直接作用在 u 上的 basis-dependent equation coefficients，然后对 u 的 residual 训练。

也就是说，本轮目标是：
从 direct R residual
    R = A_basis * u
    TeukResidual[R]
改为 extracted u-equation residual:
    C2_basis(y) u_yy + C1_basis(y) u_y + C0_basis(y) u = 0

其中不同 basis 的区别只进入 q_r=A_r/A 和 q_rr=dq_r/dr。

------------------------------------------------------------
任务 1：确认当前状态
------------------------------------------------------------

运行：

git branch --show-current
git pull
git status
git log --oneline -5

确认分支为：
feature/autoencoder-asymptotic-decoder2

确认当前包含 Stage-3 文件：
physical_ansatz/u_residual.py
config/autoencoder_stage3_ubasis.yaml
scripts/train_autoencoder_stage3_ubasis.py
scripts/debug_autoencoder_stage3_ubasis.py

------------------------------------------------------------
任务 2：修复当前 autograd 图问题
------------------------------------------------------------

在 physical_ansatz/u_residual.py 中，当前 R_yy 使用了：

create_graph=False

这会让二阶导数项对 decoder 参数的梯度被截断。请先修复：

- 在训练 residual 中，二阶导数必须使用 create_graph=True；
- 可以新增参数 create_graph=True；
- debug 或 eval 时可以允许 create_graph=False，但训练默认必须为 True；
- debug 脚本要检查二阶导 residual backward 后 up/down decoder 的梯度确实非零且 finite。

注意：
这只是必要修复，但不是解决 loss_up 爆炸的根本办法。

------------------------------------------------------------
任务 3：实现提取后的 u-basis 方程 residual
------------------------------------------------------------

新增或重构 physical_ansatz/u_residual.py，保留 direct R residual 作为 debug 参考，但新增正式训练函数：

compute_u_equation_residual(..., basis="up" or "down")

原始 Teukolsky R 方程：

Delta R_rr + (s+1) Delta_r R_r + V R = 0

令：

R = A u

定义：

q_r = A_r / A
q_rr = d q_r / dr

则 u 的 r-space 方程为：

Delta u_rr
+ [(s+1)Delta_r + 2 Delta q_r] u_r
+ [V + (s+1)Delta_r q_r + Delta(q_rr + q_r^2)] u = 0

再转到 y 坐标：

u_r = u_y y_r
u_rr = u_yy y_r^2 + u_y y_rr

因此：

C2 u_yy + C1 u_y + C0 u = 0

其中：

C2 = Delta * y_r^2

C1 = Delta * y_rr + [(s+1)Delta_r + 2Delta q_r] * y_r

C0 = V + (s+1)Delta_r q_r + Delta * (q_rr + q_r^2)

请实现：

compute_q_terms(r, a, omega, basis)

其中：

For up:
A_up = r^3 exp(i omega r_*)
q_r = 3/r + i omega rstar_r

For down:
A_down = r^{-1} exp(-i omega r_*)
q_r = -1/r - i omega rstar_r

并且：

rstar_r = (r^2 + a^2) / Delta

rstar_rr = [2 r Delta - (r^2+a^2) Delta_r] / Delta^2

所以：

up:
q_rr = -3/r^2 + i omega rstar_rr

down:
q_rr = 1/r^2 - i omega rstar_rr

注意：
- 不要在 u-equation residual 中显式构造 A_up 或 A_down；
- 不要显式构造 R_up = A_up*u_up；
- 这一步的目的正是避免 A_up ~ r^3 在无穷远爆炸；
- A_up/A_down 可以继续保留在 u_basis.py 里供 debug 和 Stage-4 使用。

------------------------------------------------------------
任务 4：u 的导数用 autograd，必须对 decoder 可微
------------------------------------------------------------

在 u-equation residual 中：

1. 先得到 f_up/f_down；
2. 用 infinity_slopes_y 得到 c_up/c_down；
3. 用 compose_u_from_f 得到 u(y)；
4. 用 autograd 计算 u_y 和 u_yy；
5. 用 C2/C1/C0 组装 residual。

这里 u_yy 的计算必须 create_graph=True，确保 loss 对 up/down decoder 参数可微。

注意：
因为 u 是光滑函数，理论上这一步比直接算 R=A*u 的二阶导稳定得多。

------------------------------------------------------------
任务 5：修改 Stage-3 训练脚本使用 u-equation residual
------------------------------------------------------------

在 scripts/train_autoencoder_stage3_ubasis.py 中：

- 默认使用 compute_stage3_loss_u_equation；
- direct R residual 只作为可选 debug 模式；
- 增加 config 开关：

stage3:
  residual_mode: u_equation   # choices: u_equation, direct_R_debug

训练时默认必须是：

residual_mode: u_equation

------------------------------------------------------------
任务 6：修正 y_eps 和采样范围
------------------------------------------------------------

当前 _build_y_grid 默认 y_eps=1e-6，太靠近无穷远。请改成从 config 读取：

stage3:
  sampling:
    y_eps: 1.0e-3

并把默认值改成 1e-3，而不是 1e-6。

当前建议：

y_eps: 1.0e-3
y_inf_min: -1.0
y_inf_max: -0.95
n_near_infinity: 32

内部仍然 clamp 到 -1 + y_eps。

这一步不是为了逃避无穷远，而是避免在训练 residual 时把 r 推到 1e6 量级导致数值污染。真正的无穷远边界条件已经由 ansatz 精确满足：
u(-1)=1, u_y(-1)=c_inf

------------------------------------------------------------
任务 7：修改 config
------------------------------------------------------------

更新 config/autoencoder_stage3_ubasis.yaml：

stage3:
  residual_mode: u_equation

  loss:
    weight_up: 1.0
    weight_down: 1.0
    normalize_residual: true
    eps: 1.0e-12

  sampling:
    y_strategy: chebyshev
    near_infinity_extra: true
    n_near_infinity: 32
    y_inf_min: -1.0
    y_inf_max: -0.95
    y_eps: 1.0e-3

如果 u_equation residual 量级仍大，可以只对 coefficient scale 做归一化，而不是除以 |R|^2，因为此时已经不再显式构造 R。

------------------------------------------------------------
任务 8：新增 u-equation debug 检查
------------------------------------------------------------

扩展 scripts/debug_autoencoder_stage3_ubasis.py，增加检查：

1. compute_u_equation_residual up finite；
2. compute_u_equation_residual down finite；
3. u-equation residual 中没有显式 A_up/A_down；
4. loss_up/loss_down 都 finite；
5. backward 后 up/down decoder 有梯度；
6. grad norm finite；
7. direct_R_debug 与 u_equation 在中等 r 区域方向上大致一致。

第 7 点只做 sanity check，不要求严格数值相等，因为 direct R residual 有巨大 scale。

------------------------------------------------------------
任务 9：重新运行 debug 和 500 epoch pilot
------------------------------------------------------------

先运行：

python -m py_compile \
  physical_ansatz/u_residual.py \
  scripts/debug_autoencoder_stage3_ubasis.py \
  scripts/train_autoencoder_stage3_ubasis.py

然后运行：

python scripts/debug_autoencoder_stage3_ubasis.py \
  --config config/autoencoder_stage3_ubasis.yaml \
  --checkpoint outputs/autoencoder_stage2_amplitude_train/20260515_192625_stage2_amplitude/checkpoints/best_model.pt \
  --device cuda

通过后，重新跑 500 epoch pilot：

python scripts/train_autoencoder_stage3_ubasis.py \
  --config config/autoencoder_stage3_ubasis.yaml \
  --checkpoint outputs/autoencoder_stage2_amplitude_train/20260515_192625_stage2_amplitude/checkpoints/best_model.pt \
  --device cuda \
  --epochs 500 \
  --verbose

报告：

- residual_mode；
- y_eps；
- epoch 1 loss_up/loss_down；
- epoch 50；
- epoch 100；
- epoch 300；
- epoch 500；
- loss_up 是否终于下降；
- loss_down 是否继续下降；
- 是否 NaN/Inf；
- gradient norm 是否 finite；
- best checkpoint 路径。

判断标准：

可以进入 5000 epoch 的条件：
- loss_up 和 loss_down 都至少下降一个明显因子；
- loss_up 不再卡在 1e29；
- 无 NaN/Inf；
- boundary ansatz 仍严格满足；
- up/down decoder 是唯一发生变化的模块。

------------------------------------------------------------
任务 10：提交和报告
------------------------------------------------------------

如果通过：

git add physical_ansatz/u_residual.py scripts/train_autoencoder_stage3_ubasis.py scripts/debug_autoencoder_stage3_ubasis.py config/autoencoder_stage3_ubasis.yaml
git commit -m "Stabilize Stage-3 u-equation residual for infinity bases"
git push

最终报告必须包含：

1. 当前 commit hash；
2. direct R residual 的问题总结；
3. 新 u-equation residual 的公式说明；
4. y_eps 设置；
5. debug 检查结果；
6. 500 epoch pilot 中 loss_up/loss_down 的变化；
7. 是否建议继续 5000 epoch；
8. 是否仍阻塞 Stage-4。