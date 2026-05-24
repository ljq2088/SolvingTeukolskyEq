你现在工作在 GitHub 仓库：

ljq2088/SolvingTeukolskyEq

当前分支必须是：

feature/autoencoder-asymptotic-decoder2

不要切换分支，不要新建分支。当前任务不是继续旧的 Stage-3 / Stage-4 方案，也不是加入 pybhpt、MMA 或谱方法 anchor。当前任务是测试一个新的纯无监督 PINN 范式：

在原 Stage-1 的 Teukolsky 径向 PINN 中，除了视界端 hard boundary 外，加入无穷远端 y=-1 的端点 Robin 正则性约束。

如果这个约束有效，后续方案将从原来的“四阶段 autoencoder + learned u-decoder + consistency polishing”调整为：

1. 纯无监督 PINN 主体训练：
   - 视界端 hard boundary；
   - 内点 Teukolsky residual；
   - 无穷远端 exact Robin boundary residual；
   - 必要时加 near-infinity weighted residual / radial curriculum。

2. 振幅网络只作为后处理/辅助模块：
   - 粗略训练 AmplitudeNet；
   - 冻结 PINN 主体；
   - 最后只训练或微调振幅网络；
   - 共享参数特征调制可以保留，但不再强依赖 learned u_up/u_down decoder。

不要再把 learned UpDecoder / DownDecoder 当作当前主线。

============================================================
一、物理与数学目标
============================================================

当前 Stage-1 的核心约定是：

R_in = P * h2 * S(y)

因此：

U(y) = R_in / P = h2 * S(y)

其中 h2 与 y 无关，所以 U 和 S 满足同一个端点 Robin 比例条件。

把：

R = P * U

代入 Teukolsky 径向方程后，可以得到 U 或 S 的 reduced 方程：

A2(y) U_yy + A1(y) U_y + A0(y) U = 0

或等价地：

D2(y) S_yy + D1(y) S_y + D0(y) S = 0

在无穷远端 y=-1，二阶导数系数退化：

D2(-1) = 0

因此端点方程退化成一阶 Robin 条件：

D1_inf * S_y(-1) + D0_inf * S(-1) = 0

等价于：

S_y(-1) = c_inf * S(-1)

其中：

c_inf = - D0_inf / D1_inf

注意：

1. 这个条件不是监督数据；
2. 不依赖 pybhpt；
3. 不依赖 MMA；
4. 不依赖谱方法 anchor；
5. 它是 Teukolsky 方程在无穷远正则奇异端点的极限条件；
6. 不能把 S(-1) 或 U(-1) 强行设成 1，因为那会改变 R_in 的全局归一化；
7. 应该约束的是导数与函数值的关系，而不是函数值本身。

训练中应该加入：

L_inf =
|S_y(-1) - c_inf S(-1)|^2
/
(|S_y(-1)|^2 + |c_inf S(-1)|^2 + eps)

或等价的非归一化版本用于诊断。

============================================================
二、核心要求
============================================================

请严格执行以下原则：

1. 不要直接在 y=-1 代入完整二阶 residual：
   D2*S_yy + D1*S_y + D0*S

   因为 y=-1 对应 r=infinity，原始系数可能出现 0*infinity 或 infinity/infinity 的数值不适定。

2. 必须先得到端点极限：
   D1_inf, D0_inf

   然后只用 Robin 条件：

   D1_inf*S_y(-1) + D0_inf*S(-1)=0

3. 如果从 x 坐标得到条件：

   A1_x_inf * S_x(0) + A0_x_inf * S(0)=0

   必须转换到 y 坐标：

   x = (y+1)/2

   S_x = 2 S_y

   所以：

   2*A1_x_inf*S_y(-1) + A0_x_inf*S(-1)=0

   因此：

   c_inf_y = - A0_x_inf / (2*A1_x_inf)

4. 如果直接从 y-space 系数 D1,D0 的极限得到：

   D1_inf*S_y + D0_inf*S = 0

   则不要再额外乘 1/2。

5. 先实现 soft Robin loss，不要先改 hard ansatz。

6. 如果 soft Robin loss 有明显效果，再考虑 hard constraint correction。

7. 本轮不加入 pybhpt anchor，不加入谱方法 anchor，不加入 MMA anchor。

============================================================
三、先检查当前仓库状态
============================================================

运行：

git branch --show-current
git pull
git status
git log --oneline -5

确认当前分支是：

feature/autoencoder-asymptotic-decoder2

确认工作区干净。如果不干净，先报告，不要覆盖已有改动。

============================================================
四、阅读相关文件
============================================================

重点阅读：

physical_ansatz/transform_y.py
physical_ansatz/prefactor.py
physical_ansatz/residual.py
physical_ansatz/teukolsky_coeffs.py
physical_ansatz/stage1_reconstruction.py
model/autoencoder_pinn.py
trainer/atlas_patch_trainer.py
scripts/train_autoencoder_stage1*
config/autoencoder_stage1*

尤其注意：

1. compose_reduced_shape_from_f(f, y, slope) 如何由 f 构造 S(y)；
2. horizon_regularity_slope 如何在视界端做 hard boundary；
3. transform_coeffs_x_to_y_S 是否已经给出 S-space 方程：

   D2*S_yy + D1*S_y + D0*S = 0

4. Stage-1 residual 当前到底是对 f、S、R/P 还是 R 计算；
5. 当前训练脚本如何采样 y；
6. 当前可视化如何计算 R/P = h2*S。

============================================================
五、实现无穷远端 Robin 极限
============================================================

新增或修改模块，建议文件：

physical_ansatz/infinity_robin.py

实现以下函数：

1. infinity_A1_A0_limit_for_S(...)
   返回 y=-1 端点处 S-space 方程的 D1_inf, D0_inf。

2. infinity_robin_slope_for_S(...)
   返回：

   c_inf = -D0_inf / D1_inf

3. infinity_robin_residual(...)
   输入 S(-1), S_y(-1), c_inf，返回：

   S_y(-1) - c_inf*S(-1)

4. infinity_robin_loss(...)
   返回相对归一化 loss。

实现要求：

A. 优先解析推导 D1_inf, D0_inf。

B. 如果短时间无法可靠解析推导，可先实现一个“极限诊断版本”：
   在 y=-1+eps_j 处计算 D2,D1,D0；
   使用 eps_j = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]；
   检查 D2 -> 0；
   检查 D1,D0 或归一化后的 D1,D0 是否收敛；
   用外推得到 D1_inf,D0_inf；
   但必须明确标记为 diagnostic/numerical limit，不要把它伪装成最终解析公式。

C. 如果 D1,D0 同时发散或同时趋零，需要寻找共同尺度 normalization。
   只要 Robin ratio c_inf = -D0/D1 收敛，就可以先使用 c_inf。
   此时报告 D1,D0 本身是否有限，以及 ratio 是否稳定。

D. 对每个参数点输出：
   - D2 near infinity 的数量级；
   - D1；
   - D0；
   - c_inf；
   - c_inf 是否随 eps 收敛；
   - 是否有 NaN/Inf。

E. 不要硬编码某一个 a, omega, lambda。
   必须支持 batch 输入。

============================================================
六、新增诊断脚本：先不训练
============================================================

新增脚本：

scripts/diagnose_infinity_robin_stage1.py

功能：

1. 加载当前 Stage-1 best checkpoint；
2. 选取 patch 0 中的若干参数点；
3. 对每个参数点，在 y=-1 精确计算：
   - S(-1)
   - S_y(-1)
   - c_inf
   - Robin residual:
     B_inf = S_y(-1) - c_inf*S(-1)

4. 同时在 y=-1+eps 处计算普通 PDE residual，用于对照；
5. 输出：
   - robin_abs
   - robin_rel
   - S(-1)
   - S_y(-1)
   - c_inf
   - median robin_rel
   - max robin_rel
   - per-point 表格

注意：

y=-1 必须精确取值，不要用 y=-1+eps 代替 Robin 约束。

但计算完整 PDE residual 时仍然只能用 y=-1+eps，不能直接在 y=-1 代入二阶方程。

运行示例：

python scripts/diagnose_infinity_robin_stage1.py \
  --checkpoint outputs/stage1_artifacts/patch_000_logw_v2/stage1_best_model.pt \
  --device cuda

如果默认路径不存在，自动从已有 outputs 中寻找最新 stage1 best checkpoint，并在报告中说明。

输出目录：

outputs/infinity_robin_diagnostics/<timestamp>/

保存：

diagnostics.json
diagnostics_summary.md
robin_residual_table.csv

============================================================
七、实现 Stage-1 + Infinity Robin 训练
============================================================

新增训练脚本：

scripts/train_stage1_with_infinity_robin.py

新增配置：

config/autoencoder_stage1_infinity_robin.yaml

训练目标：

L_total =
w_pde * L_pde
+
w_inf * L_inf
+
w_drift * L_drift

其中：

1. L_pde：
   复用原 Stage-1 内点 Teukolsky residual。
   不要重新写一套方程。
   采样要更适合当前问题：
   - 主要覆盖 near-infinity；
   - 保留中间区；
   - 保留靠近视界的少量点，避免破坏已有视界传播。

2. L_inf：
   精确在 y=-1 上计算 Robin loss：

   L_inf =
   |S_y(-1)-c_inf*S(-1)|^2
   /
   (|S_y(-1)|^2+|c_inf*S(-1)|^2+eps)

3. L_drift：
   约束当前模型不要远离原 Stage-1 checkpoint：

   L_drift =
   mean |S_new - S_old|^2 / mean(|S_old|^2 + eps)

   或者用 R/P = h2*S 计算也可以，但要保持前后一致。

建议初始配置：

stage1_infinity_robin:
  base_checkpoint: outputs/stage1_artifacts/patch_000_logw_v2/stage1_best_model.pt

  train_encoder: false
  train_rin_decoder: true
  train_amplitude_net: false
  train_up_decoder: false
  train_down_decoder: false

  optimizer:
    lr_rin_decoder: 1.0e-6
    weight_decay: 0.0

  epochs: 200
  batch_size: 8
  grad_clip: 0.1

  loss:
    weight_pde: 1.0
    weight_inf_robin: 1.0
    weight_drift: 1.0
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

  output_root: outputs/stage1_infinity_robin

runtime:
  dtype: float64

第一轮只训练 RinDecoder，不动 encoder。
如果结果明显有效，再考虑是否让 encoder 用极小学习率参与。

============================================================
八、debug 脚本
============================================================

新增：

scripts/debug_stage1_infinity_robin.py

必须检查：

1. checkpoint 可以加载；
2. 参数点可以采样；
3. y=-1 精确点可以 forward；
4. S(-1) finite；
5. S_y(-1) finite；
6. c_inf finite；
7. Robin loss finite；
8. PDE loss finite；
9. drift loss finite；
10. backward 后只有 RinDecoder 有梯度；
11. encoder 无梯度；
12. amplitude_net 无梯度；
13. up/down decoders 无梯度；
14. optimizer.step 后只有 RinDecoder 参数改变；
15. 无 NaN/Inf。

运行：

python scripts/debug_stage1_infinity_robin.py \
  --config config/autoencoder_stage1_infinity_robin.yaml \
  --device cuda

============================================================
九、训练实验顺序
============================================================

不要直接长训。

第一步：preflight

python scripts/diagnose_infinity_robin_stage1.py \
  --checkpoint outputs/stage1_artifacts/patch_000_logw_v2/stage1_best_model.pt \
  --device cuda

第二步：debug

python scripts/debug_stage1_infinity_robin.py \
  --config config/autoencoder_stage1_infinity_robin.yaml \
  --device cuda

第三步：20 epoch smoke

python scripts/train_stage1_with_infinity_robin.py \
  --config config/autoencoder_stage1_infinity_robin.yaml \
  --device cuda \
  --epochs 20 \
  --verbose

必须报告：

- initial L_pde
- initial L_inf
- initial L_drift
- epoch 20 L_pde
- epoch 20 L_inf
- epoch 20 L_drift
- Robin residual before/after
- near-infinity residual before/after
- S(-1) before/after
- S_y(-1) before/after
- Rin drift
- 是否 NaN/Inf
- best checkpoint path

第四步：如果 20 epoch 有效，再跑 200 epoch

通过条件：

1. L_inf 明显下降；
2. Robin relative residual 明显下降；
3. L_pde 不爆炸；
4. drift 不大；
5. 近无穷远可视化中斜率或截距有所改善；
6. 没有 NaN/Inf。

如果 20 epoch 后 L_inf 没下降，或者 PDE loss 爆炸，不要继续 200 epoch。

============================================================
十、可视化与评价
============================================================

新增或扩展可视化脚本：

scripts/plot_stage1_infinity_robin_effect.py

画同一批参数点的 before/after：

1. Re(R/P)
2. Im(R/P)
3. |R/P|
4. phase(R/P)
5. Robin residual per point
6. near-infinity PDE residual per y

对比对象：

- 原 Stage-1 checkpoint；
- 加 Robin 后 checkpoint；
- 如果已有 pybhpt 曲线，可以作为 optional diagnostic，但本任务的训练不得使用 pybhpt loss。

重要：
pybhpt 只能用于事后画图评估，不参与 loss。

输出目录：

outputs/stage1_infinity_robin_compare/<timestamp>/

保存：

comparison_summary.md
plots/*.png
metrics.json

评价标准：

1. Robin 条件是否改善；
2. 近无穷远端斜率是否改善；
3. 原本“趋势一致但截距偏移”的问题是否改善；
4. PDE residual 是否保持可控；
5. 是否破坏视界端行为；
6. 是否仍需要额外 near-infinity weighted residual 或 radial curriculum。

============================================================
十一、如果 soft Robin 有效，记录新范式
============================================================

如果测试显示 infinity Robin loss 明显改善远端误差，请新增文档：

docs/pinn_with_infinity_robin_paradigm.md

内容包括：

1. 为什么旧的 4-stage autoencoder 方案不再是主线；
2. 新范式：

   - Stage A: 纯无监督 PINN 主体
     * horizon hard boundary；
     * infinity Robin endpoint constraint；
     * interior residual；
     * near-infinity weighted residual；
     * optional radial curriculum。

   - Stage B: 振幅网络辅助训练
     * 使用 PINN 主体共享的参数特征；
     * 粗略学习 B_inc/B_ref；
     * 可以冻结 PINN 主体，只训练 amplitude head；
     * 可选最后微调振幅网络。

   - Stage C: 只做诊断性 consistency check
     * 不再把 learned u_up/u_down decoder 作为核心模块；
     * 谱方法/pybhpt/MMA 只用于验证，不进入纯无监督主训练。

3. 本轮实验结果：
   - L_inf before/after；
   - near-infinity error before/after；
   - plots 路径；
   - 是否建议继续。

如果 soft Robin 无效，也要写清楚失败原因：
- c_inf 不稳定；
- Robin loss 下降但远端误差不变；
- PDE residual 爆炸；
- drift 过大；
- 说明还需要 relative residual / adjoint residual / radial curriculum。

============================================================
十二、提交要求
============================================================

提交代码，不提交大型 outputs。

建议 git add：

physical_ansatz/infinity_robin.py
scripts/diagnose_infinity_robin_stage1.py
scripts/debug_stage1_infinity_robin.py
scripts/train_stage1_with_infinity_robin.py
scripts/plot_stage1_infinity_robin_effect.py
config/autoencoder_stage1_infinity_robin.yaml
docs/pinn_with_infinity_robin_paradigm.md  # 如果实验有效再加

提交信息：

git commit -m "Add infinity Robin constraint for unsupervised Stage-1 PINN"

git push

最终报告必须包含：

1. 当前 commit hash；
2. 是否成功推导/实现 c_inf；
3. D2(-1)=0 的数值或解析验证；
4. D1_inf, D0_inf, c_inf 的稳定性；
5. preflight Robin residual；
6. debug 是否全部通过；
7. 20 epoch smoke 结果；
8. 是否继续 200 epoch；
9. before/after 可视化结论；
10. 是否支持新范式：
    纯无监督 PINN + infinity Robin + 振幅网络辅助训练。

============================================================
十三、禁止事项
============================================================

本轮不要做以下事情：

1. 不要继续 Stage-4 spectral consistency polishing；
2. 不要训练 learned u_up/u_down decoders；
3. 不要加入 pybhpt anchor loss；
4. 不要加入 MMA anchor loss；
5. 不要加入谱方法 anchor loss；
6. 不要改动分支；
7. 不要重构整个模型；
8. 不要直接把 S(-1) 固定为 1；
9. 不要在 y=-1 直接计算完整二阶 residual；
10. 不要把 y=-1+eps 当成 infinity Robin 约束的替代。

本轮目标只有一个：

验证“无穷远端 Robin 正则性约束”是否能改善纯无监督 PINN 的远端截距/斜率偏移问题。