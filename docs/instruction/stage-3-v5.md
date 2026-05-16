当前分支：feature/autoencoder-asymptotic-decoder2。

现在先暂停 Stage-4 consistency 分析，也不要继续讨论 amplitude_net mismatch。注意：Stage-3 的 u_up/u_down decoder 和 AmplitudeNet 是独立模块。AmplitudeNet 只在 Stage-4 组合 R_in = B_ref u_up A_up + B_inc u_down A_down 时才重要。Stage-3 的正确性应先通过 u_up/u_down 自身与谱方法 basis 的对比来判断。

当前目标：
1. 确认 Stage-3 训练的 y 区域；
2. 增加一个和 high_omega_debug/basis_functions.png 类似的比较脚本；
3. 将谱方法的 u_up/u_down basis 和 Stage-3 decoder 输出的最终 u_up/u_down 画在同一张图中；
4. 判断 Stage-3 decoder 是否真的学到了光滑归一化 basis；
5. 根据对比结果决定是否需要重新训练一个 near-infinity-only Stage-3。

------------------------------------------------------------
任务 1：确认当前 Stage-3 训练区域
------------------------------------------------------------

先阅读：

config/autoencoder_stage3_ubasis_refine.yaml
scripts/train_autoencoder_stage3_ubasis.py

明确报告：

- n_interior；
- n_near_infinity；
- y_eps；
- Chebyshev grid 覆盖的 y 范围；
- near-infinity extra grid 的 y 范围；
- 当前训练是否覆盖整个 compact domain；
- 当前训练中靠近无穷远的有效范围是多少。

当前我预计是：

Chebyshev grid:
  y in [-1 + y_eps, 1 - y_eps]

near-infinity extra:
  y in [-1 + y_eps, -0.95]

其中 y_eps = 1e-3。

请确认这个判断。

------------------------------------------------------------
任务 2：找到谱方法绘图脚本
------------------------------------------------------------

请搜索 high_omega_debug/basis_functions.png 的来源脚本。

运行：

find . -path "*high_omega_debug*" -print
grep -R "basis_functions.png" -n .
grep -R "Left patch" -n .
grep -R "down (real u)" -n .
grep -R "up (real u)" -n .
grep -R "basis_functions" -n scripts . | head -50

目标：
找到生成你图中 basis_functions.png 的脚本。

不要重写一套谱方法逻辑；优先在已有脚本基础上修改。

------------------------------------------------------------
任务 3：理解谱方法中的坐标与 basis 定义
------------------------------------------------------------

找到脚本后，确认以下内容：

- 图中横轴 z 与当前 PINN/autoencoder 的 y 或 x 的关系；
- left patch [0,0.625] 是否对应无穷远附近；
- right patch [0.705,1] 是否对应另一个局部 patch；
- 图中的 down/up/in 分别对应什么：
  - down 是否为 u_down；
  - up 是否为 u_up；
  - in 是否为组合的 in solution；
- 谱方法中的归一化是否是：
  u_up(-1)=1
  u_down(-1)=1
- 谱方法是否已经提取了：
  A_up = r^3 exp(i omega r_*)
  A_down = r^{-1} exp(-i omega r_*)

必须在报告中写清楚：
spectral u 和 Stage-3 decoder u 是否在同一个 normalization convention 下。

------------------------------------------------------------
任务 4：新增 Stage-3 vs spectral basis 比较脚本
------------------------------------------------------------

新增脚本：

scripts/compare_stage3_ubasis_with_spectral.py

功能：
在指定参数点上，读取谱方法 u_up/u_down basis，并与 Stage-3 decoder 输出的 u_up/u_down 对比。

建议命令：

python scripts/compare_stage3_ubasis_with_spectral.py \
  --stage3-checkpoint outputs/autoencoder_stage3_ubasis_refine/20260515_222026_stage3_ubasis/checkpoints/best_model.pt \
  --config config/autoencoder_stage3_ubasis_refine.yaml \
  --spectral-dir outputs/high_omega_debug \
  --device cuda \
  --a <value> \
  --omega <value> \
  --output-dir outputs/stage3_spectral_compare/patch_000_logw_v2

如果谱方法脚本本身会生成数据，则支持：

--run-spectral

或者复用现有输出文件。

输出图建议模仿 high_omega_debug/basis_functions.png，但每个 panel 同时画：

- spectral u：实线黑色或 dashed；
- Stage-3 decoder u：彩色；
- difference：可另开下方或单独图。

至少生成：

basis_functions_stage3_vs_spectral.png

包含：

1. u_down real；
2. u_down imag；
3. u_up real；
4. u_up imag；
5. 如果有 in solution，也可画 u_in real/imag，但不要把它和 Stage-3 decoder 混淆。

------------------------------------------------------------
任务 5：计算数值误差
------------------------------------------------------------

除了画图，输出误差表：

对 left / near-infinity patch：

- median_abs_err_down_real；
- median_abs_err_down_imag；
- median_abs_err_up_real；
- median_abs_err_up_imag；
- median_rel_err_down；
- median_rel_err_up；
- max_rel_err_down；
- max_rel_err_up；
- boundary mismatch：
  u_up(-1)-1
  u_down(-1)-1
  du_up/dy(-1)-c_up
  du_down/dy(-1)-c_down

注意：
边界检查必须用 y=-1 精确点，不要用 y=-1+y_eps 的训练点。

------------------------------------------------------------
任务 6：添加 near-infinity-only Stage-3 配置，但先不训练
------------------------------------------------------------

新增配置：

config/autoencoder_stage3_ubasis_nearinf.yaml

它基于 refine config，但修改训练区域：

stage3:
  sampling:
    y_strategy: near_infinity_only
    y_eps: 1.0e-3
    y_min: -1.0
    y_max: 0.0
    n_interior: 128
    near_infinity_extra: true
    n_near_infinity: 64
    y_inf_min: -1.0
    y_inf_max: -0.95

或者如果从图中确认谱方法 left patch 只对应更小区域，就用谱方法 left patch 对应的 y 区间。

要求：
- 先只添加配置；
- 不要立刻训练；
- 等 compare 脚本确认误差后再决定是否用该配置重训 Stage-3。

------------------------------------------------------------
任务 7：不要让 amplitude_net 干扰 Stage-3 分析
------------------------------------------------------------

本轮不要再用 amplitude_net mismatch 作为 Stage-3 失败依据。

明确：
- Stage-3 的 u_up/u_down decoder 不依赖 B_inc/B_ref；
- AmplitudeNet 只在 Stage-4 consistency 里使用；
- 当前先判断 u basis 本身是否对；
- 如果后续要重新做 Stage-4，再处理 amplitude checkpoint merge 问题。

------------------------------------------------------------
任务 8：运行测试
------------------------------------------------------------

运行：

python -m py_compile \
  scripts/compare_stage3_ubasis_with_spectral.py

然后运行比较脚本。参数点优先选择生成 high_omega_debug/basis_functions.png 时使用的同一个 a, omega, lambda, l, m。

如果不知道参数，请从谱方法脚本或 metadata 中读取，不要猜。

运行后输出：

- spectral 参数；
- Stage-3 checkpoint；
- z/y 坐标映射；
- normalization convention；
- basis_functions_stage3_vs_spectral.png 路径；
- 误差表；
- 是否建议 near-infinity-only 重新训练。

------------------------------------------------------------
任务 9：提交
------------------------------------------------------------

如果新增比较脚本和配置：

git add scripts/compare_stage3_ubasis_with_spectral.py config/autoencoder_stage3_ubasis_nearinf.yaml
git commit -m "Add Stage-3 spectral basis comparison diagnostics"
git push

大型图片和 outputs 不要提交。

最终报告必须包含：

1. 当前 commit hash；
2. 当前 Stage-3 实际训练的 y 区域；
3. high_omega_debug/basis_functions.png 的生成脚本位置；
4. spectral basis 与 Stage-3 decoder 的 convention 是否一致；
5. 比较图路径；
6. u_up/u_down 的误差表；
7. Stage-3 是否已经足够好；
8. 是否需要 near-infinity-only Stage-3 重训；
9. 如果需要，建议训练区间和配置。