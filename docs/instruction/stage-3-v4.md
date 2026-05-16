当前分支：feature/autoencoder-asymptotic-decoder2。

你已经完成 Stage-3 5000 epoch 训练。报告显示：
- best_epoch = 3700
- best_val_loss = 1.21e7
- best loss_up = 1.21e7
- best loss_down = 9.68
- global min loss_down = 3.27 at epoch 4600
- final loss_up = 9.51e7
- final loss_down = 46.8
- loss_up / loss_down 后期都有反弹
- max|u_up| = 11.9–77.5，仍在 1e2 以内
- max|u_down| ≈ 1
- boundary ansatz exact check 全部通过
- near-infinity residual_up median = 9.0–81.8
- near-infinity residual_down median = 1.39–1.60

结论：
不要继续盲目延长 Stage-3 训练。
不要使用 final checkpoint。
后续必须使用 best checkpoint，即 epoch 3700 的 checkpoints/best_model.pt。

现在不要进入 Stage-4 训练。先做 Stage-4 preflight consistency diagnostic，只计算，不更新参数。

------------------------------------------------------------
任务 1：确认 git 状态和 run 目录
------------------------------------------------------------

运行：

git branch --show-current
git pull
git status
git log --oneline -5

确认当前分支是：

feature/autoencoder-asymptotic-decoder2

确认 Stage-3 run_dir：

outputs/autoencoder_stage3_ubasis_refine/20260515_222026_stage3_ubasis

检查：

summary.json
checkpoints/best_model.pt
checkpoints/latest_model.pt
diagnostics/

确认 best checkpoint 是：

outputs/autoencoder_stage3_ubasis_refine/20260515_222026_stage3_ubasis/checkpoints/best_model.pt

不要使用 latest_model.pt。

------------------------------------------------------------
任务 2：新增 Stage-4 preflight consistency diagnostic 脚本
------------------------------------------------------------

新增脚本：

scripts/diagnose_stage4_consistency.py

目标：
在不训练的情况下，检查 frozen Stage-1 R_pred、Stage-2 B_inc/B_ref、Stage-3 u_up/u_down 是否已经能在靠近无穷远区域满足：

R_pred / P
≈
(B_inc * u_down * A_down + B_ref * u_up * A_up) / P

必须使用 scattering convention：

R_in = B_ref * u_up * A_up + B_inc * u_down * A_down

其中：

A_up   = r^3 exp(i omega r_*)
A_down = r^{-1} exp(-i omega r_*)

注意：
不要写反 B_inc 和 B_ref。
B_ref 配 u_up * A_up。
B_inc 配 u_down * A_down。

------------------------------------------------------------
任务 3：输入文件和 checkpoint
------------------------------------------------------------

脚本输入：

python scripts/diagnose_stage4_consistency.py \
  --stage1-artifact outputs/stage1_artifacts/patch_000_logw_v2 \
  --stage2-checkpoint outputs/autoencoder_stage2_amplitude_train/20260515_192625_stage2_amplitude/checkpoints/best_model.pt \
  --stage3-checkpoint outputs/autoencoder_stage3_ubasis_refine/20260515_222026_stage3_ubasis/checkpoints/best_model.pt \
  --config config/autoencoder_stage3_ubasis_refine.yaml \
  --device cuda \
  --n-param 32 \
  --n-y 128 \
  --y-min -0.999 \
  --y-max -0.95 \
  --output-dir outputs/stage4_preflight/patch_000_logw_v2

其中：
- stage1 artifact 提供 R_pred、R_pred_over_P、S_pred、P、h2；
- stage2 checkpoint 提供 AmplitudeNet；
- stage3 checkpoint 提供 up_decoder/down_decoder；
- config 提供模型结构和 dtype。

如果 Stage-3 checkpoint 已经包含 Stage-2 amplitude_net，理论上可以只加载 Stage-3 checkpoint。但为了安全，脚本需要检查：
- stage3 checkpoint 中 amplitude_net 是否等于 stage2 checkpoint；
- 如果不一致，默认以 stage3 checkpoint 为整体模型，但打印 warning；
- 最好保存比较结果。

------------------------------------------------------------
任务 4：诊断计算内容
------------------------------------------------------------

对每个参数样本和 y-grid：

1. 从模型得到：
   f_up = model.predict_u_up(...)
   f_down = model.predict_u_down(...)

2. 用 infinity_slopes_y 得到：
   c_up, c_down

3. 用 compose_u_from_f 得到：
   u_up, u_down

4. 用 model.predict_amplitudes 得到：
   B_inc, B_ref

5. 用 u_basis.A_up/A_down 得到：
   A_up, A_down

6. 构造：

R_combo_over_P =
(B_ref * u_up * A_up + B_inc * u_down * A_down) / P

注意顺序：
B_ref * u_up * A_up
B_inc * u_down * A_down

7. 从 Stage-1 artifact 读取或重新插值得到：

R_pred_over_P

如果 artifact 的 y-grid 和当前 diagnostic y-grid 不一致：
- 优先重新用 Stage-1 model / RinDecoder 预测 S_pred；
- 或者使用插值，但必须说明；
- 不要把不同 y-grid 的数组直接相减。

8. 计算：

abs_err = |R_pred_over_P - R_combo_over_P|

rel_err = abs_err / (|R_pred_over_P| + eps)

分别统计：
- median_rel_err_all
- max_rel_err_all
- median_rel_err_near_inf
- max_rel_err_near_inf
- 每个 parameter sample 的 median/max
- 按 omega 排序的误差表
- low omega cases 的误差

------------------------------------------------------------
任务 5：同时诊断 up/down 贡献比例
------------------------------------------------------------

为了判断误差来自哪里，额外输出：

term_up_over_P =
B_ref * u_up * A_up / P

term_down_over_P =
B_inc * u_down * A_down / P

统计：
- median |term_up_over_P|
- median |term_down_over_P|
- median |term_up| / |term_down|
- 是否某一项完全主导；
- 如果 up term 主导且 consistency error 大，则说明 u_up 仍是瓶颈。

------------------------------------------------------------
任务 6：输出诊断文件
------------------------------------------------------------

保存：

outputs/stage4_preflight/patch_000_logw_v2/
  consistency_summary.json
  consistency_summary.md
  consistency_cases.csv
  plots/
    rel_err_vs_y_case_*.png
    abs_terms_vs_y_case_*.png
    combo_vs_rpred_case_*.png

summary.md 里必须写：

- 使用的 Stage-1 artifact；
- 使用的 Stage-2 checkpoint；
- 使用的 Stage-3 checkpoint；
- y-grid 范围；
- n_param, n_y；
- median_rel_err_all；
- max_rel_err_all；
- median_rel_err_near_inf；
- max_rel_err_near_inf；
- worst 5 cases；
- up/down contribution ratio；
- 是否建议进入 Stage-4 training。

------------------------------------------------------------
任务 7：给出进入 Stage-4 的判据
------------------------------------------------------------

诊断后按以下标准判断：

可以进入 Stage-4 low-lr polishing 的条件：

1. median_rel_err_near_inf < 0.1
   最好 < 0.05；

2. worst cases 不出现 > 1 的爆炸误差；

3. combo 和 R_pred_over_P 在 y 方向相位趋势一致；

4. up/down contribution 没有明显数值病态；

5. 所有值 finite，无 NaN/Inf。

如果满足：
下一轮进入 Stage-4 low-lr consistency polishing。

如果不满足：
不要 Stage-4。
先做 Stage-3 targeted refine：
- 只训练 up_decoder；
- 使用 near-infinity weighted u-equation residual；
- lr 降到 5e-6 或 1e-5；
- 从 stage3 best checkpoint 继续；
- loss 中加权 y in [-0.999,-0.95]；
- 不动 down_decoder，除非 down 也明显影响 consistency。

------------------------------------------------------------
任务 8：修复诊断脚本中的边界误报
------------------------------------------------------------

如果 diagnose_stage3_ubasis_solution.py 还没提交修复，确保修复：

- boundary exact check 用 y=-1 精确检查；
- PDE residual grid 继续用 y >= -1 + y_eps；
- 不要用 clipped grid 判断 u(-1)；
- autograd 检查 Re 和 Im 两部分的导数。

如果已经修复，报告 commit hash。

------------------------------------------------------------
任务 9：本轮不要做训练
------------------------------------------------------------

本轮只做 Stage-4 preflight diagnostic，不训练。

不要：
- 不要 Stage-4 optimizer；
- 不要 consistency polishing；
- 不要更新任何模型参数；
- 不要继续 Stage-3 训练；
- 不要重新训练 Stage-2；
- 不要重新导出 Stage-1 artifact。

------------------------------------------------------------
任务 10：提交并 push
------------------------------------------------------------

如果新增了诊断脚本：

git add scripts/diagnose_stage4_consistency.py scripts/diagnose_stage3_ubasis_solution.py
git commit -m "Add Stage-4 consistency preflight diagnostics"
git push

大型 outputs 不要提交。

最终报告必须包含：

1. 当前 commit hash；
2. 使用的 Stage-1 artifact；
3. 使用的 Stage-2 checkpoint；
4. 使用的 Stage-3 best checkpoint；
5. consistency median/max relative error；
6. worst 5 cases；
7. up/down contribution ratio；
8. 是否建议进入 Stage-4；
9. 如果不建议，具体是 u_up 还是其他项导致。