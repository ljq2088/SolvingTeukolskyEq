你现在在 test2 分支上工作。请先确认当前仓库里已有：
- pybhpt/compute_solution.py
- 其中存在函数 compute_pybhpt_solution(a, omega, ell=2, m=2, r_grid=None, timeout=10.0)

然后只做这一轮任务：

==================================================
目标
==================================================
把原来 07_train_all_patches_full.py 训练流程中“用于 benchmark / 可视化对比”的 Mathematica 路径，全部替换成 pybhpt。
注意：
1. 这一轮只替换 benchmark / visualization 流程
2. 不要动现有的 MMA anchor 逻辑（如果 anchor_enabled=True 仍然可以保留原逻辑）
3. Mathematica benchmark 暂时关闭
4. pybhpt benchmark 如果失败，要自动跳过，训练继续进行
5. 终端仍然保持干净的 tqdm 进度条，不要打印 pybhpt 报错到终端
6. 改完后请给我：
   - 修改文件列表
   - 每个文件的改动摘要
   - 运行 07 所需命令
   - 一个简短的 smoke run 结果摘要

==================================================
需要修改的文件
==================================================
重点文件：
1. trainer/atlas_patch_trainer.py
2. config/pinn_config.yaml
3. 如有必要，可新建一个很小的 pybhpt wrapper 文件，但优先直接复用 pybhpt/compute_solution.py
4. test/domain/07_train_all_patches_full.py 一般不需要大改，除非你觉得 CLI 需要补一个开关

==================================================
任务 A：trainer/atlas_patch_trainer.py
==================================================

【A1】替换 benchmark 依赖
当前 atlas_patch_trainer.py 里：
- import 了 MathematicaRinSampler
- __init__ 里通过 mma_cfg 初始化 self.mma_sampler
- visualize_reference() 里使用 self.mma_sampler.evaluate_rin_at_points_direct(...)
- 并由 self.viz_mma_enabled 控制

请做如下修改：

1. 在文件顶部：
   - 删除 benchmark 用途的 MathematicaRinSampler 依赖
   - 改为：
     from pybhpt.compute_solution import compute_pybhpt_solution

2. 保留 MMA anchor 逻辑所需的 MathematicaRinSampler 仅在 anchor_enabled=True 时可用。
   也就是说：
   - benchmark 用 pybhpt
   - anchor 若启用，仍可单独保留 MMA sampler
   - 不要把两者混在一个 self.mma_sampler 概念里

建议：
- 新增 self.anchor_mma_sampler（只服务 anchor）
- benchmark 不需要持久 sampler，因为 pybhpt 已经是独立函数 + timeout subprocess

【A2】在 __init__ 中增加 pybhpt benchmark 开关
当前有：
- self.viz_mma_enabled

请改成更清晰的结构，例如：
- self.viz_benchmark_backend = atlas_train_cfg.get("viz_benchmark_backend", "pybhpt")
- 允许取值：
  - "none"
  - "pybhpt"
  - "mma"   （先保留兼容，但默认不用）
- 同时把当前的 viz_mma_enabled 当作旧字段兼容：
  - 如果 yaml 里还有 viz_mma_enabled，就打印一次兼容说明，但默认仍以 viz_benchmark_backend 为准

目标：
- 默认 benchmark backend = pybhpt
- Mathematica benchmark 默认关闭

【A3】修改 visualize_reference()
当前 visualize_reference() 的 benchmark 流程是：
- 先预测模型 R_pred / Rprime_pred
- 如果 viz_mma_enabled and mma_sampler is not None:
    调 MMA 求 R_mma_r / R_mma_y
    再和模型对比作图
- 如果失败写 logs/viz_failures.jsonl

请改成：

1. 保持模型预测部分不变
2. benchmark backend 分三种：
   - "none"：
       只画模型曲线，不做任何外部参考
   - "pybhpt"：
       用 compute_pybhpt_solution(...) 在 r_grid_uniform 和 r_grid_from_y 上求参考值
   - "mma"：
       保留旧逻辑，但默认不用
3. pybhpt 的调用方式：
   - 直接把当前 r_grid_uniform / r_grid_from_y 转 numpy 传给 compute_pybhpt_solution(..., r_grid=..., timeout=...)
   - 注意 ell=l, m=m, a=float(a_t), omega=float(omega_t)
4. pybhpt 返回：
   - r_values, R_in_values
   你只需要拿第二个值作为参考解
5. 失败跳过逻辑：
   - try/except 包住 pybhpt benchmark
   - 失败时：
       - 不要 raise
       - 不要污染终端
       - 只把失败信息写入 logs/viz_failures.jsonl
       - 图仍然保存，但只画模型曲线
6. 图标题里的 mma_status 改成更通用的 benchmark_status，例如：
   - "benchmark=pybhpt-ok"
   - "benchmark=pybhpt-failed"
   - "benchmark=none"
   - "benchmark=mma-ok"
7. 图例里把 “MMA” 改成 “pybhpt” 或更通用的 “ref”

【A4】日志要保持干净
pybhpt 失败时，不要 print 到终端。
只允许：
- tqdm 进度条
- 最终训练摘要
- 必要的 run_dir / summary 路径信息

所有 pybhpt benchmark 失败信息都写入：
- logs/viz_failures.jsonl

【A5】不要破坏 anchor
当前 train_one_step() 里的 anchor 查询逻辑：
- query_mma_Rin_batch()
- _append_anchor_failures()
- dynamic anchor weight

这一轮不要重构它。
只要保证：
- anchor 继续可用
- benchmark 换成 pybhpt 后，两者互不干扰

==================================================
任务 B：config/pinn_config.yaml
==================================================

在 atlas_training 下增加或修改这些键：

1. 新增：
   viz_benchmark_backend: pybhpt

2. 兼容保留：
   viz_mma_enabled: false
   但 trainer 中应以 viz_benchmark_backend 为主，不再真正依赖 viz_mma_enabled

3. 新增 pybhpt benchmark 的 timeout 配置，例如：
   viz_pybhpt_timeout: 10.0

如果你觉得更合理，也可以单独加：
- pybhpt:
    timeout: 10.0
但要保持调用路径清楚。

==================================================
任务 C：如果你觉得需要，补一个小 helper
==================================================

如果 atlas_patch_trainer.py 里直接写 pybhpt 调用太乱，可以新建一个小文件，例如：

- pybhpt/benchmark_wrapper.py

提供一个函数，例如：
- evaluate_pybhpt_rin_at_points(a, omega, ell, m, r_query, timeout)

内部只是简单调用 compute_pybhpt_solution(...)

要求：
- 成功时返回 complex numpy array
- 失败时 raise RuntimeError / TimeoutError
- 不打印终端垃圾

但如果你觉得没必要，也可以不新建，直接在 trainer 里调用 compute_pybhpt_solution。

==================================================
任务 D：07 脚本最小兼容
==================================================

检查 test/domain/07_train_all_patches_full.py 是否需要新增 CLI 开关。
如果你认为需要，可以加一个可选参数：
- --no-benchmark
- --benchmark-backend {pybhpt,mma,none}

但这一轮不是必须。
如果不改 07，也可以，只要 trainer 从 yaml 中读到 viz_benchmark_backend=pybhpt 即可。

==================================================
任务 E：最终检查
==================================================

改完之后请你做这些检查并把结果发给我：

1. grep 检查 atlas_patch_trainer.py 中：
   - viz_mma_enabled
   - mma_sampler
   - compute_pybhpt_solution
   - benchmark_status
2. 确认：
   - train_one_step() 的 anchor 路径仍然存在
   - visualize_reference() 的 benchmark 路径已切到 pybhpt
3. 用一个小规模命令跑一次（例如 max_patches=1 或 3），确认：
   - 训练可以启动
   - tqdm 干净
   - figures 正常生成
   - logs/viz_failures.jsonl 若有失败则写入
4. 返回给我：
   - 修改文件列表
   - 关键 diff 摘要
   - 运行命令
   - smoke run 摘要

==================================================
特别注意
==================================================

- 这一轮不要改 amplitude / residual / model 接口
- 这一轮不要重写 multipatch trainer
- 这一轮不要动 atlas_predictor.py
- 这一轮只做：
  “训练中的 Mathematica benchmark -> pybhpt benchmark”
  并关闭 Mathematica benchmark