# pybhpt multiprocessing 死锁问题修复

## 问题描述

`benchmark/scripts/plot_pybhpt_solution_fixed_divU.py` 运行时会卡死进程，Ctrl+C 无法退出，只能 Ctrl+Z 然后 kill。

## 根本原因

1. **multiprocessing `spawn` 模式死锁**：
   - `compute_solution.py` 使用 `mp.get_context('spawn')` 创建子进程
   - `spawn` 模式会重新导入主模块，导致无限递归
   - 即使有 `if __name__ == "__main__":` 保护，模块级别的导入仍会执行

2. **计算量过大**：
   - 脚本参数：`viz_r_max = 1000.0`, `viz_num_points = 500`
   - pybhpt 需要计算 500 个点从 r≈2 到 r=1000
   - `timeout=30.0` 不够，实际需要 > 60 秒

## 修复方案

### 1. 修改 `compute_solution.py`（已完成）

```python
# 将 line 91 从：
ctx = mp.get_context('spawn')

# 改为：
ctx = mp.get_context('fork')
```

**原因**：
- `fork` 模式直接复制父进程内存，不重新导入模块
- 避免了 `spawn` 模式的死锁问题
- 性能更好（不需要重新初始化）

### 2. 调整脚本参数（可选）

如果仍然超时，修改 `plot_pybhpt_solution_fixed_divU.py`：

```python
# Line 66-68
viz_num_points = 200  # 从 500 减少到 200
viz_r_min = 2.0
viz_r_max = 100.0     # 从 1000 减少到 100

# Line 96
timeout=60.0          # 从 30 增加到 60
```

## 对训练的影响

### 正面影响

1. **修复潜在死锁**：训练时调用 `compute_pybhpt_solution` 不会再卡死
2. **性能提升**：`fork` 比 `spawn` 快，训练时频繁调用 pybhpt 会更快
3. **稳定性提升**：避免了 multiprocessing 的边缘情况

### 训练时的调用场景

- **Anchor 计算**（`atlas_patch_trainer.py:527`）：
  - 每个 anchor 点调用一次
  - timeout: `anchor_pybhpt_timeout`（默认 10 秒）
  
- **可视化 benchmark**（`atlas_patch_trainer.py:865, 874`）：
  - 每次可视化调用两次（r 域和 y 域）
  - timeout: `viz_pybhpt_timeout`（默认 10 秒）

训练时的 r 范围通常较小（`viz_r_min=2.0`, `viz_r_max=80.0`），不会超时。

## 验证

```bash
# 测试修复后的脚本
python benchmark/scripts/plot_pybhpt_solution_fixed_divU.py

# 如果仍然超时，检查输出
# 应该看到：
# A: config loaded
# B: before pybhpt
# [卡在这里说明 pybhpt 计算超时]
```

## 注意事项

- `fork` 模式在 Linux/macOS 上安全，Windows 不支持
- 如果需要跨平台，考虑使用 `forkserver` 模式
- 训练时的 timeout 设置（10秒）比脚本（30秒）更短，说明训练时的计算量更小
