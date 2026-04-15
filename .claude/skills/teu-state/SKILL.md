---
name: teu-state
description: 当用户问"当前训练状态是什么"、"模型保存在哪"、"参数配置是什么"、或需要了解项目运行状态时触发。
---

# Teukolsky 项目状态管理

## 快速状态检查

```bash
# 查看最新训练日志
tail -50 outputs/pinn_train.log

# 查看已保存模型
ls -lh saved_models/

# 查看输出图像
ls outputs/*.png

# 查看当前配置
cat config/pinn_config.yaml
cat config/teukolsky_radial.yaml
```

## 状态文件位置

| 状态 | 路径 |
|------|------|
| 训练日志 | `outputs/pinn_train.log` |
| PINN模型 | `saved_models/pinn_model.pt` |
| 谱方法结果 | `outputs/single_case_coeffs.pt` |
| 直接谱解 | `outputs/direct_spectral_a*.pt` |
| 训练图像 | `outputs/*.png` |

## checkpoint 结构

```python
# 保存
torch.save({
    'step': step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'config': cfg,
}, 'saved_models/pinn_model.pt')

# 加载
ckpt = torch.load('saved_models/pinn_model.pt')
model.load_state_dict(ckpt['model_state_dict'])
```

## 关键配置参数

```yaml
# config/pinn_config.yaml 核心字段
a_center / a_range        # 参数范围
hidden_dims               # 网络结构
n_interior / n_boundary   # 采样点数
n_steps / lr              # 训练超参
weight_boundary           # 边界权重
```

## 当前分支状态

```bash
git log --oneline -5      # 查看最近提交
git status                # 查看未提交修改
git stash list            # 查看暂存
```
