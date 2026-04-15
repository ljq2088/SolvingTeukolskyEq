---
name: scigym
description: 当用户想自动化科研实验循环、包装研究环境为CLI/Gym/API、或部署24/7自主实验时触发。适用场景：有模拟器/实验装置想让Agent自主运行实验；想把重复性科研流程自动化；想把研究环境发布给社区复用。
---

# SciGym — Automated Research Machine

## Pipeline

```
Phase 1: Benchmark  →  Phase 2: CLI Environment  →  Phase 3: Interface  →  Phase 4: AutoRun  →  Submit
  从论文提取指标        包装 CLI + logging              连接真实硬件/计算       Cryochamber 部署      发布
```

## 执行步骤

1. 询问用户当前处于哪个 Phase，或从头开始
2. 按对应子文档引导：
   - Phase 1: 从论文/硬件提取基准指标，建立独立可验证的 benchmark
   - Phase 2: 包装 CLI + logging（优先），按需升级到 FastAPI / Gymnasium
   - Phase 3: 连接真实硬件/计算集群接口
   - Phase 4: 部署 Cryochamber 实现 24/7 自主实验，配置 Zulip 推送
   - Submit: 生成 scigym.json，发布到社区 registry

## 核心原则

- CLI-first：Agent 直接 subprocess 调用，每步记录 `--motivation`
- Benchmark 独立：reward 不能被 hack，需有置信区间
- AutoRun 安全：参数边界检查 + N 次无改进后休眠

## 快速启动

```bash
python run_experiment.py s21 --motivation "找腔频" --qubit Q0
python show_log.py --last 10
```
