# """
# 测试PINN训练
# """
# import sys
# sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
# import os
# import torch
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from trainer.pinn_trainer import PINNTrainer

# if __name__ == "__main__":
#     cfg_path = "config/pinn_config.yaml"

#     # 使用CPU训练（可改为cuda）
#     device='cuda' if torch.cuda.is_available() else 'cpu'
#     trainer = PINNTrainer(cfg_path, device=device)

#     # 开始训练
#     trainer.train(save_dir='outputs/pinn')
"""
测试 PINN 训练入口
"""
from pathlib import Path
import sys
import torch

# 项目根目录：.../SolvingTeukolsky
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from trainer.pinn_trainer import PINNTrainer


if __name__ == "__main__":
    cfg_path = PROJECT_ROOT / "config" / "pinn_config.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = PINNTrainer(str(cfg_path), device=device)
    print(f"[train] cfg: {cfg_path}")
    print(f"[train] device: {device}")
    print(
        f"[train] parameter space: "
        f"a in [{trainer.a_center - trainer.a_range:.6f}, {trainer.a_center + trainer.a_range:.6f}], "
        f"omega in [{trainer.omega_center - trainer.omega_range:.6f}, {trainer.omega_center + trainer.omega_range:.6f}]"
    )
    print(
        f"[train] initialization: enabled={trainer.init_enabled}, "
        f"checkpoint={trainer._resolve_init_checkpoint_path() if trainer.init_enabled else None}"
    )
    print(
        f"[train] validation: mode={trainer.val_mode}, samples={trainer.val_param_samples}, "
        f"sampler={trainer.val_param_sampler}, monitor={trainer.val_monitor}"
    )
    trainer.train(save_dir=str(PROJECT_ROOT / "outputs" / "pinn"))
