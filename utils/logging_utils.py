# 日志记录工具：TensorBoard/WandB集成

import os
from pathlib import Path
from typing import Dict, Any
import json


class Logger:
    """
    Simple logger for training metrics.

    Supports:
    - Console logging
    - JSON file logging
    - TensorBoard (optional)
    - Weights & Biases (optional)
    """

    def __init__(self, log_dir: str, use_tensorboard: bool = False, use_wandb: bool = False):
        """
        Args:
            log_dir: Directory for log files
            use_tensorboard: Enable TensorBoard logging
            use_wandb: Enable Weights & Biases logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb

        # JSON log file
        self.json_log_path = self.log_dir / 'metrics.jsonl'

        # TensorBoard
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
            except ImportError:
                print("TensorBoard not available, disabling")
                self.use_tensorboard = False

        # Weights & Biases
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("Weights & Biases not available, disabling")
                self.use_wandb = False

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """
        Log metrics at given step.

        Args:
            metrics: Dict of metric name -> value
            step: Global step number
        """
        # Console
        print(f"Step {step}: {metrics}")

        # JSON file
        log_entry = {'step': step, **metrics}
        with open(self.json_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # TensorBoard
        if self.use_tensorboard:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)

        # Weights & Biases
        if self.use_wandb:
            self.wandb.log(metrics, step=step)

    def close(self):
        """Close logger and flush buffers."""
        if self.use_tensorboard:
            self.tb_writer.close()
