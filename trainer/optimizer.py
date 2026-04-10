# 优化器与学习率调度器配置
# 支持Adam/AdamW/LBFGS等，以及warmup/cosine annealing等策略

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau


def build_optimizer(model, config):
    """
    Build optimizer from config.

    Args:
        model: nn.Module to optimize
        config: Dict with optimizer settings

    Returns:
        torch.optim.Optimizer
    """
    opt_config = config.get('optimizer', {})
    opt_type = opt_config.get('type', 'adam')
    lr = opt_config.get('lr', 1e-3)
    weight_decay = opt_config.get('weight_decay', 0.0)

    if opt_type.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=opt_config.get('betas', (0.9, 0.999)),
            weight_decay=weight_decay
        )
    elif opt_type.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=opt_config.get('betas', (0.9, 0.999)),
            weight_decay=weight_decay
        )
    elif opt_type.lower() == 'lbfgs':
        optimizer = optim.LBFGS(
            model.parameters(),
            lr=lr,
            max_iter=opt_config.get('max_iter', 20),
            history_size=opt_config.get('history_size', 100)
        )
    elif opt_type.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    return optimizer


def build_scheduler(optimizer, config):
    """
    Build learning rate scheduler from config.

    Args:
        optimizer: torch.optim.Optimizer
        config: Dict with scheduler settings

    Returns:
        torch.optim.lr_scheduler or None
    """
    sched_config = config.get('scheduler', {})
    if not sched_config or sched_config.get('type') is None:
        return None

    sched_type = sched_config['type']

    if sched_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=sched_config.get('T_max', 1000),
            eta_min=sched_config.get('eta_min', 0.0)
        )
    elif sched_type == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=sched_config.get('step_size', 100),
            gamma=sched_config.get('gamma', 0.1)
        )
    elif sched_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=sched_config.get('factor', 0.5),
            patience=sched_config.get('patience', 10),
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")

    return scheduler


class WarmupScheduler:
    """
    Learning rate warmup wrapper.

    Gradually increases learning rate from 0 to target over warmup_steps.
    """

    def __init__(self, optimizer, warmup_steps, base_scheduler=None):
        """
        Args:
            optimizer: torch.optim.Optimizer
            warmup_steps: Number of warmup steps
            base_scheduler: Optional scheduler to use after warmup
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        self.current_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, metrics=None):
        """Step the scheduler."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Warmup phase: linear ramp
            lr = self.base_lr * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif self.base_scheduler is not None:
            # After warmup: use base scheduler
            if isinstance(self.base_scheduler, ReduceLROnPlateau):
                if metrics is not None:
                    self.base_scheduler.step(metrics)
            else:
                self.base_scheduler.step()

    def get_last_lr(self):
        """Get current learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]
