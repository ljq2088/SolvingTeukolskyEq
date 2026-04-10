#!/usr/bin/env python3
# 主训练脚本：加载配置→构建模型→训练循环→保存结果

import argparse
import torch
from pathlib import Path

from utils.config_loader import load_config
from model.operator_model import OperatorModel
from trainer.losses import CompositeLoss
from trainer.optimizer import build_optimizer, build_scheduler
from trainer.trainer import Trainer
from dataset.collocation import CollocationSampler
from physical_ansatz.residual import ResidualComputer
from physical_ansatz.boundary_layer import BoundaryLayerLoss
from physical_ansatz.coefficients import TeukolskyCoefficients
from physical_ansatz.mapping import CoordinateMapping


def main():
    parser = argparse.ArgumentParser(description='Train neural operator for Teukolsky equation')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    if args.device:
        config['device'] = args.device

    print("Configuration loaded:")
    print(config)

    # Build model
    model = OperatorModel(config)
    print(f"Model built with {sum(p.numel() for p in model.parameters())} parameters")

    # Build physical components
    mapping = CoordinateMapping(config)
    teukolsky_coeffs = TeukolskyCoefficients(config)
    residual_computer = ResidualComputer(config, teukolsky_coeffs, mapping)
    boundary_layer = BoundaryLayerLoss(config)

    # Build loss function
    loss_fn = CompositeLoss(config, residual_computer, boundary_layer)

    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # Build trainer
    trainer = Trainer(model, loss_fn, optimizer, scheduler, config)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Build data loaders
    # TODO: Implement proper DataLoader
    # For now, use placeholder
    train_loader = None
    val_loader = None

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
