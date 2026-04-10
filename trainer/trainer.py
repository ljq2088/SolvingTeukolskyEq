# 训练循环：epoch迭代、loss计算、梯度更新、checkpoint保存、日志记录
# 支持分布式训练和混合精度

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import os
import time
from pathlib import Path
import yaml


class Trainer:
    """
    Training loop for neural operator.

    Handles:
    - Batch iteration and loss computation
    - Gradient updates and optimization
    - Checkpointing and logging
    - Validation and early stopping
    """

    def __init__(self, model, loss_fn, optimizer, scheduler, config):
        """
        Args:
            model: OperatorModel instance
            loss_fn: CompositeLoss instance
            optimizer: torch optimizer
            scheduler: Learning rate scheduler
            config: Configuration dict
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        # Training settings
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = config.get('num_epochs', 1000)
        self.log_interval = config.get('log_interval', 10)
        self.checkpoint_interval = config.get('checkpoint_interval', 100)

        # Mixed precision training
        self.use_amp = config.get('use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None

        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.model.to(self.device)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

    def train_epoch(self, dataloader):
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader yielding batches

        Returns:
            dict with epoch metrics
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            params = batch['params'].to(self.device)
            x = batch['x_points'].to(self.device)

            # Forward pass
            if self.use_amp:
                with autocast():
                    model_output = self.model(params, x)
                    loss_dict = self.loss_fn(model_output, x, params, self.model)
                    loss = loss_dict['total']
            else:
                model_output = self.model(params, x)
                loss_dict = self.loss_fn(model_output, x, params, self.model)
                loss = loss_dict['total']

            # Backward pass
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Accumulate metrics
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if batch_idx % self.log_interval == 0:
                print(f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss.item():.6f}")

        # Average loss
        avg_loss = epoch_loss / num_batches

        return {'loss': avg_loss}

    def validate(self, dataloader):
        """
        Validate on validation set.

        Args:
            dataloader: Validation DataLoader

        Returns:
            dict with validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                params = batch['params'].to(self.device)
                x = batch['x_points'].to(self.device)

                model_output = self.model(params, x)
                loss_dict = self.loss_fn(model_output, x, params, self.model)
                loss = loss_dict['total']

                val_loss += loss.item()
                num_batches += 1

        avg_loss = val_loss / num_batches
        return {'loss': avg_loss}

    def save_checkpoint(self, filename=None):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded: {checkpoint_path}")

    def train(self, train_loader, val_loader=None):
        """
        Main training loop.

        Args:
            train_loader: Training DataLoader
            val_loader: Optional validation DataLoader
        """
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Device: {self.device}")

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader)
            epoch_time = time.time() - start_time

            print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s")
            print(f"Train loss: {train_metrics['loss']:.6f}")

            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                print(f"Val loss: {val_metrics['loss']:.6f}")

                # Save best model
                if val_metrics['loss'] < self.best_loss:
                    self.best_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pt')

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'] if val_loader else train_metrics['loss'])
                else:
                    self.scheduler.step()

            # Periodic checkpoint
            if (epoch + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint()

        print("\nTraining completed!")
