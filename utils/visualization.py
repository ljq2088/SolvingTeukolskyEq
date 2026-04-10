# 可视化工具：绘制解、残差、系数等

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path


def plot_solution(x, f_real, f_imag, params, save_path=None):
    """
    Plot complex solution f(x) = f_real + i*f_imag.

    Args:
        x: Spatial coordinate array
        f_real: Real part of solution
        f_imag: Imaginary part of solution
        params: Parameter dict (a, ω) for title
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Real part
    axes[0].plot(x, f_real, 'b-', linewidth=2)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Re[f(x)]')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"Real part (a={params.get('a', 'N/A'):.3f}, ω={params.get('omega', 'N/A')})")

    # Imaginary part
    axes[1].plot(x, f_imag, 'r-', linewidth=2)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Im[f(x)]')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Imaginary part')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_residual(x, residual, save_path=None):
    """
    Plot PDE residual.

    Args:
        x: Spatial coordinate
        residual: Complex residual array
        save_path: Optional save path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    residual_abs = np.abs(residual)
    ax.semilogy(x, residual_abs, 'k-', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('|Residual|')
    ax.set_title('PDE Residual (log scale)')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_coefficients(coeffs_real, coeffs_imag, save_path=None):
    """
    Plot Chebyshev coefficients.

    Args:
        coeffs_real: Real part of coefficients (N+1,)
        coeffs_imag: Imaginary part of coefficients
        save_path: Optional save path
    """
    n = len(coeffs_real)
    indices = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Real coefficients
    axes[0].stem(indices, coeffs_real, basefmt=' ')
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('Re[c_n]')
    axes[0].set_title('Real part of Chebyshev coefficients')
    axes[0].grid(True, alpha=0.3)

    # Imaginary coefficients
    axes[1].stem(indices, coeffs_imag, basefmt=' ', linefmt='r-', markerfmt='ro')
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('Im[c_n]')
    axes[1].set_title('Imaginary part of Chebyshev coefficients')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_training_curves(metrics_file, save_path=None):
    """
    Plot training curves from metrics JSON file.

    Args:
        metrics_file: Path to metrics.jsonl file
        save_path: Optional save path
    """
    import json

    steps = []
    losses = []

    with open(metrics_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            steps.append(entry['step'])
            losses.append(entry.get('loss', entry.get('total_loss', 0)))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(steps, losses, 'b-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
