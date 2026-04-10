#!/usr/bin/env python3
# 推理脚本：加载训练好的模型→给定(a,ω)→输出R(x)或f(x)

import argparse
import torch
import numpy as np
from pathlib import Path

from utils.config_loader import load_config
from model.operator_model import OperatorModel
from utils.visualization import plot_solution, plot_coefficients


def main():
    parser = argparse.ArgumentParser(description='Inference with trained neural operator')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--a', type=float, required=True, help='Kerr spin parameter')
    parser.add_argument('--omega_real', type=float, required=True, help='Real part of frequency')
    parser.add_argument('--omega_imag', type=float, default=0.0, help='Imaginary part of frequency')
    parser.add_argument('--n_points', type=int, default=100, help='Number of evaluation points')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cuda/cpu)')
    args = parser.parse_args()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint['config']

    # Build model
    model = OperatorModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()

    print(f"Model loaded from {args.checkpoint}")
    print(f"Epoch: {checkpoint['epoch']}, Best loss: {checkpoint['best_loss']:.6e}")

    # Prepare input
    params = torch.tensor([[args.a, args.omega_real, args.omega_imag]],
                          dtype=torch.float32, device=args.device)

    x = torch.linspace(0, 1, args.n_points, dtype=torch.float32, device=args.device)
    x = x.unsqueeze(0)  # (1, n_points)

    # Forward pass
    with torch.no_grad():
        output = model(params, x)

    # Extract results
    f_real = output['output_real'].cpu().numpy()[0]
    f_imag = output['output_imag'].cpu().numpy()[0]
    coeffs_real = output['coeffs_real'].cpu().numpy()[0]
    coeffs_imag = output['coeffs_imag'].cpu().numpy()[0]
    x_np = x.cpu().numpy()[0]

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save data
    np.savez(output_dir / 'solution.npz',
             x=x_np,
             f_real=f_real,
             f_imag=f_imag,
             coeffs_real=coeffs_real,
             coeffs_imag=coeffs_imag,
             params={'a': args.a, 'omega_real': args.omega_real, 'omega_imag': args.omega_imag})

    # Plot
    param_dict = {'a': args.a, 'omega': f"{args.omega_real:.3f} + {args.omega_imag:.3f}i"}
    plot_solution(x_np, f_real, f_imag, param_dict,
                  save_path=output_dir / 'solution.png')
    plot_coefficients(coeffs_real, coeffs_imag,
                     save_path=output_dir / 'coefficients.png')

    print(f"\nResults saved to {output_dir}")
    print(f"Solution range: Re[f] ∈ [{f_real.min():.3e}, {f_real.max():.3e}]")
    print(f"                Im[f] ∈ [{f_imag.min():.3e}, {f_imag.max():.3e}]")


if __name__ == '__main__':
    main()
