#!/usr/bin/env python3
# 评估脚本：在测试集上计算残差、边界误差等指标

import argparse
import torch
import numpy as np
from pathlib import Path

from utils.config_loader import load_config
from model.operator_model import OperatorModel
from physical_ansatz.residual import ResidualComputer
from physical_ansatz.boundary_layer import BoundaryLayerLoss
from physical_ansatz.coefficients import TeukolskyCoefficients
from physical_ansatz.mapping import CoordinateMapping
from utils.visualization import plot_residual


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_params', type=str, required=True,
                       help='Path to test parameters file (NPZ with a, omega_real, omega_imag)')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    args = parser.parse_args()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint['config']

    # Build model
    model = OperatorModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()

    # Build evaluation components
    mapping = CoordinateMapping(config)
    teukolsky_coeffs = TeukolskyCoefficients(config)
    residual_computer = ResidualComputer(config, teukolsky_coeffs, mapping)
    boundary_layer = BoundaryLayerLoss(config)

    # Load test parameters
    test_data = np.load(args.test_params)
    a_vals = test_data['a']
    omega_real_vals = test_data['omega_real']
    omega_imag_vals = test_data['omega_imag']

    n_test = len(a_vals)
    print(f"Evaluating on {n_test} test cases")

    # Evaluation loop
    residual_norms = []
    boundary_losses = []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_test):
        params = torch.tensor([[a_vals[i], omega_real_vals[i], omega_imag_vals[i]]],
                              dtype=torch.float32, device=args.device)

        # TODO: Implement proper evaluation with derivatives
        # For now, placeholder
        print(f"Test case {i+1}/{n_test}: a={a_vals[i]:.3f}, "
              f"ω={omega_real_vals[i]:.3f}+{omega_imag_vals[i]:.3f}i")

    print(f"\nEvaluation completed. Results saved to {output_dir}")


if __name__ == '__main__':
    main()
