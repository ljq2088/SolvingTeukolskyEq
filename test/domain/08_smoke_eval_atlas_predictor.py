from __future__ import annotations

import argparse
import torch

from inference.atlas_predictor import AtlasPredictor


def main():
    parser = argparse.ArgumentParser(description="Smoke test for atlas predictor.")
    parser.add_argument("--registry-json", type=str, required=True)
    parser.add_argument("--a", type=float, required=True)
    parser.add_argument("--omega", type=float, required=True)
    parser.add_argument("--n-y", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    predictor = AtlasPredictor(
        registry_json=args.registry_json,
        device=args.device,
    )

    y = torch.linspace(-0.95, 0.95, args.n_y)
    f = predictor.predict_f(a=args.a, omega=args.omega, y=y)

    print("=" * 80)
    print(f"a={args.a}, omega={args.omega}")
    print(f"y.shape = {tuple(y.shape)}")
    print(f"f.shape = {tuple(f.shape)}")
    print("first 5 values:")
    for i in range(min(5, f.numel())):
        print(i, y[i].item(), f[i].item())
    print("=" * 80)


if __name__ == "__main__":
    main()