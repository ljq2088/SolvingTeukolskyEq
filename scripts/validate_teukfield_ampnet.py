#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_teukfield_ampnet import build_model
from teukfield.training.validation import validate_random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/teukfield_ampnet_moderate_patch.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/teukfield_ampnet/validation")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg).to(args.device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    summary = validate_random(model, cfg, args.output_dir, args.device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

