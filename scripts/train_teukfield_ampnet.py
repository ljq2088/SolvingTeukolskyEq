#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from teukfield.models import TeukfieldAmpNet
from teukfield.data.anchors import SpectralBoundaryAdapter
from teukfield.training.trainer import train_dry_run


def build_model(cfg: dict) -> TeukfieldAmpNet:
    mcfg = cfg["model"]
    acfg = cfg["amplitude_head"]
    pcfg = cfg["physics"]
    return TeukfieldAmpNet(
        latent_dim=mcfg["latent_dim"],
        fourier_bands=mcfg["fourier_bands"],
        n_windows=mcfg["n_windows"],
        window_overlap=mcfg["window_overlap"],
        local_hidden_dim=mcfg["local_hidden_dim"],
        local_depth=mcfg["local_depth"],
        amp_hidden_dim=acfg["hidden_dim"],
        amp_depth=acfg["depth"],
        m=pcfg["m"],
        s=pcfg["s"],
    ).to(torch.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/teukfield_ampnet_moderate_patch.yaml")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/teukfield_ampnet/dry_run")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--boundary-patch-dir", default=None)
    parser.add_argument("--boundary-cache-size", type=int, default=0)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg).to(args.device)
    adapter = SpectralBoundaryAdapter(args.boundary_patch_dir) if args.boundary_patch_dir else None
    train_dry_run(
        model,
        cfg,
        args.steps,
        args.output_dir,
        args.device,
        spectral_adapter=adapter,
        stage=args.stage,
        boundary_cache_size=args.boundary_cache_size,
    )


if __name__ == "__main__":
    main()
