from __future__ import annotations
import sys

sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
import argparse

from trainer.multipatch_atlas_trainer import MultiPatchAtlasTrainer


def main():
    parser = argparse.ArgumentParser(description="Full multi-patch atlas training.")
    parser.add_argument("--cfg", type=str, default="config/pinn_config.yaml")
    parser.add_argument("--probe-json", type=str, default="outputs/domain/probe_l2_m2_with_ramp.json")
    parser.add_argument("--atlas-json", type=str, default="outputs/domain/atlas_l2_m2.json")
    parser.add_argument("--patch-json", type=str, default="outputs/domain/patch_cover_l2_m2.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--anchor-enabled", action="store_true")
    parser.add_argument("--n-anchor-y", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    trainer = MultiPatchAtlasTrainer(
        cfg_path=args.cfg,
        probe_json=args.probe_json,
        atlas_json=args.atlas_json,
        patch_json=args.patch_json,
        device=args.device,
        anchor_enabled=args.anchor_enabled,
        n_anchor_y=args.n_anchor_y,
        verbose=(not args.quiet),
    )
    registry_path = trainer.train_all()
    print(f"[done] atlas registry -> {registry_path}")


if __name__ == "__main__":
    main()