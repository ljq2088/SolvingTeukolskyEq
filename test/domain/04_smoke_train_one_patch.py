from __future__ import annotations

import argparse
import math
import numpy as np

from trainer.atlas_patch_trainer import AtlasPatchTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config/pinn_config.yaml")
    parser.add_argument("--probe-json", type=str, default="outputs/domain/probe_l2_m2_with_ramp.json")
    parser.add_argument("--atlas-json", type=str, default="outputs/domain/atlas_l2_m2.json")
    parser.add_argument("--patch-json", type=str, default="outputs/domain/patch_cover_l2_m2.json")
    parser.add_argument("--patch-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    trainer = AtlasPatchTrainer(
        cfg_path=args.cfg,
        probe_json=args.probe_json,
        atlas_json=args.atlas_json,
        patch_json=args.patch_json,
        patch_id=args.patch_id,
        device=args.device,
    )

    print("=" * 80)
    print(f"Patch id       : {trainer.patch.patch_id}")
    print(f"Component id   : {trainer.patch.component_id}")
    print(f"Patch center   : (u,v)=({trainer.patch.u_center:.6f}, {trainer.patch.v_center:.6f})")
    print(f"Patch halfsize : (h_u,h_v)=({trainer.patch.h_u:.6f}, {trainer.patch.h_v:.6f})")
    print(f"Patch phys ctr : (a,omega)=({trainer.patch.a_center:.6f}, {trainer.patch.omega_center:.6f})")
    print(f"Patch pool size: {len(trainer.patch_aw)}")
    print("=" * 80)

    hist = []
    for step in range(1, args.steps + 1):
        info = trainer.train_one_step()
        loss_val = float(info["loss_interior"])

        if not math.isfinite(loss_val):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss_val}")
        if not math.isfinite(info["grad_norm"]):
            raise RuntimeError(f"Non-finite grad_norm at step {step}: {info['grad_norm']}")

        hist.append(loss_val)

        if step == 1 or step % 5 == 0 or step == args.steps:
            print(
                f"[step {step:04d}] "
                f"loss_interior={info['loss_interior']:.6e}, "
                f"total={info['total_loss']:.6e}, "
                f"grad={info['grad_norm']:.6e}, "
                f"patch_pool={info['patch_pool_size']}"
            )

    hist = np.asarray(hist, dtype=float)
    print("=" * 80)
    print(f"initial loss = {hist[0]:.6e}")
    print(f"final loss   = {hist[-1]:.6e}")
    print(f"min loss     = {hist.min():.6e}")
    print(f"max loss     = {hist.max():.6e}")
    print("[check] atlas patch trainer smoke test passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()