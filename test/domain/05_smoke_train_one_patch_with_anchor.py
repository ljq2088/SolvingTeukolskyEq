from __future__ import annotations
import sys
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
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
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--n-anchor-y", type=int, default=12)
    parser.add_argument("--mma-interp-n-grid", type=int, default=1200)
    args = parser.parse_args()

    trainer = AtlasPatchTrainer(
        cfg_path=args.cfg,
        probe_json=args.probe_json,
        atlas_json=args.atlas_json,
        patch_json=args.patch_json,
        patch_id=args.patch_id,
        device=args.device,
        anchor_enabled=True,
        anchor_weight=args.anchor_weight,
        n_anchor_y=args.n_anchor_y,
        mma_interp_n_grid=args.mma_interp_n_grid,
    )

    print("=" * 80)
    print(f"Patch id       : {trainer.patch.patch_id}")
    print(f"Component id   : {trainer.patch.component_id}")
    print(f"Patch center   : (u,v)=({trainer.patch.u_center:.6f}, {trainer.patch.v_center:.6f})")
    print(f"Patch phys ctr : (a,omega)=({trainer.patch.a_center:.6f}, {trainer.patch.omega_center:.6f})")
    print(f"Patch pool size: {len(trainer.patch_aw)}")
    print(f"Anchor enabled : {trainer.anchor_enabled}")
    print(f"Anchor weight  : {trainer.anchor_weight}")
    print("=" * 80)

    hist_total = []
    hist_pde = []
    hist_anchor = []

    for step in range(1, args.steps + 1):
        info = trainer.train_one_step()

        loss_pde = float(info["loss_interior"])
        loss_anchor = float(info["loss_anchor"])
        loss_total = float(info["total_loss"])

        if not math.isfinite(loss_pde):
            raise RuntimeError(f"Non-finite PDE loss at step {step}: {loss_pde}")
        if not math.isfinite(loss_anchor):
            raise RuntimeError(f"Non-finite anchor loss at step {step}: {loss_anchor}")
        if not math.isfinite(loss_total):
            raise RuntimeError(f"Non-finite total loss at step {step}: {loss_total}")
        if not math.isfinite(info["grad_norm"]):
            raise RuntimeError(f"Non-finite grad_norm at step {step}: {info['grad_norm']}")

        hist_pde.append(loss_pde)
        hist_anchor.append(loss_anchor)
        hist_total.append(loss_total)

        print(
            f"[step {step:04d}] "
            f"pde={loss_pde:.6e}, "
            f"anchor={loss_anchor:.6e}, "
            f"total={loss_total:.6e}, "
            f"grad={info['grad_norm']:.6e}"
        )

    hist_total = np.asarray(hist_total, dtype=float)
    hist_pde = np.asarray(hist_pde, dtype=float)
    hist_anchor = np.asarray(hist_anchor, dtype=float)

    print("=" * 80)
    print(f"initial total  = {hist_total[0]:.6e}")
    print(f"final total    = {hist_total[-1]:.6e}")
    print(f"initial pde    = {hist_pde[0]:.6e}")
    print(f"final pde      = {hist_pde[-1]:.6e}")
    print(f"initial anchor = {hist_anchor[0]:.6e}")
    print(f"final anchor   = {hist_anchor[-1]:.6e}")
    print("[check] atlas patch + anchor smoke test passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()