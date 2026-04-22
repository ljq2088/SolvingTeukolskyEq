from __future__ import annotations
import sys

sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
import os
import argparse
from pathlib import Path
import random

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-pinn")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from domain.atlas_builder import (
    load_probe_grid,
    build_atlas_from_probe,
    save_atlas,
    map_to_chart,
    map_from_chart,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe-json",
        type=str,
        default="outputs/domain/probe_l2_m2.json",
    )
    parser.add_argument(
        "--atlas-json",
        type=str,
        default="outputs/domain/atlas_l2_m2.json",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="outputs/domain/atlas_l2_m2.png",
    )
    parser.add_argument("--min-component-size", type=int, default=20)
    parser.add_argument("--vertical-hole-max-gap", type=int, default=2)
    parser.add_argument("--n-check", type=int, default=50)
    args = parser.parse_args()

    probe = load_probe_grid(args.probe_json)
    atlas = build_atlas_from_probe(
        probe,
        min_component_size=args.min_component_size,
        vertical_hole_max_gap=args.vertical_hole_max_gap,
    )
    save_atlas(atlas, args.atlas_json)

    print("=" * 80)
    print(f"Loaded probe: {args.probe_json}")
    print(f"raw components  = {atlas.n_components_raw}")
    print(f"kept components = {atlas.n_components_kept}")
    print("-" * 80)
    for comp in atlas.components:
        print(
            f"[component {comp.component_id}] "
            f"raw={comp.n_points_raw}, "
            f"filled={comp.n_points_filled}, "
            f"filled_holes={comp.n_filled_hole_cells}, "
            f"a_support=[{comp.a_support[0]:.6f}, {comp.a_support[-1]:.6f}], "
            f"omega_bbox=[{comp.omega_lower[0]:.6f}, {max(comp.omega_upper):.6f}]"
        )
    print(f"saved atlas -> {args.atlas_json}")
    print("=" * 80)

    # -------- forward / inverse consistency check --------
    if len(atlas.components) > 0:
        comp = atlas.components[0]
        errs_a = []
        errs_w = []

        rng = random.Random(1234)
        for _ in range(args.n_check):
            # 从 envelope 内随机取点做映射一致性检查
            a = rng.uniform(comp.a_support[0], comp.a_support[-1])

            low = float(np.interp(a, comp.a_support, comp.omega_lower))
            up = float(np.interp(a, comp.a_support, comp.omega_upper))
            omega = rng.uniform(low, up)

            u, v = map_to_chart(comp, a, omega)
            a2, w2 = map_from_chart(comp, u, v)

            errs_a.append(abs(a2 - a))
            errs_w.append(abs(w2 - omega))

        print(f"chart check: max |Δa| = {max(errs_a):.3e}, max |Δomega| = {max(errs_w):.3e}")

    # -------- plotting --------
    a_vals = probe.a_vals
    w_vals = probe.omega_vals
    mask = probe.valid_mask.astype(float)  # 1=valid, 0=invalid

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        mask.T,
        origin="lower",
        aspect="auto",
        extent=[a_vals.min(), a_vals.max(), w_vals.min(), w_vals.max()],
    )
    plt.colorbar(im, ax=ax, label="valid mask")

    for comp in atlas.components:
        a_support = np.asarray(comp.a_support)
        w_low = np.asarray(comp.omega_lower)
        w_up = np.asarray(comp.omega_upper)

        ax.plot(a_support, w_low, lw=2.0, label=f"comp {comp.component_id} lower")
        ax.plot(a_support, w_up, lw=2.0, linestyle="--", label=f"comp {comp.component_id} upper")

    ax.set_xlabel("a")
    ax.set_ylabel("omega")
    ax.set_title("Safe domain mask and atlas envelopes")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f"saved plot  -> {plot_path}")


if __name__ == "__main__":
    main()
