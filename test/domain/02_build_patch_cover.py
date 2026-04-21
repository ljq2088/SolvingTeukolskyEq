from __future__ import annotations
import sys

sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from domain.patch_cover import (
    load_valid_chart_points,
    build_patch_cover,
    save_patch_cover,
    compute_patch_cover_counts,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe-json",
        type=str,
        default="outputs/domain/probe_l2_m2_with_ramp.json",
    )
    parser.add_argument(
        "--atlas-json",
        type=str,
        default="outputs/domain/atlas_l2_m2.json",
    )
    parser.add_argument(
        "--patch-json",
        type=str,
        default="outputs/domain/patch_cover_l2_m2.json",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="outputs/domain/patch_cover_l2_m2.png",
    )
    parser.add_argument("--component-id", type=int, default=0)
    parser.add_argument("--h-u", type=float, default=0.40)
    parser.add_argument("--h-v", type=float, default=0.40)
    args = parser.parse_args()

    uv_points, aw_points = load_valid_chart_points(
        probe_json=args.probe_json,
        atlas_json=args.atlas_json,
        component_id=args.component_id,
    )

    spec = build_patch_cover(
        uv_points=uv_points,
        aw_points=aw_points,
        atlas_json=args.atlas_json,
        probe_json=args.probe_json,
        component_id=args.component_id,
        h_u=args.h_u,
        h_v=args.h_v,
    )
    save_patch_cover(spec, args.patch_json)

    counts = compute_patch_cover_counts(uv_points, spec)

    print("=" * 80)
    print(f"Loaded safe chart points: {len(uv_points)}")
    print(f"component_id        = {spec.component_id}")
    print(f"h_u, h_v            = ({spec.h_u:.3f}, {spec.h_v:.3f})")
    print(f"n_patches           = {spec.n_patches}")
    print(f"coverage_min        = {spec.coverage_min}")
    print(f"coverage_mean       = {spec.coverage_mean:.3f}")
    print(f"coverage_max        = {spec.coverage_max}")
    print(f"n_points_overlap>=2 = {spec.n_points_with_overlap}")
    print(f"saved patch json -> {args.patch_json}")
    print("=" * 80)

    # -------- validation --------
    uncovered = int((counts == 0).sum())
    if uncovered != 0:
        raise RuntimeError(f"Found {uncovered} uncovered safe points.")
    print("[check] all safe points are covered.")

    # -------- plotting in chart space --------
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(uv_points[:, 0], uv_points[:, 1], s=8, alpha=0.6, label="safe points")

    for p in spec.patches:
        x0 = p.u_center - p.h_u
        y0 = p.v_center - p.h_v
        rect = plt.Rectangle(
            (x0, y0),
            2.0 * p.h_u,
            2.0 * p.h_v,
            fill=False,
            lw=1.2,
            alpha=0.9,
        )
        ax.add_patch(rect)
        ax.plot(p.u_center, p.v_center, "o", ms=3)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title("Patch cover on chart domain")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f"saved plot -> {plot_path}")


if __name__ == "__main__":
    main()
