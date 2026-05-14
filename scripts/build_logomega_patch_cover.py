#!/usr/bin/env python3
"""
Build log-omega chart atlas and patch cover from existing probe/atlas.

The chart mapping uses log10(omega) for the v coordinate instead of linear omega.
This produces patches with uniform log-omega span, better matching the physics.

Usage:
  python scripts/build_logomega_patch_cover.py \\
    --probe-json outputs/domain/probe_l2_m2.json \\
    --atlas-json outputs/domain/atlas_l2_m2.json \\
    --out-atlas-json outputs/domain/atlas_l2_m2_logw.json \\
    --out-patch-json outputs/domain/patch_cover_l2_m2_logw.json \\
    --component-id 0 \\
    --h-u 0.4 \\
    --h-v 0.4 \\
    --omega-chart-mode log10
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from domain.atlas_builder import load_atlas, save_atlas
from domain.patch_cover import (
    load_valid_chart_points,
    build_patch_cover,
    save_patch_cover,
    compute_patch_cover_counts,
)


def main():
    parser = argparse.ArgumentParser(description="Build log-omega chart patches")
    parser.add_argument("--probe-json", type=str, required=True)
    parser.add_argument("--atlas-json", type=str, required=True,
                        help="Existing atlas JSON (envelope reused, omega_chart_mode added)")
    parser.add_argument("--out-atlas-json", type=str, required=True)
    parser.add_argument("--out-patch-json", type=str, required=True)
    parser.add_argument("--component-id", type=int, default=0)
    parser.add_argument("--h-u", type=float, default=0.4)
    parser.add_argument("--h-v", type=float, default=0.4)
    parser.add_argument("--omega-chart-mode", type=str, default="log10",
                        choices=["linear", "log10"])
    args = parser.parse_args()

    omega_chart_mode = args.omega_chart_mode

    # 1. Load existing atlas, inject omega_chart_mode, save
    atlas = load_atlas(args.atlas_json)
    atlas.meta["omega_chart_mode"] = omega_chart_mode
    save_atlas(atlas, args.out_atlas_json)
    print(f"[1/4] Saved atlas with omega_chart_mode={omega_chart_mode} -> {args.out_atlas_json}")

    # 2. Load valid chart points using the log-omega mapping
    uv_points, aw_points = load_valid_chart_points(
        probe_json=args.probe_json,
        atlas_json=args.out_atlas_json,
        component_id=args.component_id,
        omega_chart_mode=omega_chart_mode,
    )
    print(f"[2/4] Loaded {len(uv_points)} safe chart points")
    print(f"      uv range: u=[{uv_points[:,0].min():.4f}, {uv_points[:,0].max():.4f}], "
          f"v=[{uv_points[:,1].min():.4f}, {uv_points[:,1].max():.4f}]")

    # 3. Build patch cover in log-omega chart space
    spec = build_patch_cover(
        uv_points=uv_points,
        aw_points=aw_points,
        atlas_json=args.out_atlas_json,
        probe_json=args.probe_json,
        component_id=args.component_id,
        h_u=args.h_u,
        h_v=args.h_v,
        omega_chart_mode=omega_chart_mode,
    )
    save_patch_cover(spec, args.out_patch_json)

    counts = compute_patch_cover_counts(uv_points, spec)
    uncovered = int((counts == 0).sum())

    print(f"[3/4] Patch cover built: n_patches={spec.n_patches}")
    print(f"      coverage: min={spec.coverage_min}, mean={spec.coverage_mean:.3f}, max={spec.coverage_max}")
    print(f"      overlap>=2: {spec.n_points_with_overlap}")
    if uncovered > 0:
        print(f"      WARNING: {uncovered} uncovered points!")
    else:
        print(f"      All {len(uv_points)} points covered")

    # 4. Print patch summary
    print(f"[4/4] Saved patch cover -> {args.out_patch_json}")
    print()
    print(f"{'pid':>4s}  {'u_center':>9s}  {'v_center':>9s}  {'a_center':>9s}  {'omega_center':>14s}  {'log10(w)':>10s}  {'n_pts':>6s}")
    print("-" * 85)
    for p in spec.patches:
        logw_c = np.log10(p.omega_center) if p.omega_center > 0 else float("-inf")
        print(f"{p.patch_id:4d}  {p.u_center:9.4f}  {p.v_center:9.4f}  {p.a_center:9.4f}  {p.omega_center:14.6e}  {logw_c:10.4f}  {p.n_safe_points_covered:6d}")

    # Compute per-patch omega/logomega bounds
    print()
    print(f"{'pid':>4s}  {'a_min':>8s}  {'a_max':>8s}  {'omega_min':>12s}  {'omega_max':>12s}  {'log10w_min':>10s}  {'log10w_max':>10s}")
    print("-" * 90)
    for p in spec.patches:
        u_min = max(0.0, p.u_center - p.h_u)
        u_max = min(1.0, p.u_center + p.h_u)
        v_min = max(0.0, p.v_center - p.h_v)
        v_max = min(1.0, p.v_center + p.h_v)
        a_min = atlas.components[args.component_id].a_support[0] + u_min * (atlas.components[args.component_id].a_support[-1] - atlas.components[args.component_id].a_support[0])
        a_max = atlas.components[args.component_id].a_support[0] + u_max * (atlas.components[args.component_id].a_support[-1] - atlas.components[args.component_id].a_support[0])
        # Use map_from_chart to get physical omega bounds
        from domain.atlas_builder import map_from_chart
        comp = atlas.components[args.component_id]
        _, w_min = map_from_chart(comp, p.u_center, v_min, omega_chart_mode=omega_chart_mode)
        _, w_max = map_from_chart(comp, p.u_center, v_max, omega_chart_mode=omega_chart_mode)
        # Actually need to sample the 4 corners for true bounds
        corners = [(u_min, v_min), (u_min, v_max), (u_max, v_min), (u_max, v_max)]
        ws = []
        for uc, vc in corners:
            _, w = map_from_chart(comp, uc, vc, omega_chart_mode=omega_chart_mode)
            ws.append(w)
        w_min, w_max = min(ws), max(ws)
        logw_min = np.log10(w_min) if w_min > 0 else float("-inf")
        logw_max = np.log10(w_max) if w_max > 0 else float("-inf")
        print(f"{p.patch_id:4d}  {a_min:8.4f}  {a_max:8.4f}  {w_min:12.4e}  {w_max:12.4e}  {logw_min:10.4f}  {logw_max:10.4f}")


if __name__ == "__main__":
    main()
