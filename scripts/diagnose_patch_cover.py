#!/usr/bin/env python3
"""
Diagnose patch cover coordinate system and compare linear vs logomega patches.

Usage:
  python scripts/diagnose_patch_cover.py \\
    --probe-json outputs/domain/probe_l2_m2.json \\
    --atlas-json outputs/domain/atlas_l2_m2.json \\
    --patch-json outputs/domain/patch_cover_l2_m2.json \\
    --label linear

  python scripts/diagnose_patch_cover.py \\
    --probe-json outputs/domain/probe_l2_m2.json \\
    --atlas-json outputs/domain/atlas_l2_m2_logw.json \\
    --patch-json outputs/domain/patch_cover_l2_m2_logw.json \\
    --label logomega
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from domain.atlas_builder import load_atlas, map_from_chart, omega_to_chart_coord, omega_from_chart_coord
from domain.patch_cover import load_patch_cover, load_valid_chart_points


def diagnose(probe_json, atlas_json, patch_json, label):
    print("=" * 80)
    print(f"DIAGNOSTIC: {label}")
    print("=" * 80)

    atlas = load_atlas(atlas_json)
    patch_cover = load_patch_cover(patch_json)

    omega_chart_mode = patch_cover.meta.get("omega_chart_mode", "linear")
    print(f"omega_chart_mode: {omega_chart_mode}")
    print(f"atlas meta keys: {list(atlas.meta.keys())}")
    print(f"patch meta keys: {list(patch_cover.meta.keys())}")
    print()

    # Overall stats
    print(f"n_patches:     {patch_cover.n_patches}")
    print(f"h_u:           {patch_cover.h_u}")
    print(f"h_v:           {patch_cover.h_v}")
    print(f"n_safe_points: {patch_cover.n_safe_points}")
    print()

    # Load safe points to compute per-patch stats
    uv_points, aw_points = load_valid_chart_points(
        probe_json=probe_json,
        atlas_json=atlas_json,
        component_id=patch_cover.component_id,
        omega_chart_mode=omega_chart_mode,
    )

    # Per-patch physical bounds
    comp = atlas.components[patch_cover.component_id]
    print(f"{'pid':>4s}  {'u':>8s}  {'v':>8s}  {'a_center':>9s}  {'omega_center':>14s}  "
          f"{'a_range':>18s}  {'omega_range':>18s}  {'log10w_range':>18s}  {'n_pts':>6s}")
    print("-" * 130)

    for p in patch_cover.patches:
        u_min = max(0.0, p.u_center - p.h_u)
        u_max = min(1.0, p.u_center + p.h_u)
        v_min = max(0.0, p.v_center - p.h_v)
        v_max = min(1.0, p.v_center + p.h_v)

        # Physical bounds from 4 corners
        corners = [(u_min, v_min), (u_min, v_max), (u_max, v_min), (u_max, v_max)]
        a_vals = []
        w_vals = []
        for uc, vc in corners:
            a_c, w_c = map_from_chart(comp, uc, vc, omega_chart_mode=omega_chart_mode)
            a_vals.append(a_c)
            w_vals.append(w_c)

        a_min, a_max = min(a_vals), max(a_vals)
        w_min, w_max = min(w_vals), max(w_vals)
        logw_min = np.log10(max(w_min, 1e-300))
        logw_max = np.log10(max(w_max, 1e-300))
        logw_span = logw_max - logw_min

        n_in_patch = int(np.sum(
            (np.abs(uv_points[:, 0] - p.u_center) <= p.h_u) &
            (np.abs(uv_points[:, 1] - p.v_center) <= p.h_v)
        ))

        print(f"{p.patch_id:4d}  {p.u_center:8.4f}  {p.v_center:8.4f}  {p.a_center:9.4f}  {p.omega_center:14.6e}  "
              f"[{a_min:.4f}, {a_max:.4f}]  [{w_min:.4e}, {w_max:.4e}]  [{logw_min:7.4f}, {logw_max:7.4f}] ({logw_span:.4f})  {n_in_patch:6d}")

    # Summary stats
    print()
    print("--- Summary ---")
    logw_spans = []
    w_spans = []
    for p in patch_cover.patches:
        u_min = max(0.0, p.u_center - p.h_u)
        u_max = min(1.0, p.u_center + p.h_u)
        v_min = max(0.0, p.v_center - p.h_v)
        v_max = min(1.0, p.v_center + p.h_v)
        corners = [(u_min, v_min), (u_min, v_max), (u_max, v_min), (u_max, v_max)]
        ws = []
        for uc, vc in corners:
            _, w = map_from_chart(comp, uc, vc, omega_chart_mode=omega_chart_mode)
            ws.append(w)
        w_min, w_max = min(ws), max(ws)
        w_spans.append(w_max - w_min)
        logw_spans.append(np.log10(max(w_max, 1e-300)) - np.log10(max(w_min, 1e-300)))

    logw_spans = np.array(logw_spans)
    w_spans = np.array(w_spans)
    print(f"omega span:       min={w_spans.min():.4e}, max={w_spans.max():.4e}, mean={w_spans.mean():.4e}, std={w_spans.std():.4e}")
    print(f"log10omega span:  min={logw_spans.min():.4f}, max={logw_spans.max():.4f}, mean={logw_spans.mean():.4f}, std={logw_spans.std():.4f}")
    print(f"log10w span ratio max/min: {logw_spans.max()/logw_spans.min():.2f}" if logw_spans.min() > 0 else "log10w span ratio: N/A (zero span)")

    # Patch 0 detail
    p0 = patch_cover.patches[0]
    u_min = max(0.0, p0.u_center - p0.h_u)
    u_max = min(1.0, p0.u_center + p0.h_u)
    v_min = max(0.0, p0.v_center - p0.h_v)
    v_max = min(1.0, p0.v_center + p0.h_v)
    corners = [(u_min, v_min), (u_min, v_max), (u_max, v_min), (u_max, v_max)]
    a_vals_p0 = []
    w_vals_p0 = []
    for uc, vc in corners:
        a_c, w_c = map_from_chart(comp, uc, vc, omega_chart_mode=omega_chart_mode)
        a_vals_p0.append(a_c)
        w_vals_p0.append(w_c)

    print()
    print(f"--- Patch 0 detail ---")
    print(f"  u: [{u_min:.4f}, {u_max:.4f}], v: [{v_min:.4f}, {v_max:.4f}]")
    print(f"  a: [{min(a_vals_p0):.4f}, {max(a_vals_p0):.4f}]")
    print(f"  omega: [{min(w_vals_p0):.6e}, {max(w_vals_p0):.6e}]")
    print(f"  log10 omega: [{np.log10(max(min(w_vals_p0),1e-300)):.4f}, {np.log10(max(max(w_vals_p0),1e-300)):.4f}]")
    print(f"  chart coords: u_center={p0.u_center:.4f}, v_center={p0.v_center:.4f}, h_u={p0.h_u:.4f}, h_v={p0.h_v:.4f}")
    print(f"  a_center={p0.a_center:.4f}, omega_center={p0.omega_center:.6e}")
    print(f"  n_safe_points_covered={p0.n_safe_points_covered}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Diagnose patch cover coordinates")
    parser.add_argument("--probe-json", type=str, required=True)
    parser.add_argument("--atlas-json", type=str, required=True)
    parser.add_argument("--patch-json", type=str, required=True)
    parser.add_argument("--label", type=str, default="patch_cover")
    args = parser.parse_args()

    diagnose(args.probe_json, args.atlas_json, args.patch_json, args.label)


if __name__ == "__main__":
    main()
