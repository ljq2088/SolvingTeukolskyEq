from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json

import numpy as np

from domain.atlas_builder import load_atlas, load_probe_grid, map_to_chart, map_from_chart


def _convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy(i) for i in obj]
    else:
        return obj


@dataclass
class PatchSpec:
    patch_id: int
    component_id: int
    u_center: float
    v_center: float
    h_u: float
    h_v: float
    a_center: float
    omega_center: float
    n_safe_points_covered: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatchCoverSpec:
    meta: dict[str, Any]
    atlas_json: str
    probe_json: str
    component_id: int
    h_u: float
    h_v: float
    n_safe_points: int
    n_patches: int
    coverage_min: int
    coverage_mean: float
    coverage_max: int
    n_points_with_overlap: int
    patches: list[PatchSpec]

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "atlas_json": self.atlas_json,
            "probe_json": self.probe_json,
            "component_id": self.component_id,
            "h_u": self.h_u,
            "h_v": self.h_v,
            "n_safe_points": self.n_safe_points,
            "n_patches": self.n_patches,
            "coverage_min": self.coverage_min,
            "coverage_mean": self.coverage_mean,
            "coverage_max": self.coverage_max,
            "n_points_with_overlap": self.n_points_with_overlap,
            "patches": [p.to_dict() for p in self.patches],
        }


def load_valid_chart_points(
    probe_json: str | Path,
    atlas_json: str | Path,
    component_id: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    返回：
      uv_points : shape (N, 2)
      aw_points : shape (N, 2)
    只保留原始 probe 中 valid=True 且能映到指定 component chart 的点。
    """
    probe = load_probe_grid(probe_json)
    atlas = load_atlas(atlas_json)
    comp = atlas.components[component_id]

    uv_list = []
    aw_list = []

    for i, a in enumerate(probe.a_vals):
        for j, omega in enumerate(probe.omega_vals):
            if not probe.valid_mask[i, j]:
                continue
            try:
                u, v = map_to_chart(comp, float(a), float(omega))
            except Exception:
                continue
            uv_list.append([u, v])
            aw_list.append([float(a), float(omega)])

    uv_points = np.asarray(uv_list, dtype=float)
    aw_points = np.asarray(aw_list, dtype=float)
    return uv_points, aw_points


def _points_covered_by_rect(
    uv_points: np.ndarray,
    u_center: float,
    v_center: float,
    h_u: float,
    h_v: float,
) -> np.ndarray:
    du = np.abs(uv_points[:, 0] - u_center)
    dv = np.abs(uv_points[:, 1] - v_center)
    return (du <= h_u) & (dv <= h_v)


def build_patch_cover(
    uv_points: np.ndarray,
    aw_points: np.ndarray,
    atlas_json: str | Path,
    probe_json: str | Path,
    component_id: int = 0,
    h_u: float = 0.12,
    h_v: float = 0.12,
) -> PatchCoverSpec:
    """
    在 chart 平面上做贪心矩形覆盖：
    每次从未覆盖点中选一个“离已有中心最远”的点作为新中心，
    然后用固定矩形半宽 (h_u, h_v) 覆盖。
    """
    if len(uv_points) == 0:
        raise ValueError("No valid chart points found.")

    atlas = load_atlas(atlas_json)
    comp = atlas.components[component_id]

    n = len(uv_points)
    covered = np.zeros(n, dtype=bool)
    cover_count = np.zeros(n, dtype=int)

    centers_uv = []
    cover_sizes = []

    # 第一个中心：选离 chart 中心最近的 safe 点
    chart_center = np.array([0.5, 0.5], dtype=float)
    d0 = np.sum((uv_points - chart_center[None, :]) ** 2, axis=1)
    first_idx = int(np.argmin(d0))

    while not np.all(covered):
        if len(centers_uv) == 0:
            idx = first_idx
        else:
            centers_arr = np.asarray(centers_uv, dtype=float)  # (K, 2)
            du = np.abs(uv_points[:, 0:1] - centers_arr[None, :, 0])
            dv = np.abs(uv_points[:, 1:2] - centers_arr[None, :, 1])
            # 缩放后的 L_infty 距离
            d_scaled = np.maximum(du / h_u, dv / h_v)
            min_d = np.min(d_scaled, axis=1)

            masked = np.where(covered, -1.0, min_d)
            idx = int(np.argmax(masked))

        uc, vc = uv_points[idx]
        mask_new = _points_covered_by_rect(uv_points, uc, vc, h_u, h_v)

        covered = covered | mask_new
        cover_count[mask_new] += 1

        centers_uv.append([float(uc), float(vc)])
        cover_sizes.append(int(mask_new.sum()))

    patches = []
    for pid, ((uc, vc), n_cov) in enumerate(zip(centers_uv, cover_sizes)):
        a_center, omega_center = map_from_chart(comp, uc, vc)
        patches.append(
            PatchSpec(
                patch_id=pid,
                component_id=component_id,
                u_center=float(uc),
                v_center=float(vc),
                h_u=float(h_u),
                h_v=float(h_v),
                a_center=float(a_center),
                omega_center=float(omega_center),
                n_safe_points_covered=int(n_cov),
            )
        )

    spec = PatchCoverSpec(
        meta=atlas.meta,
        atlas_json=str(atlas_json),
        probe_json=str(probe_json),
        component_id=component_id,
        h_u=float(h_u),
        h_v=float(h_v),
        n_safe_points=int(n),
        n_patches=len(patches),
        coverage_min=int(cover_count.min()),
        coverage_mean=float(cover_count.mean()),
        coverage_max=int(cover_count.max()),
        n_points_with_overlap=int((cover_count >= 2).sum()),
        patches=patches,
    )
    return spec


def save_patch_cover(spec: PatchCoverSpec, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = _convert_numpy(spec.to_dict())
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_patch_cover(path: str | Path) -> PatchCoverSpec:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    patches = [PatchSpec(**p) for p in data["patches"]]
    return PatchCoverSpec(
        meta=data["meta"],
        atlas_json=data["atlas_json"],
        probe_json=data["probe_json"],
        component_id=data["component_id"],
        h_u=data["h_u"],
        h_v=data["h_v"],
        n_safe_points=data["n_safe_points"],
        n_patches=data["n_patches"],
        coverage_min=data["coverage_min"],
        coverage_mean=data["coverage_mean"],
        coverage_max=data["coverage_max"],
        n_points_with_overlap=data["n_points_with_overlap"],
        patches=patches,
    )


def compute_patch_cover_counts(
    uv_points: np.ndarray,
    spec: PatchCoverSpec,
) -> np.ndarray:
    counts = np.zeros(len(uv_points), dtype=int)
    for p in spec.patches:
        m = _points_covered_by_rect(
            uv_points=uv_points,
            u_center=p.u_center,
            v_center=p.v_center,
            h_u=p.h_u,
            h_v=p.h_v,
        )
        counts[m] += 1
    return counts