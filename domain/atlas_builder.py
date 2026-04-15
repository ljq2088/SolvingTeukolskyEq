from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque
import json
import math
from typing import Any

import numpy as np


@dataclass
class ProbeGrid:
    meta: dict[str, Any]
    a_vals: np.ndarray              # shape (na,)
    omega_vals: np.ndarray          # shape (nw,)
    valid_mask: np.ndarray          # shape (na, nw), bool
    code_grid: np.ndarray           # shape (na, nw), dtype=object


@dataclass
class AtlasComponent:
    component_id: int
    n_points_raw: int
    n_points_filled: int
    n_filled_hole_cells: int

    a_idx_min: int
    a_idx_max: int
    omega_idx_min: int
    omega_idx_max: int

    a_support: list[float]
    omega_lower: list[float]
    omega_upper: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AtlasSpec:
    meta: dict[str, Any]
    n_components_raw: int
    n_components_kept: int
    components: list[AtlasComponent]

    def to_dict(self) -> dict[str, Any]:
        out = {
            "meta": self.meta,
            "n_components_raw": self.n_components_raw,
            "n_components_kept": self.n_components_kept,
            "components": [c.to_dict() for c in self.components],
        }
        return out


def load_probe_grid(json_path: str | Path) -> ProbeGrid:
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data["records"]

    a_vals = np.array(sorted({float(r["a"]) for r in records}), dtype=float)
    omega_vals = np.array(sorted({float(r["omega"]) for r in records}), dtype=float)

    na = len(a_vals)
    nw = len(omega_vals)

    a_to_i = {float(a): i for i, a in enumerate(a_vals)}
    w_to_j = {float(w): j for j, w in enumerate(omega_vals)}

    valid_mask = np.zeros((na, nw), dtype=bool)
    code_grid = np.empty((na, nw), dtype=object)

    for r in records:
        i = a_to_i[float(r["a"])]
        j = w_to_j[float(r["omega"])]
        valid_mask[i, j] = bool(r["valid"])
        code_grid[i, j] = str(r["code"])

    return ProbeGrid(
        meta=data["meta"],
        a_vals=a_vals,
        omega_vals=omega_vals,
        valid_mask=valid_mask,
        code_grid=code_grid,
    )


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """
    4-邻域连通分量标记。
    mask: shape (na, nw), True 表示可用
    """
    na, nw = mask.shape
    labels = np.full((na, nw), -1, dtype=int)
    current = 0

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(na):
        for j in range(nw):
            if (not mask[i, j]) or labels[i, j] != -1:
                continue

            q = deque()
            q.append((i, j))
            labels[i, j] = current

            while q:
                x, y = q.popleft()
                for dx, dy in nbrs:
                    xx = x + dx
                    yy = y + dy
                    if 0 <= xx < na and 0 <= yy < nw:
                        if mask[xx, yy] and labels[xx, yy] == -1:
                            labels[xx, yy] = current
                            q.append((xx, yy))

            current += 1

    return labels, current


def _fill_small_vertical_holes(mask: np.ndarray, max_gap: int = 2) -> tuple[np.ndarray, int]:
    """
    对每个固定 a 的列，在 omega 方向填补很小的内部空洞。
    只用于构造 chart 边界，不改变原始 valid_mask。
    """
    out = mask.copy()
    na, nw = mask.shape
    filled = 0

    for i in range(na):
        js = np.where(mask[i])[0]
        if len(js) <= 1:
            continue

        for j0, j1 in zip(js[:-1], js[1:]):
            gap = j1 - j0 - 1
            if 0 < gap <= max_gap:
                out[i, j0 + 1:j1] = True
                filled += gap

    return out, filled


def _component_to_envelope(
    component_mask: np.ndarray,
    a_vals: np.ndarray,
    omega_vals: np.ndarray,
) -> tuple[list[float], list[float], list[float], dict[str, int]]:
    """
    对单个 component 的 mask 提取包络：
    对每个 a 列，记录最小/最大可用 omega。
    """
    na, nw = component_mask.shape

    a_support = []
    omega_lower = []
    omega_upper = []

    a_idx_has = []

    for i in range(na):
        js = np.where(component_mask[i])[0]
        if len(js) == 0:
            continue
        a_support.append(float(a_vals[i]))
        omega_lower.append(float(omega_vals[js.min()]))
        omega_upper.append(float(omega_vals[js.max()]))
        a_idx_has.append(i)

    if len(a_idx_has) == 0:
        raise ValueError("component has no support")

    bbox = {
        "a_idx_min": int(min(a_idx_has)),
        "a_idx_max": int(max(a_idx_has)),
        "omega_idx_min": int(
            min(np.where(component_mask[i])[0].min() for i in a_idx_has)
        ),
        "omega_idx_max": int(
            max(np.where(component_mask[i])[0].max() for i in a_idx_has)
        ),
    }

    return a_support, omega_lower, omega_upper, bbox


def build_atlas_from_probe(
    probe: ProbeGrid,
    min_component_size: int = 20,
    vertical_hole_max_gap: int = 2,
) -> AtlasSpec:
    labels, n_comp_raw = _label_components(probe.valid_mask)

    components: list[AtlasComponent] = []

    for cid in range(n_comp_raw):
        raw_mask = (labels == cid)
        n_raw = int(raw_mask.sum())
        if n_raw < min_component_size:
            continue

        filled_mask, n_filled = _fill_small_vertical_holes(
            raw_mask,
            max_gap=vertical_hole_max_gap,
        )
        n_filled_total = int(filled_mask.sum())

        a_support, w_low, w_up, bbox = _component_to_envelope(
            component_mask=filled_mask,
            a_vals=probe.a_vals,
            omega_vals=probe.omega_vals,
        )

        comp = AtlasComponent(
            component_id=cid,
            n_points_raw=n_raw,
            n_points_filled=n_filled_total,
            n_filled_hole_cells=n_filled,
            a_idx_min=bbox["a_idx_min"],
            a_idx_max=bbox["a_idx_max"],
            omega_idx_min=bbox["omega_idx_min"],
            omega_idx_max=bbox["omega_idx_max"],
            a_support=a_support,
            omega_lower=w_low,
            omega_upper=w_up,
        )
        components.append(comp)

    return AtlasSpec(
        meta=probe.meta,
        n_components_raw=n_comp_raw,
        n_components_kept=len(components),
        components=components,
    )

import numpy as np

def _convert_numpy(obj):
    """Recursively convert numpy types to Python native types."""
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
def save_atlas(atlas: AtlasSpec, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data=atlas.to_dict()
    data=_convert_numpy(data)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_atlas(atlas_path: str | Path) -> AtlasSpec:
    atlas_path = Path(atlas_path)
    with open(atlas_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    comps = []
    for c in data["components"]:
        comps.append(AtlasComponent(**c))

    return AtlasSpec(
        meta=data["meta"],
        n_components_raw=data["n_components_raw"],
        n_components_kept=data["n_components_kept"],
        components=comps,
    )


def interp_lower_upper(comp: AtlasComponent, a: float) -> tuple[float, float]:
    a_support = np.asarray(comp.a_support, dtype=float)
    w_low = np.asarray(comp.omega_lower, dtype=float)
    w_up = np.asarray(comp.omega_upper, dtype=float)

    if a < a_support[0] or a > a_support[-1]:
        raise ValueError(f"a={a} is outside component support [{a_support[0]}, {a_support[-1]}]")

    low = float(np.interp(a, a_support, w_low))
    up = float(np.interp(a, a_support, w_up))
    return low, up


def map_to_chart(comp: AtlasComponent, a: float, omega: float) -> tuple[float, float]:
    a0 = comp.a_support[0]
    a1 = comp.a_support[-1]
    if not (a0 <= a <= a1):
        raise ValueError(f"a={a} outside support [{a0}, {a1}]")

    low, up = interp_lower_upper(comp, a)
    if not (low <= omega <= up):
        raise ValueError(f"omega={omega} outside envelope [{low}, {up}] at a={a}")

    if a1 == a0:
        u = 0.0
    else:
        u = (a - a0) / (a1 - a0)

    denom = up - low
    if denom <= 0.0:
        raise ValueError(f"degenerate omega interval at a={a}: low={low}, up={up}")

    v = (omega - low) / denom
    return float(u), float(v)


def map_from_chart(comp: AtlasComponent, u: float, v: float) -> tuple[float, float]:
    if not (0.0 <= u <= 1.0):
        raise ValueError(f"u={u} outside [0,1]")
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"v={v} outside [0,1]")

    a0 = comp.a_support[0]
    a1 = comp.a_support[-1]
    a = a0 + u * (a1 - a0)

    low, up = interp_lower_upper(comp, a)
    omega = low + v * (up - low)
    return float(a), float(omega)