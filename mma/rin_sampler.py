from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr


@dataclass
class MathematicaConfig:
    kernel_path: str
    wl_path_win: str


class MathematicaRinSampler:
    """
    用 Mathematica 的 SampleRinOnGrid[...] 生成 R_in(r) 的稠密样本，
    再对任意查询 r 做插值。
    """
    def __init__(self, kernel_path: str, wl_path_win: str):
        self.kernel_path = str(kernel_path)
        self.wl_path_win = str(wl_path_win)
        self._cache: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}

    def sample_rin_on_grid(
        self,
        s: int,
        l: int,
        m: int,
        a: float,
        omega: float,
        rmin: float,
        rmax: float,
        npts: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (
            int(s), int(l), int(m),
            round(float(a), 12),
            round(float(omega), 12),
            round(float(rmin), 12),
            round(float(rmax), 12),
            int(npts),
        )
        if key in self._cache:
            return self._cache[key]

        session = WolframLanguageSession(kernel=self.kernel_path)
        try:
            session.evaluate(wlexpr(rf'Get["{self.wl_path_win}"]'))
            expr = (
                f"SampleRinOnGrid[{int(s)}, {int(l)}, {int(m)}, "
                f"{float(a):.16g}, {float(omega):.16g}, "
                f"{float(rmin):.16g}, {float(rmax):.16g}, {int(npts)}]"
            )
            result = session.evaluate(wlexpr(expr))
            arr = np.array(result, dtype=float)

            if arr.ndim != 2 or arr.shape[1] < 3:
                raise ValueError(f"Unexpected Mathematica output shape: {arr.shape}")

            r = arr[:, 0]
            Rin = arr[:, 1] + 1j * arr[:, 2]
            self._cache[key] = (r, Rin)
            return r, Rin
        finally:
            session.terminate()

    def interpolate_rin(
        self,
        s: int,
        l: int,
        m: int,
        a: float,
        omega: float,
        r_query: np.ndarray,
        n_grid: int = 1200,
        pad_frac: float = 0.02,
    ) -> np.ndarray:
        """
        对给定 r_query 插值 Mathematica R_in。
        这里用线性插值，先在 [rmin, rmax] 上调用 SampleRinOnGrid 生成稠密样本。
        """
        r_query = np.asarray(r_query, dtype=float)
        if r_query.ndim != 1:
            raise ValueError(f"r_query must be 1D, got shape {r_query.shape}")

        rq_min = float(r_query.min())
        rq_max = float(r_query.max())
        width = max(rq_max - rq_min, 1.0e-8)
        rmin = max(1.0e-8, rq_min - pad_frac * width)
        rmax = rq_max + pad_frac * width

        r_grid, Rin_grid = self.sample_rin_on_grid(
            s=s,
            l=l,
            m=m,
            a=a,
            omega=omega,
            rmin=rmin,
            rmax=rmax,
            npts=n_grid,
        )

        re = np.interp(r_query, r_grid, Rin_grid.real)
        im = np.interp(r_query, r_grid, Rin_grid.imag)
        return re + 1j * im