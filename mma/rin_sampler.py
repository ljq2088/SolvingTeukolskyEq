from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr
def _wl_num(x: float) -> str:
    """
    把 Python float 格式化成 Mathematica 可识别的数值字面量。
    例如:
        1e-8   -> 1.*^-8
        0.125  -> 0.125
        1000.0 -> 1000.
    """
    x = float(x)

    # 0 单独处理
    if x == 0.0:
        return "0."

    s = f"{x:.17e}"   # 例如 1.00000000000000000e-08
    mant, exp = s.split("e")
    exp = int(exp)

    mant = mant.rstrip("0").rstrip(".")
    if "." not in mant:
        mant = mant + "."

    return f"{mant}*^{exp}"
def _to_float_scalar(x):
    """
    尽量把 Mathematica / numpy / 嵌套 list 返回的“数”转成 Python float。
    """
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    if isinstance(x, complex):
        if abs(x.imag) > 1e-14:
            raise ValueError(f"Expected real scalar but got complex: {x}")
        return float(x.real)

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("Empty sequence cannot be converted to float.")
        if len(x) == 1:
            return _to_float_scalar(x[0])
        raise ValueError(f"Cannot interpret sequence as scalar float: {x}")

    try:
        return float(x)
    except Exception as e:
        raise ValueError(f"Cannot convert to float scalar: {x!r}") from e


def _to_complex_scalar(x):
    """
    尽量把 Mathematica 返回对象转成 Python complex。
    支持：
      - complex
      - real scalar
      - [re, im]
      - [x] 递归展开
    """
    if isinstance(x, complex):
        return complex(x)

    if isinstance(x, (int, float, np.integer, np.floating)):
        return complex(float(x), 0.0)

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("Empty sequence cannot be converted to complex.")
        if len(x) == 1:
            return _to_complex_scalar(x[0])
        if len(x) == 2:
            re = _to_float_scalar(x[0])
            im = _to_float_scalar(x[1])
            return complex(re, im)
        raise ValueError(f"Cannot interpret sequence as complex: {x}")

    try:
        return complex(x)
    except Exception as e:
        raise ValueError(f"Cannot convert to complex scalar: {x!r}") from e

def _try_parse_numeric_table_fast(result):
    """
    快速路径：
    如果 Mathematica 返回的是 PackedArray / ndarray / 可直接转成数值表的对象，
    且形状是 (N, 3)，就直接解析。
    """
    try:
        arr = np.asarray(result, dtype=float)
    except Exception:
        return None

    if arr.ndim == 2 and arr.shape[1] >= 3:
        r = arr[:, 0]
        Rin = arr[:, 1] + 1j * arr[:, 2]
        return r, Rin

    return None
def _parse_sample_rin_result(result):
    """
    把 Mathematica 的 SampleRinOnGrid 返回解析成：
      r   : (N,) float64
      Rin : (N,) complex128

    支持以下常见格式：
      row = [r, re, im]
      row = [r, [re, im]]
      row = [r, re, [im]]
      row = [r, [re], [im]]
    """
    r_list = []
    rin_list = []

    for idx, row in enumerate(result):
        # 先尝试把一行转成普通 ndarray / list
        try:
            row_arr = np.asarray(row, dtype=object).reshape(-1)
            row_seq = row_arr.tolist()
        except Exception:
            row_seq = row

        if not isinstance(row_seq, (list, tuple)):
            raise ValueError(f"Row {idx} is not a sequence: {row!r}")

        if len(row_seq) == 3:
            r = _to_float_scalar(row_seq[0])
            re = _to_float_scalar(row_seq[1])
            im = _to_float_scalar(row_seq[2])
            rin = complex(re, im)

        elif len(row_seq) == 2:
            r = _to_float_scalar(row_seq[0])
            rin = _to_complex_scalar(row_seq[1])

        else:
            raise ValueError(
                f"Unexpected Mathematica row format at idx={idx}: "
                f"len={len(row_seq)}, row={row_seq!r}"
            )

        r_list.append(r)
        rin_list.append(rin)

    r = np.asarray(r_list, dtype=float)
    Rin = np.asarray(rin_list, dtype=np.complex128)
    return r, Rin
@dataclass
class MathematicaConfig:
    kernel_path: str
    wl_path_win: str


class MathematicaRinSampler:
    def __init__(self, kernel_path: str, wl_path_win: str, timeout_sec: float = 20.0):
        self.kernel_path = str(kernel_path)
        self.wl_path_win = str(wl_path_win)
        self.timeout_sec = float(timeout_sec)
        self._cache = {}
        self._session = None

    def _get_session(self):
        if self._session is None:
            self._session = WolframLanguageSession(kernel=self.kernel_path)
            self._session.evaluate(wlexpr(rf'Get["{self.wl_path_win}"]'))
        return self._session

    def close(self):
        if self._session is not None:
            self._session.terminate()
            self._session = None

    def _looks_like_unevaluated_call(self, result, function_name: str) -> bool:
        try:
            text = str(result)
        except Exception:
            return False
        return function_name in text

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

        session = self._get_session()
        expr = (
            f"SampleRinOnGrid[{int(s)}, {int(l)}, {int(m)}, "
            f"{_wl_num(a)}, {_wl_num(omega)}, "
            f"{_wl_num(rmin)}, {_wl_num(rmax)}, {int(npts)}]"
        )
        try:
            result = session.evaluate(wlexpr(expr))
        except Exception:
            self.close()
            raise

        # ---------------------------------------------------------
        # 快速路径：如果已经是 PackedArray / ndarray 数值表，直接解析
        # ---------------------------------------------------------
        parsed = _try_parse_numeric_table_fast(result)
        if parsed is not None:
            r, Rin = parsed
            self._cache[key] = (r, Rin)
            return r, Rin

        # ---------------------------------------------------------
        # 慢速鲁棒路径：逐行解析更复杂的 Mathematica 返回格式
        # ---------------------------------------------------------
        try:
            r, Rin = _parse_sample_rin_result(result)
        except Exception as e:
            try:
                preview = np.asarray(result, dtype=object)[:3]
            except Exception:
                preview = result
            raise ValueError(
                f"Failed to parse Mathematica SampleRinOnGrid output. "
                f"Preview of result[:3] = {preview!r}"
            ) from e

        self._cache[key] = (r, Rin)
        return r, Rin

    def interpolate_rin(
        self,
        s: int,
        l: int,
        m: int,
        a: float,
        omega: float,
        r_query: np.ndarray,
        n_grid: int = 1200,
        pad_frac: float = 0.005,
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

        # 关键修复：
        # 左端绝不能往 horizon 以下扩。对 In 解的数值域，r 必须 >= r_+ + eps
        rp = 1.0 + np.sqrt(max(1.0 - float(a) * float(a), 0.0))
        r_floor = rp + 1.0e-6

        # 左端不再减 pad，只做 clamp
        rmin = max(r_floor, rq_min)

        # 右端可以适度扩一点，便于插值边界稳定
        rmax = rq_max + pad_frac * width

        if rmax <= rmin:
            rmax = rmin + max(1.0, 0.01 * rmin)

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
    
    def evaluate_rin_at_points(
        self,
        s: int,
        l: int,
        m: int,
        a: float,
        omega: float,
        r_query: np.ndarray,
    ) -> np.ndarray:
        """
        直接在给定 r_query 点上评估 Mathematica 的 R_in。
        适合 anchor 点很少的情况，避免区间插值导致越界。
        """
        r_query = np.asarray(r_query, dtype=float)
        if r_query.ndim != 1:
            raise ValueError(f"r_query must be 1D, got shape {r_query.shape}")

        # 安全检查
        rp = 1.0 + np.sqrt(max(1.0 - float(a) * float(a), 0.0))
        if np.any(r_query <= rp):
            bad = r_query[r_query <= rp][:5]
            raise ValueError(
                f"Some query radii are not outside the horizon. "
                f"r_+={rp}, bad sample={bad}"
            )

        # 这里先沿用 SampleRinOnGrid 的思路，但区间严格由 query 本身决定，不再左端扩展
        rmin = float(r_query.min())
        rmax = float(r_query.max())

        # 稠密采样，只在合法区间内部
        r_grid, Rin_grid = self.sample_rin_on_grid(
            s=s,
            l=l,
            m=m,
            a=a,
            omega=omega,
            rmin=rmin,
            rmax=rmax,
            npts=max(2000, 20 * len(r_query)),
        )

        re = np.interp(r_query, r_grid, Rin_grid.real)
        im = np.interp(r_query, r_grid, Rin_grid.imag)
        return re + 1j * im
    def evaluate_rin_at_points_direct(
        self,
        s: int,
        l: int,
        m: int,
        a: float,
        omega: float,
        r_query: np.ndarray,
        function_name: str = "SampleRinAtPoints",
        _retry: bool = True,
    ) -> np.ndarray:
        session = self._get_session()

        r_query = np.asarray(r_query, dtype=float).reshape(-1)
        r_list = ", ".join(_wl_num(float(r)) for r in r_query)

        expr = (
            "Quiet["
            "Block[{$Messages = {}, $MessageList = {}}, "
            "Check["
            "TimeConstrained["
            f"{function_name}[{int(s)}, {int(l)}, {int(m)}, "
            f"{_wl_num(a)}, {_wl_num(omega)}, "
            f"{{{r_list}}}], {_wl_num(self.timeout_sec)}, $Aborted], "
            "$Failed]"
            "]]"
        )

        try:
            result = session.evaluate(wlexpr(expr))
        except Exception:
            self.close()
            raise
        result_str = str(result)
        if result_str == "$Failed":
            raise RuntimeError("Mathematica evaluation returned $Failed")
        if result_str == "$Aborted":
            raise TimeoutError(
                f"Mathematica evaluation timed out after {self.timeout_sec} s"
            )
        if self._looks_like_unevaluated_call(result, function_name):
            if _retry:
                self.close()
                return self.evaluate_rin_at_points_direct(
                    s=s,
                    l=l,
                    m=m,
                    a=a,
                    omega=omega,
                    r_query=r_query,
                    function_name=function_name,
                    _retry=False,
                )
            raise RuntimeError(
                f"Mathematica returned unevaluated {function_name}[...] expression"
            )

        parsed = _try_parse_numeric_table_fast(result)
        if parsed is not None:
            r_eval, Rin = parsed
        else:
            try:
                r_eval, Rin = _parse_sample_rin_result(result)
            except Exception as e:
                try:
                    preview = np.asarray(result, dtype=object)[:3]
                except Exception:
                    preview = result
                raise ValueError(
                    f"Failed to parse Mathematica {function_name} output. "
                    f"Preview of result[:3] = {preview!r}"
                ) from e

        if len(r_eval) != len(r_query):
            raise ValueError(
                f"Length mismatch: returned {len(r_eval)} points, expected {len(r_query)}"
            )

        max_diff = np.max(np.abs(r_eval - r_query))
        if max_diff > 1e-10:
            raise ValueError(f"r mismatch, max diff = {max_diff:.3e}")

        return Rin
