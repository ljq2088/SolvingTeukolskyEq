from __future__ import annotations

import json
from pathlib import Path

import torch

from utils.mode import KerrMode
from utils.amplitude import compute_smatrix


def _complex_to_json(z: complex) -> dict:
    z = complex(z)
    return {"re": float(z.real), "im": float(z.imag)}


def _complex_from_json(obj: dict) -> complex:
    return complex(float(obj["re"]), float(obj["im"]))


class SpectralCoeffCache:
    """Cache spectral coefficients B_inc and B_ref for the in-mode solution."""

    def __init__(
        self,
        cache_file: str | Path,
        physics_cfg: dict,
        device: torch.device,
        dtype: torch.dtype,
        N: int = 64,
        z_m: float = 0.3,
    ):
        self.cache_file = Path(cache_file)
        self.physics_cfg = physics_cfg
        self.device = device
        self.dtype = dtype
        self.N = int(N)
        self.z_m = float(z_m)

        problem_cfg = physics_cfg["problem"]
        self.M = float(problem_cfg.get("M", 1.0))
        self.s = int(problem_cfg.get("s", -2))
        self.ell = int(problem_cfg.get("l", problem_cfg.get("ell", 2)))
        self.m = int(problem_cfg.get("m", 2))

        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load()

    def _make_key(self, a: float, omega: float, lam: complex) -> str:
        lam = complex(lam)
        return (
            f"s={self.s}|l={self.ell}|m={self.m}|"
            f"a={a:.12e}|omega={omega:.12e}|"
            f"lam_re={lam.real:.12e}|lam_im={lam.imag:.12e}|"
            f"N={self.N}|z_m={self.z_m:.12e}"
        )

    def _load(self) -> dict:
        if not self.cache_file.exists():
            return {}
        with open(self.cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self):
        tmp = self.cache_file.with_suffix(self.cache_file.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)
        tmp.replace(self.cache_file)

    def _compute_one(self, a: float, omega: float, lam: complex) -> tuple[complex, complex]:
        mode = KerrMode(
            M=self.M, a=float(a), omega=float(omega),
            ell=self.ell, m=self.m, lam=complex(lam), s=self.s,
        )
        smat = compute_smatrix(
            mode=mode, N_in=self.N, N_out=self.N,
            z_m=self.z_m, return_profile=False,
        )
        return complex(smat["B_inc"]), complex(smat["B_ref"])

    def get_batch(
        self, a_batch: torch.Tensor, omega_batch: torch.Tensor, lambda_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Binc_list, Bref_list = [], []
        changed = False
        B = int(a_batch.shape[0])
        for i in range(B):
            a_i = float(a_batch[i].detach().cpu().item())
            omega_i = float(omega_batch[i].detach().cpu().item())
            lam_i = complex(lambda_batch[i].detach().cpu().item())
            key = self._make_key(a_i, omega_i, lam_i)
            if key not in self._cache:
                Binc, Bref = self._compute_one(a_i, omega_i, lam_i)
                self._cache[key] = {
                    "a": a_i, "omega": omega_i,
                    "lambda": _complex_to_json(lam_i),
                    "B_inc": _complex_to_json(Binc),
                    "B_ref": _complex_to_json(Bref),
                    "N": self.N, "z_m": self.z_m,
                }
                changed = True
            row = self._cache[key]
            Binc_list.append(_complex_from_json(row["B_inc"]))
            Bref_list.append(_complex_from_json(row["B_ref"]))
        if changed:
            self._save()
        cdtype = torch.complex128 if self.dtype == torch.float64 else torch.complex64
        Binc_tensor = torch.tensor(Binc_list, device=self.device, dtype=cdtype)
        Bref_tensor = torch.tensor(Bref_list, device=self.device, dtype=cdtype)
        return Binc_tensor, Bref_tensor
