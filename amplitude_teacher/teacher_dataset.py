"""Dataclass for amplitude teacher data — (a, omega) -> B_inc, B_ref."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class AmplitudeTeacher:
    a_vals: np.ndarray          # (N,) spin
    omega_vals: np.ndarray      # (N,) frequency
    B_inc: np.ndarray           # (N,) complex, incidence amplitude
    B_ref: np.ndarray           # (N,) complex, reflection amplitude
    metadata: dict = field(default_factory=dict)

    def save(self, path):
        np.savez_compressed(
            path,
            a=self.a_vals,
            omega=self.omega_vals,
            B_inc=self.B_inc,
            B_ref=self.B_ref,
            **{f"meta_{k}": v for k, v in self.metadata.items()},
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        meta = {}
        arr_keys = []
        for k in data.keys():
            if k.startswith("meta_"):
                meta[k[5:]] = data[k].item()
            else:
                arr_keys.append(k)
        return cls(
            a_vals=data["a"],
            omega_vals=data["omega"],
            B_inc=data["B_inc"],
            B_ref=data["B_ref"],
            metadata=meta,
        )
