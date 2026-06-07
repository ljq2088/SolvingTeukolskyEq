from __future__ import annotations

import numpy as np

from teukspec.atlas.patch_expert import PatchExpert
from utils.amplitude import A_down, A_in, A_up, r_of_z


class SpectralBoundaryAdapter:
    """Diagnostics-only spectral boundary adapter around existing PatchExpert."""

    def __init__(self, patch_dir: str):
        self.expert = PatchExpert(patch_dir)

    def inner_R(self, a: float, omega: float, z_values: np.ndarray):
        mode = self.expert.mode(a, omega)
        u = self.expert.eval_branches(a, omega, z_values)["in"]
        return A_in(r_of_z(z_values, mode), mode) * u

    def outer_branches(self, a: float, omega: float, z_values: np.ndarray):
        mode = self.expert.mode(a, omega)
        branches = self.expert.eval_branches(a, omega, z_values)
        r = r_of_z(z_values, mode)
        return {
            "A_down_u": A_down(r, mode) * branches["down"],
            "A_up_u": A_up(r, mode) * branches["up"],
        }

