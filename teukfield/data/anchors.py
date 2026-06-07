from __future__ import annotations

import numpy as np
import torch

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

    def torch_inner_batch(
        self,
        a: torch.Tensor,
        logw: torch.Tensor,
        z_values: torch.Tensor,
        device: str | torch.device,
    ) -> torch.Tensor:
        rows = []
        z_np = z_values.detach().cpu().numpy()
        for ai, li in zip(a.detach().cpu().numpy(), logw.detach().cpu().numpy()):
            rows.append(self.inner_R(float(ai), 10.0 ** float(li), z_np))
        return torch.tensor(np.stack(rows), dtype=torch.complex128, device=device)

    def torch_outer_batch(
        self,
        a: torch.Tensor,
        logw: torch.Tensor,
        z_values: torch.Tensor,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        down_rows, up_rows = [], []
        z_np = z_values.detach().cpu().numpy()
        for ai, li in zip(a.detach().cpu().numpy(), logw.detach().cpu().numpy()):
            branches = self.outer_branches(float(ai), 10.0 ** float(li), z_np)
            down_rows.append(branches["A_down_u"])
            up_rows.append(branches["A_up_u"])
        return (
            torch.tensor(np.stack(down_rows), dtype=torch.complex128, device=device),
            torch.tensor(np.stack(up_rows), dtype=torch.complex128, device=device),
        )
