from __future__ import annotations

import numpy as np
import torch

from teukspec.atlas.patch_expert import PatchExpert
from utils.amplitude import A_down, A_in, A_up, r_of_z
from utils.mode import KerrMode
from pybhpt_usage.compute_solution import compute_pybhpt_solution


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

    def torch_R_anchor_batch(
        self,
        a: torch.Tensor,
        logw: torch.Tensor,
        z_values: torch.Tensor,
        device: str | torch.device,
    ) -> torch.Tensor:
        rows = []
        z_np = z_values.detach().cpu().numpy()
        for ai, li in zip(a.detach().cpu().numpy(), logw.detach().cpu().numpy()):
            omega = 10.0 ** float(li)
            mode = self.expert.mode(float(ai), omega)
            r_np = r_of_z(z_np, mode)
            rows.append(self.expert.eval_R_in(float(ai), omega, r_np))
        return torch.tensor(np.stack(rows), dtype=torch.complex128, device=device)

    def torch_amplitude_anchor_batch(
        self,
        a: torch.Tensor,
        logw: torch.Tensor,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B_inc_rows, B_ref_rows = [], []
        for ai, li in zip(a.detach().cpu().numpy(), logw.detach().cpu().numpy()):
            amps = self.expert.solve_amplitudes(float(ai), 10.0 ** float(li))
            B_inc_rows.append(complex(amps["B_inc"]))
            B_ref_rows.append(complex(amps["B_ref"]))
        return (
            torch.tensor(np.asarray(B_inc_rows), dtype=torch.complex128, device=device),
            torch.tensor(np.asarray(B_ref_rows), dtype=torch.complex128, device=device),
        )


def torch_pybhpt_R_anchor_batch(
    a: torch.Tensor,
    logw: torch.Tensor,
    z_values: torch.Tensor,
    device: str | torch.device,
    timeout: float = 60.0,
) -> torch.Tensor:
    rows = []
    z_np = z_values.detach().cpu().numpy()
    for ai, li in zip(a.detach().cpu().numpy(), logw.detach().cpu().numpy()):
        omega = 10.0 ** float(li)
        mode = KerrMode(M=1.0, a=float(ai), omega=omega, ell=2, m=2, s=-2)
        r_np = r_of_z(z_np, mode)
        _, R_ref = compute_pybhpt_solution(float(ai), omega, ell=2, m=2, r_grid=r_np, timeout=timeout)
        rows.append(np.asarray(R_ref, dtype=np.complex128))
    return torch.tensor(np.stack(rows), dtype=torch.complex128, device=device)
