import torch

from teukfield.models import TeukfieldAmpNet
from teukfield.physics.reduced_equation import horizon_slope_y


def _cgrad(v, x):
    gr = torch.autograd.grad(v.real.sum(), x, create_graph=True, retain_graph=True)[0]
    gi = torch.autograd.grad(v.imag.sum(), x, create_graph=True, retain_graph=True)[0]
    return torch.complex(gr, gi)


def test_horizon_hard_constraints():
    model = TeukfieldAmpNet(local_hidden_dim=16, local_depth=1, latent_dim=16, n_windows=3).to(torch.float64)
    a = torch.tensor([0.5], dtype=torch.float64)
    logw = torch.tensor([-1.5], dtype=torch.float64)
    y = torch.ones(1, 1, dtype=torch.float64, requires_grad=True)
    out = model(y, a, logw)
    Sy = _cgrad(out["S"], y)
    omega = torch.pow(torch.tensor(10.0, dtype=torch.float64), logw)
    expected = horizon_slope_y(a, omega, out["lambda"])
    assert torch.max(torch.abs(out["S"] - 1.0)).item() < 1e-10
    assert torch.max(torch.abs(Sy.reshape(-1) - expected.reshape(-1))).item() < 1e-8

