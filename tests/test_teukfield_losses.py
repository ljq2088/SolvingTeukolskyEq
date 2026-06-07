import torch

from teukfield.losses.residuals import strong_reduced_residual
from teukfield.models import TeukfieldAmpNet


def test_strong_residual_loss_finite():
    model = TeukfieldAmpNet(local_hidden_dim=16, local_depth=1, latent_dim=16, n_windows=3).to(torch.float64)
    a = torch.tensor([0.5], dtype=torch.float64)
    logw = torch.tensor([-1.5], dtype=torch.float64)
    y = torch.linspace(-0.5, 0.8, 5, dtype=torch.float64).unsqueeze(0)
    loss, metrics = strong_reduced_residual(model, y, a, logw)
    assert torch.isfinite(loss)
    assert "res_rel_median" in metrics

