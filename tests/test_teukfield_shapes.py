import torch

from teukfield.models import TeukfieldAmpNet


def test_forward_shapes():
    model = TeukfieldAmpNet(local_hidden_dim=16, local_depth=1, latent_dim=16, n_windows=3).to(torch.float64)
    a = torch.tensor([0.49, 0.51], dtype=torch.float64)
    logw = torch.tensor([-1.5, -1.45], dtype=torch.float64)
    y = torch.linspace(-0.8, 0.9, 7, dtype=torch.float64).unsqueeze(0).expand(2, -1)
    out = model(y, a, logw)
    assert out["S"].shape == (2, 7)
    assert out["R"].shape == (2, 7)
    assert out["B_inc"].shape == (2,)
    assert torch.is_complex(out["S"])

