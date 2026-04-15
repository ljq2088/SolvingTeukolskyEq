import torch
import sys
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
from model.pinn_mlp import PINN_MLP


dtype = torch.float64
device = "cpu"

model = PINN_MLP().to(device=device, dtype=dtype)

a = torch.tensor([0.1, 0.12], dtype=dtype, device=device)
omega = torch.tensor([0.2, 0.5], dtype=dtype, device=device)
y = torch.linspace(-0.99, 0.99, 64, dtype=dtype, device=device)

out = model(a, omega, y)
print(out.shape)
print(out.dtype)