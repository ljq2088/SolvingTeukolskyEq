import torch
import sys
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
from dataset.grids import chebyshev_grid_bundle

grid = chebyshev_grid_bundle(order=8)
y = grid.y_nodes
D = grid.D
D2 = grid.D2

f = y**2
fy_true = 2.0 * y
fyy_true = 2.0 * torch.ones_like(y)

fy_num = D @ f
fyy_num = D2 @ f

print("max |fy_num - fy_true| =", torch.max(torch.abs(fy_num - fy_true)).item())
print("max |fyy_num - fyy_true| =", torch.max(torch.abs(fyy_num - fyy_true)).item())