
import torch
import torch.nn as nn
class RMSNorm(nn.Module):

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5, num_new_dim: int = None) -> None:
        super().__init__()
        if num_new_dim is not None:
            self.c_mask = float(size) /  float(size - num_new_dim)
        else:
            self.c_mask = 1
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True) * self.c_mask
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        print(x_normed)
        return self.scale * x_normed 
    
x = torch.randn((3, 3))
y = torch.randn((3, 5))
y[:, :3] = x
y[:, -2:] = 0
r1 = RMSNorm(3)
r2 = RMSNorm(5, num_new_dim=2)
x1 = r1(x)
y1 = r2(y)
print(x1)
print(y1)