import torch
import torch.nn as nn
import torch.nn.functional as F

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)
class MLP(nn.Module):
    def __init__(self, n_embd, num_new_dim: int = None) -> None:
        super().__init__()
        hidden_dim = 4 * n_embd
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)
        
        if num_new_dim is not None:
            old_n_hidden = find_multiple(int((n_embd - num_new_dim) * 8 / 3), 256)
            self.c_mask1 = torch.ones(n_embd).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.c_mask1[:, :, :, -num_new_dim:] = 0
            self.c_mask2 = torch.ones(n_hidden).unsqueeze(0).unsqueeze(0).unsqueeze(0) if n_hidden > old_n_hidden else 1
            self.c_mask2[:, :, :, -(n_hidden - old_n_hidden):] = 0
        else:
            self.c_mask1 = 1
            self.c_mask2 = 1
        
        self.c_fc1 = nn.Linear(n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_mask2 * F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_mask1 * self.c_proj(x)
        return x
    
# x = torch.randn(16)
# y = torch.zeros(20)
# y[:16] = x  

# m1 = MLP(16)
# m2 = MLP(20, num_new_dim=4)
# m2.c_proj.weight.data[:16, :] = m1.c_proj.weight.data
# m2.c_fc1.weight.data[:, :16] = m1.c_fc1.weight.data
# m2.c_fc2.weight.data[:, :16] = m1.c_fc2.weight.data
# print(m1(x))
# print(m2(y))
# print(torch.equal(m1(x), m2(y).squeeze()[:16]))

x = torch.randn(16)
y = torch.zeros(140)
y[:16] = x  

m1 = MLP(16)
m2 = MLP(140, num_new_dim=124)
m2.c_proj.weight.data[:16, :256] = m1.c_proj.weight.data
m2.c_fc1.weight.data[:256, :16] = m1.c_fc1.weight.data
m2.c_fc2.weight.data[:256, :16] = m1.c_fc2.weight.data
print(m1(x))
print(m2(y))
print(torch.equal(m1(x), m2(y).squeeze()[:16]))