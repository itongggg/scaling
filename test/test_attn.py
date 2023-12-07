import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, num_new_dim: int = None) -> None:
        super().__init__()
        self.num_new_dim = num_new_dim
        assert n_embd % n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.n_head = n_head
        self.n_embd = n_embd
        if num_new_dim is not None:
            self.cproj_mask = torch.ones(n_embd).unsqueeze(0).unsqueeze(0)
            self.cproj_mask[:, :, -num_new_dim:] = 0
        else:
            self.cproj_mask = 1

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] ,
        stage_1: bool = False,
    ) -> Tuple[torch.Tensor]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        head_size = C // self.n_head

        if stage_1:
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
            new_q = torch.clone(q)
            new_k = torch.clone(k)
            new_v = torch.clone(v)
            new_q[:,:,:self.n_embd -self.num_new_dim] = qkv[:,:,:self.n_embd-self.num_new_dim]
            new_k[:,:,:self.n_embd -self.num_new_dim] = qkv[:,:,self.n_embd-self.num_new_dim:2*(self.n_embd-self.num_new_dim)]
            new_v[:,:,:self.n_embd -self.num_new_dim] = qkv[:,:,2*(self.n_embd-self.num_new_dim):3*(self.n_embd-self.num_new_dim)]
            new_q[:,:,-self.num_new_dim:] = 0
            new_k[:,:,-self.num_new_dim:] = 0
            new_v[:,:,-self.num_new_dim:] = 0
            # new_q = neq_q * self.cproj_mask
            # new_k = neq_k * self.cproj_mask
            # new_v = neq_v * self.cproj_mask
            q = new_q
            k = new_k
            v = new_v
            del new_q, new_k, new_v

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        else:
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # return q, k, v
        k = k.view(B, T, self.n_head, head_size)
        q = q.view(B, T, self.n_head, head_size)
        v = v.view(B, T, self.n_head, head_size)
        # return q, k ,v
        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)


        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        if stage_1:
            y = self.c_proj(y)
            return self.cproj_mask * y
        else:    
            y = self.c_proj(y)
        return y
    
torch.set_printoptions(threshold=50000)

mask = torch.ones(1, 1, 4, 4)
mask = torch.tril(mask)
print(mask)
orig_attn = CausalSelfAttention(128*3, 3)
x = torch.randn((1, 4, 128*3))
y = torch.randn((1, 4, 128*4))
y[:, :, :128*3] = x
y[:, :, -128:] = 0
new_attn = CausalSelfAttention(128*4, 4, num_new_dim=128)
new_attn.c_attn.weight.data[:128*3*3, :128*3] = orig_attn.c_attn.weight.data
new_attn.c_proj.weight.data[:128*3, :128*3] = orig_attn.c_proj.weight.data

# q1, k1, v1 = orig_attn(x, mask=mask)
# q2, k2, v2 = new_attn(y, mask=mask, stage_1=True)
# print(q1)
# print(q2[:, :, :4*3])
# print(k1)
# print(k2[:, :, :4*3])
# print(v1)
# print(v2[:, :, :4*3])
# q2p = q2[:, :, :4*3]
# k2p = k2[:, :, :4*3]
# v2p = v2[:, :, :4*3]
# print(torch.equal(q1, q2p))
# print(torch.equal(k1, k2p))
# print(torch.equal(v1, v2p))
# print(torch.allclose(q1, q2[:, :, :4*3], atol=1e-6))
# print(torch.allclose(k1, k2[:, :, :4*3], atol=1e-6))
# print(torch.allclose(v1, v2[:, :, :4*3], atol=1e-6))
# print(torch.equal(q1, q2[:, :, :4*3]))
# print(torch.equal(k1, k2[:, :, :4*3]))
# print(torch.equal(v1, v2[:, :, :4*3]))
# print(torch.eq(q1, q2[:, :, :4*3]))
# print(torch.eq(k1, k2[:, :, :4*3]))
# print(torch.eq(v1, v2[:, :, :4*3]))
x_o = orig_attn(x, mask=mask)
y_o = new_attn(y, mask=mask, stage_1=True)
print(torch.allclose(x_o, y_o[:, :, :128*3], atol=1e-7))
# print(x_o)
# print(y_o)
print(torch.equal(x_o, y_o[:, :, :128*3]))
print(x_o - y_o[:, :, :128*3])