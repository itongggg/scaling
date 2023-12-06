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
        mask: MaskCache,
        stage_1: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        head_size = C // self.n_head
        if stage_1:
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
            q[:,:,:self.n_embd -self.num_new_dim] = qkv[:,:,:self.n_embd-self.num_new_dim]
            k[:,:,:self.n_embd -self.num_new_dim] = qkv[:,:,self.n_embd-self.num_new_dim:2*(self.n_embd-self.num_new_dim)]
            v[:,:,:self.n_embd -self.num_new_dim] = qkv[:,:,2*(self.n_embd-self.num_new_dim):3*(self.n_embd-self.num_new_dim)]
            q[:,:,-self.num_new_dim:] = 0
            k[:,:,-self.num_new_dim:] = 0
            v[:,:,-self.num_new_dim:] = 0

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        else:
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        
        k = k.view(B, T, self.n_head, head_size)
        q = q.view(B, T, self.n_head, head_size)
        v = v.view(B, T, self.n_head, head_size)

        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)


        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        if stage_1:
            y = y * self.cproj_mask

        return y
    
orig_attn = CausalSelfAttention(128*3, 3)
x = torch.randn((1, 4, 128*3))
y = torch.randn((1, 4, 128*4))
y[:, :, :128*3] = x
y[:, :, -128:] = 0
new_attn = CausalSelfAttention(128*4, 4, num_new_dim=128)
new_attn.c_attn.weight.data[:128*3*3, :128*3] = orig_attn.c_attn.weight.data
new_attn.c_proj.weight.data[:128*3, :128*3] = orig_attn.c_proj.weight.data