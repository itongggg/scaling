import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LLaMAConfig, num_new_dim: int = None) -> None:
        super().__init__()
        self.num_new_dim = num_new_dim
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # self.Q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # self.K = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # self.V = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        if num_new_dim is not None:
            self.c_mask = torch.ones(config.n_embd).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        else:
            self.c_mask = 1

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: MaskCache,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
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

            k = k.view(B, T, self.n_head, head_size)
            q = q.view(B, T, self.n_head, head_size)
            v = v.view(B, T, self.n_head, head_size)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        
        k = k.view(B, T, self.n_head, head_size)
        q = q.view(B, T, self.n_head, head_size)
        v = v.view(B, T, self.n_head, head_size)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=2)
                cache_v = torch.roll(cache_v, -1, dims=2)
            k = cache_k.index_copy(2, input_pos, k)
            v = cache_v.index_copy(2, input_pos, v)
            kv_cache = k, v

        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)

        return y, kv_cache