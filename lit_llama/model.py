"""Full definition of a LLaMA Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
"""
# mypy: ignore-errors
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from loguru import logger
import torch
import gc
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self
from lit_llama.lowrank import proximal
from lit_llama.utils import find_multiple


MaskCache = torch.Tensor
RoPECache = torch.Tensor
KVCache = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class LLaMAConfig:
    block_size: int = 2048
    vocab_size: int = 32000
    padded_vocab_size: Optional[int] = None
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096

    def __post_init__(self):
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, 64)

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**llama_configs[name])



llama_configs = {
    "s": dict(n_layer=2, n_head=1, n_embd=16),
    "s1": dict(n_layer=2, n_head=1, n_embd=20),
    "500M": dict(n_layer=32, n_head=8, n_embd=1024),
    "300M": dict(n_layer=24, n_head=7, n_embd=896),
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
}


class LLaMA(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=RMSNorm(config.n_embd),
            )
        )

        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[MaskCache] = None
        self.kv_caches: List[KVCache] = []
        self.mask_grown = False
        self.hooks = {}

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def forward(
        self, idx: torch.Tensor, max_seq_length: Optional[int] = None, input_pos: Optional[torch.Tensor] = None, res: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[KVCache]]]:
        B, T = idx.size()

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        assert T <= max_seq_length, f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert T <= block_size, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        if self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        if input_pos is not None:
            rope = self.rope_cache.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            rope = self.rope_cache[:T]
            mask = self.mask_cache[:, :, :T, :T]

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        if input_pos is None:  # proxy for use_cache=False
            if res:
                for i, block in enumerate(self.transformer.h):
                    if i in self.new_block_index:
                        x_new, _ = block(x, rope, mask, max_seq_length)
                        x = 0.5 * x + 0.5 * x_new
                    else:
                        x, _ = block(x, rope, mask, max_seq_length)
            else:
                for block in self.transformer.h:
                    x, _ = block(x, rope, mask, max_seq_length)
        else:
            if not self.kv_caches:
                head_size = self.config.n_embd // self.config.n_head
                cache_shape = (B, self.config.n_head, max_seq_length, head_size)
                self.kv_caches = [
                    (torch.zeros(cache_shape, device=x.device, dtype=x.dtype), torch.zeros(cache_shape, device=x.device, dtype=x.dtype))
                    for _ in range(self.config.n_layer)
                ]
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(x, rope, mask, max_seq_length, input_pos, self.kv_caches[i])

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (b, t, vocab_size)

        return logits
    
    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(LLaMAConfig.from_name(name))

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=self.config.n_embd // self.config.n_head,
            dtype=idx.dtype,
            device=idx.device,
        )

    def build_mask_cache(self, idx: torch.Tensor) -> MaskCache:
        ones = torch.ones((self.config.block_size, self.config.block_size), device=idx.device, dtype=torch.bool)
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def reset_cache(self) -> None:
        self.kv_caches.clear()
        if self.mask_cache.device.type == "xla":
            # https://github.com/Lightning-AI/lit-parrot/pull/83#issuecomment-1558150179
            self.rope_cache = None
            self.mask_cache = None
    
    @torch.no_grad()
    def grow_model(self, new_config: LLaMAConfig, way: str = "smooth"):
        """Grow the model to a new config, by adding more layers and increasing the embedding size.
        """
        def copy_weight():
            old_block_rm1 = old_blocks[0].rms_1.scale
            old_block_rm2 = old_blocks[0].rms_2.scale

            old_block_attn = old_blocks[0].attn.c_attn.weight
            old_block_proj = old_blocks[0].attn.c_proj.weight

            old_block_mlp1 = old_blocks[0].mlp.c_fc1.weight
            old_block_mlp2 = old_blocks[0].mlp.c_fc2.weight
            old_block_mlp3 = old_blocks[0].mlp.c_proj.weight

            new_block = Block(new_config)
            new_block.rms_1.scale[:self.config.n_embd].copy_(old_block_rm1)
            new_block.rms_2.scale[:self.config.n_embd].copy_(old_block_rm2)
            new_block.attn.c_attn.weight[:3 * self.config.n_embd, :self.config.n_embd].copy_(old_block_attn)
            new_block.attn.c_proj.weight[:self.config.n_embd, :self.config.n_embd].copy_(old_block_proj)
            new_block.mlp.c_fc1.weight[:old_n_hidden, :self.config.n_embd].copy_(old_block_mlp1)
            new_block.mlp.c_fc2.weight[:old_n_hidden, :self.config.n_embd].copy_(old_block_mlp2)
            new_block.mlp.c_proj.weight[:self.config.n_embd, :old_n_hidden].copy_(old_block_mlp3)

            return new_block

        assert new_config.n_embd >= self.config.n_embd
        assert new_config.n_layer >= self.config.n_layer
        assert new_config.block_size == self.config.block_size
        self.old_block_index = []
        self.new_block_index = []
        self.emb_grow = new_config.n_embd - self.config.n_embd
        self.layer_grow = new_config.n_layer - self.config.n_layer
        self.head_grow = new_config.n_head - self.config.n_head
        self.vocab_grow = new_config.padded_vocab_size - self.config.padded_vocab_size

        # grow the embedding layer
        old_wte = self.transformer.wte.weight
        new_wte = nn.Embedding(new_config.padded_vocab_size, new_config.n_embd)
        new_wte.weight[: self.config.padded_vocab_size, : self.config.n_embd].copy_(old_wte)
        self.transformer.wte = new_wte

        # grow the linear head
        old_lm_head = self.lm_head.weight
        new_lm_head = nn.Linear(new_config.n_embd, new_config.padded_vocab_size, bias=False)
        new_lm_head.weight[: self.config.padded_vocab_size, : self.config.n_embd].copy_(old_lm_head)
        self.lm_head = new_lm_head

        old_transforemr_lnf = self.transformer.ln_f.scale
        new_transformer_lnf = RMSNorm(new_config.n_embd)
        new_transformer_lnf.scale[:self.config.n_embd].copy_(old_transforemr_lnf)
        self.transformer.ln_f = new_transformer_lnf
        # grow the transformer blocks
        new_blocks = [] 
        old_blocks = self.transformer.h
        
        old_n_hidden = find_multiple(int(self.config.n_embd * 4 * 2 / 3), 256)
        
        if new_config.n_layer == self.config.n_layer:
            for i in range(new_config.n_layer):
                self.old_block_index.append(i)
                new_blocks.append(copy_weight())
                del old_blocks[0]

        if way == "smooth":
            for i in range(new_config.n_layer):
                if i > 2 * self.config.n_layer - 1:
                    self.new_block_index.append(i)
                    new_blocks.append(Block(new_config))
                else:
                    if i % 2 == 0 or i > 2 * (new_config.n_layer - self.config.n_layer) - 1:
                        new_blocks.append(copy_weight())
                        self.old_block_index.append(i)
                        del old_blocks[0]
                    else:
                        new_blocks.append(Block(new_config))
                        self.new_block_index.append(i)
        if way == "up":
            for i in range(new_config.n_layer):
                if i < self.config.n_layer:
                    self.old_block_index.append(i)
                    new_blocks.append(copy_weight())
                    del old_blocks[0]
                else:
                    self.new_block_index.append(i)
                    new_blocks.append(Block(new_config))
        if way == "down":
            for i in range(new_config.n_layer):
                if i >= self.layer_grow:
                    self.old_block_index.append(i)
                    new_blocks.append(copy_weight())
                    del old_blocks[0]
                else:
                    self.new_block_index.append(i)
                    new_blocks.append(Block(new_config))
        if way == "middle":
            middle = int(self.config.n_layer / 2)
            for i in range(new_config.n_layer):
                if i >= middle and i < middle + self.layer_grow:
                    self.new_block_index.append(i)
                    new_blocks.append(Block(new_config))
                else:
                    self.old_block_index.append(i)
                    new_blocks.append(copy_weight())
                    del old_blocks[0]
        self.transformer.h = nn.ModuleList(new_blocks)
        self.config = new_config


    @torch.no_grad()
    def _init_new_weights(self, low_rank=True, new_block="random"):
        orgin_dim = self.config.n_embd - self.emb_grow
        orgin_hidden = find_multiple(int(orgin_dim * 8 / 3), 256)
        for i, block in enumerate(self.transformer.h):
            if i in self.new_block_index:
                if new_block == "random":
                    block.apply(self._init_weights)
                if new_block == "up":
                    index = min(num for num in self.old_block_index if num > i)
                    block.attn.c_attn.weight.data = self.transformer.h[index].attn.c_attn.weight.data
                    block.attn.c_proj.weight.data = self.transformer.h[index].attn.c_proj.weight.data
                    block.mlp.c_fc1.weight.data = self.transformer.h[index].mlp.c_fc1.weight.data
                    block.mlp.c_fc2.weight.data = self.transformer.h[index].mlp.c_fc2.weight.data
                    block.mlp.c_proj.weight.data = self.transformer.h[index].mlp.c_proj.weight.data
                if new_block == "down":
                    index = max(num for num in self.old_block_index if num < i)
                    block.attn.c_attn.weight.data = self.transformer.h[index].attn.c_attn.weight.data
                    block.attn.c_proj.weight.data = self.transformer.h[index].attn.c_proj.weight.data
                    block.mlp.c_fc1.weight.data = self.transformer.h[index].mlp.c_fc1.weight.data
                    block.mlp.c_fc2.weight.data = self.transformer.h[index].mlp.c_fc2.weight.data
                    block.mlp.c_proj.weight.data = self.transformer.h[index].mlp.c_proj.weight.data  
                if new_block == "linear":
                    i1 = min(num for num in self.old_block_index if num > i)
                    i2 = max(num for num in self.old_block_index if num < i)
                    block.attn.c_attn.weight.data = 0.5 * self.transformer.h[i1].attn.c_attn.weight.data + 0.5 * self.transformer.h[i2].attn.c_attn.weight.data
                    block.attn.c_proj.weight.data = 0.5 * self.transformer.h[i1].attn.c_proj.weight.data + 0.5 * self.transformer.h[i2].attn.c_proj.weight.data
                    block.mlp.c_fc1.weight.data = 0.5 * self.transformer.h[i1].mlp.c_fc1.weight.data + 0.5 * self.transformer.h[i2].mlp.c_fc1.weight.data
                    block.mlp.c_fc2.weight.data = 0.5 * self.transformer.h[i1].mlp.c_fc2.weight.data + 0.5 * self.transformer.h[i2].mlp.c_fc2.weight.data
                    block.mlp.c_proj.weight.data = 0.5 * self.transformer.h[i1].mlp.c_proj.weight.data + 0.5 * self.transformer.h[i2].mlp.c_proj.weight.data
            else:
                if low_rank:
                    # block.attn.c_attn.weight = proximal(block.attn.c_attn.weight, 0.1, 0.1, (3 * orgin_dim, orgin_dim))
                    Wq = block.attn.c_attn.weight.data[:self.config.n_embd, :self.config.n_embd]
                    Wk = block.attn.c_attn.weight.data[self.config.n_embd:2*self.config.n_embd, :self.config.n_embd]
                    Wv = block.attn.c_attn.weight.data[2*self.config.n_embd:3*self.config.n_embd, :self.config.n_embd]
                    Wq = proximal(Wq, 0.1, 1, (orgin_dim, orgin_dim))
                    Wk = proximal(Wk, 0.1, 1, (orgin_dim, orgin_dim))
                    Wv = proximal(Wv, 0.1, 1, (orgin_dim, orgin_dim))
                    block.attn.c_attn.weight[:self.config.n_embd, :self.config.n_embd] = Wq
                    block.attn.c_attn.weight[self.config.n_embd:2*self.config.n_embd, :self.config.n_embd] = Wk
                    block.attn.c_attn.weight[2*self.config.n_embd:3*self.config.n_embd, :self.config.n_embd] = Wv
                    block.attn.c_proj.weight.data = proximal(block.attn.c_proj.weight, 0.1, 1, (orgin_dim, orgin_dim))
                    block.mlp.c_fc1.weight.data = proximal(block.mlp.c_fc1.weight, 0.1, 1, (orgin_hidden, orgin_dim))
                    block.mlp.c_fc2.weight.data = proximal(block.mlp.c_fc2.weight, 0.1, 1, (orgin_hidden, orgin_dim))
                    block.mlp.c_proj.weight.data = proximal(block.mlp.c_proj.weight, 0.1, 1, (orgin_dim, orgin_hidden))
                    
    
    def freeze_old_params(self):
        def create_hook(shape):
            def hook(grad):
                grad_clone = grad.clone()
                if type(shape) is not int:
                    grad_clone[:shape[0], :shape[1]] = 0
                else:
                    grad_clone[:shape] = 0
                return grad_clone
            return hook
        origin_dim = self.config.n_embd - self.emb_grow
        origin_hidden = find_multiple(int(origin_dim * 8 / 3), 256)
        for i, block in enumerate(self.transformer.h):
            if i in self.old_block_index:
                hook_id_rms_1 = f"{i}_rms_1_scale"
                hook_id_rms_2 = f"{i}_rms_2_scale"
                hook_id_c_attn = f"{i}_c_attn_weight"
                hook_id_c_proj = f"{i}_c_proj_weight"
                hook_id_mlp_fc1 = f"{i}_mlp_fc1"
                hook_id_mlp_fc2 = f"{i}_mlp_fc2"
                hook_id_mlp_proj = f"{i}_mlp_proj"

                hook_rms = create_hook(origin_dim)
                self.hooks[hook_id_rms_1] = block.rms_1.scale.register_hook(hook_rms)
                self.hooks[hook_id_rms_2] = block.rms_2.scale.register_hook(hook_rms)

                hook_c_attn = create_hook((3 * origin_dim, origin_dim))
                self.hooks[hook_id_c_attn] = block.attn.c_attn.weight.register_hook(hook_c_attn)

                hook_c_proj = create_hook((origin_dim, origin_dim))
                self.hooks[hook_id_c_proj] = block.attn.c_proj.weight.register_hook(hook_c_proj)

                hook_mlp_fc = create_hook((origin_hidden, origin_dim))
                self.hooks[hook_id_mlp_fc1] = block.mlp.c_fc1.weight.register_hook(hook_mlp_fc)
                self.hooks[hook_id_mlp_fc2] = block.mlp.c_fc2.weight.register_hook(hook_mlp_fc)

                hook_mlp_proj = create_hook((origin_dim, origin_hidden))
                self.hooks[hook_id_mlp_proj] = block.mlp.c_proj.weight.register_hook(hook_mlp_proj)
        lnf_handle = self.transformer.ln_f.scale.register_hook(create_hook(origin_dim))
        lm_head_handle = self.lm_head.weight.register_hook(create_hook((self.config.padded_vocab_size, origin_dim)))
        wte_handle = self.transformer.wte.weight.register_hook(create_hook((self.config.padded_vocab_size, origin_dim)))
        self.hooks["lnf"] = lnf_handle
        self.hooks["lm_head"] = lm_head_handle
        self.hooks["wte"] = wte_handle
                

    def unfreeze_old_params(self):
        for hook in self.hooks.values():
            hook.remove()
        self.hooks = {}

    def register_forward_hook_for_old_block(self):
        def hook_fn(module, input, output):
            print(f"Output of {module.__class__.__name__}: {output}")
        for i, block in enumerate(self.transformer.h):
            if i in self.old_block_index:
                block.attn.register_forward_hook(hook_fn)
                block.mlp.register_forward_hook(hook_fn)
    
class Block(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: MaskCache,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        h, new_kv_cache = self.attn(self.rms_1(x), rope, mask, max_seq_length, input_pos, kv_cache)
        x = x + h
        x = x + self.mlp(self.rms_2(x))
        return x, new_kv_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # self.Wq = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # self.Wv = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
    
    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: MaskCache,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        head_size = C // self.n_head
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


class MLP(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)
        
        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed 


def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.half()
    return cache


def apply_rope(x: torch.Tensor, rope_cache: RoPECache) -> torch.Tensor:
    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # cast because the reference does
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)