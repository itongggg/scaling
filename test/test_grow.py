import os
import sys
import torch
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from lit_llama.model import Block, LLaMA, LLaMAConfig

model_name = "300M"
model_config = LLaMAConfig.from_name(model_name)

model = LLaMA(model_config)


old_block_attn = [model.transformer['h'][i].attn.c_attn.weight.detach() for i in range(24)]
old_block_mlp = [model.transformer['h'][i].mlp.c_fc1.weight.detach() for i in range(24)]

# print(model)
new_model_name = "500M"

new_model_config = LLaMAConfig.from_name(new_model_name)

model.grow_model(new_model_config)

print(model)
# print(model.transformer['ln_f'].scale.shape)
# print(model.transformer['h'][0].rms_1.scale.shape)
# print(model.new_block_index)
# print(model.old_block_index)
blcok_0_attn_new = model.transformer['h'][0].attn.c_attn.weight.detach()


old_layers = [0,2,4,6,8,10,12,14,] + [i for i in range(16, 32)]
for index, i in enumerate(old_layers):
    block_i_attn = model.transformer['h'][i].attn.c_attn.weight.detach()
    old_attn = old_block_attn[index]
    print(torch.equal(block_i_attn[:2688, :896], old_attn))


for index, i in enumerate(old_layers):
    block_i_mlp = model.transformer['h'][i].mlp.c_fc1.weight.detach()
    old_mlp = old_block_mlp[index]
    print(torch.equal(block_i_mlp[:2560, :896], old_mlp))