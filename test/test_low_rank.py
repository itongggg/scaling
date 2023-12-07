import torch

import matplotlib.pyplot as plt
# torch.linalg.matrix_rank(t)



ck_path = "/workspace/checkpoints/300m-base/iter-049999.pt"
model = torch.load(ck_path)

layer_numbers = []
attn_list_sum = []
c_proj_list_sum = []    
mlp_c_fc1_list_sum = []
mlp_c_fc2_list_sum = []
mlp_c_proj_list_sum = []



attn_list_g5 = []
c_proj_list_g5 = []    
mlp_c_fc1_list_g5 = []
mlp_c_fc2_list_g5 = []
mlp_c_proj_list_g5 = []

for key in model.keys():
    if key.contains("wte"):
        continue
    if key.startswith("transformer"): 
        if key.endswith("weight"):
            
            rank = torch.linalg.matrix_rank(model[key])
            _, S, _ = torch.linalg.svd(model[key])
            layer_number = int(key.split('.')[2]) 
            layer_numbers.append(layer_number)
            if key.contains("attn"):
                if key.contains("c_attn"):
                    attn_list_sum.append(torch.sum(S).item())
                    attn_list_g5.append((S > 5).sum.item())
                else:
                    c_proj_list_sum.append(torch.sum(S).item())
                    c_proj_list_g5.append((S > 5).sum.item())
            else:
                if key.contains("c_fc1"):
                    mlp_c_fc1_list_sum.append(torch.sum(S).item())
                    mlp_c_fc1_list_g5.append((S > 5).sum.item())
                elif key.contains("c_fc2"):
                    mlp_c_fc2_list_sum.append(torch.sum(S).item())
                    mlp_c_fc2_list_g5.append((S > 5).sum.item())
                else:
                    mlp_c_proj_list_sum.append(torch.sum(S).item())
                    mlp_c_proj_list_g5.append((S > 5).sum.item())
                
    # Create a figure with subplots
fig, axs = plt.subplots(2, 5, figsize=(25, 10))

# Plotting the sum of singular values for each component
axs[0, 0].plot(layer_numbers, attn_list_sum, marker='o')
axs[0, 0].set_title("Sum of Singular Values (attn)")
axs[0, 1].plot(layer_numbers, c_proj_list_sum, marker='o')
axs[0, 1].set_title("Sum of Singular Values (c_proj)")
axs[0, 2].plot(layer_numbers, mlp_c_fc1_list_sum, marker='o')
axs[0, 2].set_title("Sum of Singular Values (mlp_c_fc1)")
axs[0, 3].plot(layer_numbers, mlp_c_fc2_list_sum, marker='o')
axs[0, 3].set_title("Sum of Singular Values (mlp_c_fc2)")
axs[0, 4].plot(layer_numbers, mlp_c_proj_list_sum, marker='o')
axs[0, 4].set_title("Sum of Singular Values (mlp_c_proj)")

# Plotting the count of singular values > 5 for each component
axs[1, 0].plot(layer_numbers, attn_list_g5, marker='o')
axs[1, 0].set_title("Count of Singular Values > 5 (attn)")
axs[1, 1].plot(layer_numbers, c_proj_list_g5, marker='o')
axs[1, 1].set_title("Count of Singular Values > 5 (c_proj)")
axs[1, 2].plot(layer_numbers, mlp_c_fc1_list_g5, marker='o')
axs[1, 2].set_title("Count of Singular Values > 5 (mlp_c_fc1)")
axs[1, 3].plot(layer_numbers, mlp_c_fc2_list_g5, marker='o')
axs[1, 3].set_title("Count of Singular Values > 5 (mlp_c_fc2)")
axs[1, 4].plot(layer_numbers, mlp_c_proj_list_g5, marker='o')
axs[1, 4].set_title("Count of Singular Values > 5 (mlp_c_proj)")

# Setting labels for axes
for ax in axs.flat:
    ax.set(xlabel='Layer Number', ylabel='Value')

# Adjust layout
plt.tight_layout()
plt.show()