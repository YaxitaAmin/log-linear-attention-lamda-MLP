import sys
import os

# Add project paths dynamically
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'src', 'flash-linear-attention'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import hattention.modeling_hattention as mh
mh.LAMBDA_MODE_TYPE = 'mlp_softplus'

from mqar_minimal import generate_mqar, MQARModel, train
from hattention.base import get_num_levels

device = 'cuda'
seq_len = 128
num_kv_pairs = 8
vocab_size = 128
num_levels = get_num_levels(seq_len, base=2)

print("Training model first...")
# train a model
model = MQARModel(
    vocab_size=vocab_size, d_model=64, num_heads=1,
    state_size=64, num_levels=num_levels, n_layers=2,
    lambda_mode='mlp_softplus', mlp_hidden_dim=64,
).to(device)

# quick train
import torch.nn.functional as F
train_x, train_y = generate_mqar(10000, seq_len, num_kv_pairs, vocab_size, seed=0)
train_x, train_y = train_x.to(device), train_y.to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(2000):
    idx = torch.randperm(10000, device=device)[:64]
    x, y = train_x[idx], train_y[idx]
    logits = model(x)
    sl = logits[:, :-1]; sy = y[:, 1:]
    mask = sy != -100
    loss = F.cross_entropy(sl[mask].contiguous(), sy[mask].contiguous())
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if (step+1) % 500 == 0:
        print(f'Step {step+1} | Loss: {loss.item():.4f}')

print("Visualizing lambda...")
# capture lambda values
lambdas = []
def hook(module, input, output):
    lambdas.append(output.detach().cpu())
model.layers[0].lambda_module.register_forward_hook(hook)

val_x, val_y = generate_mqar(10, seq_len, num_kv_pairs, vocab_size, seed=99)
val_x = val_x.to(device)
model.eval()
with torch.no_grad():
    model(val_x[:1])

L = lambdas[0][0, :, 0, :].numpy()  # (seq_len, num_levels)

# mark KV pair positions and query positions
kv_positions = list(range(0, num_kv_pairs*2, 1))
query_start = seq_len - num_kv_pairs * 2

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(L.T, aspect='auto', cmap='viridis')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('λ value', fontsize=11)

# mark regions
ax.axvspan(0, num_kv_pairs*2-0.5, alpha=0.15, color='green', label='KV pairs')
ax.axvspan(query_start-0.5, seq_len-0.5, alpha=0.15, color='red', label='Query region')

ax.set_xlabel('Token position', fontsize=12)
ax.set_ylabel('Fenwick tree memory level (ℓ)', fontsize=12)
ax.set_title('MLP-λ Weights Across Memory Levels\n(Trained model, MQAR task, MLP Softplus)', fontsize=13)
ax.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('figs/lambda_heatmap_trained.png', dpi=150, bbox_inches='tight')
print('Saved figs/lambda_heatmap_trained.png!')
