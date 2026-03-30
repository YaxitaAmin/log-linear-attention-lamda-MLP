"""
Minimal self-contained MQAR experiment for Log-Linear Attention.
No HuggingFace. Zoology-style data. Official hattention Triton kernel.
"""
import sys
sys.path.insert(0, "/scratch/zt1/project/msml612/user/yaxita/log-linear-attention/flame/3rdparty/flash-linear-attention")
sys.path.insert(1, "/scratch/zt1/project/msml612/user/yaxita/log-linear-attention")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
from einops import rearrange

from hattention.base import HType, HStruct, get_num_levels
from hattention.kernel import hattention_kernel
from hattention.lambda_mlp import LambdaMLPSoftplus, LambdaMLPSoftmax

LEVEL_BASE = 2
HTYPE      = HType.WEAK
HSTRUCT    = HStruct.MAMBA2


def generate_mqar(n_samples, seq_len, num_kv_pairs, vocab_size=8192, seed=42):
    np.random.seed(seed)
    half        = vocab_size // 2
    query_start = seq_len - num_kv_pairs * 2
    input_ids   = np.zeros((n_samples, seq_len), dtype=np.int64)
    labels      = np.full((n_samples, seq_len), -100, dtype=np.int64)
    key_choices   = np.arange(1, half)
    value_choices = np.arange(half, vocab_size)

    for b in range(n_samples):
        keys   = np.random.choice(key_choices,   size=num_kv_pairs, replace=False)
        values = np.random.choice(value_choices, size=num_kv_pairs, replace=False)
        kv     = dict(zip(keys.tolist(), values.tolist()))
        for i, (k, v) in enumerate(zip(keys, values)):
            input_ids[b, i*2]     = k
            input_ids[b, i*2 + 1] = v
        qkeys = np.random.choice(keys, size=num_kv_pairs, replace=False)
        for i, qk in enumerate(qkeys):
            pos = query_start + i * 2
            input_ids[b, pos]     = int(qk)
            labels[b, pos + 1]    = kv[int(qk)]

    mask = input_ids == 0
    input_ids[mask] = np.random.randint(1, vocab_size, mask.sum())
    return torch.tensor(input_ids), torch.tensor(labels)


class HAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, state_size, num_levels,
                 lambda_mode="fixed", mlp_hidden_dim=64):
        super().__init__()
        self.num_heads  = num_heads
        self.state_size = state_size
        self.head_dim   = d_model // num_heads
        self.num_levels = num_levels

        self.q_proj   = nn.Linear(d_model, num_heads * state_size, bias=False)
        self.k_proj   = nn.Linear(d_model, num_heads * state_size, bias=False)
        self.v_proj   = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.dl_proj  = nn.Linear(d_model, num_heads * num_levels,  bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

        A = torch.arange(1, num_heads + 1, dtype=torch.float32)
        self.A_log   = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.dt_proj = nn.Linear(d_model, num_heads, bias=False)
        dt = torch.exp(torch.rand(num_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        self.L = nn.Parameter(torch.ones(num_heads, num_levels))
        self.lambda_mode  = lambda_mode
        self.lambda_fixed = (lambda_mode == "fixed")
        if lambda_mode == "mlp_softplus":
            self.lambda_module = LambdaMLPSoftplus(num_levels=num_levels, hidden_dim=mlp_hidden_dim)
            self.lambda_fixed = False
        elif lambda_mode == "mlp_softmax":
            self.lambda_module = LambdaMLPSoftmax(num_levels=num_levels, hidden_dim=mlp_hidden_dim)
            self.lambda_fixed = False
        else:
            self.lambda_module = None

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        Q  = rearrange(self.q_proj(x),  "b t (h d) -> b t h d", h=self.num_heads)
        K  = rearrange(self.k_proj(x),  "b t (h d) -> b t h d", h=self.num_heads)
        V  = rearrange(self.v_proj(x),  "b t (h d) -> b t h d", h=self.num_heads)
        dl = rearrange(self.dl_proj(x), "b t (h l) -> b t h l", h=self.num_heads)

        A  = -torch.exp(self.A_log.float())
        dt = self.dt_proj(x)
        dt = F.softplus(dt + self.dt_bias)
        g  = A[None, None, :] * dt

        if self.lambda_fixed:
            L = F.softplus(self.L[None, None, :, :] * dl)
        else:
            L = self.lambda_module(dl)

        Y = hattention_kernel(
            q=Q, k=K, v=V,
            b=None, g=g, l=L,
            scale=None,
            head_first=False,
            level_base=LEVEL_BASE,
            htype=HTYPE,
            hstruct=HSTRUCT,
        )

        Y = rearrange(Y, "b t h d -> b t (h d)")
        return self.norm(self.out_proj(Y) + x)


class MQARModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, state_size,
                 num_levels, n_layers=2, lambda_mode="fixed", mlp_hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            HAttentionLayer(d_model, num_heads, state_size, num_levels,
                            lambda_mode, mlp_hidden_dim)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        for layer in self.layers:
            nn.init.normal_(layer.out_proj.weight, std=0.02/math.sqrt(2 * 2))

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


def train(lambda_mode, model_dim, num_kv_pairs, seq_len, batch_size,
          max_steps, lr, mlp_hidden_dim, seed, device):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    vocab_size = 8192
    num_levels = get_num_levels(seq_len, base=LEVEL_BASE)
    n_train    = 10000
    n_val      = 1000

    print(f"Generating data (num_levels={num_levels})...")
    train_x, train_y = generate_mqar(n_train, seq_len, num_kv_pairs, vocab_size, seed=seed)
    val_x,   val_y   = generate_mqar(n_val,   seq_len, num_kv_pairs, vocab_size, seed=seed+1)
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x,   val_y   = val_x.to(device),   val_y.to(device)

    model = MQARModel(
        vocab_size=vocab_size,
        d_model=model_dim,
        num_heads=1,
        state_size=model_dim,
        num_levels=num_levels,
        n_layers=2,
        lambda_mode=lambda_mode,
        mlp_hidden_dim=mlp_hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}")
    print(f"Mode: {lambda_mode} | dim: {model_dim} | kv: {num_kv_pairs} | seed: {seed}")
    print(f"Params: {total_params:,} | num_levels: {num_levels}")
    print(f"{'='*60}\n")

    best_acc = 0.0
    for step in range(max_steps):
        model.train()
        idx    = torch.randperm(n_train, device=device)[:batch_size]
        x, y   = train_x[idx], train_y[idx]
        logits = model(x)
        sl     = logits[:, :-1].contiguous()
        sy     = y[:, 1:].contiguous()
        mask   = sy != -100
        if mask.sum() == 0:
            continue
        loss = F.cross_entropy(sl[mask], sy[mask])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                correct, total = 0, 0
                for i in range(0, n_val, batch_size):
                    xb, yb = val_x[i:i+batch_size], val_y[i:i+batch_size]
                    sl     = model(xb)[:, :-1]
                    sy     = yb[:, 1:]
                    mask   = sy != -100
                    preds  = sl.argmax(-1)
                    correct += (preds[mask] == sy[mask]).sum().item()
                    total   += mask.sum().item()
                acc = correct / total * 100

            print(f"Step {step+1:5d} | Loss: {loss.item():.4f} | Val Acc: {acc:.1f}%")
            if acc > best_acc:
                best_acc = acc
            if acc >= 99.0:
                print(f"Early stop at step {step+1}!")
                break

    print(f"\nBest Val Acc: {best_acc:.1f}%")
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_mode",    type=str,   default="fixed",
                        choices=["fixed", "mlp_softplus", "mlp_softmax"])
    parser.add_argument("--mlp_hidden_dim", type=int,   default=64)
    parser.add_argument("--model_dim",      type=int,   default=64)
    parser.add_argument("--num_kv_pairs",   type=int,   default=4)
    parser.add_argument("--seq_len",        type=int,   default=256)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--max_steps",      type=int,   default=20000)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--seed",           type=int,   default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train(
        lambda_mode=args.lambda_mode,
        model_dim=args.model_dim,
        num_kv_pairs=args.num_kv_pairs,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        mlp_hidden_dim=args.mlp_hidden_dim,
        seed=args.seed,
        device=device,
    )
