"""
train_selective_copy.py  —  Selective Copying benchmark
for Adaptive Memory Decay for Log-Linear Attention

Task: given a sequence of tokens where some are "marked" with a special flag,
the model must copy only the marked tokens to the output (ignoring noise tokens).
This is a standard long-range memory stress test used in Mamba, Hyena, S4 papers.

Sequence structure:
  [marked_token, noise, noise, ..., noise, marked_token, noise, ..., SEPARATOR, ?, ?, ...]
  - num_tokens marked tokens scattered randomly in a noise sequence
  - model must output them in order at the end

Usage:
    python train_selective_copy.py --lambda_mode mlp_softplus --seq_len 256
    python train_selective_copy.py --lambda_mode fixed --seq_len 1024
"""

import sys
sys.path.insert(0, "/mnt/e/log-linear-attention-lamda-MLP/flame/3rdparty/flash-linear-attention")
sys.path.insert(1, "/mnt/e/log-linear-attention-lamda-MLP")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
import os
import json
from einops import rearrange

from hattention.base import HType, HStruct, get_num_levels
from hattention.kernel import hattention_kernel
from hattention.lambda_mlp import LambdaMLPSoftplus, LambdaMLPSoftmax

LEVEL_BASE = 2
HTYPE      = HType.WEAK
HSTRUCT    = HStruct.MAMBA2

# ── task config ───────────────────────────────────────────────────────────────
# vocab layout:
#   0          = padding (unused)
#   1          = separator token
#   2          = blank/query placeholder
#   3..N_DATA  = data tokens (the ones to copy)
#   rest       = noise tokens

SEPARATOR   = 1
BLANK       = 2
DATA_START  = 3
DATA_END    = 66    # 63 data tokens
NOISE_START = 67
VOCAB_SIZE  = 128


# ── data generation ───────────────────────────────────────────────────────────

def generate_selective_copy(n_samples, seq_len, num_tokens=16, seed=42):
    """
    Generate selective copying task.

    Sequence layout:
      positions 0..seq_len-num_tokens-2 : input phase (noise + marked tokens)
      position  seq_len-num_tokens-1    : SEPARATOR
      positions seq_len-num_tokens..end : output phase (blanks, labels = marked tokens in order)

    The model sees the full sequence and must predict the marked tokens
    at the output positions.
    """
    np.random.seed(seed)

    input_len  = seq_len - num_tokens - 1   # input phase length
    total      = seq_len

    input_ids = np.zeros((n_samples, total), dtype=np.int64)
    labels    = np.full((n_samples, total), -100, dtype=np.int64)

    data_tokens  = np.arange(DATA_START, DATA_END)
    noise_tokens = np.arange(NOISE_START, VOCAB_SIZE)

    for b in range(n_samples):
        # pick num_tokens distinct data tokens to remember
        chosen = np.random.choice(data_tokens, size=num_tokens, replace=False)

        # scatter them randomly in the input phase
        positions = np.sort(np.random.choice(input_len, size=num_tokens, replace=False))

        # fill input phase with noise
        seq = np.random.choice(noise_tokens, size=input_len)

        # place marked tokens at chosen positions
        for i, pos in enumerate(positions):
            seq[pos] = chosen[i]

        input_ids[b, :input_len] = seq

        # separator
        input_ids[b, input_len] = SEPARATOR

        # output phase: blanks as input, labels = chosen tokens in order
        for i in range(num_tokens):
            out_pos = input_len + 1 + i
            input_ids[b, out_pos] = BLANK
            labels[b, out_pos]    = int(chosen[i])

    return torch.tensor(input_ids), torch.tensor(labels)


# ── model (same HAttentionLayer as MQAR) ──────────────────────────────────────

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
            self.lambda_fixed  = False
        elif lambda_mode == "mlp_softmax":
            self.lambda_module = LambdaMLPSoftmax(num_levels=num_levels, hidden_dim=mlp_hidden_dim)
            self.lambda_fixed  = False
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


class SelectiveCopyModel(nn.Module):
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
            nn.init.normal_(layer.out_proj.weight, std=0.02 / math.sqrt(2 * 2))

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


# ── training ──────────────────────────────────────────────────────────────────

def train(lambda_mode, model_dim, num_tokens, seq_len, batch_size,
          max_steps, lr, mlp_hidden_dim, seed, device, output_dir="results/selective_copy"):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run_name = f"{lambda_mode}_tok{num_tokens}_seq{seq_len}_seed{seed}"
    run_dir  = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    num_levels = get_num_levels(seq_len, base=LEVEL_BASE)
    n_train    = 10000
    n_val      = 1000

    print(f"Generating data (num_levels={num_levels})...")
    train_x, train_y = generate_selective_copy(n_train, seq_len, num_tokens, seed=seed)
    val_x,   val_y   = generate_selective_copy(n_val,   seq_len, num_tokens, seed=seed+1)
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x,   val_y   = val_x.to(device),   val_y.to(device)

    model = SelectiveCopyModel(
        vocab_size=VOCAB_SIZE,
        d_model=model_dim,
        num_heads=1,
        state_size=model_dim,
        num_levels=num_levels,
        n_layers=2,
        lambda_mode=lambda_mode,
        mlp_hidden_dim=mlp_hidden_dim,
    ).to(device)

    optimizer    = torch.optim.Adam(model.parameters(), lr=lr)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}")
    print(f"Mode: {lambda_mode} | dim: {model_dim} | tokens: {num_tokens} | seq: {seq_len} | seed: {seed}")
    print(f"Params: {total_params:,} | num_levels: {num_levels}")
    print(f"Output: {run_dir}")
    print(f"{'='*60}\n")

    best_acc    = 0.0
    best_step   = 0
    log_history = []

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
                acc = correct / total * 100 if total > 0 else 0.0

            print(f"Step {step+1:5d} | Loss: {loss.item():.4f} | Val Acc: {acc:.1f}%")
            log_history.append({"step": step+1, "loss": round(loss.item(), 4), "val_acc": round(acc, 2)})

            if acc > best_acc:
                best_acc  = acc
                best_step = step + 1
                torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
                print(f"  -> Saved best model (acc={best_acc:.1f}%)")

            if acc >= 99.0:
                print(f"Early stop at step {step+1}!")
                break

    print(f"\nBest Val Acc: {best_acc:.1f}% at step {best_step}")

    results = {
        "lambda_mode":    lambda_mode,
        "model_dim":      model_dim,
        "num_tokens":     num_tokens,
        "seq_len":        seq_len,
        "seed":           seed,
        "max_steps":      max_steps,
        "lr":             lr,
        "mlp_hidden_dim": mlp_hidden_dim,
        "total_params":   total_params,
        "best_val_acc":   round(best_acc, 2),
        "best_step":      best_step,
        "log_history":    log_history,
    }
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {run_dir}/result.json")
    return best_acc


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_mode",    type=str, default="mlp_softplus",
                        choices=["fixed", "mlp_softplus", "mlp_softmax"])
    parser.add_argument("--mlp_hidden_dim", type=int, default=64)
    parser.add_argument("--model_dim",      type=int, default=64)
    parser.add_argument("--num_tokens",     type=int, default=16,
                        help="Number of tokens to selectively copy")
    parser.add_argument("--seq_len",        type=int, default=256)
    parser.add_argument("--batch_size",     type=int, default=64)
    parser.add_argument("--max_steps",      type=int, default=10000)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--seed",           type=int, default=0)
    parser.add_argument("--output_dir",     type=str, default="results/selective_copy")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train(
        lambda_mode=args.lambda_mode,
        model_dim=args.model_dim,
        num_tokens=args.num_tokens,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        mlp_hidden_dim=args.mlp_hidden_dim,
        seed=args.seed,
        device=device,
        output_dir=args.output_dir,
    )
