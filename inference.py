import sys
sys.path.insert(0, "/mnt/e/log-linear-attention-lamda-MLP/flame/3rdparty/flash-linear-attention")
sys.path.insert(1, "/mnt/e/log-linear-attention-lamda-MLP")

import math
import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from hattention.base import HType, HStruct, get_num_levels
from hattention.kernel import hattention_kernel
from hattention.lambda_mlp import LambdaMLPSoftplus, LambdaMLPSoftmax

LEVEL_BASE = 2
HTYPE      = HType.WEAK
HSTRUCT    = HStruct.MAMBA2

SCP_RESULTS_DIR  = "/mnt/e/log-linear-attention-lamda-MLP/results/selective_copy"
MQAR_RESULTS_DIR = "/mnt/e/log-linear-attention-lamda-MLP/results/mqar"


# ── shared attention layer ────────────────────────────────────────────────────

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
            q=Q, k=K, v=V, b=None, g=g, l=L,
            scale=None, head_first=False,
            level_base=LEVEL_BASE, htype=HTYPE, hstruct=HSTRUCT,
        )
        Y = rearrange(Y, "b t h d -> b t (h d)")
        return self.norm(self.out_proj(Y) + x)


# ── models ────────────────────────────────────────────────────────────────────

class SelectiveCopyModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, state_size,
                 num_levels, n_layers=2, lambda_mode="fixed", mlp_hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers    = nn.ModuleList([
            HAttentionLayer(d_model, num_heads, state_size, num_levels, lambda_mode, mlp_hidden_dim)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


class MQARModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, state_size,
                 num_levels, n_layers=2, lambda_mode="fixed", mlp_hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers    = nn.ModuleList([
            HAttentionLayer(d_model, num_heads, state_size, num_levels, lambda_mode, mlp_hidden_dim)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


# ── data generators ───────────────────────────────────────────────────────────

def generate_selective_copy(n_samples, seq_len, num_tokens, vocab_size=128, seed=42):
    np.random.seed(seed)
    SEPARATOR, BLANK, DATA_START, DATA_END = 1, 2, 3, 66
    NOISE_START = 67

    input_len    = seq_len - num_tokens - 1
    input_ids    = np.zeros((n_samples, seq_len), dtype=np.int64)
    labels       = np.full((n_samples, seq_len), -100, dtype=np.int64)
    data_tokens  = np.arange(DATA_START, min(DATA_END, vocab_size))
    noise_tokens = np.arange(NOISE_START, vocab_size)

    for b in range(n_samples):
        chosen    = np.random.choice(data_tokens, size=num_tokens, replace=False)
        positions = np.sort(np.random.choice(input_len, size=num_tokens, replace=False))
        seq       = np.random.choice(noise_tokens, size=input_len)
        for i, pos in enumerate(positions):
            seq[pos] = chosen[i]
        input_ids[b, :input_len] = seq
        input_ids[b, input_len]  = SEPARATOR
        for i in range(num_tokens):
            out_pos = input_len + 1 + i
            input_ids[b, out_pos] = BLANK
            labels[b, out_pos]    = int(chosen[i])

    return torch.tensor(input_ids), torch.tensor(labels)


def generate_mqar(n_samples, seq_len, num_kv_pairs, vocab_size=128, seed=42):
    np.random.seed(seed)
    half          = vocab_size // 2
    query_start   = seq_len - num_kv_pairs * 2
    input_ids     = np.zeros((n_samples, seq_len), dtype=np.int64)
    labels        = np.full((n_samples, seq_len), -100, dtype=np.int64)
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


# ── loader ────────────────────────────────────────────────────────────────────

def load_model(ckpt_path, task, lambda_mode, train_seq_len, eval_seq_len, mlp_hidden_dim, device):
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    vocab_size = state_dict["embedding.weight"].shape[0]
    d_model    = state_dict["embedding.weight"].shape[1]
    # FIXED: use train_seq_len to match checkpoint's num_levels
    num_levels = get_num_levels(train_seq_len, base=LEVEL_BASE)

    cls = SelectiveCopyModel if task == "scp" else MQARModel
    model = cls(
        vocab_size=vocab_size, d_model=d_model,
        num_heads=1, state_size=d_model,
        num_levels=num_levels, n_layers=2,
        lambda_mode=lambda_mode, mlp_hidden_dim=mlp_hidden_dim,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, vocab_size


def get_data(task, n_samples, eval_seq_len, num_tokens, num_kv_pairs, vocab_size, data_seed):
    if task == "scp":
        return generate_selective_copy(n_samples, eval_seq_len, num_tokens,
                                       vocab_size=vocab_size, seed=data_seed)
    else:
        return generate_mqar(n_samples, eval_seq_len, num_kv_pairs,
                             vocab_size=vocab_size, seed=data_seed)


# ── evaluate ──────────────────────────────────────────────────────────────────

def evaluate(model, x, y, batch_size, device):
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb    = x[i:i+batch_size].to(device)
            yb    = y[i:i+batch_size].to(device)
            sl    = model(xb)[:, :-1]
            sy    = yb[:, 1:]
            mask  = sy != -100
            preds = sl.argmax(-1)
            correct += (preds[mask] == sy[mask]).sum().item()
            total   += mask.sum().item()
    return correct / total * 100 if total > 0 else 0.0


# ── display ───────────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'='*58}")
    print(f"  {title}")
    print(f"{'='*58}")

def print_bar(acc):
    bar = "█" * int(acc / 2) + "░" * (50 - int(acc / 2))
    print(f"  [{bar}] {acc:.2f}%")


# ── subcommands ───────────────────────────────────────────────────────────────

def cmd_eval(args, device):
    """Single checkpoint eval, with optional generalization to a different seq_len."""
    results_dir = SCP_RESULTS_DIR if args.task == "scp" else MQAR_RESULTS_DIR
    eval_seq    = args.eval_seq_len or args.train_seq_len

    if args.task == "scp":
        run_name = f"{args.lambda_mode}_tok{args.num_tokens}_seq{args.train_seq_len}_seed{args.seed}"
    else:
        run_name = f"{args.lambda_mode}_kv{args.num_kv_pairs}_seq{args.train_seq_len}_seed{args.seed}"

    ckpt = os.path.join(results_dir, run_name, "best_model.pt")
    if not os.path.exists(ckpt):
        print(f"\n  [error] checkpoint not found:\n  {ckpt}")
        return

    print_section(
        f"eval | task={args.task} | mode={args.lambda_mode} | "
        f"train_seq={args.train_seq_len} | eval_seq={eval_seq} | seed={args.seed}"
    )
    if eval_seq != args.train_seq_len:
        print(f"  [generalization test] trained on seq={args.train_seq_len}, testing on seq={eval_seq}")

    model, vocab_size = load_model(ckpt, args.task, args.lambda_mode,
                                   args.train_seq_len, eval_seq, args.mlp_hidden_dim, device)
    print(f"  vocab={vocab_size} | levels={get_num_levels(args.train_seq_len, LEVEL_BASE)} | samples={args.n_samples}")

    x, y = get_data(args.task, args.n_samples, eval_seq,
                    args.num_tokens, args.num_kv_pairs, vocab_size, args.data_seed)
    acc  = evaluate(model, x, y, args.batch_size, device)

    # show saved training acc if available
    rjson = os.path.join(results_dir, run_name, "result.json")
    if os.path.exists(rjson):
        with open(rjson) as f:
            saved = json.load(f).get("best_val_acc")
        print(f"\n  best training acc : {saved:.2f}%")

    print(f"  eval acc          : {acc:.2f}%")
    print_bar(acc)


def cmd_compare(args, device):
    """Compare all 3 lambda modes side by side for a given config."""
    results_dir = SCP_RESULTS_DIR if args.task == "scp" else MQAR_RESULTS_DIR
    eval_seq    = args.eval_seq_len or args.train_seq_len
    modes       = ["fixed", "mlp_softplus", "mlp_softmax"]

    print_section(
        f"compare | task={args.task} | train_seq={args.train_seq_len} | "
        f"eval_seq={eval_seq} | seed={args.seed}"
    )

    rows = []
    for mode in modes:
        if args.task == "scp":
            run_name = f"{mode}_tok{args.num_tokens}_seq{args.train_seq_len}_seed{args.seed}"
        else:
            run_name = f"{mode}_kv{args.num_kv_pairs}_seq{args.train_seq_len}_seed{args.seed}"

        ckpt = os.path.join(results_dir, run_name, "best_model.pt")
        if not os.path.exists(ckpt):
            print(f"  [skip] {mode} — not found")
            rows.append((mode, None))
            continue

        model, vocab_size = load_model(ckpt, args.task, mode,
                                       args.train_seq_len, eval_seq, args.mlp_hidden_dim, device)
        x, y = get_data(args.task, args.n_samples, eval_seq,
                        args.num_tokens, args.num_kv_pairs, vocab_size, args.data_seed)
        acc  = evaluate(model, x, y, args.batch_size, device)
        rows.append((mode, acc))

    print(f"\n  {'rank':<6} {'mode':<18} {'acc':>8}")
    print(f"  {'-'*34}")
    ranked = sorted([(m, a) for m, a in rows if a is not None],
                    key=lambda r: r[1], reverse=True)
    for rank, (mode, acc) in enumerate(ranked, 1):
        bar = "█" * int(acc / 4)
        print(f"  {rank:<6} {mode:<18} {acc:>7.2f}%  {bar}")


def cmd_sweep_eval(args, device):
    """Grid eval over multiple train_seq_lens x seeds — great for generalization tables."""
    results_dir = SCP_RESULTS_DIR if args.task == "scp" else MQAR_RESULTS_DIR
    seq_lens    = [int(s) for s in args.seq_lens.split(",")]
    seeds       = [int(s) for s in args.seeds.split(",")]
    eval_seq    = args.eval_seq_len

    print_section(
        f"sweep_eval | task={args.task} | mode={args.lambda_mode} | eval_seq={eval_seq}"
    )
    print(f"  train_seq_lens : {seq_lens}")
    print(f"  seeds          : {seeds}\n")
    print(f"  {'train_seq':<12} {'seed':<8} {'acc':>8}")
    print(f"  {'-'*30}")

    for seq in seq_lens:
        accs = []
        for seed in seeds:
            if args.task == "scp":
                run_name = f"{args.lambda_mode}_tok{args.num_tokens}_seq{seq}_seed{seed}"
            else:
                run_name = f"{args.lambda_mode}_kv{args.num_kv_pairs}_seq{seq}_seed{seed}"

            ckpt = os.path.join(results_dir, run_name, "best_model.pt")
            if not os.path.exists(ckpt):
                print(f"  {seq:<12} {seed:<8} {'missing':>8}")
                continue

            model, vocab_size = load_model(ckpt, args.task, args.lambda_mode,
                                           seq, eval_seq, args.mlp_hidden_dim, device)
            x, y = get_data(args.task, args.n_samples, eval_seq,
                            args.num_tokens, args.num_kv_pairs, vocab_size, args.data_seed)
            acc  = evaluate(model, x, y, args.batch_size, device)
            accs.append(acc)
            print(f"  {seq:<12} {seed:<8} {acc:>7.2f}%")

        if accs:
            print(f"  {seq:<12} {'mean':<8} {sum(accs)/len(accs):>7.2f}%")
        print()


# ── cli ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inference CLI — selective copy (scp) and mqar",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("command", choices=["eval", "compare", "sweep_eval"],
                        help=(
                            "eval        — single checkpoint eval\n"
                            "compare     — all 3 modes side by side\n"
                            "sweep_eval  — grid over train_seq_lens x seeds"
                        ))
    parser.add_argument("task", choices=["scp", "mqar"])

    # model args
    parser.add_argument("--lambda_mode",    type=str, default="fixed",
                        choices=["fixed", "mlp_softplus", "mlp_softmax"])
    parser.add_argument("--mlp_hidden_dim", type=int, default=64)

    # seq args
    parser.add_argument("--train_seq_len",  type=int, default=256,
                        help="seq_len the checkpoint was trained on")
    parser.add_argument("--eval_seq_len",   type=int, default=None,
                        help="seq_len to eval on (default: same as train_seq_len)")

    # checkpoint / data
    parser.add_argument("--seed",           type=int, default=0)
    parser.add_argument("--data_seed",      type=int, default=777)
    parser.add_argument("--n_samples",      type=int, default=1000)
    parser.add_argument("--batch_size",     type=int, default=64)

    # task-specific
    parser.add_argument("--num_tokens",     type=int, default=16,   help="[scp] tokens to copy")
    parser.add_argument("--num_kv_pairs",   type=int, default=4,    help="[mqar] kv pairs")

    # sweep_eval specific
    parser.add_argument("--seq_lens",       type=str, default="256,512",
                        help="[sweep_eval] comma-separated train seq_lens")
    parser.add_argument("--seeds",          type=str, default="0,1,2",
                        help="[sweep_eval] comma-separated seeds")

    args   = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    if args.command == "eval":
        cmd_eval(args, device)
    elif args.command == "compare":
        cmd_compare(args, device)
    elif args.command == "sweep_eval":
        cmd_sweep_eval(args, device)

    print("\ndone!\n")


if __name__ == "__main__":
    main()