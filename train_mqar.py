import sys
sys.path.insert(0, "/scratch/zt1/project/msml612/user/yaxita/log-linear-attention/flame/3rdparty/flash-linear-attention")
sys.path.insert(1, "/scratch/zt1/project/msml612/user/yaxita/log-linear-attention")

import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse


def generate_mqar_dataset(n_samples, seq_len, num_kv_pairs, seed=42):
    np.random.seed(seed)
    vocab_size  = 8192
    half        = vocab_size // 2
    query_start = seq_len - num_kv_pairs * 2
    input_ids   = torch.zeros(n_samples, seq_len, dtype=torch.long)
    labels      = torch.full((n_samples, seq_len), -100, dtype=torch.long)

    key_choices   = np.arange(1, half)
    value_choices = np.arange(half, vocab_size)

    for b in range(n_samples):
        keys   = np.random.choice(key_choices,   size=num_kv_pairs, replace=False)
        values = np.random.choice(value_choices, size=num_kv_pairs, replace=False)
        kv_dict = dict(zip(keys.tolist(), values.tolist()))

        for i, (k, v) in enumerate(zip(keys, values)):
            input_ids[b, i*2]     = int(k)
            input_ids[b, i*2 + 1] = int(v)

        query_keys = np.random.choice(keys, size=num_kv_pairs, replace=False)
        for i, qk in enumerate(query_keys):
            pos = query_start + i * 2
            input_ids[b, pos] = int(qk)
            # answer position stays 0, gets randomized below
            labels[b, pos + 1] = kv_dict[int(qk)]

    # replace zeros with random tokens (Zoology style)
    mask = input_ids == 0
    input_ids[mask] = torch.randint(1, vocab_size, (mask.sum(),))
    return input_ids, labels


def train_mqar(lambda_mode, num_kv_pairs, seq_len, batch_size,
               max_steps, lr, mlp_hidden_dim, model_dim, seed, device):

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # set lambda mode before importing model
    import hattention.modeling_hattention as mh
    mh.LAMBDA_MODE_TYPE      = lambda_mode
    mh.LAMBDA_MLP_HIDDEN_DIM = mlp_hidden_dim

    from hattention.base import get_num_levels
    from mqar_model import MQARModel

    n_train = 10000
    n_val   = 1000
    print("Generating datasets...")
    train_x, train_y = generate_mqar_dataset(n_train, seq_len, num_kv_pairs, seed=seed)
    val_x,   val_y   = generate_mqar_dataset(n_val,   seq_len, num_kv_pairs, seed=seed+1)
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x,   val_y   = val_x.to(device),   val_y.to(device)
    print("Done!")

    num_levels = get_num_levels(seq_len, base=2)
    model = MQARModel(
        vocab_size=8192,
        d_model=64,
        num_heads=1,
        state_size=model_dim,
        num_levels=num_levels,
        n_layers=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Mode: {lambda_mode} | dim: {model_dim} | kv: {num_kv_pairs} | seed: {seed} | params: {total_params:,}")
    print(f"{'='*60}\n")

    best_acc = 0.0
    for step in range(max_steps):
        model.train()
        idx = torch.randperm(n_train, device=device)[:batch_size]
        x, y = train_x[idx], train_y[idx]

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
                    xb = val_x[i:i+batch_size]
                    yb = val_y[i:i+batch_size]
                    logits = model(xb)
                    sl     = logits[:, :-1].contiguous()
                    sy     = yb[:, 1:].contiguous()
                    mask   = sy != -100
                    preds  = sl.argmax(-1)
                    correct += (preds[mask] == sy[mask]).sum().item()
                    total   += mask.sum().item()
                acc = correct / total * 100

            print(f"Step {step+1:5d} | Loss: {loss.item():.4f} | Val Acc: {acc:.1f}%")
            if acc > best_acc:
                best_acc = acc
            if acc >= 99.0:
                print(f"Early stop at step {step+1} with acc {acc:.1f}%")
                break

    print(f"\nBest accuracy: {best_acc:.1f}%")
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_mode",    type=str,   default="fixed",
                        choices=["fixed", "mlp_softplus", "mlp_softmax"])
    parser.add_argument("--mlp_hidden_dim", type=int,   default=64)
    parser.add_argument("--model_dim",      type=int,   default=16,
                        help="state_size (16, 32, or 64)")
    parser.add_argument("--num_kv_pairs",   type=int,   default=4)
    parser.add_argument("--seq_len",        type=int,   default=256)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--max_steps",      type=int,   default=20000)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--seed",           type=int,   default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_mqar(
        lambda_mode=args.lambda_mode,
        num_kv_pairs=args.num_kv_pairs,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        mlp_hidden_dim=args.mlp_hidden_dim,
        model_dim=args.model_dim,
        seed=args.seed,
        device=device,
    )
