import sys
sys.path.insert(0, "/scratch/zt1/project/msml612/user/yaxita/log-linear-attention/flame/3rdparty/flash-linear-attention")
sys.path.insert(1, "/scratch/zt1/project/msml612/user/yaxita/log-linear-attention")

import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse


def generate_mqar_dataset(n_samples, seq_len, num_kv_pairs, seed=42):
    """
    MQAR: vocab_size=64, random key-value pairs.
    Keys: 1..31, Values: 32..63
    Format: [k1 v1 ... kn vn] [PAD...] [q1 a1 ... qn an]
    """
    random.seed(seed)
    torch.manual_seed(seed)
    vocab_size  = max(64, num_kv_pairs * 4 + 2)
    half        = vocab_size // 2
    query_start = seq_len - num_kv_pairs * 2
    input_ids   = torch.zeros(n_samples, seq_len, dtype=torch.long)
    labels      = torch.full((n_samples, seq_len), -100, dtype=torch.long)

    for b in range(n_samples):
        keys    = random.sample(range(1, half), num_kv_pairs)
        values  = random.sample(range(half, vocab_size), num_kv_pairs)
        kv_dict = dict(zip(keys, values))
        for i, (k, v) in enumerate(zip(keys, values)):
            input_ids[b, i*2]     = k
            input_ids[b, i*2 + 1] = v
        query_keys = random.sample(keys, num_kv_pairs)
        for i, qk in enumerate(query_keys):
            pos = query_start + i * 2
            input_ids[b, pos]     = qk
            input_ids[b, pos + 1] = kv_dict[qk]
            labels[b, pos + 1]    = kv_dict[qk]

    return input_ids, labels


def train_mqar(lambda_mode, num_kv_pairs, seq_len, batch_size,
               max_steps, lr, mlp_hidden_dim, model_dim, seed, device):

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    for k in list(__import__('sys').modules.keys()):
        if "hattention" in k:
            del __import__('sys').modules[k]

    import hattention.modeling_hattention as mh
    mh.LAMBDA_MODE_TYPE      = lambda_mode
    mh.LAMBDA_MLP_HIDDEN_DIM = mlp_hidden_dim

    from hattention.configuration_hattention import HAttentionConfig
    from hattention.modeling_hattention import HAttentionForCausalLM

    n_train = 10000
    n_val   = 1000
    train_x, train_y = generate_mqar_dataset(n_train, seq_len, num_kv_pairs, seed=seed)
    val_x,   val_y   = generate_mqar_dataset(n_val,   seq_len, num_kv_pairs, seed=seed+1)
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x,   val_y   = val_x.to(device),   val_y.to(device)

    # paper's exact config: state_size=16, head_dim=model_dim
    config = HAttentionConfig(
        hidden_size=64,
        num_heads=1,
        num_hidden_layers=2,
        vocab_size=max(64, num_kv_pairs * 4 + 2),
        state_size=model_dim,
        head_dim=model_dim,
        expand=1,
        chunk_size=64,
        fuse_cross_entropy=False,
        fuse_norm=False,
        residual_in_fp32=False,
    )

    model = HAttentionForCausalLM(config).to(device)
    model.backbone.gradient_checkpointing = False
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

        out  = model(input_ids=x)
        sl   = out.logits[:, :-1].contiguous()
        sy   = y[:, 1:].contiguous()
        mask = sy != -100
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
                    out  = model(input_ids=xb)
                    sl   = out.logits[:, :-1].contiguous()
                    sy   = yb[:, 1:].contiguous()
                    mask = sy != -100
                    preds = sl.argmax(-1)
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
                        help="state_size and head_dim (16, 32, or 64)")
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
