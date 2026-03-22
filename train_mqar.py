import sys
# Use the correct fla from flame submodule — MUST be first!
sys.path.insert(0, "/scratch/zt1/project/msml612/user/yaxita/log-linear-attention/flame/3rdparty/flash-linear-attention")
sys.path.insert(1, "/scratch/zt1/project/msml612/user/yaxita/log-linear-attention")

import torch
import numpy as np
import random
import argparse


def generate_mqar_batch(batch_size, seq_len, num_kv_pairs, vocab_size, device):
    key_vocab = vocab_size // 2
    value_vocab = vocab_size // 2
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    for b in range(batch_size):
        keys = random.sample(range(1, key_vocab), num_kv_pairs)
        values = [random.randint(1, value_vocab - 1) + key_vocab for _ in range(num_kv_pairs)]
        kv_dict = dict(zip(keys, values))
        pos = 0
        for k, v in zip(keys, values):
            if pos + 1 >= seq_len:
                break
            input_ids[b, pos] = k
            input_ids[b, pos + 1] = v
            pos += 2
        query_keys = random.sample(keys, min(num_kv_pairs, (seq_len - pos) // 2))
        for qk in query_keys:
            if pos + 1 >= seq_len:
                break
            input_ids[b, pos] = qk
            labels[b, pos + 1] = kv_dict[qk]
            input_ids[b, pos + 1] = kv_dict[qk]
            pos += 2
    return input_ids, labels


def train_mqar(lambda_mode, hidden_dim, num_heads, seed, num_kv_pairs,
               seq_len, batch_size, max_steps, lr, mlp_hidden_dim, device):

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # patch constants BEFORE importing model
    import importlib, sys
    for k in list(sys.modules.keys()):
        if "hattention" in k:
            del sys.modules[k]
    import hattention.modeling_hattention as mh
    mh.LAMBDA_MODE_TYPE = lambda_mode
    mh.LAMBDA_MLP_HIDDEN_DIM = mlp_hidden_dim

    from hattention.configuration_hattention import HAttentionConfig
    from hattention.modeling_hattention import HAttentionForCausalLM

    vocab_size = 128
    config = HAttentionConfig(
        hidden_size=64,
        num_heads=num_heads,
        num_hidden_layers=2,
        vocab_size=vocab_size,
        state_size=64,
        head_dim=64,
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
    print(f"\n{'='*55}")
    print(f"Mode: {lambda_mode} | dim: {hidden_dim} | seed: {seed} | params: {total_params:,}")
    print(f"{'='*55}")

    best_acc = 0.0
    for step in range(max_steps):
        model.train()
        input_ids, labels = generate_mqar_batch(
            batch_size, seq_len, num_kv_pairs, vocab_size, device)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                input_ids, labels = generate_mqar_batch(
                    batch_size * 2, seq_len, num_kv_pairs, vocab_size, device)
                outputs = model(input_ids=input_ids, labels=labels)
                logits = outputs.logits
                mask = labels != -100
                preds = logits.argmax(dim=-1)
                acc = (preds[mask] == labels[mask]).float().mean().item() * 100
            print(f"Step {step+1:5d} | Loss: {loss.item():.4f} | Acc: {acc:.1f}%")
            if acc > best_acc:
                best_acc = acc
            if acc >= 99.0:
                print(f"Early stop at step {step+1} with acc {acc:.1f}%")
                break

    print(f"Best accuracy: {best_acc:.1f}%")
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_mode", type=str, default="fixed",
                        choices=["fixed", "mlp_softplus", "mlp_softmax"])
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--mlp_hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_kv_pairs", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_mqar(
        lambda_mode=args.lambda_mode,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        seed=args.seed,
        num_kv_pairs=args.num_kv_pairs,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        mlp_hidden_dim=args.mlp_hidden_dim,
        device=device,
    )
