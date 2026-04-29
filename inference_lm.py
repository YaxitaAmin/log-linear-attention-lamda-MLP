"""
evaluate.py - Load checkpoints and compute perplexity on WikiText-103 validation set
Usage:
    python evaluate.py --checkpoint_dir results/lm/mlp_softplus_dim512_seed0 --mode mlp_softplus --hidden_size 512
    python evaluate.py --checkpoint_dir results/lm/fixed_dim512_seed0       --mode fixed       --hidden_size 512

    # Compare all at once:
    python evaluate.py --compare
"""

import torch
import argparse
import os
import json
import numpy as np
from tqdm import tqdm


# ─── model loader ────────────────────────────────────────────────────────────

def load_model(mode: str, hidden_size: int, num_layers: int, num_heads: int, device):
    import hattention.modeling_hattention as mh

    if mode in ("mlp_softplus", "mlp_softmax", "fixed"):
        if mode == "mlp_softplus":
            mh.LAMBDA_MODE_TYPE = "mlp_softplus"
        elif mode == "mlp_softmax":
            mh.LAMBDA_MODE_TYPE = "mlp_softmax"
        elif mode == "fixed":
            mh.LAMBDA_MODE_TYPE = "fixed"
        mh.LAMBDA_MLP_HIDDEN_DIM = 64

        from hattention.modeling_hattention import HAttentionForCausalLM
        from hattention.configuration_hattention import HAttentionConfig

        config = HAttentionConfig(
            residual_in_fp32=False,
            fuse_norm=False,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_heads=num_heads,
            head_dim=64,
            expand=1,
            intermediate_size=hidden_size * 2,
            vocab_size=50257,
        )
        model = HAttentionForCausalLM(config)

    elif mode == "softmax":
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_heads,
            vocab_size=50257,
        )
        model = GPT2LMHeadModel(config)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return model.to(device)


# ─── evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_dir: str,
    mode: str,
    hidden_size: int = 256,
    num_layers: int = 6,
    num_heads: int = 4,
    seq_len: int = 512,
    batch_size: int = 8,
    device_str: str = "auto",
    split: str = "validation",
    ckpt_name: str = "best_model.pt",
) -> dict:
    """Load best_model.pt from checkpoint_dir and return loss/ppl on `split`."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "auto"
        else device_str
    )

    # ── load stored result metadata if present ──────────────────────────────
    result_path = os.path.join(checkpoint_dir, "result.json")
    stored = {}
    if os.path.exists(result_path):
        with open(result_path) as f:
            stored = json.load(f)
        print(f"  [stored] best_val_ppl = {stored.get('best_val_ppl', '?'):.2f}")

    # ── load model ──────────────────────────────────────────────────────────
    # auto-detect checkpoint filename
    for candidate in [ckpt_name, "best_model.pt", "model.pt", "checkpoint.pt"]:
        ckpt_path = os.path.join(checkpoint_dir, candidate)
        if os.path.exists(ckpt_path):
            break
    else:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir}. "
            f"Files present: {os.listdir(checkpoint_dir)}"
        )

    print(f"  Loading model ({mode}, hidden={hidden_size}) …")
    model = load_model(mode, hidden_size, num_layers, num_heads, device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # ── load data ───────────────────────────────────────────────────────────
    from data_utils import get_wikitext103_dataloader

    loader, data_size = get_wikitext103_dataloader(split, seq_len, batch_size, device)
    print(f"  Evaluating on {split} ({data_size} tokens) …")

    # ── inference loop ──────────────────────────────────────────────────────
    losses = []
    for batch in tqdm(loader, desc="  batches", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        outputs   = model(input_ids=input_ids, labels=labels)
        losses.append(outputs.loss.item())

    loss = float(np.mean(losses))
    ppl  = float(np.exp(loss))

    return {
        "checkpoint_dir": checkpoint_dir,
        "mode":           mode,
        "hidden_size":    hidden_size,
        "split":          split,
        "loss":           loss,
        "ppl":            ppl,
        "stored_ppl":     stored.get("best_val_ppl"),
        "num_params":     num_params,
    }


# ─── compare helper ──────────────────────────────────────────────────────────

# Edit this list to match YOUR actual folder names / paths
COMPARE_RUNS = [
    # (checkpoint_dir,                            mode,           hidden_size)
    ("results/lm/fixed_dim512_seed0",             "fixed",        512),
    ("results/lm/mlp_softmax_dim512_seed0",       "mlp_softmax",  512),
    ("results/lm/mlp_softplus_dim512_seed0",      "mlp_softplus", 512),
    ("results/lm/fixed_256_s1",                   "fixed",        256),
    ("results/lm/fixed_256_s2",                   "fixed",        256),
    ("results/lm/mlp_softplus_256_s1",            "mlp_softplus", 256),
    ("results/lm/mlp_softplus_256_s2",            "mlp_softplus", 256),
    ("results/lm/mlp_softmax_256_s1",             "mlp_softmax",  256),
    ("results/lm/mlp_softmax_256_s2",             "mlp_softmax",  256),
]


def run_compare(seq_len: int, batch_size: int):
    rows = []
    for ckpt_dir, mode, hidden in COMPARE_RUNS:
        if not os.path.isdir(ckpt_dir):
            print(f"\n[SKIP] {ckpt_dir} not found")
            continue
        print(f"\n{'─'*60}")
        print(f"  {ckpt_dir}  ({mode}  h={hidden})")
        try:
            r = evaluate_checkpoint(
                checkpoint_dir=ckpt_dir,
                mode=mode,
                hidden_size=hidden,
                seq_len=seq_len,
                batch_size=batch_size,
            )
            rows.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── pretty table ────────────────────────────────────────────────────────
    if not rows:
        print("\nNo results to display.")
        return

    print(f"\n{'='*72}")
    print(f"  {'DIR':<40} {'MODE':<14} {'H':>4}  {'PPL':>8}  {'STORED':>8}")
    print(f"{'─'*72}")
    for r in sorted(rows, key=lambda x: x["ppl"]):
        stored_str = f"{r['stored_ppl']:.2f}" if r["stored_ppl"] else "  n/a "
        marker = " ◀ BEST" if r == min(rows, key=lambda x: x["ppl"]) else ""
        print(
            f"  {os.path.basename(r['checkpoint_dir']):<40} "
            f"{r['mode']:<14} {r['hidden_size']:>4}  "
            f"{r['ppl']:>8.2f}  {stored_str:>8}{marker}"
        )
    print(f"{'='*72}\n")

    # ── save results ────────────────────────────────────────────────────────
    out_path = "eval_comparison.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Full results saved to {out_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate language model checkpoints on WikiText-103"
    )

    # single-run args
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Path to folder containing best_model.pt")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["mlp_softplus", "mlp_softmax", "fixed", "softmax"],
                        help="Model variant used during training")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers",  type=int, default=6)
    parser.add_argument("--num_heads",   type=int, default=4)
    parser.add_argument("--split", type=str, default="validation",
                        choices=["validation", "test"],
                        help="Dataset split to evaluate on")

    # shared args
    parser.add_argument("--seq_len",    type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device",     type=str, default="auto")

    parser.add_argument("--ckpt_name", type=str, default="best_model.pt",
                        help="Checkpoint filename inside checkpoint_dir (default: best_model.pt, auto-detects model.pt too)")
    parser.add_argument("--compare", action="store_true",
                        help="Run all runs in COMPARE_RUNS table and print summary")

    args = parser.parse_args()

    if args.compare:
        run_compare(args.seq_len, args.batch_size)
        return

    if args.checkpoint_dir is None or args.mode is None:
        parser.error("--checkpoint_dir and --mode are required unless --compare is set")

    print(f"\n{'='*60}")
    print(f"  Checkpoint : {args.checkpoint_dir}")
    print(f"  Mode       : {args.mode}")
    print(f"  Hidden     : {args.hidden_size}")
    print(f"  Split      : {args.split}")
    print(f"{'='*60}\n")

    result = evaluate_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        mode=args.mode,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device_str=args.device,
        split=args.split,
        ckpt_name=args.ckpt_name,
    )

    print(f"\n{'='*60}")
    print(f"  ✓ Loss : {result['loss']:.4f}")
    print(f"  ✓ PPL  : {result['ppl']:.2f}")
    if result["stored_ppl"]:
        delta = result["ppl"] - result["stored_ppl"]
        sign  = "+" if delta > 0 else ""
        print(f"  Δ vs stored : {sign}{delta:.2f}  (stored={result['stored_ppl']:.2f})")
    print(f"{'='*60}\n")

    # save individual result
    out = os.path.join(args.checkpoint_dir, "eval_result.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()