import torch
import argparse
import os
import json
import random
import numpy as np
from data_utils import get_wikitext103_dataloader

def log(log_path, msg):
    print(msg)
    with open(log_path, "a") as f:
        f.write(msg + "\n")
        
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader, train_size = get_wikitext103_dataloader("train", args.seq_len, args.batch_size, device)
    val_loader, val_size = get_wikitext103_dataloader("validation", args.seq_len, args.batch_size, device)
    print(f"Train size: {train_size}, Val size: {val_size}")

    import hattention.modeling_hattention as mh
    if args.mode == "mlp_softplus":
        mh.LAMBDA_MODE_TYPE = "mlp_softplus"
    elif args.mode == "mlp_softmax":
        mh.LAMBDA_MODE_TYPE = "mlp_softmax"
    elif args.mode == "fixed":
        mh.LAMBDA_MODE_TYPE = "fixed"
    mh.LAMBDA_MLP_HIDDEN_DIM = 64

    print(f"Loading model: {args.mode}")
    if args.mode in ["mlp_softplus", "mlp_softmax", "fixed"]:
        from hattention.modeling_hattention import HAttentionForCausalLM
        from hattention.configuration_hattention import HAttentionConfig
        config = HAttentionConfig(
            residual_in_fp32=False,
            fuse_norm=False,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_heads=args.num_heads,
            head_dim=64,
            expand=1,
            intermediate_size=args.hidden_size * 2,
            vocab_size=50257,
        )
        model = HAttentionForCausalLM(config)
    elif args.mode == "softmax":
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(
            n_embd=args.hidden_size,
            n_layer=args.num_layers,
            n_head=args.num_heads,
            vocab_size=50257,
        )
        model = GPT2LMHeadModel(config)

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    warmup_steps = 500
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (args.max_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = f"{args.output_dir}/train_log.txt"

    step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    train_iter = iter(train_loader)

    while step < args.max_steps:
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            ppl = torch.exp(loss).item()
            log(log_path, f"Step {step} | train loss: {loss.item():.4f} | ppl: {ppl:.2f}")

        if step % args.val_every == 0 and step > 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch in val_loader:
                    input_ids = val_batch["input_ids"].to(device)
                    labels = val_batch["labels"].to(device)
                    outputs = model(input_ids=input_ids, labels=labels)
                    val_losses.append(outputs.loss.item())
            val_loss = sum(val_losses) / len(val_losses)
            val_ppl = torch.exp(torch.tensor(val_loss)).item()
            log(log_path, f"Step {step} | val loss: {val_loss:.4f} | val ppl: {val_ppl:.2f}")
            

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"{args.output_dir}/best_model.pt")
                print(f"Saved best model at step {step}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at step {step}")
                    break

        step += 1

    best_ppl = torch.exp(torch.tensor(best_val_loss)).item()
    log(log_path, f"Done! Best val ppl: {best_ppl:.2f}")


    results = {
        "mode": args.mode,
        "hidden_size": args.hidden_size,
        "seq_len": args.seq_len,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "best_val_loss": best_val_loss,
        "best_val_ppl": best_ppl,
    }
    with open(f"{args.output_dir}/result.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_dir}/result.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["mlp_softplus", "mlp_softmax", "fixed", "softmax"])
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/lm")
    args = parser.parse_args()
    train(args)
