# import torch
# import argparse
# import os
# import json
# import random
# import numpy as np
# from data_utils import get_wikitext103_dataloader

# def set_seed(seed):
#     torch.manual_seed(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.cuda.manual_seed_all(seed)

# def train(args):
#     set_seed(args.seed)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     print("Loading data...")
#     train_loader, train_size = get_wikitext103_dataloader("train", args.seq_len, args.batch_size, device)
#     val_loader, val_size = get_wikitext103_dataloader("validation", args.seq_len, args.batch_size, device)
#     print(f"Train size: {train_size}, Val size: {val_size}")

#     import hattention.modeling_hattention as mh
#     if args.mode == "mlp_softplus":
#         mh.LAMBDA_MODE_TYPE = "mlp_softplus"
#     elif args.mode == "mlp_softmax":
#         mh.LAMBDA_MODE_TYPE = "mlp_softmax"
#     elif args.mode == "fixed":
#         mh.LAMBDA_MODE_TYPE = "fixed"
#     mh.LAMBDA_MLP_HIDDEN_DIM = 64

#     print(f"Loading model: {args.mode}")
#     if args.mode in ["mlp_softplus", "mlp_softmax", "fixed"]:
#         from hattention.modeling_hattention import HAttentionForCausalLM
#         from hattention.configuration_hattention import HAttentionConfig
#         config = HAttentionConfig(
#             residual_in_fp32=False,
#             fuse_norm=False,
#             hidden_size=args.hidden_size,
#             num_hidden_layers=args.num_layers,
#             num_heads=args.num_heads,
#             head_dim=64,
#             expand=1,
#             intermediate_size=args.hidden_size * 2,
#             vocab_size=50257,
#         )
#         model = HAttentionForCausalLM(config)
#     elif args.mode == "softmax":
#         from transformers import GPT2Config, GPT2LMHeadModel
#         config = GPT2Config(
#             n_embd=args.hidden_size,
#             n_layer=args.num_layers,
#             n_head=args.num_heads,
#             vocab_size=50257,
#         )
#         model = GPT2LMHeadModel(config)

#     model = model.to(device)
#     num_params = sum(p.numel() for p in model.parameters())
#     print(f"Model params: {num_params:,}")

#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
#     warmup_steps = 500
#     def lr_lambda(step):
#         if step < warmup_steps:
#             return step / warmup_steps
#         progress = (step - warmup_steps) / (args.max_steps - warmup_steps)
#         return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item())
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

#     os.makedirs(args.output_dir, exist_ok=True)

#     step = 0
#     best_val_loss = float("inf")
#     patience_counter = 0
#     train_iter = iter(train_loader)

#     while step < args.max_steps:
#         model.train()
#         try:
#             batch = next(train_iter)
#         except StopIteration:
#             train_iter = iter(train_loader)
#             batch = next(train_iter)

#         input_ids = batch["input_ids"].to(device)
#         labels = batch["labels"].to(device)

#         outputs = model(input_ids=input_ids, labels=labels)
#         loss = outputs.loss

#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         scheduler.step()

#         if step % 100 == 0:
#             ppl = torch.exp(loss).item()
#             print(f"Step {step} | train loss: {loss.item():.4f} | ppl: {ppl:.2f}")

#         if step % args.val_every == 0 and step > 0:
#             model.eval()
#             val_losses = []
#             with torch.no_grad():
#                 for val_batch in val_loader:
#                     input_ids = val_batch["input_ids"].to(device)
#                     labels = val_batch["labels"].to(device)
#                     outputs = model(input_ids=input_ids, labels=labels)
#                     val_losses.append(outputs.loss.item())
#             val_loss = sum(val_losses) / len(val_losses)
#             val_ppl = torch.exp(torch.tensor(val_loss)).item()
#             print(f"Step {step} | val loss: {val_loss:.4f} | val ppl: {val_ppl:.2f}")

#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 patience_counter = 0
#                 torch.save(model.state_dict(), f"{args.output_dir}/best_model.pt")
#                 print(f"Saved best model at step {step}")
#             else:
#                 patience_counter += 1
#                 if patience_counter >= args.patience:
#                     print(f"Early stopping at step {step}")
#                     break

#         step += 1

#     best_ppl = torch.exp(torch.tensor(best_val_loss)).item()
#     print(f"Done! Best val ppl: {best_ppl:.2f}")

#     results = {
#         "mode": args.mode,
#         "hidden_size": args.hidden_size,
#         "seq_len": args.seq_len,
#         "max_steps": args.max_steps,
#         "seed": args.seed,
#         "best_val_loss": best_val_loss,
#         "best_val_ppl": best_ppl,
#     }
#     with open(f"{args.output_dir}/result.json", "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"Results saved to {args.output_dir}/result.json")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", type=str, required=True, choices=["mlp_softplus", "mlp_softmax", "fixed", "softmax"])
#     parser.add_argument("--hidden_size", type=int, default=256)
#     parser.add_argument("--num_layers", type=int, default=6)
#     parser.add_argument("--num_heads", type=int, default=4)
#     parser.add_argument("--seq_len", type=int, default=512)
#     parser.add_argument("--batch_size", type=int, default=8)
#     parser.add_argument("--max_steps", type=int, default=20000)
#     parser.add_argument("--val_every", type=int, default=500)
#     parser.add_argument("--patience", type=int, default=10)
#     parser.add_argument("--lr", type=float, default=3e-4)
#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--output_dir", type=str, default="results/lm")
#     args = parser.parse_args()
#     train(args)

import torch
import argparse
import os
import json
import random
import numpy as np
from data_utils import get_wikitext103_dataloader

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(args):
    set_seed(args.seed)

    # ── GPU setup ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # A100 optimizations: TF32 for matmul and convolutions
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ── Data ───────────────────────────────────────────────────────────────────
    print("Loading data...")
    train_loader, train_size = get_wikitext103_dataloader(
        "train", args.seq_len, args.batch_size, device,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader, val_size = get_wikitext103_dataloader(
        "validation", args.seq_len, args.batch_size, device,
        num_workers=args.num_workers, pin_memory=True
    )
    print(f"Train size: {train_size}, Val size: {val_size}")

    # ── Model ──────────────────────────────────────────────────────────────────
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

    # torch.compile — big free speedup on A100 (PyTorch 2.x)
    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # ── Optimizer & Scheduler ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.1, fused=True  # fused=True is faster on CUDA
    )
    warmup_steps = 500
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (args.max_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # bf16 GradScaler (A100 natively supports bf16 — faster than fp16)
    use_amp = args.bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))  # scaler only needed for fp16
    print(f"AMP enabled: {use_amp}, dtype: {amp_dtype}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Training Loop ──────────────────────────────────────────────────────────
    step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    train_iter = iter(train_loader)

    # Gradient accumulation: effective_batch = batch_size * grad_accum_steps
    effective_batch = args.batch_size * args.grad_accum_steps
    print(f"Effective batch size: {effective_batch} "
          f"(batch={args.batch_size} x accum={args.grad_accum_steps})")

    while step < args.max_steps:
        model.train()
        optimizer.zero_grad()

        # Gradient accumulation loop
        accum_loss = 0.0
        for accum_step in range(args.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / args.grad_accum_steps  # normalize

            if amp_dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()

        # Optimizer step
        if amp_dtype == torch.float16:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        if step % 100 == 0:
            ppl = torch.exp(torch.tensor(accum_loss * args.grad_accum_steps)).item()
            print(f"Step {step} | train loss: {accum_loss * args.grad_accum_steps:.4f} | ppl: {ppl:.2f}")

        # ── Validation ────────────────────────────────────────────────────────
        if step % args.val_every == 0 and step > 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch in val_loader:
                    input_ids = val_batch["input_ids"].to(device, non_blocking=True)
                    labels = val_batch["labels"].to(device, non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                        outputs = model(input_ids=input_ids, labels=labels)
                    val_losses.append(outputs.loss.item())
            val_loss = sum(val_losses) / len(val_losses)
            val_ppl = torch.exp(torch.tensor(val_loss)).item()
            print(f"Step {step} | val loss: {val_loss:.4f} | val ppl: {val_ppl:.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save unwrapped model (needed if torch.compile is used)
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save(raw_model.state_dict(), f"{args.output_dir}/best_model.pt")
                print(f"Saved best model at step {step}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at step {step}")
                    break

        step += 1

    # ── Results ────────────────────────────────────────────────────────────────
    best_ppl = torch.exp(torch.tensor(best_val_loss)).item()
    print(f"Done! Best val ppl: {best_ppl:.2f}")

    results = {
        "mode": args.mode,
        "hidden_size": args.hidden_size,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": effective_batch,
        "bf16": args.bf16,
        "compile": args.compile,
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
    parser.add_argument("--mode", type=str, required=True,
                        choices=["mlp_softplus", "mlp_softmax", "fixed", "softmax"])
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)        # was 8
    parser.add_argument("--grad_accum_steps", type=int, default=1)   # new
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/lm")
    parser.add_argument("--bf16", action="store_true", default=True)  # new
    parser.add_argument("--compile", action="store_true", default=True) # new
    parser.add_argument("--num_workers", type=int, default=4)         # new
    args = parser.parse_args()
    train(args)
