"""
Language modeling training script for Phase 2 validation.
Trains Log-Linear Attention variants on TinyShakespeare.
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from torch.optim import Adam

from hattention.configuration_hattention import HAttentionConfig
from hattention.modeling_hattention import HAttentionForCausalLM
from hattention.data_utils import get_tiny_shakespeare_dataloader


def compute_perplexity(loss):
    """Compute perplexity from loss."""
    return torch.exp(torch.tensor(loss))


def train_lm(
    lambda_mode: str = "fixed",
    mlp_hidden_dim: int = 64,
    model_dim: int = 256,
    seq_len: int = 1024,
    batch_size: int = 8,
    max_steps: int = 5000,
    val_steps: int = 500,
    lr: float = 1e-4,
    seed: int = 0,
    output_dir: str = "results/lm",
    device: str = "cuda",
):
    """
    Train a Log-Linear Attention model on TinyShakespeare.
    
    Args:
        lambda_mode: "fixed", "mlp_softplus", or "mlp_softmax"
        mlp_hidden_dim: Hidden dimension for MLP λ (32, 64, or 128)
        model_dim: Model hidden dimension (smaller for quick iteration)
        seq_len: Sequence length (1024 for TinyShakespeare)
        batch_size: Batch size
        max_steps: Maximum training steps
        val_steps: Validation frequency
        lr: Learning rate
        seed: Random seed
        output_dir: Output directory for results
        device: "cuda" or "cpu"
    """
    
    # Set seed
    torch.manual_seed(seed)
    
    # Create output dir
    run_name = f"{lambda_mode}_dim{mlp_hidden_dim}_seed{seed}"
    result_dir = Path(output_dir) / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training: {lambda_mode} | dim: {mlp_hidden_dim} | seed: {seed}")
    print(f"Results: {result_dir}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading TinyShakespeare...")
    train_loader, num_train_samples = get_tiny_shakespeare_dataloader(
        split="train",
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
    )
    val_loader, num_val_samples = get_tiny_shakespeare_dataloader(
        split="validation",
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
    )
    print(f"✓ Loaded: {num_train_samples} train samples, {num_val_samples} val samples")
    
    # Create model
    print(f"Creating HAttentionForCausalLM (lambda_mode={lambda_mode})...")
    config = HAttentionConfig(
        hidden_size=model_dim,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=model_dim * 2,
        vocab_size=50257,  # GPT-2 vocab
        max_position_embeddings=seq_len,
    )
    
    model = HAttentionForCausalLM(config)
    
    # Patch lambda mode BEFORE moving to device
    # (Follow partner's convention from train_mqar.py)
    import hattention.modeling_hattention as mh
    mh.LAMBDA_MODE_TYPE = lambda_mode
    mh.LAMBDA_MLP_HIDDEN_DIM = mlp_hidden_dim
    
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {total_params:,} parameters")
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Training loop
    print(f"Starting training ({max_steps} steps)...\n")
    
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 5
    
    step = 0
    train_iter = iter(train_loader)
    
    while step < max_steps:
        model.train()
        
        # Training step
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
        step += 1
        
        # Validation
        if step % val_steps == 0:
            model.eval()
            val_loss_steps = []
            
            with torch.no_grad():
                val_step = 0
                for val_batch in val_loader:
                    if val_step >= 10:  # Sample 10 validation batches
                        break
                    
                    val_input_ids = val_batch["input_ids"].to(device)
                    val_labels = val_batch["labels"].to(device)
                    
                    val_outputs = model(val_input_ids, labels=val_labels)
                    val_loss_steps.append(val_outputs.loss.item())
                    val_step += 1
            
            avg_val_loss = sum(val_loss_steps) / len(val_loss_steps)
            val_losses.append(avg_val_loss)
            
            avg_train_loss = sum(train_losses[-val_steps:]) / val_steps
            train_ppl = compute_perplexity(avg_train_loss)
            val_ppl = compute_perplexity(avg_val_loss)
            
            print(
                f"Step {step:5d} | "
                f"Train Loss: {avg_train_loss:.4f} (PPL: {train_ppl:.2f}) | "
                f"Val Loss: {avg_val_loss:.4f} (PPL: {val_ppl:.2f})"
            )
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at step {step}")
                    break
    
    print(f"\n✓ Training complete. Best val loss: {best_val_loss:.4f}")
    
    # Save results
    results = {
        "lambda_mode": lambda_mode,
        "mlp_hidden_dim": mlp_hidden_dim,
        "seed": seed,
        "final_train_loss": train_losses[-1],
        "best_val_loss": best_val_loss,
        "best_val_ppl": float(compute_perplexity(best_val_loss)),
        "total_steps": step,
        "total_params": total_params,
    }
    
    with open(result_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save loss history
    with open(result_dir / "train_losses.json", "w") as f:
        json.dump(train_losses, f)
    with open(result_dir / "val_losses.json", "w") as f:
        json.dump(val_losses, f)
    
    print(f"✓ Results saved to {result_dir}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lambda_mode",
        type=str,
        default="fixed",
        choices=["fixed", "mlp_softplus", "mlp_softmax"],
    )
    parser.add_argument("--mlp_hidden_dim", type=int, default=64)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--val_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/lm")
    
    args = parser.parse_args()
    
    train_lm(
        lambda_mode=args.lambda_mode,
        mlp_hidden_dim=args.mlp_hidden_dim,
        model_dim=args.model_dim,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        val_steps=args.val_steps,
        lr=args.lr,
        seed=args.seed,
        output_dir=args.output_dir,
    )
