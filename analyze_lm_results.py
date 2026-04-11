"""
Analyze and visualize Phase 2 language modeling results.
Generates loss curves, perplexity tables, and comparison plots.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_results(output_dir: str):
    """Load all experiment results."""
    output_dir = Path(output_dir)
    
    results = defaultdict(list)
    
    for run_dir in sorted(output_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        results_file = run_dir / "results.json"
        if not results_file.exists():
            continue
        
        with open(results_file, "r") as f:
            run_results = json.load(f)
        
        lambda_mode = run_results["lambda_mode"]
        results[lambda_mode].append(run_results)
    
    return results


def print_perplexity_table(results):
    """Print summary table of validation perplexities."""
    print("\n" + "="*70)
    print("PHASE 2: Language Modeling Validation Results")
    print("="*70)
    print(f"{'Lambda Mode':<20} {'Best Val PPL':<20} {'Std Dev':<20}")
    print("-"*70)
    
    for mode in ["fixed", "mlp_softplus", "mlp_softmax"]:
        if mode not in results or not results[mode]:
            print(f"{mode:<20} {'N/A':<20} {'N/A':<20}")
            continue
        
        ppls = [r["best_val_ppl"] for r in results[mode]]
        mean_ppl = np.mean(ppls)
        std_ppl = np.std(ppls)
        
        print(f"{mode:<20} {mean_ppl:<20.2f} {std_ppl:<20.2f}")
    
    print("="*70)


def plot_loss_curves(results, output_dir: str):
    """Plot training and validation loss curves."""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    modes = ["fixed", "mlp_softplus", "mlp_softmax"]
    
    for ax, mode in zip(axes, modes):
        if mode not in results or not results[mode]:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(mode)
            continue
        
        for run in results[mode]:
            run_dir = output_dir / f"{mode}_dim{run['mlp_hidden_dim']}_seed{run['seed']}"
            
            loss_file = run_dir / "train_losses.json"
            if loss_file.exists():
                with open(loss_file, "r") as f:
                    train_losses = json.load(f)
                
                ax.plot(train_losses, alpha=0.5, label=f"seed {run['seed']}")
        
        ax.set_title(f"λ Mode: {mode}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=150)
    print(f"✓ Loss curves saved to {output_dir / 'loss_curves.png'}")
    plt.close()


def plot_ppl_comparison(results, output_dir: str):
    """Plot perplexity comparison across modes."""
    output_dir = Path(output_dir)
    
    modes = ["fixed", "mlp_softplus", "mlp_softmax"]
    ppls_by_mode = {}
    
    for mode in modes:
        if mode in results:
            ppls_by_mode[mode] = [r["best_val_ppl"] for r in results[mode]]
        else:
            ppls_by_mode[mode] = []
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(modes))
    means = [np.mean(ppls_by_mode[m]) if ppls_by_mode[m] else 0 for m in modes]
    stds = [np.std(ppls_by_mode[m]) if ppls_by_mode[m] else 0 for m in modes]
    
    ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, color=["blue", "green", "red"])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Validation Perplexity (lower is better)")
    ax.set_title("Phase 2: Language Modeling Perplexity Comparison")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_dir / "ppl_comparison.png", dpi=150)
    print(f"✓ PPL comparison saved to {output_dir / 'ppl_comparison.png'}")
    plt.close()


def analyze_lm_results(output_dir: str = "results/lm"):
    """Main analysis function."""
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return
    
    print(f"Loading results from: {output_dir}")
    results = load_results(str(output_dir))
    
    if not results:
        print("No results found!")
        return
    
    # Print table
    print_perplexity_table(results)
    
    # Generate plots
    plot_loss_curves(results, str(output_dir))
    plot_ppl_comparison(results, str(output_dir))
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results/lm")
    args = parser.parse_args()
    
    analyze_lm_results(args.output_dir)
