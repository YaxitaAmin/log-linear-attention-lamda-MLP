"""
Sweep results plotter — highlights mlp_softmax and mlp_softplus vs fixed baseline.
Run from the selective_copy folder (where sweep_summary.json and run folders live).

Usage:
    python plots.py

Outputs:
    bar_strip.png      — mean ± std of best_val_acc per model × seq_len
    learning_curves.png — mean val_acc over epochs per model type
"""

import json
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── config ────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(".")   # change if running from elsewhere

MODEL_STYLES = {
    "fixed":        dict(color="#B4B2A9", label="fixed (baseline)", zorder=1),
    "mlp_softplus": dict(color="#1D9E75", label="mlp_softplus",     zorder=3),
    "mlp_softmax":  dict(color="#7F77DD", label="mlp_softmax",      zorder=3),
}

SEQ_LENS = [256, 512, 1024]

# ── helpers ───────────────────────────────────────────────────────────────────

def parse_run_name(name):
    """Return (model, seq_len, seed) from a folder name."""
    m = re.match(r"^(fixed|mlp_softplus|mlp_softmax)_tok\d+_seq(\d+)_seed(\d+)$", name)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None, None, None


def load_runs(results_dir):
    runs = []
    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir():
            continue
        model, seq_len, seed = parse_run_name(folder.name)
        if model is None:
            continue
        rfile = folder / "result.json"
        if not rfile.exists():
            rfile = folder / "results.json"
        if not rfile.exists():
            continue
        with open(rfile) as f:
            data = json.load(f)
        # Look for where the script does: data = json.load(f)
# Add these lines immediately after:

        if 'log_history' in data and (not data.get('val_accs')):
            # Extract val_acc from each dictionary in the log_history list
            data['val_accs'] = [entry['val_acc'] for entry in data['log_history'] if 'val_acc' in entry]
    
            # Optional: If the script also needs the 'steps', extract those too
            data['steps'] = [entry['step'] for entry in data['log_history'] if 'step' in entry]
        # try common key variants for the per-epoch val acc list
        val_accs = (
            data.get("val_accs")
            or data.get("val_acc")
            or data.get("val_accuracies")
            or data.get("validation_accs")
            or []
        )
        runs.append(dict(
            model=model,
            seq_len=seq_len,
            seed=seed,
            best_val_acc=data["best_val_acc"],
            val_accs=val_accs,
        ))
    return runs

# ── plot 1: bar + strip ───────────────────────────────────────────────────────

def plot_bar_strip(runs):
    # group by (model, seq_len)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in runs:
        groups[(r["model"], r["seq_len"])].append(r["best_val_acc"])

    seq_lens_present = sorted({r["seq_len"] for r in runs})
    models = list(MODEL_STYLES.keys())
    n_seq = len(seq_lens_present)
    n_models = len(models)

    fig, axes = plt.subplots(1, n_seq, figsize=(4 * n_seq, 5), sharey=True)
    if n_seq == 1:
        axes = [axes]

    fig.suptitle("Best val acc: model variants vs baseline", fontsize=13, fontweight="normal", y=1.01)

    for ax, seq in zip(axes, seq_lens_present):
        ax.set_title(f"seq_len = {seq}", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(range(n_models))
        ax.set_xticklabels([MODEL_STYLES[m]["label"] for m in models], fontsize=9, rotation=15, ha="right")
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.set_ylabel("best val acc (%)" if ax is axes[0] else "", fontsize=10)

        for i, model in enumerate(models):
            vals = groups.get((model, seq), [])
            if not vals:
                continue
            style = MODEL_STYLES[model]
            mean, std = np.mean(vals), np.std(vals)

            # bar
            ax.bar(i, mean, width=0.5, color=style["color"], alpha=0.85,
                   zorder=style["zorder"], linewidth=0)
            # error bar
            ax.errorbar(i, mean, yerr=std, fmt="none", color="black",
                        capsize=5, linewidth=1.2, zorder=4)
            # strip (individual seeds)
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color="white", edgecolors="black", s=28, linewidth=0.8,
                       zorder=5)
            # mean label
            ax.text(i, mean + std + 0.8, f"{mean:.1f}", ha="center",
                    fontsize=8, color="#444441")

        # baseline reference line (mean of fixed for this seq_len)
        baseline = groups.get(("fixed", seq), [])
        if baseline:
            ax.axhline(np.mean(baseline), color=MODEL_STYLES["fixed"]["color"],
                       linewidth=1, linestyle="--", zorder=0, alpha=0.7)

    # shared y floor
    all_vals = [r["best_val_acc"] for r in runs]
    axes[0].set_ylim(min(all_vals) - 3, max(all_vals) + 5)

    plt.tight_layout()
    out = Path("bar_strip.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ── plot 2: learning curves ───────────────────────────────────────────────────

def plot_learning_curves(runs):
    from collections import defaultdict

    # group val_accs by (model, seq_len) — average across seeds
    groups = defaultdict(list)
    for r in runs:
        if r["val_accs"]:
            groups[(r["model"], r["seq_len"])].append(r["val_accs"])

    seq_lens_present = sorted({k[1] for k in groups})
    models = list(MODEL_STYLES.keys())

    if not seq_lens_present:
        sample_keys = list(runs[0].keys()) if runs else []
        print("No val_accs found. Keys in first run dict:", sample_keys)
        print("Skipping learning_curves.png — check the key name in your result.json files.")
        return

    fig, axes = plt.subplots(1, len(seq_lens_present),
                             figsize=(5 * len(seq_lens_present), 4), sharey=True)
    if len(seq_lens_present) == 1:
        axes = [axes]

    fig.suptitle("Val acc over epochs (mean across seeds)", fontsize=13,
                 fontweight="normal", y=1.01)

    handles = []
    for ax, seq in zip(axes, seq_lens_present):
        ax.set_title(f"seq_len = {seq}", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("epoch", fontsize=10)
        ax.set_ylabel("val acc (%)" if ax is axes[0] else "", fontsize=10)

        for model in models:
            curves = groups.get((model, seq), [])
            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            arr = np.array([c[:min_len] for c in curves])
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            epochs = np.arange(1, min_len + 1)
            style = MODEL_STYLES[model]

            line, = ax.plot(epochs, mean, color=style["color"],
                            linewidth=2 if model != "fixed" else 1.2,
                            linestyle="-" if model != "fixed" else "--",
                            label=style["label"], zorder=style["zorder"])
            ax.fill_between(epochs, mean - std, mean + std,
                            color=style["color"], alpha=0.15, zorder=1)
            if ax is axes[0]:
                handles.append(line)

    axes[-1].legend(handles=handles, loc="lower right", fontsize=9,
                    frameon=False)
    plt.tight_layout()
    out = Path("learning_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    runs = load_runs(RESULTS_DIR)
    if not runs:
        print("No runs found. Make sure you're running from the selective_copy folder.")
    else:
        print(f"Loaded {len(runs)} runs.")
        plot_bar_strip(runs)
        plot_learning_curves(runs)