import sys
sys.path.insert(0, "/mnt/e/log-linear-attention-lamda-MLP/flame/3rdparty/flash-linear-attention")
sys.path.insert(1, "/mnt/e/log-linear-attention-lamda-MLP")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from einops import rearrange

from hattention.base import HType, HStruct, get_num_levels
from hattention.kernel import hattention_kernel
from hattention.lambda_mlp import LambdaMLPSoftplus, LambdaMLPSoftmax

LEVEL_BASE = 2
HTYPE      = HType.WEAK
HSTRUCT    = HStruct.MAMBA2


# ── model (same as mqar) ──────────────────────────────────────────────────────

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
            self.lambda_fixed = False
        elif lambda_mode == "mlp_softmax":
            self.lambda_module = LambdaMLPSoftmax(num_levels=num_levels, hidden_dim=mlp_hidden_dim)
            self.lambda_fixed = False
        else:
            self.lambda_module = None

        self.norm = nn.LayerNorm(d_model)
        self._captured_lambda = None

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

        self._captured_lambda = L.detach().cpu()

        Y = hattention_kernel(
            q=Q, k=K, v=V,
            b=None, g=g, l=L,
            scale=None,
            head_first=False,
            level_base=LEVEL_BASE,
            htype=HTYPE,
            hstruct=HSTRUCT,
        )

        Y = rearrange(Y, "b t h d -> b t (h d)")
        return self.norm(self.out_proj(Y) + x)


class SelectiveCopyModel(nn.Module):
    """
    Model for selective copy task.
    Input tokens -> embedding -> HAttention layers -> lm_head.
    Adjust vocab_size, d_model, num_heads etc. to match your training setup.
    """
    def __init__(self, vocab_size, d_model, num_heads, state_size,
                 num_levels, n_layers=2, lambda_mode="fixed", mlp_hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            HAttentionLayer(d_model, num_heads, state_size, num_levels,
                            lambda_mode, mlp_hidden_dim)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


# ── selective copy data generator ─────────────────────────────────────────────

def generate_selective_copy(n_samples, seq_len, num_tokens, vocab_size=256, seed=42):
    """
    Selective copy task:
      - num_tokens random tokens are placed at random positions in the first half
      - the model must copy them in order at the end of the sequence
      - the rest is filled with a noise/blank token (0)
    """
    np.random.seed(seed)
    token_vocab = np.arange(1, vocab_size)  # reserve 0 as noise/blank

    input_ids = np.zeros((n_samples, seq_len), dtype=np.int64)
    labels    = np.full((n_samples, seq_len), -100, dtype=np.int64)

    copy_region   = seq_len // 2          # tokens to copy placed in first half
    target_start  = seq_len - num_tokens  # copy target written at end

    for b in range(n_samples):
        positions  = np.random.choice(copy_region, size=num_tokens, replace=False)
        positions  = np.sort(positions)
        tokens     = np.random.choice(token_vocab, size=num_tokens, replace=False)
        input_ids[b, positions] = tokens
        # fill non-selected positions with random noise so the model can't cheat
        noise_mask = np.ones(copy_region, dtype=bool)
        noise_mask[positions] = False
        noise_positions = np.where(noise_mask)[0]
        input_ids[b, noise_positions] = np.random.randint(1, vocab_size, noise_positions.shape)
        # target region
        input_ids[b, target_start:] = tokens
        labels[b, target_start:]    = tokens

    return torch.tensor(input_ids), torch.tensor(labels)


# ── config ────────────────────────────────────────────────────────────────────

RESULTS_DIR = "/mnt/e/log-linear-attention-lamda-MLP/results/selective_copy"
OUTPUT_DIR  = "/mnt/e/log-linear-attention-lamda-MLP/results/selective_copy_lambda_heatmaps"

MODEL_DIM  = 64
VOCAB_SIZE = 256
BATCH_VIZ  = 8

# mirrors sweep_selective_copy.py — adjust seeds/seqs if needed
CONFIGS = [
    # (lambda_mode, num_tokens, seq_len, seeds)
    ("fixed",        16, 256,  [0, 1, 2, 3, 4]),
    ("fixed",        16, 512,  [0, 1, 2, 3, 4]),
    ("fixed",        16, 1024, [0, 1, 2]),
    ("mlp_softplus", 16, 256,  [0, 1, 2, 3, 4]),
    ("mlp_softplus", 16, 512,  [0, 1, 2, 3, 4]),
    ("mlp_softplus", 16, 1024, [0, 1, 2]),
    ("mlp_softmax",  16, 256,  [0, 1, 2, 3, 4]),
    ("mlp_softmax",  16, 512,  [0, 1, 2, 3, 4]),
    ("mlp_softmax",  16, 1024, [0, 1, 2]),
]

CMAP_AVG   = "YlOrRd"
CMAP_TOKEN = "viridis"


# ── model loader ──────────────────────────────────────────────────────────────

def load_model(lambda_mode, num_tokens, seq_len, seed, device="cpu"):
    run_name = f"{lambda_mode}_tok{num_tokens}_seq{seq_len}_seed{seed}"
    ckpt     = os.path.join(RESULTS_DIR, run_name, "best_model.pt")

    if not os.path.exists(ckpt):
        print(f"  [skip] not found: {ckpt}")
        return None

    # Auto-detect vocab size from checkpoint
    state_dict = torch.load(ckpt, map_location=device, weights_only=True)
    vocab_size  = state_dict["embedding.weight"].shape[0]
    print(f"  [info] detected vocab_size={vocab_size}")

    num_levels = get_num_levels(seq_len, base=LEVEL_BASE)
    model = SelectiveCopyModel(
        vocab_size=vocab_size,   # use detected value!
        d_model=MODEL_DIM,
        num_heads=1,
        state_size=MODEL_DIM,
        num_levels=num_levels,
        n_layers=2,
        lambda_mode=lambda_mode,
        mlp_hidden_dim=64,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  [ok] loaded {run_name}")
    return model
# ── lambda extractors ─────────────────────────────────────────────────────────

def get_lambda_fixed(model):
    """Fixed mode: return softplus(L) — shape (num_heads, num_levels) per layer."""
    lambdas = []
    for layer in model.layers:
        L = F.softplus(layer.L).detach().cpu().numpy()
        lambdas.append(L)
    return lambdas


def get_lambda_mlp(model, seq_len, num_tokens, device="cpu"):
    """
    Mlp modes: run a sample batch and capture lambda per token.
    Returns:
      per_layer_avg  — list of (T, num_levels) averaged over batch & heads
      per_layer_full — list of full (B, T, H, num_levels) tensors
    """
    vocab_size = model.embedding.num_embeddings  # auto-detect from loaded model!
    x, _ = generate_selective_copy(BATCH_VIZ, seq_len, num_tokens, vocab_size=vocab_size, seed=999)
    x = x.to(device)
    with torch.no_grad():
        model(x)
    per_layer_avg  = []
    per_layer_full = []
    for layer in model.layers:
        L = layer._captured_lambda          # (B, T, H, num_levels)
        avg = L.mean(dim=(0, 2)).numpy()    # (T, num_levels)
        per_layer_avg.append(avg)
        per_layer_full.append(L.numpy())
    return per_layer_avg, per_layer_full


# ── plotting helpers ──────────────────────────────────────────────────────────

def plot_fixed_heatmap(lambdas_per_layer, title, save_path):
    """Heatmap of heads x levels for each layer (fixed mode)."""
    n_layers = len(lambdas_per_layer)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    for i, (ax, L) in enumerate(zip(axes, lambdas_per_layer)):
        im = ax.imshow(L, aspect="auto", cmap=CMAP_AVG, interpolation="nearest")
        ax.set_title(f"Layer {i} — fixed λ\n(heads × levels)", fontsize=11)
        ax.set_xlabel("Hierarchy level")
        ax.set_ylabel("Head")
        ax.set_xticks(range(L.shape[1]))
        ax.set_yticks(range(L.shape[0]))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mlp_avg_heatmap(lambdas_per_layer, title, save_path):
    """Summary heatmap: mean over tokens & batch -> (1, num_levels) per layer."""
    n_layers = len(lambdas_per_layer)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    for i, (ax, L) in enumerate(zip(axes, lambdas_per_layer)):
        L_avg = L.mean(axis=0, keepdims=True)   # (1, num_levels)
        im = ax.imshow(L_avg, aspect="auto", cmap=CMAP_AVG, interpolation="nearest")
        ax.set_title(f"Layer {i} — avg λ\n(mean over tokens & batch)", fontsize=11)
        ax.set_xlabel("Hierarchy level")
        ax.set_yticks([])
        ax.set_xticks(range(L_avg.shape[1]))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mlp_token_heatmap(lambdas_per_layer, title, save_path, num_tokens=None, seq_len=None):
    """
    Per-token heatmap: (num_levels, T) — levels on y-axis, token positions on x.
    Optionally marks the target copy region with a vertical dashed line.
    """
    n_layers = len(lambdas_per_layer)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]
    for i, (ax, L) in enumerate(zip(axes, lambdas_per_layer)):
        im = ax.imshow(L.T, aspect="auto", cmap=CMAP_TOKEN, interpolation="nearest")
        ax.set_title(f"Layer {i} — per-token λ\n(levels × token positions)", fontsize=11)
        ax.set_xlabel("Token position")
        ax.set_ylabel("Hierarchy level")
        ax.set_yticks(range(L.shape[1]))

        # mark where the copy target region starts
        if num_tokens is not None and seq_len is not None:
            target_start = seq_len - num_tokens
            ax.axvline(x=target_start - 0.5, color="red", linestyle="--",
                       linewidth=1.5, label=f"copy target start ({target_start})")
            ax.legend(fontsize=7, loc="upper left")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_seed_comparison(all_seed_lambdas, mode, num_tokens, seq_len, save_dir):
    """
    Overlay lambda values across seeds (mean over heads) — line plot per layer.
    Works for both fixed and mlp modes (both store per-layer (*, num_levels) arrays).
    """
    n_layers = len(all_seed_lambdas[0])
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_seed_lambdas)))
    for i, ax in enumerate(axes):
        for s_idx, L_layers in enumerate(all_seed_lambdas):
            L = L_layers[i]            # (num_heads, num_levels) or (T, num_levels)
            L_mean = L.mean(axis=0)    # (num_levels,)
            ax.plot(L_mean, marker="o", color=colors[s_idx],
                    label=f"seed {s_idx}", linewidth=2)
        ax.set_title(f"Layer {i} — λ per level across seeds", fontsize=11)
        ax.set_xlabel("Hierarchy level")
        ax.set_ylabel("λ value")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    title = f"{mode} | tok={num_tokens} | seq={seq_len} | seed comparison"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"seed_comparison_{mode}_tok{num_tokens}_seq{seq_len}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved seed comparison -> {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from collections import defaultdict
    seed_lambda_store = defaultdict(list)   # (mode, tok, seq) -> list of per-layer lambdas

    for (mode, tok, seq, seeds) in CONFIGS:
        print(f"\n{'='*58}")
        print(f"Config: mode={mode} | tok={tok} | seq={seq} | seeds={seeds}")
        print(f"{'='*58}")

        config_dir = os.path.join(OUTPUT_DIR, f"{mode}_tok{tok}_seq{seq}")
        os.makedirs(config_dir, exist_ok=True)

        for seed in seeds:
            print(f"\n  seed={seed}")
            model = load_model(mode, tok, seq, seed, device)
            if model is None:
                continue

            run_label = f"{mode} | tok={tok} | seq={seq} | seed={seed}"
            base_name = f"{mode}_tok{tok}_seq{seq}_seed{seed}"

            if mode == "fixed":
                lambdas = get_lambda_fixed(model)
                seed_lambda_store[(mode, tok, seq)].append(lambdas)

                plot_fixed_heatmap(
                    lambdas,
                    title=run_label,
                    save_path=os.path.join(config_dir, f"{base_name}_fixed_heatmap.png")
                )
                print(f"    saved fixed heatmap")

            else:
                avg_lambdas, full_lambdas = get_lambda_mlp(model, seq, tok, device)
                seed_lambda_store[(mode, tok, seq)].append(avg_lambdas)

                plot_mlp_avg_heatmap(
                    avg_lambdas,
                    title=f"{run_label} [avg over tokens]",
                    save_path=os.path.join(config_dir, f"{base_name}_avg_heatmap.png")
                )

                # per-token heatmap with red line marking the copy target region
                plot_mlp_token_heatmap(
                    avg_lambdas,
                    title=f"{run_label} [per token]",
                    save_path=os.path.join(config_dir, f"{base_name}_token_heatmap.png"),
                    num_tokens=tok,
                    seq_len=seq,
                )
                print(f"    saved avg + token heatmaps")

        # seed comparison plot
        store_key = (mode, tok, seq)
        if len(seed_lambda_store[store_key]) > 1:
            plot_seed_comparison(
                seed_lambda_store[store_key],
                mode, tok, seq,
                config_dir,
            )

    print(f"\n\nAll done! Heatmaps saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()