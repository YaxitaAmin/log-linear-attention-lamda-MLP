import sys
sys.path.insert(0, "/mnt/e/log-linear-attention-lamda-MLP/flame/3rdparty/flash-linear-attention")
sys.path.insert(1, "/mnt/e/log-linear-attention-lamda-MLP")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from einops import rearrange

from hattention.base import HType, HStruct, get_num_levels
from hattention.kernel import hattention_kernel
from hattention.lambda_mlp import LambdaMLPSoftplus, LambdaMLPSoftmax

LEVEL_BASE = 2
HTYPE      = HType.WEAK
HSTRUCT    = HStruct.MAMBA2


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


class MQARModel(nn.Module):
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


def generate_mqar(n_samples, seq_len, num_kv_pairs, vocab_size=128, seed=42):
    np.random.seed(seed)
    half        = vocab_size // 2
    query_start = seq_len - num_kv_pairs * 2
    input_ids   = np.zeros((n_samples, seq_len), dtype=np.int64)
    labels      = np.full((n_samples, seq_len), -100, dtype=np.int64)
    key_choices   = np.arange(1, half)
    value_choices = np.arange(half, vocab_size)

    for b in range(n_samples):
        keys   = np.random.choice(key_choices,   size=num_kv_pairs, replace=False)
        values = np.random.choice(value_choices, size=num_kv_pairs, replace=False)
        kv     = dict(zip(keys.tolist(), values.tolist()))
        for i, (k, v) in enumerate(zip(keys, values)):
            input_ids[b, i*2]     = k
            input_ids[b, i*2 + 1] = v
        qkeys = np.random.choice(keys, size=num_kv_pairs, replace=False)
        for i, qk in enumerate(qkeys):
            pos = query_start + i * 2
            input_ids[b, pos]     = int(qk)
            labels[b, pos + 1]    = kv[int(qk)]

    mask = input_ids == 0
    input_ids[mask] = np.random.randint(1, vocab_size, mask.sum())
    return torch.tensor(input_ids), torch.tensor(labels)



RESULTS_DIR  = "/mnt/e/log-linear-attention-lamda-MLP/results/mqar"
OUTPUT_DIR   = "/mnt/e/log-linear-attention-lamda-MLP/results/lambda_heatmaps"
MODEL_DIM    = 64
VOCAB_SIZE   = 128
BATCH_VIZ    = 8   

CONFIGS = [
    ("fixed",       4, 128, [0,1,2,3,4]),
    ("mlp_softplus",4, 128, [0,1,2,3,4]),
    ("mlp_softmax", 4, 128, [0,1,2,3,4]),

    ("fixed",        4, 256, [0,1]),
    ("fixed",        8, 256, [0,1]),
    ("fixed",       16, 256, [0,1]),
    ("fixed",       32, 256, [0,1]),
    ("mlp_softplus", 4, 256, [0,1]),
    ("mlp_softplus", 8, 256, [0,1]),
    ("mlp_softplus",16, 256, [0,1]),
    ("mlp_softplus",32, 256, [0,1]),
    ("mlp_softmax",  4, 256, [0,1]),
    ("mlp_softmax",  8, 256, [0,1]),
    ("mlp_softmax", 16, 256, [0,1]),
    ("mlp_softmax", 32, 256, [0,1]),

    # seq=512, kv=4, fixed + mlp_softplus, seed0+seed1
    ("fixed",        4, 512, [0,1]),
    ("mlp_softplus", 4, 512, [0,1]),
]

CMAP_AVG     = "YlOrRd"
CMAP_TOKEN   = "viridis"

def load_model(lambda_mode, kv, seq, seed, device="cpu"):
    run_name = f"{lambda_mode}_kv{kv}_seq{seq}_seed{seed}"
    ckpt     = os.path.join(RESULTS_DIR, run_name, "best_model.pt")
    if not os.path.exists(ckpt):
        print(f"  [skip] not found: {ckpt}")
        parent = os.path.dirname(ckpt)
        if os.path.exists(parent):
            print(f"         folder exists but contains: {os.listdir(parent)}")
        else:
            print(f"         folder missing: {parent}")
            similar = [d for d in os.listdir(RESULTS_DIR) if run_name[:12] in d]
            if similar:
                print(f"         similar folders: {similar[:5]}")
        return None

    num_levels = get_num_levels(seq, base=LEVEL_BASE)
    model = MQARModel(
        vocab_size=VOCAB_SIZE,
        d_model=MODEL_DIM,
        num_heads=1,
        state_size=MODEL_DIM,
        num_levels=num_levels,
        n_layers=2,
        lambda_mode=lambda_mode,
        mlp_hidden_dim=64,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"  [ok] loaded {run_name}")
    return model


def get_lambda_fixed(model):
    """For fixed mode: return softplus(L) shape (num_heads, num_levels)."""
    lambdas = []
    for layer in model.layers:
        L = F.softplus(layer.L).detach().cpu().numpy()  # (num_heads, num_levels)
        lambdas.append(L)
    return lambdas  # list over layers


def get_lambda_mlp(model, seq_len, kv, device="cpu"):
    """
    For mlp modes: run a sample batch, capture lambda per token.
    Returns per-layer list of arrays shape (T, num_levels) averaged over batch & heads.
    """
    x, _ = generate_mqar(BATCH_VIZ, seq_len, kv, seed=999)
    x = x.to(device)
    with torch.no_grad():
        model(x)
    per_layer_avg  = []
    per_layer_full = []
    for layer in model.layers:
        L = layer._captured_lambda  # (B, T, H, num_levels)
        avg_over_batch_head = L.mean(dim=(0, 2)).numpy()   # (T, num_levels)
        per_layer_avg.append(avg_over_batch_head)
        per_layer_full.append(L.numpy())
    return per_layer_avg, per_layer_full


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_fixed_heatmap(lambdas_per_layer, title, save_path):
    """lambdas_per_layer: list of (num_heads, num_levels) arrays."""
    n_layers = len(lambdas_per_layer)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    for i, (ax, L) in enumerate(zip(axes, lambdas_per_layer)):
        im = ax.imshow(L, aspect="auto", cmap=CMAP_AVG, interpolation="nearest")
        ax.set_title(f"Layer {i} - fixed λ\n(heads × levels)", fontsize=11)
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
    """lambdas_per_layer: list of (T, num_levels) arrays (averaged over batch & heads)."""
    n_layers = len(lambdas_per_layer)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    for i, (ax, L) in enumerate(zip(axes, lambdas_per_layer)):
        # average over tokens for the "summary" view
        L_avg = L.mean(axis=0, keepdims=True)  # (1, num_levels)
        im = ax.imshow(L_avg, aspect="auto", cmap=CMAP_AVG, interpolation="nearest")
        ax.set_title(f"Layer {i} - avg λ\n(mean over tokens & batch)", fontsize=11)
        ax.set_xlabel("Hierarchy level")
        ax.set_yticks([])
        ax.set_xticks(range(L_avg.shape[1]))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mlp_token_heatmap(lambdas_per_layer, title, save_path):
    """lambdas_per_layer: list of (T, num_levels) arrays."""
    n_layers = len(lambdas_per_layer)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]
    for i, (ax, L) in enumerate(zip(axes, lambdas_per_layer)):
        im = ax.imshow(L.T, aspect="auto", cmap=CMAP_TOKEN, interpolation="nearest")
        ax.set_title(f"Layer {i} - per-token λ\n(levels × token positions)", fontsize=11)
        ax.set_xlabel("Token position")
        ax.set_ylabel("Hierarchy level")
        ax.set_yticks(range(L.shape[1]))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_seed_comparison(all_seed_lambdas, mode, kv, seq, save_dir):
    """
    For fixed mode: overlay all seeds' lambda values per level.
    all_seed_lambdas: list of per-layer arrays (one per seed).
    """
    n_layers = len(all_seed_lambdas[0])
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_seed_lambdas)))
    for i, ax in enumerate(axes):
        for s_idx, L_layers in enumerate(all_seed_lambdas):
            L = L_layers[i]  # (num_heads, num_levels)
            L_mean = L.mean(axis=0)  # (num_levels,)
            ax.plot(L_mean, marker="o", color=colors[s_idx],
                    label=f"seed{s_idx}", linewidth=2)
        ax.set_title(f"Layer {i} - λ per level across seeds", fontsize=11)
        ax.set_xlabel("Hierarchy level")
        ax.set_ylabel("λ value (softplus)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    title = f"{mode} | kv={kv} | seq={seq} | seed comparison"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"seed_comparison_{mode}_kv{kv}_seq{seq}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved seed comparison -> {save_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # group configs by (mode, kv, seq) for seed comparison plots
    from collections import defaultdict
    seed_lambda_store = defaultdict(list)  # key: (mode,kv,seq) -> list of per-layer lambdas

    for (mode, kv, seq, seeds) in CONFIGS:
        print(f"\n{'='*55}")
        print(f"Config: mode={mode} | kv={kv} | seq={seq} | seeds={seeds}")
        print(f"{'='*55}")

        config_dir = os.path.join(OUTPUT_DIR, f"{mode}_kv{kv}_seq{seq}")
        os.makedirs(config_dir, exist_ok=True)

        for seed in seeds:
            print(f"\n  seed={seed}")
            model = load_model(mode, kv, seq, seed, device)
            if model is None:
                continue

            run_label = f"{mode} | kv={kv} | seq={seq} | seed={seed}"
            base_name = f"{mode}_kv{kv}_seq{seq}_seed{seed}"

            if mode == "fixed":
                lambdas = get_lambda_fixed(model)
                seed_lambda_store[(mode, kv, seq)].append(lambdas)

                # heatmap: heads x levels per layer
                plot_fixed_heatmap(
                    lambdas,
                    title=run_label,
                    save_path=os.path.join(config_dir, f"{base_name}_fixed_heatmap.png")
                )
                print(f"    saved fixed heatmap")

            else:
                avg_lambdas, full_lambdas = get_lambda_mlp(model, seq, kv, device)
                seed_lambda_store[(mode, kv, seq)].append(avg_lambdas)

                # avg heatmap (summary view)
                plot_mlp_avg_heatmap(
                    avg_lambdas,
                    title=f"{run_label} [avg over tokens]",
                    save_path=os.path.join(config_dir, f"{base_name}_avg_heatmap.png")
                )

                # per-token heatmap (levels x tokens)
                plot_mlp_token_heatmap(
                    avg_lambdas,
                    title=f"{run_label} [per token]",
                    save_path=os.path.join(config_dir, f"{base_name}_token_heatmap.png")
                )
                print(f"    saved avg + token heatmaps")

        # seed comparison plot (all seeds for this config)
        store_key = (mode, kv, seq)
        if len(seed_lambda_store[store_key]) > 1:
            plot_seed_comparison(
                seed_lambda_store[store_key],
                mode, kv, seq,
                config_dir
            )

    print(f"\n\nAll done! Heatmaps saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()