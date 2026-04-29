"""
live_demo.py  —  Adaptive Memory Decay for Log-Linear Attention
Live inference demo for professor presentation!

Usage:
    python live_demo.py \
        --softplus_ckpt path/to/mlp_softplus_kv8_seq256_seed0/best_model.pt \
        --fixed_ckpt    path/to/fixed_kv8_seq256_seed0/best_model.pt \
        --kv 8 --seq_len 256

Then open http://localhost:7860 in your browser!
"""

import sys
sys.path.insert(0, "/mnt/e/log-linear-attention-lamda-MLP/flame/3rdparty/flash-linear-attention")
sys.path.insert(1, "/mnt/e/log-linear-attention-lamda-MLP")

import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from einops import rearrange
import gradio as gr

from hattention.base import HType, HStruct, get_num_levels
from hattention.kernel import hattention_kernel
from hattention.lambda_mlp import LambdaMLPSoftplus, LambdaMLPSoftmax

# ── constants ────────────────────────────────────────────────────────────────
LEVEL_BASE = 2
HTYPE      = HType.WEAK
HSTRUCT    = HStruct.MAMBA2
VOCAB_SIZE = 128

# ── model (copy from your training code) ─────────────────────────────────────

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
            self.lambda_fixed  = False
        elif lambda_mode == "mlp_softmax":
            self.lambda_module = LambdaMLPSoftmax(num_levels=num_levels, hidden_dim=mlp_hidden_dim)
            self.lambda_fixed  = False
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

        self._captured_lambda = L.detach().cpu()   # capture for visualization!

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


# ── data generation ──────────────────────────────────────────────────────────

def generate_mqar(n_samples, seq_len, num_kv_pairs, vocab_size=128, seed=42):
    np.random.seed(seed)
    half          = vocab_size // 2
    query_start   = seq_len - num_kv_pairs * 2
    input_ids     = np.zeros((n_samples, seq_len), dtype=np.int64)
    labels        = np.full((n_samples, seq_len), -100, dtype=np.int64)
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


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path, lambda_mode, num_levels, model_dim=64, mlp_hidden_dim=64):
    model = MQARModel(
        vocab_size=VOCAB_SIZE,
        d_model=model_dim,
        num_heads=1,
        state_size=model_dim,
        num_levels=num_levels,
        n_layers=2,
        lambda_mode=lambda_mode,
        mlp_hidden_dim=mlp_hidden_dim,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print(f"  loaded {lambda_mode} from {ckpt_path}")
    return model


# ── inference + lambda extraction ────────────────────────────────────────────

def run_inference(model, input_ids, labels, num_kv_pairs, seq_len):
    """Run model, return predictions, accuracy, and captured lambdas per layer."""
    with torch.no_grad():
        logits = model(input_ids)           # (1, T, vocab)

    # compute accuracy on query positions only
    sl   = logits[:, :-1]                   # (1, T-1, vocab)
    sy   = labels[:, 1:]                    # (1, T-1)
    mask = sy != -100
    preds = sl.argmax(-1)

    correct = (preds[mask] == sy[mask]).sum().item()
    total   = mask.sum().item()
    acc     = correct / total * 100 if total > 0 else 0.0

    # decode predictions at query positions
    query_start = seq_len - num_kv_pairs * 2
    pred_values = []
    true_values = []
    for i in range(num_kv_pairs):
        pos = query_start + i * 2          # query key position
        if pos + 1 < seq_len:
            pred_tok = preds[0, pos].item()
            true_tok = sy[0, pos].item() if mask[0, pos] else -1
            pred_values.append(pred_tok)
            true_values.append(true_tok)

    # capture lambdas per layer
    lambdas = []
    for layer in model.layers:
        if layer._captured_lambda is not None:
            L = layer._captured_lambda  # (B, T, H, num_levels)
            # average over batch & heads -> (T, num_levels)
            lambdas.append(L.mean(dim=(0, 2)).numpy())
        else:
            lambdas.append(None)

    return acc, pred_values, true_values, lambdas


# ── plotting ──────────────────────────────────────────────────────────────────

def make_heatmap_figure(lambdas_softplus, lambdas_fixed, seq_len, num_kv_pairs, seed):
    """
    Side-by-side heatmap: mlp-softplus (left) vs fixed (right), both layers.
    Returns a matplotlib figure.
    """
    query_start = seq_len - num_kv_pairs * 2
    n_layers    = len(lambdas_softplus)

    fig = plt.figure(figsize=(18, 4 * n_layers))
    fig.patch.set_facecolor("#0f0f0f")

    gs = gridspec.GridSpec(n_layers, 4, figure=fig,
                           wspace=0.35, hspace=0.5)

    for li in range(n_layers):
        # ── mlp-softplus ──
        ax1 = fig.add_subplot(gs[li, 0:2])
        L_sp = lambdas_softplus[li]         # (T, num_levels)
        im1  = ax1.imshow(
            L_sp.T, aspect="auto",
            cmap="plasma", interpolation="nearest",
            origin="upper"
        )
        ax1.axvline(query_start - 0.5, color="cyan", linewidth=1.5,
                    linestyle="--", label="query region starts")
        ax1.set_title(f"MLP-softplus λ  (layer {li})\nper-token × per-level",
                      color="white", fontsize=12, pad=8)
        ax1.set_xlabel("Token position", color="white", fontsize=10)
        ax1.set_ylabel("Fenwick tree level", color="white", fontsize=10)
        ax1.tick_params(colors="white")
        for spine in ax1.spines.values():
            spine.set_edgecolor("#444")
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.035, pad=0.04)
        cbar1.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar1.ax.yaxis.get_ticklabels(), color="white")
        cbar1.set_label("λ magnitude", color="white", fontsize=9)
        ax1.legend(fontsize=8, loc="upper left",
                   facecolor="#222", edgecolor="#555",
                   labelcolor="white")

        # ── fixed λ ──
        ax2 = fig.add_subplot(gs[li, 2:4])
        if lambdas_fixed[li] is not None:
            L_fx = lambdas_fixed[li]        # (T, num_levels)
            im2  = ax2.imshow(
                L_fx.T, aspect="auto",
                cmap="plasma", interpolation="nearest",
                origin="upper"
            )
        else:
            # fixed mode: show a uniform bar
            dummy = np.ones((1, lambdas_softplus[li].shape[1]))
            im2   = ax2.imshow(dummy, aspect="auto", cmap="plasma",
                               interpolation="nearest")
        ax2.axvline(query_start - 0.5, color="cyan", linewidth=1.5, linestyle="--")
        ax2.set_title(f"Fixed λ  (layer {li})\nuniform across all tokens",
                      color="white", fontsize=12, pad=8)
        ax2.set_xlabel("Token position", color="white", fontsize=10)
        ax2.set_ylabel("Fenwick tree level", color="white", fontsize=10)
        ax2.tick_params(colors="white")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#444")
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.035, pad=0.04)
        cbar2.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="white")
        cbar2.set_label("λ magnitude", color="white", fontsize=9)

    fig.suptitle(
        f"Adaptive vs Fixed λ  |  kv={num_kv_pairs}  seq={seq_len}  seed={seed}",
        color="white", fontsize=15, fontweight="bold", y=1.01
    )
    return fig


# ── global state (loaded once) ────────────────────────────────────────────────
MODELS = {}   # filled in main()
ARGS   = None


# ── gradio callback ───────────────────────────────────────────────────────────

def run_demo(num_kv_pairs_choice, seq_len_choice, seed_choice):
    kv      = int(num_kv_pairs_choice)
    seq_len = int(seq_len_choice)
    seed    = int(seed_choice)

    # pick whatever checkpoint is loaded that matches kv/seq best
    model_sp  = MODELS.get("softplus")
    model_fx  = MODELS.get("fixed")

    # generate a fresh mqar sample
    x, labels = generate_mqar(1, seq_len, kv, seed=seed)

    # ── run softplus inference ──
    acc_sp, pred_sp, true_sp, lam_sp = run_inference(
        model_sp, x, labels, kv, seq_len
    )

    # ── run fixed inference ──
    acc_fx, pred_fx, true_fx, lam_fx = run_inference(
        model_fx, x, labels, kv, seq_len
    )

    # ── build pretty result text ──
    lines = []
    lines.append("=" * 52)
    lines.append(f"  MQAR task  |  kv={kv}  seq_len={seq_len}  seed={seed}")
    lines.append("=" * 52)
    lines.append("")
    lines.append(f"  MLP-softplus accuracy : {acc_sp:.1f}%")
    lines.append(f"  Fixed-λ    accuracy   : {acc_fx:.1f}%")
    lines.append("")
    lines.append("  Query predictions (softplus):")
    for i, (p, t) in enumerate(zip(pred_sp, true_sp)):
        status = "✓" if p == t else "✗"
        lines.append(f"    query {i+1}: predicted={p:3d}  true={t:3d}  {status}")
    lines.append("")
    lines.append("  Query predictions (fixed-λ):")
    for i, (p, t) in enumerate(zip(pred_fx, true_fx)):
        status = "✓" if p == t else "✗"
        lines.append(f"    query {i+1}: predicted={p:3d}  true={t:3d}  {status}")
    lines.append("")
    lines.append("  Key insight: MLP-softplus learns WHICH Fenwick")
    lines.append("  tree level to weight per token. Fixed-λ cannot!")
    lines.append("=" * 52)
    result_text = "\n".join(lines)

    # ── build heatmap figure ──
    fig = make_heatmap_figure(lam_sp, lam_fx, seq_len, kv, seed)

    return result_text, fig


# ── gradio ui ─────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="Adaptive Memory Decay — Live Demo",
        theme=gr.themes.Base(
            primary_hue="red",
            neutral_hue="slate",
        ),
        css="""
        .gr-box { border-radius: 10px !important; }
        #title  { text-align: center; color: #e63946; font-size: 2em;
                  font-weight: bold; margin-bottom: 0.2em; }
        #sub    { text-align: center; color: #aaa; margin-bottom: 1.5em; }
        """
    ) as demo:

        gr.Markdown("# 🔬 Adaptive Memory Decay for Log-Linear Attention", elem_id="title")
        gr.Markdown(
            "**University of Maryland** — Yaxita Amin, Helen Li, Mengfan Zhang  \n"
            "Live inference: MLP-softplus λ vs Fixed λ on the MQAR task",
            elem_id="sub"
        )

        with gr.Row():
            kv_slider  = gr.Radio(
                choices=["4", "8", "16", "32"],
                value="8",
                label="Number of KV pairs",
                info="Higher = harder recall task"
            )
            seq_slider = gr.Radio(
                choices=["128", "256", "512"],
                value="256",
                label="Sequence length",
                info="Longer = fixed λ collapses more"
            )
            seed_input = gr.Slider(
                minimum=0, maximum=99, step=1, value=42,
                label="Sample seed",
                info="Different seed = different MQAR sample"
            )

        run_btn = gr.Button("▶  Run Inference", variant="primary", size="lg")

        with gr.Row():
            result_box = gr.Textbox(
                label="Inference Results",
                lines=20,
                max_lines=30,
                show_copy_button=True,
            )

        heatmap_plot = gr.Plot(label="λ Heatmaps: MLP-Softplus (left) vs Fixed (right)")

        run_btn.click(
            fn=run_demo,
            inputs=[kv_slider, seq_slider, seed_input],
            outputs=[result_box, heatmap_plot],
        )

        gr.Markdown(
            "> **What to look at:** The left heatmap (MLP-softplus) shows λ varying "
            "meaningfully across tokens and Fenwick tree levels — the model learned "
            "to *selectively* weight memory. The right heatmap (Fixed λ) is nearly "
            "uniform — it cannot adapt to input content."
        )

    return demo


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    global MODELS, ARGS

    parser = argparse.ArgumentParser(description="Live demo for adaptive lambda MQAR")
    parser.add_argument("--softplus_ckpt", type=str, required=True,
                        help="Path to best_model.pt for mlp_softplus")
    parser.add_argument("--fixed_ckpt",    type=str, required=True,
                        help="Path to best_model.pt for fixed lambda")
    parser.add_argument("--kv",            type=int, default=8,
                        help="kv pairs the checkpoints were trained with")
    parser.add_argument("--seq_len",       type=int, default=256,
                        help="seq_len the checkpoints were trained with")
    parser.add_argument("--model_dim",     type=int, default=64)
    parser.add_argument("--mlp_hidden",    type=int, default=64)
    parser.add_argument("--port",          type=int, default=7860)
    ARGS = parser.parse_args()

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    num_levels = get_num_levels(ARGS.seq_len, base=LEVEL_BASE)

    print(f"\nDevice      : {device}")
    print(f"Num levels  : {num_levels}")
    print(f"Loading checkpoints...\n")

    MODELS["softplus"] = load_model(
        ARGS.softplus_ckpt, "mlp_softplus", num_levels,
        ARGS.model_dim, ARGS.mlp_hidden
    )
    MODELS["fixed"] = load_model(
        ARGS.fixed_ckpt, "fixed", num_levels,
        ARGS.model_dim, ARGS.mlp_hidden
    )

    print("\nModels loaded! Starting Gradio demo...\n")
    demo = build_ui()
    demo.launch(server_port=ARGS.port, share=False, inbrowser=True)


if __name__ == "__main__":
    main()