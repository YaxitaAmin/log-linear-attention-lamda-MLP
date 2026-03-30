import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from hattention.base import HType, HStruct, get_num_levels
from hattention.kernel import hattention_kernel
from hattention.mamba_apis import compute_lambda_maybe_fixed
from hattention.base import HType, HStruct
import hattention.modeling_hattention as mh
from hattention.modeling_hattention import (
    MAX_NUM_LEVELS, LAMBDA_LEVEL_BASE, LAMBDA_HTYPE, LAMBDA_HSTRUCT
)
from hattention.lambda_mlp import LambdaMLPSoftplus, LambdaMLPSoftmax


class HAttentionMixer(nn.Module):
    def __init__(self, d_model, num_heads, state_size, num_levels):
        super().__init__()
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.state_size = state_size
        self.num_levels = num_levels
        self.head_dim   = d_model // num_heads

        self.q_proj   = nn.Linear(d_model, num_heads * state_size, bias=False)
        self.k_proj   = nn.Linear(d_model, num_heads * state_size, bias=False)
        self.v_proj   = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.g_proj   = nn.Linear(d_model, num_heads, bias=True)
        self.dl_proj  = nn.Linear(d_model, num_heads * num_levels, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

        self.L = nn.Parameter(torch.ones(num_heads, num_levels))
        self.lambda_fixed = (mh.LAMBDA_MODE_TYPE == "fixed")

        if mh.LAMBDA_MODE_TYPE == "mlp_softplus":
            self.lambda_module = LambdaMLPSoftplus(
                num_levels=num_levels,
                hidden_dim=mh.LAMBDA_MLP_HIDDEN_DIM)
            self.lambda_fixed = False
        elif mh.LAMBDA_MODE_TYPE == "mlp_softmax":
            self.lambda_module = LambdaMLPSoftmax(
                num_levels=num_levels,
                hidden_dim=mh.LAMBDA_MLP_HIDDEN_DIM)
            self.lambda_fixed = False
        else:
            self.lambda_module = None

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, _ = x.shape
        Q  = rearrange(self.q_proj(x),  "b t (h d) -> b t h d", h=self.num_heads)
        K  = rearrange(self.k_proj(x),  "b t (h d) -> b t h d", h=self.num_heads)
        V  = rearrange(self.v_proj(x),  "b t (h d) -> b t h d", h=self.num_heads)
        g  = -torch.nn.functional.softplus(self.g_proj(x))  # must be negative!
        dl = rearrange(self.dl_proj(x), "b t (h l) -> b t h l", h=self.num_heads)

        # compute lambda
        L = compute_lambda_maybe_fixed(
            L=self.L,
            dl=dl,
            lambda_mode="positive",
            lambda_level_max=MAX_NUM_LEVELS,
            lambda_level_fixed=self.lambda_fixed,
            lambda_level_module=self.lambda_module,
        )

        # fast Triton kernel
        Y = hattention_kernel(
            q=Q, k=K, v=V,
            b=None,
            g=g,
            l=L,
            scale=None,
            head_first=False,
            level_base=LAMBDA_LEVEL_BASE,
            htype=LAMBDA_HTYPE,
            hstruct=LAMBDA_HSTRUCT,
        )

        Y = rearrange(Y, "b t h d -> b t (h d)")
        return self.norm(self.out_proj(Y) + x)


class MQARModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, state_size,
                 num_levels, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            HAttentionMixer(d_model, num_heads, state_size, num_levels)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)
