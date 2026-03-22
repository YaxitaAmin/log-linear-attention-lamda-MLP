import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaMLPSoftplus(nn.Module):
    """
    MLP-parameterized lambda with softplus activation.
    Replaces the linear lambda projection in Log-Linear Attention.

    Architecture:
        h_t = GELU(dl_t @ W1 + b1)
        lambda_t = softplus(h_t @ W2 + b2)

    Input dl has shape (batch, seqlen, nheads, num_levels) — the
    input-dependent projection already computed by in_proj.
    W2 rows learn per-level importance, replacing the static L parameter.

    Identity initialization: bias of W2 set so softplus(0.54) ≈ 1.0,
    making MLP output ≈ 1.0 at start of training — identical to baseline.
    """

    def __init__(self, num_levels: int, hidden_dim: int):
        super().__init__()
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim

        self.W1 = nn.Linear(num_levels, hidden_dim, bias=True)
        self.W2 = nn.Linear(hidden_dim, num_levels, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.zeros_(self.W1.bias)
        nn.init.zeros_(self.W2.weight)
        nn.init.constant_(self.W2.bias, 0.54)

    def forward(self, dl: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dl: (batch, seqlen, nheads, num_levels)  [training]
                (batch, nheads, num_levels)           [decoding]
        Returns:
            lambda: same shape as dl
        """
        h = F.gelu(self.W1(dl))
        return F.softplus(self.W2(h))


class LambdaMLPSoftmax(nn.Module):
    """
    MLP-parameterized lambda with softmax-across-levels activation.
    Forces model to trade off between memory levels explicitly.

    Architecture:
        h_t = GELU(dl_t @ W1 + b1)
        lambda_t = softmax(h_t @ W2 + b2)

    Uniform initialization: all levels get equal weight at start,
    model learns to differentiate as training progresses.
    """

    def __init__(self, num_levels: int, hidden_dim: int):
        super().__init__()
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim

        self.W1 = nn.Linear(num_levels, hidden_dim, bias=True)
        self.W2 = nn.Linear(hidden_dim, num_levels, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.zeros_(self.W1.bias)
        nn.init.zeros_(self.W2.weight)
        nn.init.zeros_(self.W2.bias)

    def forward(self, dl: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dl: (batch, seqlen, nheads, num_levels)  [training]
                (batch, nheads, num_levels)           [decoding]
        Returns:
            lambda: same shape as dl, sums to 1 across levels dim
        """
        h = F.gelu(self.W1(dl))
        return F.softmax(self.W2(h), dim=-1)
