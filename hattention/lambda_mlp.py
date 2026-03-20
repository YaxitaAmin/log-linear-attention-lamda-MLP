import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LambdaMLPSoftplus(nn.Module):
    """
    MLP-parameterized lambda with softplus activation.
    Replaces the linear lambda projection in Log-Linear Attention.
    
    Architecture:
        h_t = GELU(x_t @ W1 + b1)
        lambda_t = softplus(h_t @ W2 + b2)
    
    Identity initialization ensures lambda ≈ 1.0 at start of training,
    making MLP initially identical to the baseline.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_levels: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels

        self.W1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W2 = nn.Linear(hidden_dim, num_levels, bias=True)

        self._init_weights()

    def _init_weights(self):
        # Standard init for W1
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.zeros_(self.W1.bias)

        # Identity init for W2: bias set so softplus output ≈ 1.0 at start
        # softplus(0.54) ≈ 1.0
        nn.init.zeros_(self.W2.weight)
        nn.init.constant_(self.W2.bias, 0.54)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seqlen, nheads, input_dim)
        Returns:
            lambda: (batch, seqlen, nheads, num_levels)
        """
        h = F.gelu(self.W1(x))
        return F.softplus(self.W2(h))


class LambdaMLPSoftmax(nn.Module):
    """
    MLP-parameterized lambda with softmax-across-levels activation.
    Forces the model to trade off between memory levels.
    
    Architecture:
        h_t = GELU(x_t @ W1 + b1)
        lambda_t = softmax(h_t @ W2 + b2)  softmax across levels
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_levels: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels

        self.W1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W2 = nn.Linear(hidden_dim, num_levels, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.zeros_(self.W1.bias)
        # Uniform init for softmax: all levels equal weight at start
        nn.init.zeros_(self.W2.weight)
        nn.init.zeros_(self.W2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seqlen, nheads, input_dim)
        Returns:
            lambda: (batch, seqlen, nheads, num_levels)
        """
        h = F.gelu(self.W1(x))
        return F.softmax(self.W2(h), dim=-1)