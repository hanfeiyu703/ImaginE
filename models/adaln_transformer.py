"""
Adaptive Layer Normalization (AdaLN) Block.

Inspired by LeWorldModel's predictor architecture. AdaLN injects the type
hypothesis embedding into each layer via learned affine parameters (gamma, beta),
analogous to how LeWM injects actions.

Key design: AdaLN MLP weights are zero-initialized so type conditioning
takes effect progressively during training, improving stability.

Since ImaginE's Imagination Predictor operates on single-vector spans (N=1),
we use a direct (B, D) interface with a linear mixing layer instead of MHSA.
Self-attention with N=1 degenerates to identity, wasting ~4M parameters.
"""

import torch
import torch.nn as nn


class AdaLN(nn.Module):
    """Adaptive Layer Normalization conditioned on type hypothesis."""

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) hidden features
            cond: (B, C) conditioning signal (type hypothesis embedding)
        Returns:
            (B, D) normalized and modulated features
        """
        params = self.proj(cond)  # (B, 2D)
        gamma, beta = params.chunk(2, dim=-1)  # each (B, D)
        return (1 + gamma) * self.norm(x) + beta


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class AdaLNBlock(nn.Module):
    """Single AdaLN-conditioned block operating on (B, D) vectors.

    Architecture per block:
        h' = AdaLN(h, cond)
        h  = h + Linear(h')
        h  = h + FFN(LN(h))

    This replaces the full Transformer block (which includes MHSA) because
    ImaginE's predictor operates on single vectors per span (N=1), making
    self-attention degenerate to identity + linear.
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        ffn_expansion: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.adaln = AdaLN(hidden_dim, cond_dim)
        self.linear_mix = nn.Linear(hidden_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ffn_expansion, dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) hidden states
            cond: (B, C) type hypothesis conditioning
        Returns:
            (B, D) updated hidden states
        """
        h = self.adaln(x, cond)
        h = x + self.dropout1(self.linear_mix(h))
        h = h + self.ffn(self.norm2(h))
        return h


# Backward-compatible alias
AdaLNTransformerBlock = AdaLNBlock
