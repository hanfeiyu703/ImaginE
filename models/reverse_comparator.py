"""
Reverse Comparator.

Compares imagined textual representations against real text span
representations using a 4-way feature combination + MLP scorer.

Unlike the forward Comparator (which uses attention over image patches),
the reverse direction compares against a single span vector directly:

    feat = [z_imag_text_k; z_span; z_imag_text_k - z_span; z_imag_text_k * z_span]
    s_k  = MLP(feat)
"""

import torch
import torch.nn as nn


class ReverseComparator(nn.Module):
    """Computes reverse imagination-reality consistency scores."""

    def __init__(self, shared_dim: int = 384, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(4 * shared_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z_imag_text: torch.Tensor,
        z_span: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_imag_text: (B, K, d) imagined text representations for K types
            z_span:      (B, d) real text span representations
        Returns:
            scores: (B, K) reverse imagination-reality compatibility scores
        """
        K = z_imag_text.size(1)
        z_span_exp = z_span.unsqueeze(1).expand_as(z_imag_text)  # (B, K, d)

        feat = torch.cat([
            z_imag_text,
            z_span_exp,
            z_imag_text - z_span_exp,
            z_imag_text * z_span_exp,
        ], dim=-1)  # (B, K, 4d)

        scores = self.score_mlp(feat).squeeze(-1)  # (B, K)
        return scores
