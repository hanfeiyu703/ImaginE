"""
Imagination-Reality Comparator.

For each type hypothesis k, compares the imagined visual z_imag_k against
the real image patches Z_v via attention-based matching:

    alpha_k  = softmax(z_imag_k · Z_v^T / sqrt(d))
    z_attend = alpha_k · Z_v
    s_k      = MLP([z_imag_k; z_attend; z_imag_k - z_attend; z_imag_k ⊙ z_attend])

This produces a scalar compatibility score s_k per type hypothesis.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImaginationRealityComparator(nn.Module):
    """Computes imagination-reality consistency scores for all type hypotheses."""

    def __init__(self, shared_dim: int = 384, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.scale = math.sqrt(shared_dim)
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
        z_imag: torch.Tensor,
        z_v: torch.Tensor,
        region_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_imag: (B, K, d) imagined visual representations for K types
            z_v:    (B, N_m, d) real image patch embeddings
        Returns:
            scores:     (B, K) imagination-reality compatibility scores
            z_attended: (B, K, d) attention-weighted real visual features
        """
        B, K, d = z_imag.shape
        N_m = z_v.size(1)

        # Attention: (B, K, N_m)
        attn_logits = torch.bmm(z_imag, z_v.transpose(1, 2)) / self.scale
        if region_mask is not None:
            mask = (region_mask.unsqueeze(1).expand(-1, K, -1) >= 0.5)
            attn_logits = attn_logits.masked_fill(~mask, -1e4)
            attn_weights = F.softmax(attn_logits, dim=-1)
            attn_weights = attn_weights * mask.to(attn_logits.dtype)
            norm = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            attn_weights = attn_weights / norm
        else:
            attn_weights = F.softmax(attn_logits, dim=-1)  # (B, K, N_m)

        # Attended visual features: (B, K, d)
        z_attended = torch.bmm(attn_weights, z_v)

        # 4-way feature combination for scoring
        feat = torch.cat([
            z_imag,
            z_attended,
            z_imag - z_attended,
            z_imag * z_attended,
        ], dim=-1)  # (B, K, 4d)

        scores = self.score_mlp(feat).squeeze(-1)  # (B, K)

        return scores, z_attended
