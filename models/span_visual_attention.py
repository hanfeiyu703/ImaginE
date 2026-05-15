"""
Span-Visual Attention.

Extracts span-relevant visual features via learned cross-attention:
    q = W_q · z_span,  k = W_k · z_v
    z_v_span = softmax(q · k^T / sqrt(d)) · z_v

Each text span "queries" the image patches to obtain a span-specific
visual representation, which serves as the input to the Reverse
Imagination Predictor.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpanVisualAttention(nn.Module):
    """Cross-attention from text spans to image patches with learned Q/K projections."""

    def __init__(self, shared_dim: int = 384, attn_dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(shared_dim, shared_dim)
        self.k_proj = nn.Linear(shared_dim, shared_dim)
        self.scale = math.sqrt(shared_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        z_span: torch.Tensor,
        z_v: torch.Tensor,
        region_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z_span: (B, S, d) text span representations (queries)
            z_v:    (B, N_m, d) image patch embeddings (keys/values)
        Returns:
            z_v_span: (B, S, d) span-specific visual features
        """
        q = self.q_proj(z_span)   # (B, S, d)
        k = self.k_proj(z_v)      # (B, N_m, d)
        attn_logits = torch.bmm(q, k.transpose(1, 2)) / self.scale
        if region_mask is not None:
            mask = (region_mask.unsqueeze(1).expand(-1, z_span.size(1), -1) >= 0.5)
            attn_logits = attn_logits.masked_fill(~mask, -1e4)
            attn_weights = F.softmax(attn_logits, dim=-1)
            attn_weights = attn_weights * mask.to(attn_logits.dtype)
            norm = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            attn_weights = attn_weights / norm
        else:
            attn_weights = F.softmax(attn_logits, dim=-1)  # (B, S, N_m)
        attn_weights = self.attn_dropout(attn_weights)
        z_v_span = torch.bmm(attn_weights, z_v)  # (B, S, d)
        return z_v_span
