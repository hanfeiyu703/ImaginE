"""
Text Encoder: RoBERTa-base + Span Encoder + Projection.

The span representation follows PAR-MNER:
    H_(s,e) = [avg_pool(h_s:h_e); h_e - h_s; h_s ⊙ h_e]

This produces a 3*d_t dimensional vector per span, then projected to d_shared.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from utils import hf_local_files_only


def _gather_tokens(
    token_reprs: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Gather token representations at given indices.

    Args:
        token_reprs: (B, L, D)
        indices: (B, S) token positions to gather
    Returns:
        (B, S, D)
    """
    indices = indices.clamp(0, token_reprs.size(1) - 1)
    expanded = indices.unsqueeze(-1).expand(-1, -1, token_reprs.size(-1))
    return torch.gather(token_reprs, 1, expanded)


class SpanAvgPoolFast(nn.Module):
    """Vectorized span average pooling using cumulative sums."""

    def forward(
        self,
        token_reprs: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            token_reprs: (B, L, D)
            starts: (B, S) span start indices
            ends: (B, S) span end indices (inclusive)
        Returns:
            (B, S, D) averaged span representations
        """
        B, L, D = token_reprs.shape
        S = starts.size(1)

        starts = starts.clamp(0, L - 1)
        ends = ends.clamp(0, L - 1)

        # Cumulative sum for efficient range averaging
        pad = torch.zeros(B, 1, D, device=token_reprs.device)
        cumsum = torch.cat([pad, token_reprs.cumsum(dim=1)], dim=1)  # (B, L+1, D)

        ends_p1 = (ends + 1).clamp(max=L)  # (B, S)
        lengths = (ends_p1 - starts).clamp(min=1).unsqueeze(-1).float()  # (B, S, 1)

        # Gather cumsum values
        s_idx = starts.unsqueeze(-1).expand(B, S, D)
        e_idx = ends_p1.unsqueeze(-1).expand(B, S, D)

        sum_e = torch.gather(cumsum, 1, e_idx)  # (B, S, D)
        sum_s = torch.gather(cumsum, 1, s_idx)  # (B, S, D)

        return (sum_e - sum_s) / lengths


class TextEncoder(nn.Module):
    """RoBERTa backbone + Span Encoder + Projection to shared space."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        hidden_size: int = 768,
        shared_dim: int = 384,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            local_files_only=hf_local_files_only(),
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.span_avg_fast = SpanAvgPoolFast()
        self.projection = nn.Sequential(
            nn.LayerNorm(3 * hidden_size),
            nn.Linear(3 * hidden_size, shared_dim),
            nn.Dropout(0.2),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (B, L)
            attention_mask: (B, L)
            span_indices: (B, num_spans, 2) start/end inclusive
        Returns:
            z_span: (B, num_spans, d_shared) projected span representations
            token_reprs: (B, L, d_text) raw token representations
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_reprs = outputs.last_hidden_state  # (B, L, 768)

        B, num_spans, _ = span_indices.shape
        D = token_reprs.size(-1)

        starts = span_indices[:, :, 0]
        ends = span_indices[:, :, 1]

        h_start = _gather_tokens(token_reprs, starts)
        h_end = _gather_tokens(token_reprs, ends)
        h_avg = self.span_avg_fast(token_reprs, starts, ends)

        span_reprs = torch.cat(
            [h_avg, h_end - h_start, h_start * h_end], dim=-1
        )  # (B, num_spans, 3*768)

        z_span = self.projection(span_reprs)  # (B, num_spans, d_shared)
        return z_span, token_reprs
