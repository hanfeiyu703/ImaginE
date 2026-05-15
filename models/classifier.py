"""
Entity Classifier.

Fuses five signal sources for the final entity type prediction:
    1. z_span   — textual span representation (from text understanding)
    2. z_v_cls  — global visual feature (from passive visual observation)
    3. s_fwd    — forward imagination-reality scores (text imagines image)
    4. s_rev    — reverse imagination-reality scores (image imagines text)
    5. w        — span width embedding (encodes span length information)

    h_final = MLP_fusion([z_span; z_v_cls; s_fwd; s_rev; w])
    y_hat   = softmax(MLP_cls(h_final))
"""

import torch
import torch.nn as nn


class EntityClassifier(nn.Module):
    """MLP-based entity type classifier with five-way signal fusion."""

    def __init__(
        self,
        shared_dim: int = 384,
        num_types: int = 5,
        hidden_dim: int = 256,
        max_span_length: int = 5,
        width_embed_dim: int = 32,
    ):
        super().__init__()
        self.width_embedding = nn.Embedding(max_span_length + 2, width_embed_dim)
        # z_span + z_v_cls + fwd_scores + rev_scores + width_embed
        input_dim = 2 * shared_dim + 2 * num_types + width_embed_dim

        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.cls_head = nn.Linear(hidden_dim, num_types)

    def forward(
        self,
        z_span: torch.Tensor,
        z_v_cls: torch.Tensor,
        imag_scores: torch.Tensor,
        reverse_scores: torch.Tensor,
        span_widths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_span:         (B, num_spans, d) textual span representations
            z_v_cls:        (B, d) global visual feature — broadcast over spans
            imag_scores:    (B, num_spans, K) forward imagination-reality scores
            reverse_scores: (B, num_spans, K) reverse imagination-reality scores
            span_widths:    (B, num_spans) span widths in tokens
        Returns:
            logits: (B, num_spans, K) classification logits
        """
        num_spans = z_span.size(1)

        z_v_expanded = z_v_cls.unsqueeze(1).expand(-1, num_spans, -1)
        w_embed = self.width_embedding(
            span_widths.clamp(0, self.width_embedding.num_embeddings - 1)
        )

        fused = torch.cat(
            [z_span, z_v_expanded, imag_scores, reverse_scores, w_embed], dim=-1
        )

        h = self.fusion(fused)  # (B, num_spans, hidden_dim)
        logits = self.cls_head(h)  # (B, num_spans, K)
        return logits
