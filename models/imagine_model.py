"""
ImaginE: Full model integrating all modules (Bidirectional Imagination).

Pipeline:
    1. Text Encoder  → token reprs → Span Encoder → z_span
    2. Image Encoder  → patch reprs → z_v, z_v_cls
    --- Forward Imagination (text imagines image) ---
    3a. Imagination Predictor: z_span → z_imag (B, S, K, d)
    4a. Comparator: z_imag vs z_v → scores (B, S, K), z_attended
    --- Reverse Imagination (image imagines text) ---
    3b. Span-Visual Attention: z_span query z_v → z_v_span
    3c. Reverse Imagination: z_v_span → z_imag_text (B, S, K, d)
    4b. Reverse Comparator: z_imag_text vs z_span → reverse_scores (B, S, K)
    --- Classification ---
    5. Classifier: [z_span; z_v_cls; scores; reverse_scores; w] → logits
"""

import torch
import torch.nn as nn

from .text_encoder import TextEncoder
from .image_encoder import ImageEncoder
from .imagination import ImaginationPredictor
from .comparator import ImaginationRealityComparator
from .span_visual_attention import SpanVisualAttention
from .reverse_imagination import ReverseImaginationPredictor
from .reverse_comparator import ReverseComparator
from .classifier import EntityClassifier
from config import ModelConfig


class ImaginEModel(nn.Module):
    """Complete ImaginE model with bidirectional imagination."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # --- Backbone Encoders ---
        self.text_encoder = TextEncoder(
            model_name=config.text_model_name,
            hidden_size=config.text_hidden_size,
            shared_dim=config.shared_dim,
        )
        self.image_encoder = ImageEncoder(
            model_name=config.image_model_name,
            hidden_size=config.image_hidden_size,
            shared_dim=config.shared_dim,
        )

        # --- Forward: Entity Imagination World Model ---
        self.imagination = ImaginationPredictor(
            shared_dim=config.shared_dim,
            type_embed_dim=config.type_embed_dim,
            pred_hidden_dim=config.pred_hidden_dim,
            num_layers=config.pred_num_layers,
            ffn_expansion=config.pred_ffn_expansion,
            dropout=config.pred_dropout,
            num_types=config.num_types,
        )

        # --- Forward: Imagination-Reality Comparator ---
        self.comparator = ImaginationRealityComparator(
            shared_dim=config.shared_dim,
            hidden_dim=config.comparator_hidden_dim,
        )

        # --- Reverse: Span-Visual Attention ---
        self.span_visual_attention = SpanVisualAttention(
            shared_dim=config.shared_dim,
        )

        # --- Reverse: Imagination Predictor (image → imagined text) ---
        shared_type_embed = (
            self.imagination.type_embedding if config.share_type_embedding else None
        )
        self.reverse_imagination = ReverseImaginationPredictor(
            shared_dim=config.shared_dim,
            type_embed_dim=config.type_embed_dim,
            pred_hidden_dim=config.rev_pred_hidden_dim,
            num_layers=config.rev_pred_num_layers,
            ffn_expansion=config.rev_pred_ffn_expansion,
            dropout=config.rev_pred_dropout,
            num_types=config.num_types,
            type_embedding=shared_type_embed,
        )

        # --- Reverse: Comparator (imagined text vs real text) ---
        self.reverse_comparator = ReverseComparator(
            shared_dim=config.shared_dim,
            hidden_dim=config.rev_comparator_hidden_dim,
        )

        # --- Visual Relevance Gate ---
        # Learns to suppress reverse imagination when image is irrelevant.
        # Input: [z_span; z_v_cls; z_span * z_v_cls] → scalar gate ∈ [0, 1]
        self.visual_gate = nn.Sequential(
            nn.Linear(3 * config.shared_dim, config.shared_dim),
            nn.GELU(),
            nn.Linear(config.shared_dim, 1),
        )
        # bias=2 → sigmoid(2)≈0.88: default to "reverse ON", learn to turn off
        nn.init.constant_(self.visual_gate[-1].bias, 2.0)

        # --- Entity Classifier ---
        self.classifier = EntityClassifier(
            shared_dim=config.shared_dim,
            num_types=config.num_types,
            hidden_dim=config.classifier_hidden_dim,
            max_span_length=config.max_span_length,
            width_embed_dim=config.width_embed_dim,
        )

    def init_type_embeddings_from_clip(self, clip_model_name: str | None = None):
        """Initialize imagination type embeddings from CLIP text encoder.

        When type embeddings are shared, this initializes both forward and
        reverse predictors at once.
        """
        if clip_model_name is None:
            clip_model_name = self.config.image_model_name
        self.imagination.init_type_embeddings_from_clip(clip_model_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        span_indices: torch.Tensor,
        span_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids:      (B, L) tokenized text
            attention_mask:  (B, L) attention mask
            pixel_values:    (B, 3, 224, 224) image pixels
            span_indices:    (B, S, 2) span start/end indices (inclusive)
            span_labels:     (B, S) ground-truth type ids (optional, for loss)
        Returns:
            dict with keys:
                logits:             (B, S, K) entity type logits
                z_imag:             (B, S, K, d) imagined visual representations
                z_attended:         (B, S, K, d) attended real visual features
                scores:             (B, S, K) forward imagination-reality scores
                z_imag_text:        (B, S, K, d) imagined textual representations
                reverse_scores:     (B, S, K) reverse imagination-reality scores (ungated)
                visual_relevance:   (B, S) per-span visual relevance gate ∈ [0, 1]
                z_span:             (B, S, d) span representations
                z_v:                (B, N_m, d) visual patch representations
                z_v_cls:            (B, d) global visual feature
                z_v_span:           (B, S, d) span-specific visual features
        """
        B, S, _ = span_indices.shape

        # 1. Encode text → span representations
        z_span, _ = self.text_encoder(input_ids, attention_mask, span_indices)
        # z_span: (B, S, d_shared)

        # 2. Encode image → patch representations
        z_v, z_v_cls = self.image_encoder(pixel_values)
        # z_v: (B, N_m, d_shared), z_v_cls: (B, d_shared)

        # ===== Forward Imagination (text imagines image) =====

        # 3a. For each span, imagine visual for all K types
        z_span_flat = z_span.reshape(B * S, -1)
        z_imag_flat = self.imagination(z_span_flat)  # (B*S, K, d)
        K = z_imag_flat.size(1)
        d = z_imag_flat.size(2)
        z_imag = z_imag_flat.reshape(B, S, K, d)

        # 4a. Compare forward imagination vs reality
        N_m = z_v.size(1)
        z_v_expanded = z_v.repeat_interleave(S, dim=0)  # (B*S, N_m, d)
        scores_flat, z_attended_flat = self.comparator(z_imag_flat, z_v_expanded)
        scores = scores_flat.reshape(B, S, K)
        z_attended = z_attended_flat.reshape(B, S, K, d)

        # ===== Reverse Imagination (image imagines text) =====

        # 3b. Extract span-relevant visual features via cross-attention
        z_v_span = self.span_visual_attention(z_span, z_v)  # (B, S, d)

        # 3c. For each span, imagine textual for all K types
        z_v_span_flat = z_v_span.reshape(B * S, -1)
        z_imag_text_flat = self.reverse_imagination(z_v_span_flat)  # (B*S, K, d)
        z_imag_text = z_imag_text_flat.reshape(B, S, K, d)

        # 4b. Compare reverse imagination vs reality (imagined text vs real span)
        reverse_scores = self.reverse_comparator(
            z_imag_text_flat.reshape(B * S, K, d),
            z_span_flat,
        )  # (B*S, K)
        reverse_scores = reverse_scores.reshape(B, S, K)

        # ===== Visual Relevance Gate =====
        # Suppress reverse scores when image is irrelevant to the text span.
        z_v_exp = z_v_cls.unsqueeze(1).expand(B, S, -1)  # (B, S, d)
        gate_input = torch.cat([z_span, z_v_exp, z_span * z_v_exp], dim=-1)
        visual_relevance = torch.sigmoid(self.visual_gate(gate_input)).squeeze(-1)  # (B, S)
        reverse_scores_gated = reverse_scores * visual_relevance.unsqueeze(-1)  # (B, S, K)

        # 5. Classify: fuse all signals
        span_widths = span_indices[:, :, 1] - span_indices[:, :, 0] + 1  # (B, S)
        logits = self.classifier(
            z_span, z_v_cls, scores, reverse_scores_gated, span_widths
        )  # (B, S, K)

        return {
            "logits": logits,
            "z_imag": z_imag,
            "z_attended": z_attended,
            "scores": scores,
            "z_imag_text": z_imag_text,
            "reverse_scores": reverse_scores,
            "visual_relevance": visual_relevance,
            "z_span": z_span,
            "z_v": z_v,
            "z_v_cls": z_v_cls,
            "z_v_span": z_v_span,
        }

    def get_encoder_params(self) -> list:
        """Parameters of backbone encoders (for lower learning rate)."""
        return list(self.text_encoder.backbone.parameters()) + \
               list(self.image_encoder.backbone.parameters())

    def get_new_module_params(self) -> list:
        """Parameters of new modules (for higher learning rate).

        Deduplicates by tensor id to avoid double-updating shared parameters
        (e.g. type_embedding shared between forward and reverse predictors).
        """
        encoder_param_ids = set(
            id(p) for p in self.get_encoder_params()
        )
        seen: set[int] = set()
        params: list[nn.Parameter] = []
        for p in self.parameters():
            pid = id(p)
            if pid not in encoder_param_ids and p.requires_grad and pid not in seen:
                params.append(p)
                seen.add(pid)
        return params
