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
import torch.nn.functional as F

from .text_encoder import TextEncoder
from .image_encoder import ImageEncoder
from .imagination import ImaginationPredictor
from .comparator import ImaginationRealityComparator
from .span_visual_attention import SpanVisualAttention
from .reverse_imagination import ReverseImaginationPredictor
from .reverse_comparator import ReverseComparator
from .classifier import EntityClassifier
from .dream_registers import (
    RegisterAugmentedVisualBlock,
    SpanTypeConditionedRegisterBlock,
    VisualRegisterGenerator,
)
from config import (
    ModelConfig,
    get_coarse_to_fine_transition,
    get_fine_to_coarse_ids,
)


def _compatible_num_heads(embed_dim: int) -> int:
    for num_heads in (8, 6, 4, 3, 2):
        if embed_dim % num_heads == 0:
            return num_heads
    return 1


class ImaginEModel(nn.Module):
    """Complete ImaginE model with bidirectional imagination."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.use_dream_registers = (
            config.use_dream_registers and config.visual_backend == "vinvl"
        )
        self.use_span_type_registers = (
            self.use_dream_registers and config.use_span_type_registers
        )
        self.use_hierarchical_fine_logits = (
            config.task == "fmnerg" and config.use_hierarchical_fine_logits
        )
        self.use_clip_patch_fallback = (
            config.visual_backend == "vinvl" and config.use_clip_patch_fallback
        )
        self.use_knowledge_gate = config.knowledge_injection == "gated_span"
        self.use_set_prediction_aux = config.use_set_prediction_aux

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
            visual_backend=config.visual_backend,
            vinvl_feature_dim=config.vinvl_feature_dim,
        )
        if self.use_clip_patch_fallback:
            self.clip_patch_encoder = ImageEncoder(
                model_name=config.image_model_name,
                hidden_size=config.image_hidden_size,
                shared_dim=config.shared_dim,
                freeze_backbone=True,
                visual_backend="raw_image",
                vinvl_feature_dim=config.vinvl_feature_dim,
            )
        else:
            self.clip_patch_encoder = None

        if self.use_knowledge_gate:
            self.knowledge_projection = nn.Sequential(
                nn.LayerNorm(config.text_hidden_size),
                nn.Linear(config.text_hidden_size, config.shared_dim),
            )
            self.knowledge_attention = nn.MultiheadAttention(
                config.shared_dim,
                num_heads=_compatible_num_heads(config.shared_dim),
                dropout=config.pred_dropout,
                batch_first=True,
            )
            self.knowledge_norm = nn.LayerNorm(config.shared_dim)
        else:
            self.knowledge_projection = None
            self.knowledge_attention = None
            self.knowledge_norm = None

        # --- DreamPRVR-style global visual registers (VinVL regions only) ---
        if self.use_dream_registers:
            self.visual_register_generator = VisualRegisterGenerator(
                shared_dim=config.shared_dim,
                num_registers=config.num_visual_registers,
                dropout=config.pred_dropout,
            )
            self.register_augmented_visual = RegisterAugmentedVisualBlock(
                shared_dim=config.shared_dim,
                dropout=config.pred_dropout,
            )
            if self.use_span_type_registers:
                self.span_type_register_block = SpanTypeConditionedRegisterBlock(
                    shared_dim=config.shared_dim,
                    dropout=config.pred_dropout,
                )
            else:
                self.span_type_register_block = None
        else:
            self.visual_register_generator = None
            self.register_augmented_visual = None
            self.span_type_register_block = None

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
            num_grounding_classes=config.max_regions + 1,
            num_coarse_types=config.num_coarse_types,
            use_region_pointer=(
                config.use_region_pointer and config.visual_backend == "vinvl"
            ),
            use_type_aware_region_pointer=(
                config.use_type_aware_region_pointer
                and config.use_region_pointer
                and config.visual_backend == "vinvl"
            ),
            use_knowledge_gate=self.use_knowledge_gate,
            knowledge_dropout=getattr(config, "knowledge_dropout", 0.2),
            knowledge_gate_init=config.knowledge_gate_init,
        )
        if self.use_set_prediction_aux:
            self.set_queries = nn.Embedding(config.set_aux_queries, config.shared_dim)
            self.set_context = nn.Linear(2 * config.shared_dim, config.shared_dim)
            self.set_hidden = nn.Sequential(
                nn.LayerNorm(config.shared_dim),
                nn.Linear(config.shared_dim, config.shared_dim),
                nn.GELU(),
                nn.Dropout(config.pred_dropout),
            )
            self.set_start_head = nn.Linear(config.shared_dim, config.max_seq_length)
            self.set_end_head = nn.Linear(config.shared_dim, config.max_seq_length)
            self.set_type_head = nn.Linear(config.shared_dim, config.num_types)
            self.set_grounding_head = nn.Linear(
                config.shared_dim,
                config.max_regions + 1,
            )
        else:
            self.set_queries = None
            self.set_context = None
            self.set_hidden = None
            self.set_start_head = None
            self.set_end_head = None
            self.set_type_head = None
            self.set_grounding_head = None
        fine_to_coarse = torch.tensor(
            get_fine_to_coarse_ids(config.task),
            dtype=torch.long,
        )
        self.register_buffer("fine_to_coarse_ids", fine_to_coarse, persistent=False)
        coarse_to_fine = torch.tensor(
            get_coarse_to_fine_transition(config.task),
            dtype=torch.float,
        )
        self.register_buffer("coarse_to_fine_transition", coarse_to_fine, persistent=False)

    def _apply_hierarchical_fine_logits(
        self,
        fine_logits: torch.Tensor,
        coarse_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        """Inject TIGER-style coarse priors into FMNERG fine-type logits."""
        if (
            not self.use_hierarchical_fine_logits
            or coarse_logits is None
            or self.fine_to_coarse_ids.numel() != fine_logits.size(-1)
            or int(self.fine_to_coarse_ids.max().item()) >= coarse_logits.size(-1)
        ):
            return fine_logits

        if (
            self.config.use_coarse_fine_transition
            and self.config.transition_prior_weight > 0
            and self.coarse_to_fine_transition.shape
            == (coarse_logits.size(-1), fine_logits.size(-1))
        ):
            transition = self.coarse_to_fine_transition.to(
                device=fine_logits.device,
                dtype=torch.float32,
            )
            coarse_probs = F.softmax(coarse_logits.float(), dim=-1)
            fine_prior = torch.matmul(coarse_probs, transition).clamp(min=1e-8)
            return fine_logits + self.config.transition_prior_weight * fine_prior.log().to(
                fine_logits.dtype
            )

        if self.config.coarse_prior_weight <= 0:
            return fine_logits

        coarse_log_prior = F.log_softmax(coarse_logits.float(), dim=-1).to(fine_logits.dtype)
        parent_ids = self.fine_to_coarse_ids.to(fine_logits.device)
        gathered_prior = coarse_log_prior.gather(
            dim=-1,
            index=parent_ids.view(1, 1, -1).expand(
                fine_logits.size(0),
                fine_logits.size(1),
                -1,
            ),
        )
        return fine_logits + self.config.coarse_prior_weight * gathered_prior

    def _build_knowledge_context(
        self,
        z_span: torch.Tensor,
        knowledge_input_ids: torch.Tensor | None,
        knowledge_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Let span states softly read independently encoded caption/knowledge tokens."""
        if (
            not self.use_knowledge_gate
            or knowledge_input_ids is None
            or knowledge_attention_mask is None
            or self.knowledge_projection is None
            or self.knowledge_attention is None
            or self.knowledge_norm is None
        ):
            return None

        knowledge_attention_mask = knowledge_attention_mask.to(device=z_span.device)
        knowledge_outputs = self.text_encoder.backbone(
            input_ids=knowledge_input_ids.to(device=z_span.device),
            attention_mask=knowledge_attention_mask,
        )
        knowledge_tokens = self.knowledge_projection(knowledge_outputs.last_hidden_state)

        valid_mask = knowledge_attention_mask > 0
        has_knowledge = valid_mask.any(dim=1)
        effective_mask = valid_mask.clone()
        if not bool(has_knowledge.all().item()):
            effective_mask[~has_knowledge, 0] = True
        knowledge_tokens = knowledge_tokens * effective_mask.unsqueeze(-1).to(
            dtype=knowledge_tokens.dtype
        )
        context, _ = self.knowledge_attention(
            query=z_span,
            key=knowledge_tokens,
            value=knowledge_tokens,
            key_padding_mask=~effective_mask,
            need_weights=False,
        )
        context = self.knowledge_norm(context)
        context = context * has_knowledge.to(dtype=context.dtype).view(-1, 1, 1)
        return context

    def _build_set_aux_outputs(
        self,
        z_span: torch.Tensor,
        z_v_cls: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        if (
            not self.use_set_prediction_aux
            or self.set_queries is None
            or self.set_context is None
            or self.set_hidden is None
        ):
            return None

        batch_size = z_span.size(0)
        sample_context = torch.cat([z_span.mean(dim=1), z_v_cls], dim=-1)
        sample_context = self.set_context(sample_context).unsqueeze(1)
        query_states = self.set_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        hidden = self.set_hidden(query_states + sample_context)
        return {
            "start_logits": self.set_start_head(hidden),
            "end_logits": self.set_end_head(hidden),
            "type_logits": self.set_type_head(hidden),
            "grounding_logits": self.set_grounding_head(hidden),
        }

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
        pixel_values: torch.Tensor | None,
        span_indices: torch.Tensor,
        region_features: torch.Tensor | None = None,
        region_mask: torch.Tensor | None = None,
        span_labels: torch.Tensor | None = None,
        knowledge_input_ids: torch.Tensor | None = None,
        knowledge_attention_mask: torch.Tensor | None = None,
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
                groundable_logits:  (B, S) grounding existence logits
                coarse_logits:      (B, S, C) coarse type logits
        """
        B, S, _ = span_indices.shape

        # 1. Encode text → span representations
        z_span, _ = self.text_encoder(input_ids, attention_mask, span_indices)
        # z_span: (B, S, d_shared)

        # 2. Encode image → patch representations
        z_v, z_v_cls = self.image_encoder(
            pixel_values=pixel_values,
            region_features=region_features,
            region_mask=region_mask,
        )
        clip_patch_tokens = None
        clip_patch_mask = None
        if self.clip_patch_encoder is not None and pixel_values is not None:
            clip_patch_tokens, _clip_cls = self.clip_patch_encoder(pixel_values=pixel_values)
            clip_patch_mask = torch.ones(
                clip_patch_tokens.size()[:2],
                dtype=torch.float32,
                device=clip_patch_tokens.device,
            )
        # z_v: (B, N_m, d_shared), z_v_cls: (B, d_shared)
        visual_region_mask = region_mask
        if visual_region_mask is not None and visual_region_mask.size(1) != z_v.size(1):
            visual_region_mask = None

        visual_registers = None
        register_summary = None
        if self.use_dream_registers:
            visual_registers = self.visual_register_generator(z_v, visual_region_mask)
            z_v = self.register_augmented_visual(z_v, visual_registers, visual_region_mask)
            if visual_region_mask is not None:
                weights = visual_region_mask.to(z_v.dtype).unsqueeze(-1)
                denom = weights.sum(dim=1).clamp(min=1.0)
                z_v_cls = (z_v * weights).sum(dim=1) / denom
            else:
                z_v_cls = z_v.mean(dim=1)
            register_summary = visual_registers.mean(dim=1)

        # ===== Forward Imagination (text imagines image) =====

        # 3a. For each span, imagine visual for all K types
        z_span_flat = z_span.reshape(B * S, -1)
        z_imag_flat = self.imagination(z_span_flat)  # (B*S, K, d)
        K = z_imag_flat.size(1)
        d = z_imag_flat.size(2)
        z_imag = z_imag_flat.reshape(B, S, K, d)
        if self.span_type_register_block is not None and visual_registers is not None:
            z_imag = self.span_type_register_block(
                z_imag,
                z_v,
                visual_registers,
                visual_region_mask,
            )
            z_imag_flat = z_imag.reshape(B * S, K, d)

        # 4a. Compare forward imagination vs reality
        N_m = z_v.size(1)
        z_v_expanded = z_v.repeat_interleave(S, dim=0)  # (B*S, N_m, d)
        region_mask_expanded = None
        if visual_region_mask is not None:
            region_mask_expanded = visual_region_mask.repeat_interleave(S, dim=0)
        scores_flat, z_attended_flat = self.comparator(
            z_imag_flat,
            z_v_expanded,
            region_mask=region_mask_expanded,
        )
        scores = scores_flat.reshape(B, S, K)
        z_attended = z_attended_flat.reshape(B, S, K, d)

        # ===== Reverse Imagination (image imagines text) =====

        # 3b. Extract span-relevant visual features via cross-attention
        z_v_span = self.span_visual_attention(z_span, z_v, region_mask=visual_region_mask)  # (B, S, d)

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
        knowledge_context = self._build_knowledge_context(
            z_span,
            knowledge_input_ids,
            knowledge_attention_mask,
        )
        classifier_outputs = self.classifier(
            z_span,
            z_v_cls,
            scores,
            reverse_scores_gated,
            span_widths,
            region_tokens=z_v,
            region_mask=visual_region_mask,
            clip_patch_tokens=clip_patch_tokens,
            clip_patch_mask=clip_patch_mask,
            type_ids_for_region=(
                span_labels
                if (
                    self.config.use_type_aware_region_pointer
                    and span_labels is not None
                )
                else None
            ),
            knowledge_context=knowledge_context,
        )
        logits = self._apply_hierarchical_fine_logits(
            classifier_outputs["type_logits"],
            classifier_outputs["coarse_logits"],
        )
        grounding_logits = classifier_outputs["grounding_logits"]
        groundable_logits = classifier_outputs["groundable_logits"]
        coarse_logits = classifier_outputs["coarse_logits"]
        set_aux_outputs = self._build_set_aux_outputs(z_span, z_v_cls)

        return {
            "logits": logits,
            "grounding_logits": grounding_logits,
            "groundable_logits": groundable_logits,
            "coarse_logits": coarse_logits,
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
            "visual_registers": visual_registers,
            "register_summary": register_summary,
            "fused_states": classifier_outputs["fused_states"],
            "knowledge_gate": classifier_outputs.get("knowledge_gate"),
            "clip_patch_logits": classifier_outputs.get("clip_patch_logits"),
            "set_aux_outputs": set_aux_outputs,
        }

    def get_encoder_params(self) -> list:
        """Parameters of backbone encoders (for lower learning rate)."""
        params = list(self.text_encoder.backbone.parameters())
        if self.image_encoder.backbone is not None:
            params.extend(list(self.image_encoder.backbone.parameters()))
        if self.clip_patch_encoder is not None and self.clip_patch_encoder.backbone is not None:
            params.extend(list(self.clip_patch_encoder.backbone.parameters()))
        return params

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
