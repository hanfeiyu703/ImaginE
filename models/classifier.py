"""
Shared fusion trunk with separate entity-typing and grounding heads.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class EntityClassifier(nn.Module):
    """Fuse multimodal span features, then branch into type and grounding heads."""

    def __init__(
        self,
        shared_dim: int = 384,
        num_types: int = 5,
        hidden_dim: int = 256,
        max_span_length: int = 5,
        width_embed_dim: int = 32,
        num_grounding_classes: int = 37,
        num_coarse_types: int = 5,
        use_region_pointer: bool = False,
        use_type_aware_region_pointer: bool = False,
        use_knowledge_gate: bool = False,
        knowledge_dropout: float = 0.2,
        knowledge_gate_init: float = -2.0,
    ):
        super().__init__()
        self.num_grounding_classes = num_grounding_classes
        self.use_region_pointer = use_region_pointer
        self.use_type_aware_region_pointer = use_type_aware_region_pointer
        self.use_knowledge_gate = use_knowledge_gate
        self.width_embedding = nn.Embedding(max_span_length + 2, width_embed_dim)
        input_dim = 2 * shared_dim + 2 * num_types + width_embed_dim

        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.type_head = nn.Linear(hidden_dim, num_types)
        self.coarse_head = nn.Linear(hidden_dim, num_coarse_types)
        self.grounding_head = nn.Linear(hidden_dim, num_grounding_classes)
        self.region_query = nn.Linear(hidden_dim, shared_dim)
        self.region_type_embedding = (
            nn.Embedding(num_types, shared_dim)
            if use_type_aware_region_pointer
            else None
        )
        self.patch_query = nn.Linear(hidden_dim, shared_dim)
        self.no_region_head = nn.Linear(hidden_dim, 1)
        self.groundable_head = nn.Linear(hidden_dim, 1)
        if use_knowledge_gate:
            self.knowledge_dropout = nn.Dropout(knowledge_dropout)
            self.knowledge_adapter = nn.Linear(shared_dim, hidden_dim)
            self.knowledge_gate = nn.Linear(hidden_dim + shared_dim, 1)
            nn.init.constant_(self.knowledge_gate.bias, knowledge_gate_init)
        else:
            self.knowledge_dropout = None
            self.knowledge_adapter = None
            self.knowledge_gate = None

    def _region_pointer_logits(
        self,
        fused_states: torch.Tensor,
        region_tokens: torch.Tensor | None,
        region_mask: torch.Tensor | None,
        type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Point from each fused span state to VinVL regions plus a no-region class."""
        if region_tokens is None:
            return self.grounding_head(fused_states)

        batch_size, num_spans, _ = fused_states.shape
        max_region_classes = self.num_grounding_classes - 1
        if max_region_classes < 1:
            return self.grounding_head(fused_states)

        region_tokens = region_tokens[:, :max_region_classes]
        num_regions = region_tokens.size(1)

        query = self.region_query(fused_states)
        if (
            self.use_type_aware_region_pointer
            and self.region_type_embedding is not None
            and type_ids is not None
        ):
            safe_type_ids = type_ids.clamp(0, self.region_type_embedding.num_embeddings - 1)
            query = query + self.region_type_embedding(safe_type_ids)
        region_logits = torch.matmul(query, region_tokens.transpose(1, 2))
        region_logits = region_logits / math.sqrt(max(query.size(-1), 1))

        if region_mask is None or region_mask.shape[:2] != region_tokens.shape[:2]:
            valid_mask = torch.ones(
                batch_size,
                num_regions,
                dtype=torch.bool,
                device=fused_states.device,
            )
        else:
            valid_mask = region_mask[:, :num_regions].to(device=fused_states.device) >= 0.5

        invalid_value = -1e4
        region_logits = region_logits.masked_fill(
            ~valid_mask.unsqueeze(1),
            invalid_value,
        )

        if num_regions < max_region_classes:
            pad = region_logits.new_full(
                (batch_size, num_spans, max_region_classes - num_regions),
                invalid_value,
            )
            region_logits = torch.cat([region_logits, pad], dim=-1)

        no_region_logit = self.no_region_head(fused_states)
        return torch.cat([region_logits, no_region_logit], dim=-1)

    def _clip_patch_logits(
        self,
        fused_states: torch.Tensor,
        patch_tokens: torch.Tensor | None,
        patch_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Point from each fused span state to CLIP patch tokens."""
        if patch_tokens is None:
            return None

        query = self.patch_query(fused_states)
        patch_logits = torch.matmul(query, patch_tokens.transpose(1, 2))
        patch_logits = patch_logits / math.sqrt(max(query.size(-1), 1))

        if patch_mask is not None and patch_mask.shape[:2] == patch_tokens.shape[:2]:
            valid_mask = patch_mask.to(device=fused_states.device) >= 0.5
            patch_logits = patch_logits.masked_fill(~valid_mask.unsqueeze(1), -1e4)
        return patch_logits

    def forward(
        self,
        z_span: torch.Tensor,
        z_v_cls: torch.Tensor,
        imag_scores: torch.Tensor,
        reverse_scores: torch.Tensor,
        span_widths: torch.Tensor,
        region_tokens: torch.Tensor | None = None,
        region_mask: torch.Tensor | None = None,
        clip_patch_tokens: torch.Tensor | None = None,
        clip_patch_mask: torch.Tensor | None = None,
        type_ids_for_region: torch.Tensor | None = None,
        knowledge_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        num_spans = z_span.size(1)
        z_v_expanded = z_v_cls.unsqueeze(1).expand(-1, num_spans, -1)
        width_embed = self.width_embedding(
            span_widths.clamp(0, self.width_embedding.num_embeddings - 1)
        )

        fused = torch.cat(
            [z_span, z_v_expanded, imag_scores, reverse_scores, width_embed], dim=-1
        )
        fused_states = self.fusion(fused)
        task_states = fused_states
        knowledge_gate = None
        if (
            self.use_knowledge_gate
            and knowledge_context is not None
            and self.knowledge_adapter is not None
            and self.knowledge_gate is not None
        ):
            knowledge_context = knowledge_context.to(dtype=fused_states.dtype)
            knowledge_hidden = self.knowledge_adapter(
                self.knowledge_dropout(knowledge_context)
                if self.knowledge_dropout is not None
                else knowledge_context
            )
            knowledge_gate = torch.sigmoid(
                self.knowledge_gate(torch.cat([fused_states, knowledge_context], dim=-1))
            )
            task_states = fused_states + knowledge_gate * knowledge_hidden

        type_logits = self.type_head(task_states)
        coarse_logits = self.coarse_head(task_states)
        region_type_ids = type_ids_for_region
        if self.use_type_aware_region_pointer and region_type_ids is None:
            region_type_ids = type_logits.detach().argmax(dim=-1)
        if self.use_region_pointer:
            grounding_logits = self._region_pointer_logits(
                fused_states,
                region_tokens,
                region_mask,
                type_ids=region_type_ids,
            )
        else:
            grounding_logits = self.grounding_head(fused_states)
        return {
            "type_logits": type_logits,
            "coarse_logits": coarse_logits,
            "grounding_logits": grounding_logits,
            "groundable_logits": self.groundable_head(task_states).squeeze(-1),
            "fused_states": fused_states,
            "knowledge_gate": (
                knowledge_gate.squeeze(-1)
                if knowledge_gate is not None
                else None
            ),
            "clip_patch_logits": self._clip_patch_logits(
                fused_states,
                clip_patch_tokens,
                clip_patch_mask,
            ),
        }
