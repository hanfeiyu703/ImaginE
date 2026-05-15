"""
DreamPRVR-style lightweight visual registers.

The modules here keep the first-stage idea small: derive a few global semantic
registers from VinVL regions, then let region tokens read those registers before
the normal ImaginE comparator/grounding path.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _pick_num_heads(shared_dim: int) -> int:
    """Pick a small valid attention-head count for smoke-test dimensions too."""
    for num_heads in (8, 4, 2):
        if shared_dim % num_heads == 0:
            return num_heads
    return 1


def _valid_region_mask(
    region_tokens: torch.Tensor,
    region_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Return a boolean (B, N) valid-token mask; mismatched masks become all-valid."""
    batch_size, num_regions, _ = region_tokens.shape
    if (
        region_mask is None
        or region_mask.size(0) != batch_size
        or region_mask.size(1) != num_regions
    ):
        return torch.ones(
            batch_size,
            num_regions,
            dtype=torch.bool,
            device=region_tokens.device,
        )
    return region_mask.to(device=region_tokens.device) >= 0.5


class VisualRegisterGenerator(nn.Module):
    """Generate global visual registers via learnable queries over region tokens."""

    def __init__(
        self,
        shared_dim: int = 384,
        num_registers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_registers < 1:
            raise ValueError("num_registers must be at least 1.")
        self.num_registers = num_registers
        num_heads = _pick_num_heads(shared_dim)

        self.query_tokens = nn.Parameter(
            torch.randn(1, num_registers, shared_dim) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            shared_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(shared_dim)
        self.ffn = nn.Sequential(
            nn.Linear(shared_dim, shared_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim * 2, shared_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(shared_dim)

    def forward(
        self,
        region_tokens: torch.Tensor,
        region_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            region_tokens: (B, N, d) VinVL region tokens
            region_mask:   (B, N) valid-region mask
        Returns:
            registers: (B, R, d)
        """
        batch_size = region_tokens.size(0)
        valid_mask = _valid_region_mask(region_tokens, region_mask)
        has_valid = valid_mask.any(dim=1)

        safe_mask = valid_mask.clone()
        safe_mask[~has_valid] = True
        key_padding_mask = ~safe_mask

        queries = self.query_tokens.expand(batch_size, -1, -1)
        attended, _ = self.cross_attn(
            query=queries,
            key=region_tokens,
            value=region_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        registers = self.attn_norm(queries + attended)
        registers = self.ffn_norm(registers + self.ffn(registers))

        return registers * has_valid.to(registers.dtype).view(batch_size, 1, 1)


class RegisterAugmentedVisualBlock(nn.Module):
    """Enhance region tokens by letting them attend to regions plus global registers."""

    def __init__(
        self,
        shared_dim: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        num_heads = _pick_num_heads(shared_dim)
        self.region_attn = nn.MultiheadAttention(
            shared_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(shared_dim)
        self.ffn = nn.Sequential(
            nn.Linear(shared_dim, shared_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim * 2, shared_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(shared_dim)

    def forward(
        self,
        region_tokens: torch.Tensor,
        visual_registers: torch.Tensor,
        region_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            region_tokens:    (B, N, d) original region tokens
            visual_registers: (B, R, d) generated global registers
            region_mask:      (B, N) valid-region mask
        Returns:
            enhanced region tokens: (B, N, d)
        """
        batch_size, _, _ = region_tokens.shape
        valid_mask = _valid_region_mask(region_tokens, region_mask)
        has_valid = valid_mask.any(dim=1)

        safe_mask = valid_mask.clone()
        safe_mask[~has_valid] = True
        register_mask = torch.zeros(
            batch_size,
            visual_registers.size(1),
            dtype=torch.bool,
            device=region_tokens.device,
        )
        key_padding_mask = torch.cat([~safe_mask, register_mask], dim=1)

        memory = torch.cat([region_tokens, visual_registers], dim=1)
        attended, _ = self.region_attn(
            query=region_tokens,
            key=memory,
            value=memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        enhanced = self.attn_norm(region_tokens + attended)
        enhanced = self.ffn_norm(enhanced + self.ffn(enhanced))

        return enhanced * valid_mask.to(enhanced.dtype).unsqueeze(-1)


class SpanTypeConditionedRegisterBlock(nn.Module):
    """Let each span/type query read region tokens plus global visual registers."""

    def __init__(
        self,
        shared_dim: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        num_heads = _pick_num_heads(shared_dim)
        self.cross_attn = nn.MultiheadAttention(
            shared_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(shared_dim)
        self.ffn = nn.Sequential(
            nn.Linear(shared_dim, shared_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim * 2, shared_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(shared_dim)

    def forward(
        self,
        span_type_queries: torch.Tensor,
        region_tokens: torch.Tensor,
        visual_registers: torch.Tensor,
        region_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            span_type_queries: (B, S, K, d) span/type imagination queries
            region_tokens:      (B, N, d) region tokens after global-register update
            visual_registers:   (B, R, d) global visual registers
            region_mask:        (B, N) valid-region mask
        Returns:
            conditioned queries: (B, S, K, d)
        """
        batch_size, num_spans, num_types, hidden_dim = span_type_queries.shape
        valid_mask = _valid_region_mask(region_tokens, region_mask)
        has_valid = valid_mask.any(dim=1)

        safe_mask = valid_mask.clone()
        safe_mask[~has_valid] = True
        register_mask = torch.zeros(
            batch_size,
            visual_registers.size(1),
            dtype=torch.bool,
            device=region_tokens.device,
        )
        key_padding_mask = torch.cat([~safe_mask, register_mask], dim=1)
        memory = torch.cat([region_tokens, visual_registers], dim=1)

        queries = span_type_queries.reshape(
            batch_size * num_spans,
            num_types,
            hidden_dim,
        )
        memory_expanded = memory.repeat_interleave(num_spans, dim=0)
        key_padding_mask = key_padding_mask.repeat_interleave(num_spans, dim=0)

        attended, _ = self.cross_attn(
            query=queries,
            key=memory_expanded,
            value=memory_expanded,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        enhanced = self.attn_norm(queries + attended)
        enhanced = self.ffn_norm(enhanced + self.ffn(enhanced))

        valid_gate = has_valid.repeat_interleave(num_spans).view(-1, 1, 1)
        conditioned = torch.where(valid_gate, enhanced, queries)
        return conditioned.reshape(batch_size, num_spans, num_types, hidden_dim)
