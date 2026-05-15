"""
Reverse Imagination Predictor.

Symmetric counterpart to ImaginationPredictor for the reverse direction:
    Forward:  z_span  + e_k → z_imag_visual  (text imagines image)
    Reverse:  z_v_span + e_k → z_imag_text   (image imagines text)

Shares the type embedding table with the forward predictor to enforce
consistent type semantics across both imagination directions.
"""

import torch
import torch.nn as nn

from .adaln_transformer import AdaLNBlock


class ReverseImaginationPredictor(nn.Module):
    """Type-Conditioned Reverse Imagination Predictor.

    Architecture mirrors ImaginationPredictor:
        1. MLP_in:   d_shared -> d_pred  (input projection)
        2. N layers of AdaLNBlock conditioned on e_k
        3. MLP_out:  d_pred -> d_shared  (output projection)
    """

    def __init__(
        self,
        shared_dim: int = 384,
        type_embed_dim: int = 128,
        pred_hidden_dim: int = 512,
        num_layers: int = 4,
        ffn_expansion: int = 2,
        dropout: float = 0.1,
        num_types: int = 5,
        type_embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        self.num_types = num_types
        self.shared_dim = shared_dim

        if type_embedding is not None:
            self.type_embedding = type_embedding
        else:
            self.type_embedding = nn.Embedding(num_types, type_embed_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(shared_dim, pred_hidden_dim),
            nn.GELU(),
        )

        self.layers = nn.ModuleList([
            AdaLNBlock(
                hidden_dim=pred_hidden_dim,
                cond_dim=type_embed_dim,
                ffn_expansion=ffn_expansion,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(pred_hidden_dim),
            nn.Linear(pred_hidden_dim, shared_dim),
        )

    def forward(self, z_v_span: torch.Tensor) -> torch.Tensor:
        """Imagine textual representations for ALL type hypotheses in parallel.

        Args:
            z_v_span: (B, d_shared) span-relevant visual features
        Returns:
            z_imag_text: (B, K, d_shared) imagined text for each of K types
        """
        B = z_v_span.size(0)
        device = z_v_span.device
        K = self.num_types

        z_v_rep = z_v_span.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)

        type_ids = (
            torch.arange(K, device=device)
            .unsqueeze(0)
            .expand(B, K)
            .reshape(B * K)
        )
        e_k = self.type_embedding(type_ids)  # (B*K, d_type)

        h = self.input_proj(z_v_rep)  # (B*K, d_pred)
        for layer in self.layers:
            h = layer(h, e_k)

        z_imag_text = self.output_proj(h)  # (B*K, d_shared)
        return z_imag_text.reshape(B, K, self.shared_dim)
