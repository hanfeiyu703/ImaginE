"""
Imagination Predictor — the core module of ImaginE.

Analogous to LeWorldModel's predictor:
  LeWM:    z_{t+1} = Predictor(z_t, a_t)
  ImaginE: z_imag_k = Predictor(z_span, e_k)

Given an entity span representation and a candidate type hypothesis,
the predictor "imagines" what the visual representation should look
like under that type assumption, producing a type-specific imagined
visual embedding in the shared latent space.
"""

import logging

import torch
import torch.nn as nn

from .adaln_transformer import AdaLNBlock

logger = logging.getLogger(__name__)


class ImaginationPredictor(nn.Module):
    """Type-Conditioned Imagination Predictor.

    Architecture:
        1. MLP_in:   d_shared -> d_pred  (input projection)
        2. N layers of AdaLNBlock conditioned on e_k
        3. MLP_out:  d_pred -> d_shared  (output projection)

    Operates on (B, D) vectors directly — no sequence dimension needed.
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
    ):
        super().__init__()
        self.num_types = num_types
        self.shared_dim = shared_dim

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

    def forward_single_type(
        self, z_span: torch.Tensor, type_id: int
    ) -> torch.Tensor:
        """Imagine visual representation for a single type hypothesis.

        Args:
            z_span: (B, d_shared) entity span representations
            type_id: integer type index
        Returns:
            z_imag: (B, d_shared) imagined visual representation
        """
        B = z_span.size(0)
        device = z_span.device

        type_ids = torch.full((B,), type_id, dtype=torch.long, device=device)
        e_k = self.type_embedding(type_ids)  # (B, d_type)

        h = self.input_proj(z_span)  # (B, d_pred)
        for layer in self.layers:
            h = layer(h, e_k)

        return self.output_proj(h)  # (B, d_shared)

    @torch.no_grad()
    def init_type_embeddings_from_clip(
        self, clip_model_name: str = "openai/clip-vit-base-patch16"
    ) -> None:
        """Initialize type embeddings using CLIP text encoder.

        Encodes semantic descriptions of each entity type with CLIP and
        projects to type_embed_dim, giving the embedding table a meaningful
        starting point instead of random initialization.
        """
        from transformers import CLIPModel, CLIPTokenizer

        type_descriptions = [
            "not a named entity",
            "a person or human being",
            "a geographical location or place",
            "an organization or company",
            "a miscellaneous named entity like a movie or event",
        ]

        tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        clip_model = CLIPModel.from_pretrained(clip_model_name)

        inputs = tokenizer(
            type_descriptions, padding=True, return_tensors="pt"
        )
        text_features = clip_model.get_text_features(**inputs)  # (K, 512)
        text_features = nn.functional.normalize(text_features, dim=-1)

        clip_dim = text_features.size(-1)
        type_embed_dim = self.type_embedding.embedding_dim

        proj = nn.Linear(clip_dim, type_embed_dim, bias=False)
        nn.init.xavier_uniform_(proj.weight)
        projected = proj(text_features)  # (K, type_embed_dim)

        assert projected.size(0) == self.num_types, (
            f"Expected {self.num_types} type descriptions, got {projected.size(0)}"
        )
        self.type_embedding.weight.data.copy_(projected)

        del clip_model, tokenizer
        logger.info(
            "Initialized type embeddings from CLIP text encoder "
            f"(clip_dim={clip_dim} -> type_embed_dim={type_embed_dim})"
        )

    def forward(self, z_span: torch.Tensor) -> torch.Tensor:
        """Imagine visual representations for ALL type hypotheses in parallel.

        Args:
            z_span: (B, d_shared) entity span representations
        Returns:
            z_imag: (B, K, d_shared) imagined visual for each of K types
        """
        B = z_span.size(0)
        device = z_span.device
        K = self.num_types

        # Repeat span for each type: (B*K, d_shared)
        z_span_rep = z_span.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)

        # Build type ids: (B*K,)
        type_ids = (
            torch.arange(K, device=device)
            .unsqueeze(0)
            .expand(B, K)
            .reshape(B * K)
        )
        e_k = self.type_embedding(type_ids)  # (B*K, d_type)

        h = self.input_proj(z_span_rep)  # (B*K, d_pred)
        for layer in self.layers:
            h = layer(h, e_k)

        z_imag = self.output_proj(h)  # (B*K, d_shared)
        return z_imag.reshape(B, K, self.shared_dim)  # (B, K, d_shared)
