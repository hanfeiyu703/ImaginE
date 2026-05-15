"""
Image Encoder: ViT/CLIP-ViT + Projection to shared space.

Supports both standard ViT (google/vit-base-patch16-224) and
CLIP ViT (openai/clip-vit-base-patch16). Both produce 197 patch
embeddings (196 patches + 1 CLS) with hidden_size 768.

A linear projection maps from d_v (768) to d_shared.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig

from utils import hf_local_files_only


def _load_vision_backbone(model_name: str):
    """Load the appropriate vision model based on the model config type."""
    local_files_only = hf_local_files_only()
    config = AutoConfig.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model_type = getattr(config, "model_type", "")

    if model_type == "clip":
        from transformers import CLIPModel
        full_clip = CLIPModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        vision_backbone = full_clip.vision_model
        del full_clip.text_model, full_clip.text_projection
        return vision_backbone
    elif model_type == "clip_vision_model":
        from transformers import CLIPVisionModel
        return CLIPVisionModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
    else:
        from transformers import ViTModel
        return ViTModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )


class ImageEncoder(nn.Module):
    """ViT/CLIP-ViT backbone + Projection to shared latent space."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        hidden_size: int = 768,
        shared_dim: int = 384,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = _load_vision_backbone(model_name)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, shared_dim),
        )

        for module in self.backbone.modules():
            if hasattr(module, 'position_ids'):
                module.register_forward_pre_hook(self._clone_position_ids_hook)

    @staticmethod
    def _clone_position_ids_hook(module, args):
        if hasattr(module, 'position_ids'):
            module.position_ids = module.position_ids.clone()

    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_values: (B, 3, 224, 224)
        Returns:
            z_v: (B, N_patches, d_shared)  — spatial patch embeddings (no CLS)
            z_v_cls: (B, d_shared)          — CLS token (global visual feature)
        """
        outputs = self.backbone(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state  # (B, 197, 768)

        z_v = self.projection(hidden[:, 1:, :])               # (B, 196, d_shared) patches only
        z_v_cls = self.projection(hidden[:, :1, :]).squeeze(1) # (B, d_shared) CLS only

        return z_v, z_v_cls
