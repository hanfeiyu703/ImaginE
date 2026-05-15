"""
Image / region encoder with a unified visual-token interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig


def _load_vision_backbone(model_name: str):
    """Load the appropriate raw-image backbone from Hugging Face."""
    config = AutoConfig.from_pretrained(model_name)
    model_type = getattr(config, "model_type", "")

    if model_type == "clip":
        from transformers import CLIPModel

        full_clip = CLIPModel.from_pretrained(model_name)
        vision_backbone = full_clip.vision_model
        del full_clip.text_model, full_clip.text_projection
        return vision_backbone
    if model_type == "clip_vision_model":
        from transformers import CLIPVisionModel

        return CLIPVisionModel.from_pretrained(model_name)

    from transformers import ViTModel

    return ViTModel.from_pretrained(model_name)


class ImageEncoder(nn.Module):
    """Encode either raw images or VinVL region proposals into shared visual tokens."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        hidden_size: int = 768,
        shared_dim: int = 384,
        freeze_backbone: bool = False,
        visual_backend: str = "raw_image",
        vinvl_feature_dim: int = 2048,
    ):
        super().__init__()
        self.visual_backend = visual_backend

        if visual_backend == "raw_image":
            self.backbone = _load_vision_backbone(model_name)
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            self.projection = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, shared_dim),
            )
            for module in self.backbone.modules():
                if hasattr(module, "position_ids"):
                    module.register_forward_pre_hook(self._clone_position_ids_hook)
        elif visual_backend == "vinvl":
            self.backbone = None
            self.projection = nn.Sequential(
                nn.LayerNorm(vinvl_feature_dim),
                nn.Linear(vinvl_feature_dim, shared_dim),
            )
        else:
            raise ValueError(f"Unsupported visual backend: {visual_backend}")

    @staticmethod
    def _clone_position_ids_hook(module, _args):
        if hasattr(module, "position_ids"):
            module.position_ids = module.position_ids.clone()

    def _forward_raw_image(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state
        z_v = self.projection(hidden[:, 1:, :])
        z_v_cls = self.projection(hidden[:, :1, :]).squeeze(1)
        return z_v, z_v_cls

    def _forward_vinvl(
        self,
        region_features: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_v = self.projection(region_features)
        weights = region_mask.unsqueeze(-1)
        denom = weights.sum(dim=1).clamp(min=1.0)
        z_v_cls = (z_v * weights).sum(dim=1) / denom
        return z_v, z_v_cls

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        region_features: torch.Tensor | None = None,
        region_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z_v: visual tokens (B, N, d_shared)
            z_v_cls: pooled visual representation (B, d_shared)
        """
        if self.visual_backend == "raw_image":
            if pixel_values is None:
                raise ValueError("pixel_values must be provided for raw_image backend.")
            return self._forward_raw_image(pixel_values)

        if region_features is None or region_mask is None:
            raise ValueError("region_features and region_mask must be provided for vinvl backend.")
        return self._forward_vinvl(region_features, region_mask)
