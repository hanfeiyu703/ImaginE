import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    import torch.nn as nn

    from config import ModelConfig
    from evaluate import _apply_fine_rerank, _grounding_prediction
    from models.classifier import EntityClassifier
    from models.imagine_model import ImaginEModel

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class DummyTextBackbone(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.embedding = nn.Embedding(128, hidden_size)

        def forward(self, input_ids, attention_mask):
            del attention_mask
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


    class DummyVisionBackbone(nn.Module):
        def __init__(self, hidden_size: int, num_tokens: int = 3):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_tokens = num_tokens

        def forward(self, pixel_values):
            batch_size = pixel_values.size(0)
            device = pixel_values.device
            return SimpleNamespace(
                last_hidden_state=torch.zeros(
                    batch_size,
                    self.num_tokens + 1,
                    self.hidden_size,
                    device=device,
                )
            )


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class HIndexTigerEnhancementTest(unittest.TestCase):
    def test_region_pointer_masks_invalid_and_all_mask_regions(self):
        classifier = EntityClassifier(
            shared_dim=4,
            num_types=3,
            hidden_dim=6,
            max_span_length=3,
            width_embed_dim=2,
            num_grounding_classes=5,
            num_coarse_types=3,
            use_region_pointer=True,
        )
        batch_size, num_spans, num_types = 2, 3, 3
        outputs = classifier(
            z_span=torch.randn(batch_size, num_spans, 4),
            z_v_cls=torch.randn(batch_size, 4),
            imag_scores=torch.randn(batch_size, num_spans, num_types),
            reverse_scores=torch.randn(batch_size, num_spans, num_types),
            span_widths=torch.ones(batch_size, num_spans, dtype=torch.long),
            region_tokens=torch.randn(batch_size, 4, 4),
            region_mask=torch.tensor(
                [[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
            ),
        )

        grounding_logits = outputs["grounding_logits"]
        self.assertEqual(grounding_logits.shape, (2, 3, 5))
        self.assertTrue((grounding_logits[0, :, 1] < -999.0).all())
        self.assertTrue((grounding_logits[0, :, 3] < -999.0).all())
        self.assertTrue((grounding_logits[1, :, :4] < -999.0).all())
        self.assertTrue(torch.isfinite(grounding_logits[:, :, -1]).all())

    def test_type_aware_region_pointer_accepts_gold_and_predicted_types(self):
        classifier = EntityClassifier(
            shared_dim=4,
            num_types=3,
            hidden_dim=6,
            max_span_length=3,
            width_embed_dim=2,
            num_grounding_classes=5,
            num_coarse_types=3,
            use_region_pointer=True,
            use_type_aware_region_pointer=True,
        )
        common_kwargs = dict(
            z_span=torch.randn(2, 3, 4),
            z_v_cls=torch.randn(2, 4),
            imag_scores=torch.randn(2, 3, 3),
            reverse_scores=torch.randn(2, 3, 3),
            span_widths=torch.ones(2, 3, dtype=torch.long),
            region_tokens=torch.randn(2, 4, 4),
            region_mask=torch.ones(2, 4),
        )

        gold_outputs = classifier(
            **common_kwargs,
            type_ids_for_region=torch.tensor([[1, 2, 1], [2, 1, 0]], dtype=torch.long),
        )
        pred_outputs = classifier(**common_kwargs)

        self.assertEqual(gold_outputs["grounding_logits"].shape, (2, 3, 5))
        self.assertEqual(pred_outputs["grounding_logits"].shape, (2, 3, 5))
        self.assertTrue(torch.isfinite(gold_outputs["grounding_logits"]).all())
        self.assertTrue(torch.isfinite(pred_outputs["grounding_logits"]).all())

    def test_soft_groundable_decode_calibrates_region_and_no_region(self):
        region_boxes = torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [10.0, 10.0, 20.0, 20.0],
                [20.0, 20.0, 30.0, 30.0],
            ]
        )
        region_mask = torch.tensor([1.0, 1.0, 0.0])
        grounding_logits = torch.tensor([0.2, 0.3, 50.0, 0.6])

        high_groundable = _grounding_prediction(
            grounding_logits,
            region_boxes,
            region_mask,
            groundable_logit=torch.tensor(5.0),
            decode_mode="soft_groundable",
        )
        low_groundable = _grounding_prediction(
            grounding_logits,
            region_boxes,
            region_mask,
            groundable_logit=torch.tensor(-5.0),
            decode_mode="soft_groundable",
        )

        self.assertTrue(high_groundable["groundable"])
        self.assertEqual(high_groundable["region_index"], 1)
        self.assertFalse(low_groundable["groundable"])
        self.assertIsNone(low_groundable["region_index"])

    def test_clip_patch_fallback_decodes_when_detector_predicts_no_region(self):
        region_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        region_mask = torch.tensor([1.0])
        grounding_logits = torch.tensor([0.1, 2.0])
        clip_patch_logits = torch.tensor([0.2, 2.5, -1.0])
        clip_patch_boxes = torch.tensor(
            [
                [0.0, 0.0, 5.0, 5.0],
                [5.0, 5.0, 10.0, 10.0],
                [10.0, 10.0, 15.0, 15.0],
            ],
            dtype=torch.float32,
        )
        clip_patch_mask = torch.tensor([1.0, 1.0, 0.0])

        pred = _grounding_prediction(
            grounding_logits,
            region_boxes,
            region_mask,
            groundable_logit=torch.tensor(1.5),
            decode_mode="argmax",
            clip_patch_logits=clip_patch_logits,
            clip_patch_boxes=clip_patch_boxes,
            clip_patch_mask=clip_patch_mask,
            use_clip_patch_fallback=True,
            clip_fallback_threshold=0.5,
        )

        self.assertTrue(pred["groundable"])
        self.assertEqual(pred["region_index"], "clip_patch:1")
        self.assertEqual(pred["box"], [5.0, 5.0, 10.0, 10.0])

    @patch("models.image_encoder._load_vision_backbone")
    @patch("models.text_encoder.AutoModel.from_pretrained")
    def test_hierarchical_fine_logits_use_parent_coarse_prior(
        self,
        mock_text_model,
        mock_vision_loader,
    ):
        mock_text_model.return_value = DummyTextBackbone(hidden_size=8)
        mock_vision_loader.return_value = DummyVisionBackbone(hidden_size=8)
        config = ModelConfig(
            task="fmnerg",
            visual_backend="raw_image",
            text_hidden_size=8,
            image_hidden_size=8,
            shared_dim=4,
            num_types=6,
            num_coarse_types=3,
            max_span_length=3,
            type_embed_dim=4,
            pred_hidden_dim=8,
            rev_pred_hidden_dim=8,
            comparator_hidden_dim=8,
            rev_comparator_hidden_dim=8,
            classifier_hidden_dim=8,
            width_embed_dim=4,
            use_hierarchical_fine_logits=True,
            coarse_prior_weight=0.5,
        )
        model = ImaginEModel(config)
        model.fine_to_coarse_ids = torch.tensor([0, 1, 1, 2, 2, 0])

        fine_logits = torch.zeros(1, 1, 6)
        coarse_logits = torch.tensor([[[2.0, 1.0, -1.0]]])
        calibrated = model._apply_hierarchical_fine_logits(fine_logits, coarse_logits)

        coarse_log_prior = torch.log_softmax(coarse_logits, dim=-1)
        expected = 0.5 * coarse_log_prior[:, :, [0, 1, 1, 2, 2, 0]]
        self.assertTrue(torch.allclose(calibrated - fine_logits, expected, atol=1e-6))

    @patch("models.image_encoder._load_vision_backbone")
    @patch("models.text_encoder.AutoModel.from_pretrained")
    def test_hierarchical_fine_logits_can_use_transition_prior(
        self,
        mock_text_model,
        mock_vision_loader,
    ):
        mock_text_model.return_value = DummyTextBackbone(hidden_size=8)
        mock_vision_loader.return_value = DummyVisionBackbone(hidden_size=8)
        config = ModelConfig(
            task="fmnerg",
            visual_backend="raw_image",
            text_hidden_size=8,
            image_hidden_size=8,
            shared_dim=4,
            num_types=6,
            num_coarse_types=3,
            max_span_length=3,
            type_embed_dim=4,
            pred_hidden_dim=8,
            rev_pred_hidden_dim=8,
            comparator_hidden_dim=8,
            rev_comparator_hidden_dim=8,
            classifier_hidden_dim=8,
            width_embed_dim=4,
            use_hierarchical_fine_logits=True,
            coarse_prior_weight=0.0,
            use_coarse_fine_transition=True,
            transition_prior_weight=0.7,
        )
        model = ImaginEModel(config)
        model.fine_to_coarse_ids = torch.tensor([0, 1, 1, 2, 2, 0])
        model.coarse_to_fine_transition = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.6, 0.4, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.25, 0.75, 0.0],
            ],
            dtype=torch.float32,
        )

        fine_logits = torch.zeros(1, 1, 6)
        coarse_logits = torch.tensor([[[2.0, 1.0, -1.0]]])
        calibrated = model._apply_hierarchical_fine_logits(fine_logits, coarse_logits)

        expected_prior = (
            torch.softmax(coarse_logits, dim=-1)
            @ model.coarse_to_fine_transition
        ).clamp(min=1e-8)
        expected = 0.7 * expected_prior.log()
        self.assertTrue(torch.allclose(calibrated - fine_logits, expected, atol=1e-6))

    def test_fine_rerank_uses_parent_coarse_scores(self):
        fine_logits = torch.zeros(1, 2, 52)
        coarse_logits = torch.zeros(1, 2, 9)
        coarse_logits[0, 0, 1] = 2.0
        coarse_logits[0, 1, 4] = 2.0
        reranked = _apply_fine_rerank(
            fine_logits,
            coarse_logits,
            task="fmnerg",
            lambda_value=0.3,
        )

        self.assertEqual(reranked.shape, fine_logits.shape)
        self.assertTrue(torch.isfinite(reranked).all())
        self.assertFalse(torch.allclose(reranked, fine_logits))


if __name__ == "__main__":
    unittest.main()
