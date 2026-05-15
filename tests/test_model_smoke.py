import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    import torch.nn as nn
    from config import ModelConfig
    from losses.imagine_loss import ImaginELoss
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
        def __init__(self, hidden_size: int, num_tokens: int = 5):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_tokens = num_tokens

        def forward(self, pixel_values):
            batch_size = pixel_values.size(0)
            device = pixel_values.device
            base = torch.arange(
                (self.num_tokens + 1) * self.hidden_size,
                device=device,
                dtype=torch.float32,
            ).view(1, self.num_tokens + 1, self.hidden_size)
            return SimpleNamespace(last_hidden_state=base.expand(batch_size, -1, -1))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class ModelSmokeTest(unittest.TestCase):
    @patch("models.image_encoder._load_vision_backbone")
    @patch("models.text_encoder.AutoModel.from_pretrained")
    def test_raw_image_backend_forward_shapes(self, mock_text_model, mock_vision_loader):
        mock_text_model.return_value = DummyTextBackbone(hidden_size=8)
        mock_vision_loader.return_value = DummyVisionBackbone(hidden_size=8, num_tokens=5)

        config = ModelConfig(
            task="mner",
            visual_backend="raw_image",
            text_hidden_size=8,
            image_hidden_size=8,
            shared_dim=4,
            num_types=5,
            max_span_length=3,
            max_regions=4,
            type_embed_dim=4,
            pred_hidden_dim=8,
            rev_pred_hidden_dim=8,
            comparator_hidden_dim=8,
            rev_comparator_hidden_dim=8,
            classifier_hidden_dim=8,
            width_embed_dim=4,
        )
        model = ImaginEModel(config)

        outputs = model(
            input_ids=torch.randint(0, 32, (2, 6)),
            attention_mask=torch.ones(2, 6, dtype=torch.long),
            pixel_values=torch.randn(2, 3, 32, 32),
            span_indices=torch.tensor(
                [[[1, 1], [2, 3], [4, 4]], [[1, 2], [3, 3], [4, 4]]],
                dtype=torch.long,
            ),
        )

        self.assertEqual(outputs["logits"].shape, (2, 3, 5))
        self.assertEqual(outputs["coarse_logits"].shape, (2, 3, 5))
        self.assertEqual(outputs["grounding_logits"].shape, (2, 3, 5))
        self.assertEqual(outputs["groundable_logits"].shape, (2, 3))
        self.assertEqual(outputs["z_v"].shape, (2, 5, 4))

    @patch("models.text_encoder.AutoModel.from_pretrained")
    def test_vinvl_backend_forward_and_masked_attention(self, mock_text_model):
        mock_text_model.return_value = DummyTextBackbone(hidden_size=8)

        config = ModelConfig(
            task="gmner",
            visual_backend="vinvl",
            text_hidden_size=8,
            image_hidden_size=8,
            shared_dim=4,
            vinvl_feature_dim=6,
            num_types=5,
            max_span_length=3,
            max_regions=4,
            type_embed_dim=4,
            pred_hidden_dim=8,
            rev_pred_hidden_dim=8,
            comparator_hidden_dim=8,
            rev_comparator_hidden_dim=8,
            classifier_hidden_dim=8,
            width_embed_dim=4,
        )
        model = ImaginEModel(config)

        outputs = model(
            input_ids=torch.randint(0, 32, (2, 6)),
            attention_mask=torch.ones(2, 6, dtype=torch.long),
            pixel_values=torch.zeros(2, 3, 224, 224),
            region_features=torch.randn(2, 4, 6),
            region_mask=torch.tensor(
                [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
            ),
            span_indices=torch.tensor(
                [[[1, 1], [2, 3], [4, 4]], [[1, 2], [3, 3], [4, 4]]],
                dtype=torch.long,
            ),
        )

        self.assertEqual(outputs["logits"].shape, (2, 3, 5))
        self.assertEqual(outputs["coarse_logits"].shape, (2, 3, 5))
        self.assertEqual(outputs["grounding_logits"].shape, (2, 3, 5))
        self.assertEqual(outputs["groundable_logits"].shape, (2, 3))
        self.assertTrue(torch.isfinite(outputs["scores"]).all())
        self.assertTrue(torch.isfinite(outputs["reverse_scores"]).all())
        self.assertTrue(torch.isfinite(outputs["groundable_logits"]).all())
        self.assertGreater(len(model.get_encoder_params()), 0)
        self.assertGreater(len(model.get_new_module_params()), 0)
        self.assertTrue(
            torch.allclose(
                outputs["z_v_span"][1],
                torch.zeros_like(outputs["z_v_span"][1]),
                atol=1e-6,
            )
        )

    @patch("models.image_encoder._load_vision_backbone")
    @patch("models.text_encoder.AutoModel.from_pretrained")
    def test_vinvl_backend_with_clip_patch_fallback(
        self,
        mock_text_model,
        mock_vision_loader,
    ):
        mock_text_model.return_value = DummyTextBackbone(hidden_size=8)
        mock_vision_loader.return_value = DummyVisionBackbone(hidden_size=8, num_tokens=4)

        config = ModelConfig(
            task="gmner",
            visual_backend="vinvl",
            text_hidden_size=8,
            image_hidden_size=8,
            shared_dim=4,
            vinvl_feature_dim=6,
            num_types=5,
            max_span_length=3,
            max_regions=4,
            type_embed_dim=4,
            pred_hidden_dim=8,
            rev_pred_hidden_dim=8,
            comparator_hidden_dim=8,
            rev_comparator_hidden_dim=8,
            classifier_hidden_dim=8,
            width_embed_dim=4,
            use_region_pointer=True,
            use_clip_patch_fallback=True,
        )
        model = ImaginEModel(config)

        outputs = model(
            input_ids=torch.randint(0, 32, (2, 6)),
            attention_mask=torch.ones(2, 6, dtype=torch.long),
            pixel_values=torch.randn(2, 3, 224, 224),
            region_features=torch.randn(2, 4, 6),
            region_mask=torch.tensor(
                [[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
            ),
            span_indices=torch.tensor(
                [[[1, 1], [2, 3], [4, 4]], [[1, 2], [3, 3], [4, 4]]],
                dtype=torch.long,
            ),
        )

        self.assertEqual(outputs["clip_patch_logits"].shape, (2, 3, 4))
        self.assertTrue(torch.isfinite(outputs["clip_patch_logits"]).all())
        self.assertTrue(torch.isfinite(outputs["grounding_logits"]).all())

    @patch("models.text_encoder.AutoModel.from_pretrained")
    def test_vinvl_backend_with_dream_registers(self, mock_text_model):
        mock_text_model.return_value = DummyTextBackbone(hidden_size=8)

        config = ModelConfig(
            task="gmner",
            visual_backend="vinvl",
            text_hidden_size=8,
            image_hidden_size=8,
            shared_dim=4,
            vinvl_feature_dim=6,
            use_dream_registers=True,
            num_visual_registers=2,
            use_span_type_registers=True,
            num_types=5,
            max_span_length=3,
            max_regions=4,
            type_embed_dim=4,
            pred_hidden_dim=8,
            rev_pred_hidden_dim=8,
            comparator_hidden_dim=8,
            rev_comparator_hidden_dim=8,
            classifier_hidden_dim=8,
            width_embed_dim=4,
        )
        model = ImaginEModel(config)

        outputs = model(
            input_ids=torch.randint(0, 32, (2, 6)),
            attention_mask=torch.ones(2, 6, dtype=torch.long),
            pixel_values=torch.zeros(2, 3, 224, 224),
            region_features=torch.randn(2, 4, 6),
            region_mask=torch.tensor(
                [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
            ),
            span_indices=torch.tensor(
                [[[1, 1], [2, 3], [4, 4]], [[1, 2], [3, 3], [4, 4]]],
                dtype=torch.long,
            ),
        )

        self.assertEqual(outputs["logits"].shape, (2, 3, 5))
        self.assertEqual(outputs["coarse_logits"].shape, (2, 3, 5))
        self.assertEqual(outputs["grounding_logits"].shape, (2, 3, 5))
        self.assertEqual(outputs["groundable_logits"].shape, (2, 3))
        self.assertEqual(outputs["visual_registers"].shape, (2, 2, 4))
        self.assertEqual(outputs["register_summary"].shape, (2, 4))
        self.assertEqual(outputs["z_v"].shape, (2, 4, 4))
        self.assertTrue(torch.isfinite(outputs["logits"]).all())
        self.assertTrue(torch.isfinite(outputs["grounding_logits"]).all())
        self.assertTrue(torch.isfinite(outputs["groundable_logits"]).all())
        self.assertTrue(torch.isfinite(outputs["visual_registers"]).all())
        self.assertTrue(
            torch.allclose(
                outputs["visual_registers"][1],
                torch.zeros_like(outputs["visual_registers"][1]),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs["z_v"][1],
                torch.zeros_like(outputs["z_v"][1]),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs["z_v_span"][1],
                torch.zeros_like(outputs["z_v_span"][1]),
                atol=1e-6,
            )
        )

        grounding_labels = torch.zeros(2, 3, 5)
        grounding_labels[:, :, -1] = 1.0
        groundable_labels = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        loss_dict = ImaginELoss(num_types=5)(
            logits=outputs["logits"],
            z_imag=outputs["z_imag"],
            z_attended=outputs["z_attended"],
            scores=outputs["scores"],
            z_imag_text=outputs["z_imag_text"],
            reverse_scores=outputs["reverse_scores"],
            z_span=outputs["z_span"],
            labels=torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.long),
            span_mask=torch.ones(2, 3, dtype=torch.float32),
            grounding_logits=outputs["grounding_logits"],
            grounding_labels=grounding_labels,
            groundable_logits=outputs["groundable_logits"],
            groundable_labels=groundable_labels,
            register_summary=outputs["register_summary"],
            visual_gate=outputs["visual_relevance"],
            task="gmner",
        )
        self.assertTrue(torch.isfinite(loss_dict["loss"]).item())
        loss_dict["loss"].backward()
        self.assertIsNotNone(model.visual_register_generator.query_tokens.grad)
        self.assertTrue(
            torch.isfinite(model.visual_register_generator.query_tokens.grad).all()
        )
        self.assertIsNotNone(model.span_type_register_block.cross_attn.in_proj_weight.grad)
        self.assertTrue(
            torch.isfinite(
                model.span_type_register_block.cross_attn.in_proj_weight.grad
            ).all()
        )

    @patch("models.text_encoder.AutoModel.from_pretrained")
    def test_knowledge_gate_and_set_aux_forward_shapes(self, mock_text_model):
        mock_text_model.return_value = DummyTextBackbone(hidden_size=8)

        config = ModelConfig(
            task="fmnerg",
            visual_backend="vinvl",
            text_hidden_size=8,
            image_hidden_size=8,
            shared_dim=4,
            vinvl_feature_dim=6,
            num_types=6,
            num_coarse_types=3,
            max_seq_length=8,
            max_span_length=3,
            max_regions=4,
            type_embed_dim=4,
            pred_hidden_dim=8,
            rev_pred_hidden_dim=8,
            comparator_hidden_dim=8,
            rev_comparator_hidden_dim=8,
            classifier_hidden_dim=8,
            width_embed_dim=4,
            use_region_pointer=True,
            use_type_aware_region_pointer=True,
            knowledge_injection="gated_span",
            knowledge_dropout=0.0,
            use_set_prediction_aux=True,
            set_aux_queries=5,
        )
        model = ImaginEModel(config)

        outputs = model(
            input_ids=torch.randint(0, 32, (2, 8)),
            attention_mask=torch.ones(2, 8, dtype=torch.long),
            pixel_values=torch.zeros(2, 3, 224, 224),
            region_features=torch.randn(2, 4, 6),
            region_mask=torch.ones(2, 4),
            span_indices=torch.tensor(
                [[[1, 1], [2, 3], [4, 4]], [[1, 2], [3, 3], [4, 4]]],
                dtype=torch.long,
            ),
            span_labels=torch.tensor([[1, 2, 0], [2, 1, 0]], dtype=torch.long),
            knowledge_input_ids=torch.randint(0, 32, (2, 8)),
            knowledge_attention_mask=torch.tensor(
                [[1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=torch.long,
            ),
        )

        self.assertEqual(outputs["logits"].shape, (2, 3, 6))
        self.assertEqual(outputs["knowledge_gate"].shape, (2, 3))
        self.assertTrue(torch.isfinite(outputs["knowledge_gate"]).all())
        self.assertIsNotNone(outputs["set_aux_outputs"])
        self.assertEqual(outputs["set_aux_outputs"]["type_logits"].shape, (2, 5, 6))
        self.assertEqual(outputs["set_aux_outputs"]["start_logits"].shape, (2, 5, 8))


if __name__ == "__main__":
    unittest.main()
