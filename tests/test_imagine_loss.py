import unittest

try:
    import torch
    from losses.imagine_loss import ImaginELoss

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class ImagineLossRegisterTest(unittest.TestCase):
    def setUp(self):
        self.criterion = ImaginELoss(
            num_types=5,
            register_weight=0.1,
            qsp_weight=0.05,
        )

    def _assert_finite_scalar(self, value):
        self.assertEqual(value.dim(), 0)
        self.assertTrue(torch.isfinite(value).item())

    def test_auxiliary_losses_without_entities_are_finite(self):
        z_span = torch.randn(2, 3, 4)
        labels = torch.zeros(2, 3, dtype=torch.long)
        span_mask = torch.ones(2, 3, dtype=torch.float32)
        register_summary = torch.randn(2, 4)

        l_register = self.criterion.register_alignment_loss(
            register_summary,
            z_span,
            labels,
            span_mask,
        )
        l_qsp = self.criterion.query_similarity_preservation_loss(
            z_span,
            labels,
            span_mask,
        )

        self._assert_finite_scalar(l_register)
        self._assert_finite_scalar(l_qsp)
        self.assertAlmostEqual(l_register.item(), 0.0, places=6)
        self.assertAlmostEqual(l_qsp.item(), 0.0, places=6)

    def test_auxiliary_losses_with_single_entity_are_finite(self):
        z_span = torch.randn(1, 3, 4)
        labels = torch.tensor([[1, 0, 0]], dtype=torch.long)
        span_mask = torch.ones(1, 3, dtype=torch.float32)
        register_summary = torch.randn(1, 4)

        l_register = self.criterion.register_alignment_loss(
            register_summary,
            z_span,
            labels,
            span_mask,
        )
        l_qsp = self.criterion.query_similarity_preservation_loss(
            z_span,
            labels,
            span_mask,
        )

        self._assert_finite_scalar(l_register)
        self._assert_finite_scalar(l_qsp)

    def test_register_loss_skips_zero_register_summary(self):
        z_span = torch.randn(2, 3, 4)
        labels = torch.tensor([[1, 0, 0], [2, 0, 0]], dtype=torch.long)
        span_mask = torch.ones(2, 3, dtype=torch.float32)
        register_summary = torch.zeros(2, 4)

        l_register = self.criterion.register_alignment_loss(
            register_summary,
            z_span,
            labels,
            span_mask,
        )

        self._assert_finite_scalar(l_register)
        self.assertAlmostEqual(l_register.item(), 0.0, places=6)

    def test_auxiliary_losses_with_multiple_entities_are_finite(self):
        z_span = torch.randn(2, 4, 4)
        labels = torch.tensor(
            [[1, 2, 0, 0], [3, 4, 0, 0]],
            dtype=torch.long,
        )
        span_mask = torch.ones(2, 4, dtype=torch.float32)
        register_summary = torch.randn(2, 4)

        l_register = self.criterion.register_alignment_loss(
            register_summary,
            z_span,
            labels,
            span_mask,
        )
        l_qsp = self.criterion.query_similarity_preservation_loss(
            z_span,
            labels,
            span_mask,
        )

        self._assert_finite_scalar(l_register)
        self._assert_finite_scalar(l_qsp)
        self.assertGreaterEqual(l_register.item(), 0.0)
        self.assertGreaterEqual(l_qsp.item(), 0.0)

    def test_groundable_loss_uses_detector_miss_style_labels(self):
        groundable_logits = torch.tensor([[0.8, -0.4, 0.0]], requires_grad=True)
        groundable_labels = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        labels = torch.tensor([[1, 2, 0]], dtype=torch.long)
        span_mask = torch.ones(1, 3, dtype=torch.float32)

        l_groundable = self.criterion.groundable_loss(
            groundable_logits,
            groundable_labels,
            labels,
            span_mask,
        )

        self._assert_finite_scalar(l_groundable)
        self.assertGreater(l_groundable.item(), 0.0)
        l_groundable.backward()
        self.assertIsNotNone(groundable_logits.grad)

    def test_grounding_loss_skips_detector_miss_region_kl(self):
        grounding_logits = torch.randn(1, 2, 4, requires_grad=True)
        grounding_labels = torch.zeros(1, 2, 4)
        grounding_labels[:, :, -1] = 1.0
        labels = torch.tensor([[1, 2]], dtype=torch.long)
        span_mask = torch.ones(1, 2, dtype=torch.float32)
        groundable_labels = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

        l_ground = self.criterion.grounding_loss(
            grounding_logits,
            grounding_labels,
            labels,
            span_mask,
            groundable_labels=groundable_labels,
        )

        plain = self.criterion.grounding_loss(
            grounding_logits,
            grounding_labels,
            labels,
            span_mask,
        )

        self._assert_finite_scalar(l_ground)
        self._assert_finite_scalar(plain)
        self.assertAlmostEqual(l_ground.item(), 0.0, places=6)
        self.assertGreater(plain.item(), 0.0)

    def test_no_region_consistency_default_off_and_finite(self):
        criterion = ImaginELoss(num_types=5, no_region_consistency_weight=0.0)
        grounding_logits = torch.randn(2, 3, 4, requires_grad=True)
        groundable_logits = torch.randn(2, 3, requires_grad=True)
        labels = torch.tensor([[1, 0, 2], [0, 3, 0]], dtype=torch.long)
        span_mask = torch.ones(2, 3, dtype=torch.float32)

        loss = criterion.no_region_consistency_loss(
            grounding_logits,
            groundable_logits,
            labels,
            span_mask,
        )

        self._assert_finite_scalar(loss)
        self.assertGreaterEqual(loss.item(), 0.0)
        (criterion.no_region_consistency_weight * loss).backward()
        if grounding_logits.grad is not None:
            self.assertTrue(torch.allclose(grounding_logits.grad, torch.zeros_like(grounding_logits.grad)))

    def test_clip_patch_grounding_loss_is_finite(self):
        criterion = ImaginELoss(num_types=5, clip_patch_weight=0.1)
        clip_patch_logits = torch.randn(1, 3, 4, requires_grad=True)
        clip_patch_labels = torch.tensor(
            [[[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        labels = torch.tensor([[1, 2, 0]], dtype=torch.long)
        span_mask = torch.ones(1, 3, dtype=torch.float32)

        loss = criterion.clip_patch_grounding_loss(
            clip_patch_logits,
            clip_patch_labels,
            labels,
            span_mask,
        )

        self._assert_finite_scalar(loss)
        self.assertGreaterEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(clip_patch_logits.grad)

    def test_cb_focal_task_loss_is_finite(self):
        criterion = ImaginELoss(
            num_types=4,
            fine_loss_type="cb_focal",
            fine_class_counts=[20, 2, 5, 1],
            fine_focal_gamma=1.5,
            fine_class_balance_beta=0.999,
        )
        logits = torch.randn(2, 3, 4, requires_grad=True)
        labels = torch.tensor([[0, 1, 2], [3, 0, 0]], dtype=torch.long)
        span_mask = torch.ones(2, 3, dtype=torch.float32)

        loss = criterion.task_loss(logits, labels, span_mask)

        self._assert_finite_scalar(loss)
        self.assertGreaterEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_region_hard_negative_loss_is_finite(self):
        criterion = ImaginELoss(num_types=5, region_hard_negative_weight=0.05)
        grounding_logits = torch.tensor(
            [[[0.2, 1.0, -0.5, 0.1], [1.2, 0.1, -0.1, 0.0]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        grounding_labels = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        labels = torch.tensor([[1, 2]], dtype=torch.long)
        span_mask = torch.ones(1, 2, dtype=torch.float32)

        loss = criterion.region_hard_negative_loss(
            grounding_logits,
            grounding_labels,
            labels,
            span_mask,
        )

        self._assert_finite_scalar(loss)
        self.assertGreaterEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(grounding_logits.grad)

    def test_set_prediction_aux_loss_handles_empty_and_entity_cases(self):
        criterion = ImaginELoss(num_types=4, set_aux_weight=0.05)
        set_outputs = {
            "start_logits": torch.randn(2, 5, 8, requires_grad=True),
            "end_logits": torch.randn(2, 5, 8, requires_grad=True),
            "type_logits": torch.randn(2, 5, 4, requires_grad=True),
            "grounding_logits": torch.randn(2, 5, 3, requires_grad=True),
        }
        labels = torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.long)
        span_indices = torch.tensor(
            [[[1, 2], [3, 3], [0, 0]], [[0, 0], [0, 0], [0, 0]]],
            dtype=torch.long,
        )
        span_mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        grounding_labels = torch.zeros(2, 3, 3)
        grounding_labels[:, :, -1] = 1.0

        loss = criterion.set_prediction_aux_loss(
            set_outputs,
            labels,
            span_indices,
            span_mask,
            grounding_labels=grounding_labels,
        )

        self._assert_finite_scalar(loss)
        self.assertGreaterEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(set_outputs["type_logits"].grad)

    def test_forward_returns_register_components_for_gmner(self):
        batch_size, num_spans, num_types, dim = 2, 3, 5, 4
        logits = torch.randn(batch_size, num_spans, num_types, requires_grad=True)
        z_imag = torch.randn(batch_size, num_spans, num_types, dim, requires_grad=True)
        z_attended = torch.randn(batch_size, num_spans, num_types, dim)
        scores = torch.randn(batch_size, num_spans, num_types, requires_grad=True)
        z_imag_text = torch.randn(batch_size, num_spans, num_types, dim, requires_grad=True)
        reverse_scores = torch.randn(batch_size, num_spans, num_types, requires_grad=True)
        z_span = torch.randn(batch_size, num_spans, dim, requires_grad=True)
        labels = torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.long)
        span_mask = torch.ones(batch_size, num_spans, dtype=torch.float32)
        register_summary = torch.randn(batch_size, dim, requires_grad=True)
        groundable_logits = torch.randn(batch_size, num_spans, requires_grad=True)
        groundable_labels = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float32,
        )

        loss_dict = self.criterion(
            logits=logits,
            z_imag=z_imag,
            z_attended=z_attended,
            scores=scores,
            z_imag_text=z_imag_text,
            reverse_scores=reverse_scores,
            z_span=z_span,
            labels=labels,
            span_mask=span_mask,
            groundable_logits=groundable_logits,
            groundable_labels=groundable_labels,
            grounding_logits=torch.randn(batch_size, num_spans, 6, requires_grad=True),
            grounding_labels=torch.softmax(torch.randn(batch_size, num_spans, 6), dim=-1),
            register_summary=register_summary,
            task="gmner",
        )

        self._assert_finite_scalar(loss_dict["loss"])
        self._assert_finite_scalar(loss_dict["l_groundable"])
        self._assert_finite_scalar(loss_dict["l_register"])
        self._assert_finite_scalar(loss_dict["l_qsp"])
        self._assert_finite_scalar(loss_dict["l_no_region_consistency"])
        self._assert_finite_scalar(loss_dict["l_clip_patch"])
        self._assert_finite_scalar(loss_dict["l_coarse"])
        self._assert_finite_scalar(loss_dict["l_coarse_fine"])

    def test_fmnerg_coarse_fine_losses_are_finite(self):
        fine_to_coarse = [0, 1, 1, 2, 2, 3]
        criterion = ImaginELoss(
            num_types=6,
            num_coarse_types=4,
            fine_to_coarse_ids=fine_to_coarse,
            coarse_weight=0.2,
            coarse_fine_weight=0.1,
        )
        batch_size, num_spans, num_types, dim = 2, 3, 6, 4
        logits = torch.randn(batch_size, num_spans, num_types, requires_grad=True)
        coarse_logits = torch.randn(batch_size, num_spans, 4, requires_grad=True)
        labels = torch.tensor([[1, 3, 0], [5, 0, 0]], dtype=torch.long)
        coarse_labels = torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.long)
        span_mask = torch.ones(batch_size, num_spans, dtype=torch.float32)

        l_coarse = criterion.coarse_task_loss(
            coarse_logits,
            coarse_labels,
            span_mask,
        )
        l_coarse_fine = criterion.coarse_fine_consistency_loss(
            logits,
            coarse_logits,
            labels,
            span_mask,
        )

        self._assert_finite_scalar(l_coarse)
        self._assert_finite_scalar(l_coarse_fine)
        self.assertGreaterEqual(l_coarse.item(), 0.0)
        self.assertGreaterEqual(l_coarse_fine.item(), 0.0)
        (l_coarse + l_coarse_fine).backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(coarse_logits.grad)

    def test_fmnerg_coarse_fine_losses_skip_empty_entity_cases(self):
        criterion = ImaginELoss(
            num_types=6,
            num_coarse_types=4,
            fine_to_coarse_ids=[0, 1, 1, 2, 2, 3],
            coarse_weight=0.1,
            coarse_fine_weight=0.02,
        )
        logits = torch.randn(2, 3, 6, requires_grad=True)
        coarse_logits = torch.randn(2, 3, 4, requires_grad=True)
        labels = torch.zeros(2, 3, dtype=torch.long)
        span_mask = torch.zeros(2, 3, dtype=torch.float32)

        l_coarse = criterion.coarse_task_loss(
            coarse_logits,
            torch.zeros(2, 3, dtype=torch.long),
            span_mask,
        )
        l_coarse_fine = criterion.coarse_fine_consistency_loss(
            logits,
            coarse_logits,
            labels,
            span_mask,
        )

        self._assert_finite_scalar(l_coarse)
        self._assert_finite_scalar(l_coarse_fine)
        self.assertAlmostEqual(l_coarse.item(), 0.0, places=6)
        self.assertAlmostEqual(l_coarse_fine.item(), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
