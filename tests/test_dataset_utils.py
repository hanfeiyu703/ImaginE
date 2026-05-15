import os
import tempfile
import unittest

import numpy as np
import torch

from config import (
    FMNERG_FINE_TO_COARSE,
    get_coarse_entity_type_to_id,
    get_coarse_entity_types,
    get_entity_types,
    get_fine_to_coarse_ids,
)
from data.dataset import TwitterMNERDataset
from data.grounding import (
    build_aspect_iou_map,
    build_grounding_supervision,
    load_vinvl_features,
)
from data.span_utils import (
    compute_word_token_bounds,
    enumerate_candidate_spans_with_words,
    extract_entity_spans,
    token_span_to_word_span,
    word_span_to_token_span,
)


class FakeProcessor:
    max_span_length = 2

    def process_image(self, image_path):
        del image_path
        return torch.zeros(3, 224, 224)

    def tokenize_and_align(self, words, labels):
        del labels
        return {
            "input_ids": torch.arange(len(words) + 2, dtype=torch.long),
            "attention_mask": torch.ones(len(words) + 2, dtype=torch.long),
            "word_ids": [None] + list(range(len(words))) + [None],
        }

    def compute_word_token_bounds(self, word_ids):
        return compute_word_token_bounds(word_ids)

    def word_span_to_token_span(self, word_ids, word_start, word_end):
        return word_span_to_token_span(word_ids, word_start, word_end)

    def enumerate_candidate_spans_with_words(self, word_ids):
        return enumerate_candidate_spans_with_words(word_ids, self.max_span_length)

    def extract_entity_spans(self, words, labels):
        del words
        return extract_entity_spans(labels)


class DatasetUtilityTest(unittest.TestCase):
    def test_fmnerg_label_space_matches_reference(self):
        entity_types = get_entity_types("fmnerg")
        self.assertEqual(len(entity_types), 52)
        self.assertIn("actor", entity_types)
        self.assertEqual(FMNERG_FINE_TO_COARSE["actor"], "person")
        coarse_types = get_coarse_entity_types("fmnerg")
        self.assertEqual(len(coarse_types), 9)
        self.assertIn("person", coarse_types)
        fine_to_coarse = get_fine_to_coarse_ids("fmnerg")
        self.assertEqual(len(fine_to_coarse), len(entity_types))
        self.assertEqual(fine_to_coarse[entity_types.index("actor")], coarse_types.index("person"))

    def test_fmnerg_dataset_emits_coarse_span_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "train.txt")
            with open(data_file, "w", encoding="utf-8") as handle:
                handle.write(
                    "IMGID:sample.jpg\n"
                    "Pak\tB-building\tB-building_other\n"
                    "Webber\tB-person\tB-intellectual\n"
                    "wins\tO\tO\n\n"
                )

            dataset = TwitterMNERDataset(
                data_file=data_file,
                image_dir=tmpdir,
                processor=FakeProcessor(),
                max_spans=8,
                is_train=False,
                task="fmnerg",
                visual_backend="raw_image",
                max_regions=3,
                vinvl_feature_dim=4,
            )
            item = dataset[0]
            coarse_to_id = get_coarse_entity_type_to_id("fmnerg")

            self.assertIn(coarse_to_id["building"], item["coarse_labels"].tolist())
            self.assertIn(coarse_to_id["person"], item["coarse_labels"].tolist())

    def test_caption_context_does_not_create_candidate_spans(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "train.txt")
            caption_file = os.path.join(tmpdir, "captions.txt")
            with open(data_file, "w", encoding="utf-8") as handle:
                handle.write("IMGID:sample.jpg\nAlice\tB-PER\nruns\tO\n\n")
            with open(caption_file, "w", encoding="utf-8") as handle:
                handle.write("sample.jpg : Bob appears in a caption\n")

            dataset = TwitterMNERDataset(
                data_file=data_file,
                image_dir=tmpdir,
                processor=FakeProcessor(),
                max_spans=16,
                is_train=False,
                task="gmner",
                visual_backend="raw_image",
                max_regions=3,
                vinvl_feature_dim=4,
                append_caption=True,
                caption_files=[caption_file],
                caption_max_words=8,
            )
            item = dataset[0]
            valid_spans = int(item["span_mask"].sum().item())

            self.assertEqual(valid_spans, 3)
            self.assertEqual(item["metadata"]["caption"], "Bob appears in a caption")
            for span in item["span_indices"][:valid_spans].tolist():
                self.assertLessEqual(span[-1], 2)

    def test_knowledge_context_is_separate_from_candidate_spans(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "train.txt")
            knowledge_file = os.path.join(tmpdir, "knowledge.txt")
            with open(data_file, "w", encoding="utf-8") as handle:
                handle.write("IMGID:sample.jpg\nAlice\tB-PER\nruns\tO\n\n")
            with open(knowledge_file, "w", encoding="utf-8") as handle:
                handle.write("sample.jpg : Bob appears in separate knowledge\n")

            dataset = TwitterMNERDataset(
                data_file=data_file,
                image_dir=tmpdir,
                processor=FakeProcessor(),
                max_spans=16,
                is_train=False,
                task="gmner",
                visual_backend="raw_image",
                max_regions=3,
                vinvl_feature_dim=4,
                knowledge_files=[knowledge_file],
                knowledge_max_words=8,
            )
            item = dataset[0]
            valid_spans = int(item["span_mask"].sum().item())

            self.assertEqual(valid_spans, 3)
            self.assertEqual(
                item["metadata"]["knowledge"],
                "Bob appears in separate knowledge",
            )
            self.assertGreater(int(item["knowledge_attention_mask"].sum().item()), 0)
            for span in item["span_indices"][:valid_spans].tolist():
                self.assertLessEqual(span[-1], 2)

    def test_word_token_alignment_helpers(self):
        word_ids = [None, 0, 1, 1, 2, None]

        bounds = compute_word_token_bounds(word_ids)
        self.assertEqual(bounds, [(0, 1, 1), (1, 2, 3), (2, 4, 4)])
        self.assertEqual(word_span_to_token_span(word_ids, 1, 2), (2, 4))
        self.assertEqual(token_span_to_word_span(2, 3, bounds), (1, 1))

        spans = enumerate_candidate_spans_with_words(word_ids, max_span_length=3)
        self.assertIn(((1, 1), (0, 0)), spans)
        self.assertIn(((1, 3), (0, 1)), spans)
        self.assertIn(((2, 4), (1, 2)), spans)

    def test_multi_box_merge_and_soft_target_normalization(self):
        annotation_box_map = {
            "Alice": [
                [0, 0, 10, 10],
                [100, 100, 110, 110],
            ]
        }
        proposal_boxes = np.array(
            [
                [0, 0, 10, 10],
                [20, 20, 30, 30],
                [100, 100, 110, 110],
            ],
            dtype=np.float32,
        )

        aspect_iou_map = build_aspect_iou_map(annotation_box_map, proposal_boxes)
        supervision = build_grounding_supervision(
            entity_name="Alice",
            annotation_box_map=annotation_box_map,
            aspect_iou_map=aspect_iou_map,
            max_regions=3,
        )

        np.testing.assert_allclose(
            supervision["distribution"],
            np.array([0.5, 0.0, 0.5, 0.0], dtype=np.float32),
        )
        self.assertTrue(supervision["groundable"])
        self.assertFalse(supervision["detector_miss"])

    def test_detector_miss_does_not_override_detected_boxes_for_same_name(self):
        annotation_box_map = {
            "Alice": [
                [0, 0, 10, 10],
                [100, 100, 110, 110],
            ]
        }
        proposal_boxes = np.array(
            [
                [100, 100, 110, 110],
                [200, 200, 210, 210],
            ],
            dtype=np.float32,
        )

        aspect_iou_map = build_aspect_iou_map(annotation_box_map, proposal_boxes)
        supervision = build_grounding_supervision(
            entity_name="Alice",
            annotation_box_map=annotation_box_map,
            aspect_iou_map=aspect_iou_map,
            max_regions=2,
        )

        np.testing.assert_allclose(
            supervision["distribution"],
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        self.assertFalse(supervision["detector_miss"])

    def test_load_vinvl_features_applies_padding_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez(
                os.path.join(tmpdir, "sample.jpg.npz"),
                num_boxes=np.array(2),
                box_features=np.array(
                    [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
                    dtype=np.float32,
                ),
                bounding_boxes=np.array(
                    [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]],
                    dtype=np.float32,
                ),
            )

            features, boxes, mask = load_vinvl_features(
                vinvl_dir=tmpdir,
                image_id="sample.jpg",
                max_regions=4,
                feature_dim=4,
                normalize=False,
            )

            self.assertEqual(features.shape, (4, 4))
            self.assertEqual(boxes.shape, (4, 4))
            np.testing.assert_allclose(mask, np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32))
            np.testing.assert_allclose(features[2:], 0.0)
            np.testing.assert_allclose(boxes[2:], 0.0)

    def test_clip_patch_distribution_uses_patch_iou(self):
        patch_boxes = torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [10.0, 0.0, 20.0, 10.0],
            ],
            dtype=torch.float32,
        )
        patch_mask = torch.tensor([1.0, 1.0], dtype=torch.float32)
        dist = TwitterMNERDataset._patch_distribution_for_boxes(
            [[0.0, 0.0, 10.0, 10.0]],
            patch_boxes,
            patch_mask,
        )

        np.testing.assert_allclose(dist, np.array([1.0, 0.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
