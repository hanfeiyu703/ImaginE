import unittest

from evaluation_metrics import compute_grounded_metrics


class GroundingMetricTest(unittest.TestCase):
    def assert_subtask_f1(self, metrics, name, expected):
        self.assertAlmostEqual(metrics["subtasks"][name]["f1"], expected, places=6)

    def test_gmner_exact_hit(self):
        gold = [[{
            "entity": "Alice",
            "type": "PER",
            "coarse_type": "PER",
            "word_span": (0, 0),
            "groundable": True,
            "box": None,
            "gt_boxes": [[0.0, 0.0, 3.0, 1.0]],
            "detector_miss": False,
        }]]
        pred = [[{
            "entity": "Alice",
            "type": "PER",
            "coarse_type": "PER",
            "word_span": (0, 0),
            "groundable": True,
            "box": [0.0, 0.0, 3.0, 1.0],
            "region_index": 0,
        }]]

        metrics = compute_grounded_metrics(pred, gold, task="gmner")
        self.assertEqual(metrics["tp"], 1)
        self.assertAlmostEqual(metrics["f1"], 1.0, places=6)
        self.assert_subtask_f1(metrics, "mner", 1.0)
        self.assert_subtask_f1(metrics, "eeg", 1.0)

    def test_gmner_type_error_still_counts_for_eeg(self):
        gold = [[{
            "entity": "Alice",
            "type": "PER",
            "coarse_type": "PER",
            "word_span": (0, 0),
            "groundable": True,
            "box": None,
            "gt_boxes": [[0.0, 0.0, 3.0, 1.0]],
            "detector_miss": False,
        }]]
        pred = [[{
            "entity": "Alice",
            "type": "LOC",
            "coarse_type": "LOC",
            "word_span": (0, 0),
            "groundable": True,
            "box": [0.0, 0.0, 3.0, 1.0],
            "region_index": 0,
        }]]

        metrics = compute_grounded_metrics(pred, gold, task="gmner")
        self.assertAlmostEqual(metrics["f1"], 0.0, places=6)
        self.assert_subtask_f1(metrics, "mner", 0.0)
        self.assert_subtask_f1(metrics, "eeg", 1.0)

    def test_gmner_region_error_none_and_detector_miss(self):
        gold = [
            [{
                "entity": "Alice",
                "type": "PER",
                "coarse_type": "PER",
                "word_span": (0, 0),
                "groundable": True,
                "box": None,
                "gt_boxes": [[0.0, 0.0, 3.0, 1.0]],
                "detector_miss": False,
            }],
            [{
                "entity": "Paris",
                "type": "LOC",
                "coarse_type": "LOC",
                "word_span": (1, 1),
                "groundable": False,
                "box": None,
                "gt_boxes": [],
                "detector_miss": False,
            }],
            [{
                "entity": "Bob",
                "type": "PER",
                "coarse_type": "PER",
                "word_span": (2, 2),
                "groundable": True,
                "box": None,
                "gt_boxes": [[10.0, 10.0, 13.0, 11.0]],
                "detector_miss": True,
            }],
        ]
        pred = [
            [{
                "entity": "Alice",
                "type": "PER",
                "coarse_type": "PER",
                "word_span": (0, 0),
                "groundable": True,
                "box": [10.0, 10.0, 13.0, 11.0],
                "region_index": 0,
            }],
            [{
                "entity": "Paris",
                "type": "LOC",
                "coarse_type": "LOC",
                "word_span": (1, 1),
                "groundable": False,
                "box": None,
                "region_index": None,
            }],
            [{
                "entity": "Bob",
                "type": "PER",
                "coarse_type": "PER",
                "word_span": (2, 2),
                "groundable": False,
                "box": None,
                "region_index": None,
            }],
        ]

        metrics = compute_grounded_metrics(pred, gold, task="gmner")
        self.assertEqual(metrics["tp"], 1)
        self.assertAlmostEqual(metrics["f1"], 1 / 3, places=6)
        self.assert_subtask_f1(metrics, "mner", 1.0)
        self.assert_subtask_f1(metrics, "eeg", 1 / 3)

    def test_gmner_iou_boundary_is_strictly_greater_than_half(self):
        gold = [[{
            "entity": "Alice",
            "type": "PER",
            "coarse_type": "PER",
            "word_span": (0, 0),
            "groundable": True,
            "box": None,
            "gt_boxes": [[0.0, 0.0, 3.0, 1.0]],
            "detector_miss": False,
        }]]
        pred = [[{
            "entity": "Alice",
            "type": "PER",
            "coarse_type": "PER",
            "word_span": (0, 0),
            "groundable": True,
            "box": [1.0, 0.0, 4.0, 1.0],
            "region_index": 0,
        }]]

        metrics = compute_grounded_metrics(pred, gold, task="gmner")
        self.assertAlmostEqual(metrics["f1"], 0.0, places=6)
        self.assert_subtask_f1(metrics, "mner", 1.0)
        self.assert_subtask_f1(metrics, "eeg", 0.0)

    def test_fmnerg_fine_and_grounding_subtasks(self):
        gold = [
            [{
                "entity": "Taylor Swift",
                "type": "musician",
                "coarse_type": "person",
                "word_span": (0, 1),
                "groundable": True,
                "box": None,
                "gt_boxes": [[0.0, 0.0, 3.0, 1.0]],
                "detector_miss": False,
            }],
            [{
                "entity": "Taylor Swift",
                "type": "musician",
                "coarse_type": "person",
                "word_span": (0, 1),
                "groundable": True,
                "box": None,
                "gt_boxes": [[0.0, 0.0, 3.0, 1.0]],
                "detector_miss": False,
            }],
        ]
        pred = [
            [{
                "entity": "Taylor Swift",
                "type": "musician",
                "coarse_type": "person",
                "word_span": (0, 1),
                "groundable": True,
                "box": [10.0, 10.0, 13.0, 11.0],
                "region_index": 0,
            }],
            [{
                "entity": "Taylor Swift",
                "type": "actor",
                "coarse_type": "person",
                "word_span": (0, 1),
                "groundable": True,
                "box": [0.0, 0.0, 3.0, 1.0],
                "region_index": 0,
            }],
        ]

        metrics = compute_grounded_metrics(pred, gold, task="fmnerg")
        self.assertAlmostEqual(metrics["f1"], 0.0, places=6)
        self.assert_subtask_f1(metrics, "fmner", 0.5)
        self.assert_subtask_f1(metrics, "eeg", 0.5)

    def test_fmnerg_duplicate_keys_follow_last_write_wins(self):
        gold = [[{
            "entity": "Taylor Swift",
            "type": "musician",
            "coarse_type": "person",
            "word_span": (0, 1),
            "groundable": True,
            "box": None,
            "gt_boxes": [[0.0, 0.0, 3.0, 1.0]],
            "detector_miss": False,
        }]]
        pred = [[
            {
                "entity": "Taylor Swift",
                "type": "musician",
                "coarse_type": "person",
                "word_span": (0, 1),
                "groundable": True,
                "box": [0.0, 0.0, 3.0, 1.0],
                "region_index": 0,
            },
            {
                "entity": "Taylor Swift",
                "type": "musician",
                "coarse_type": "person",
                "word_span": (0, 1),
                "groundable": True,
                "box": [10.0, 10.0, 13.0, 11.0],
                "region_index": 1,
            },
        ]]

        metrics = compute_grounded_metrics(pred, gold, task="fmnerg")
        self.assertAlmostEqual(metrics["f1"], 0.0, places=6)
        self.assert_subtask_f1(metrics, "fmner", 1.0)
        self.assert_subtask_f1(metrics, "eeg", 0.0)

    def test_fmnerg_object_detection_fault_is_penalized(self):
        gold = [[{
            "entity": "Taylor Swift",
            "type": "musician",
            "coarse_type": "person",
            "word_span": (0, 1),
            "groundable": True,
            "box": None,
            "gt_boxes": [[0.0, 0.0, 3.0, 1.0]],
            "detector_miss": True,
        }]]
        pred = [[{
            "entity": "Taylor Swift",
            "type": "musician",
            "coarse_type": "person",
            "word_span": (0, 1),
            "groundable": False,
            "box": None,
            "region_index": None,
        }]]

        metrics = compute_grounded_metrics(pred, gold, task="fmnerg")
        self.assertAlmostEqual(metrics["f1"], 0.0, places=6)
        self.assert_subtask_f1(metrics, "fmner", 1.0)
        self.assert_subtask_f1(metrics, "eeg", 0.0)


if __name__ == "__main__":
    unittest.main()
