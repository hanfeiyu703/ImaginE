import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from scripts.diagnose_vinvl_coverage import diagnose_split


class CoverageDiagnosticTest(unittest.TestCase):
    def test_diagnose_split_counts_covered_groundable_entity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            split_dir = os.path.join(data_dir, "Twitter10000_v2.0", "txt")
            xml_dir = os.path.join(tmpdir, "xml")
            vinvl_dir = os.path.join(tmpdir, "vinvl")
            os.makedirs(split_dir)
            os.makedirs(xml_dir)
            os.makedirs(vinvl_dir)

            with open(os.path.join(split_dir, "train.txt"), "w", encoding="utf-8") as handle:
                handle.write("IMGID:sample.jpg\nAlice\tB-PER\nruns\tO\n\n")
            with open(os.path.join(xml_dir, "sample.xml"), "w", encoding="utf-8") as handle:
                handle.write(
                    "<annotation><object><name>Alice</name><bndbox>"
                    "<xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax>"
                    "</bndbox></object></annotation>"
                )
            np.savez(
                os.path.join(vinvl_dir, "sample.jpg.npz"),
                num_boxes=np.array(1),
                box_features=np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
                bounding_boxes=np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            )

            stats = diagnose_split(
                SimpleNamespace(
                    task="gmner",
                    dataset="Twitter10000_v2.0",
                    data_dir=data_dir,
                    image_dir=tmpdir,
                    vinvl_dir=vinvl_dir,
                    annotation_dir=xml_dir,
                    max_regions=2,
                    vinvl_feature_dim=4,
                    normalize_vinvl=False,
                ),
                "train",
            )

            self.assertEqual(stats["total_entities"], 1)
            self.assertEqual(stats["groundable_entities"], 1)
            self.assertEqual(stats["covered_groundable"], 1)
            self.assertEqual(stats["detector_miss"], 0)
            self.assertEqual(stats["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
