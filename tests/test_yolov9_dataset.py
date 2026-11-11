"""Pure preparation checks: no simulator, training, or GPU required."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_yolov9_dataset import choose_split, view_group, yolo_box


class DatasetTests(unittest.TestCase):
    def test_box_conversion(self):
        self.assertEqual(yolo_box([0, 0, 640, 480], 640, 480), (0.5, 0.5, 1.0, 1.0))

    def test_bad_boxes(self):
        for box in ([0, 0, 0, 2], [-1, 0, 2, 2], [639, 0, 2, 2], [float("nan"), 0, 2, 2]):
            with self.assertRaises(ValueError):
                yolo_box(box, 640, 480)

    def test_related_views_grouped(self):
        self.assertEqual(view_group("zoom_bottle_a020_near.png", {}),
                         view_group("zoom_can_a020_far.png", {}))
        self.assertEqual(view_group("table_001.png", {"table_001.png": {"azimuth_deg": 20}}),
                         view_group("zoom_bottle_a020_near.png", {}))

    def test_disjoint_split_and_class_coverage(self):
        records = [{"group": str(i // 2), "annotations": [{"category_id": 1}, {"category_id": 7}]}
                   for i in range(10)]
        train, val = choose_split(records, {1, 7})
        self.assertEqual((len(train), len(val)), (8, 2))
        self.assertFalse({r["group"] for r in train} & {r["group"] for r in val})
        self.assertEqual(choose_split(records, {1, 7}), (train, val))


if __name__ == "__main__":
    unittest.main()