"""CPU-only perception tests; no ROS, simulator, services, or motion.

The model regression is skipped unless TABLETOP_TEST_NPZ names an explicitly
trusted local recording. It uses the existing pinned local checkpoint only.
"""
import builtins
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perception.tabletop_rgbd import locate_targets, locate_tray, roi_mask, unique_target, world_points
from perception.yolov9_tabletop import TabletopYOLOv9


def target(label="Blue Tray", class_id=0, confidence=.8, box=None):
    return {"label": label, "class_id": class_id, "confidence": confidence,
            "bbox_xyxy": [0, 0, 100, 100] if box is None else box}


def calibrated_tray(floor=.008, focal=500.):
    """101x101 blue plane viewed downwards from world Z=1 m.

    Optical Z is positive depth, not world height or radial distance. The
    180-degree X rotation maps optical +Z down and optical +Y to world -Y.
    At the default floor, adjacent samples are 1.984 mm apart in world XY.
    """
    rgb = np.full((101, 101, 3), [20, 100, 200], dtype=np.uint8)
    depth = np.full((101, 101), 1. - floor, dtype=float)
    k = np.array([[focal, 0., 50.], [0., focal, 50.], [0., 0., 1.]])
    transform = np.diag([1., -1., -1., 1.])
    transform[:3, 3] = [.45, -.25, 1.]
    return rgb, depth, k, transform, target()


class TargetGateTests(unittest.TestCase):
    def test_exact_label_and_class_with_inclusive_confidence_threshold(self):
        for label, class_id in (("Water Bottle", 6), ("Blue Tray", 0)):
            selected = target(label, class_id, .5)
            distractors = [target(label.lower(), class_id), target(label, class_id + 1),
                           target(label, class_id, .4999), target("Orange Can", 5)]
            with self.subTest(label=label):
                self.assertIs(unique_target(distractors + [selected], label, class_id), selected)

    def test_missing_wrong_label_wrong_class_and_low_confidence_rejected(self):
        for label, class_id in (("Water Bottle", 6), ("Blue Tray", 0)):
            for objects in ([], [target(label.lower(), class_id)],
                            [target(label, class_id + 1)], [target(label, class_id, .4999)],
                            [target(label, class_id, float("nan"))]):
                with self.subTest(label=label, objects=objects), self.assertRaisesRegex(ValueError, "found 0"):
                    unique_target(objects, label, class_id)

    def test_multiple_valid_targets_are_ambiguous_even_with_different_scores(self):
        for label, class_id in (("Water Bottle", 6), ("Blue Tray", 0)):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "found 2"):
                unique_target([target(label, class_id, .99), target(label, class_id, .5)], label, class_id)

    def test_locate_targets_gates_both_targets_before_geometry(self):
        rgb, depth, k, transform, tray = calibrated_tray()
        bottle = target("Water Bottle", 6)
        for objects in ([tray], [bottle], [bottle, bottle, tray], [bottle, tray, tray]):
            with self.subTest(objects=objects), patch("perception.tabletop_rgbd.locate_bottle") as fit:
                with self.assertRaisesRegex(ValueError, "Need exactly one"):
                    locate_targets(rgb, depth, k, transform, objects)
                fit.assert_not_called()


class ROITests(unittest.TestCase):
    def test_shape_dtype_xy_order_and_inclusive_endpoints(self):
        expected = np.zeros((4, 7), dtype=bool)
        expected[1:3, 2:6] = True
        actual = roi_mask((4, 7), [2, 1, 5, 2])
        self.assertEqual(actual.shape, (4, 7))
        self.assertEqual(actual.dtype, np.dtype(bool))
        np.testing.assert_array_equal(actual, expected)

    def test_single_pixel(self):
        mask = roi_mask((4, 7), [3, 2, 3, 2])
        self.assertEqual(np.count_nonzero(mask), 1)
        self.assertTrue(mask[2, 3])

    def test_partially_outside_box_is_clipped(self):
        expected = np.zeros((4, 7), dtype=bool)
        expected[:3, :4] = True
        np.testing.assert_array_equal(roi_mask((4, 7), [-2, -3, 3, 2]), expected)
        self.assertTrue(roi_mask((4, 7), [-2, -3, 99, 99]).all())

    def test_wholly_outside_box_has_no_pixels(self):
        # Negative upper bounds must not become Python's from-the-end slices.
        for box in ([-5, 0, -2, 3], [0, -5, 6, -2], [7, 0, 9, 3], [0, 4, 6, 8]):
            with self.subTest(box=box):
                self.assertFalse(roi_mask((4, 7), box).any())

    def test_reversed_bounds_are_empty(self):
        for box in ([5, 1, 2, 3], [1, 3, 5, 1]):
            with self.subTest(box=box):
                self.assertFalse(roi_mask((4, 7), box).any())


class CalibratedTrayTests(unittest.TestCase):
    def test_deprojection_uses_intrinsics_and_optical_to_world_transform(self):
        _, depth, k, transform, _ = calibrated_tray()
        world = world_points(depth, k, transform)
        self.assertEqual(world.shape, (101, 101, 3))
        np.testing.assert_allclose(world[50, 50], [.45, -.25, .008], atol=1e-12)
        np.testing.assert_allclose(world[0, 100], [.5492, -.1508, .008], atol=1e-12)
        np.testing.assert_allclose(world_points(depth, k.ravel(), transform), world)

    def test_valid_empty_floor(self):
        result = locate_tray(*calibrated_tray())
        np.testing.assert_allclose(result["center_xy"], [.45, -.25], atol=1e-12)
        np.testing.assert_allclose(result["visible_extent_xy"], [.194432, .194432], atol=1e-12)
        self.assertAlmostEqual(result["floor_z"], .008)
        self.assertEqual(result["rim_z"], .06)  # Known geometry, not a measured rim.
        self.assertGreaterEqual(result["floor_points"], 25)

    def test_invalid_transform_rejected(self):
        rgb, depth, k, transform, tray = calibrated_tray()
        nonfinite = transform.copy()
        nonfinite[0, 3] = np.nan
        for bad in (np.eye(3), np.zeros((4, 3)), nonfinite):
            with self.subTest(shape=bad.shape), self.assertRaisesRegex(ValueError, "Invalid calibration/transform"):
                locate_tray(rgb, depth, k, bad, tray)

    def test_nonpositive_focal_lengths_rejected(self):
        rgb, depth, k, transform, tray = calibrated_tray()
        for axis in (0, 1):
            for value in (0., -500.):
                bad = k.copy()
                bad[axis, axis] = value
                with self.subTest(axis=axis, value=value), self.assertRaisesRegex(ValueError, "Invalid calibration/transform"):
                    locate_tray(rgb, depth, bad, transform, tray)

    def test_no_usable_depth_rejected(self):
        rgb, depth, k, transform, tray = calibrated_tray()
        for value in (np.nan, np.inf, 0.):
            with self.subTest(depth=value), self.assertRaisesRegex(ValueError, "No blue tray depth surface"):
                locate_tray(rgb, np.full_like(depth, value), k, transform, tray)

    def test_nonblue_surface_rejected(self):
        rgb, depth, k, transform, tray = calibrated_tray()
        rgb[:] = [200, 40, 20]
        with self.assertRaisesRegex(ValueError, "No blue tray depth surface"):
            locate_tray(rgb, depth, k, transform, tray)

    def test_sparse_component_rejected(self):
        rgb, depth, k, transform, tray = calibrated_tray()
        rgb[:] = 0
        rgb[47:54, 47:54] = [20, 100, 200]
        with self.assertRaisesRegex(ValueError, "Too few tray depth points"):
            locate_tray(rgb, depth, k, transform, tray)

    def test_extent_too_small_or_large_rejected(self):
        for focal in (800., 300.):
            with self.subTest(focal=focal), self.assertRaisesRegex(ValueError, "Tray extent inconsistent"):
                locate_tray(*calibrated_tray(focal=focal))

    def test_hidden_floor_rejected(self):
        with self.assertRaisesRegex(ValueError, "Tray floor not visible"):
            locate_tray(*calibrated_tray(floor=.04))

    def test_wrong_floor_height_rejected(self):
        with self.assertRaisesRegex(ValueError, "Tray floor not flat at expected height"):
            locate_tray(*calibrated_tray(floor=.014))

    def test_nonflat_floor_rejected_even_when_median_is_correct(self):
        rgb, depth, k, transform, tray = calibrated_tray()
        depth[:, ::5] = 1. - .014  # 20% raised samples; median stays at 8 mm.
        with self.assertRaisesRegex(ValueError, "Tray floor not flat at expected height"):
            locate_tray(rgb, depth, k, transform, tray)

    def test_nonblue_central_obstacle_rejected(self):
        rgb, depth, k, transform, tray = calibrated_tray()
        rgb[49:52, 49:52] = [200, 40, 20]
        depth[49:52, 49:52] = 1. - .04
        with self.assertRaisesRegex(ValueError, "Tray placement area contains an obstacle"):
            locate_tray(rgb, depth, k, transform, tray)


class RGBDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        real_import = builtins.__import__
        forbidden = {"rclpy", "sensor_msgs", "std_msgs", "tf2_ros", "isaacsim", "omni", "torch"}

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in forbidden:
                raise AssertionError(f"Script import attempted runtime dependency: {name}")
            return real_import(name, *args, **kwargs)

        spec = importlib.util.spec_from_file_location("wrist_yolo_under_test", ROOT / "scripts/wrist_yolo_ros.py")
        if spec is None or spec.loader is None:
            raise AssertionError("Cannot load wrist publisher script")
        cls.script = importlib.util.module_from_spec(spec)
        old_path = sys.path[:]
        try:
            with patch("builtins.__import__", side_effect=guarded_import):
                spec.loader.exec_module(cls.script)  # Not __main__: no ROS/model startup.
        finally:
            sys.path[:] = old_path

    def message(self, **changes):
        fields = dict(height=2, width=2, step=8, encoding="rgb8", is_bigendian=0,
                      data=bytearray([1, 2, 3, 4, 5, 6, 250, 251,
                                      7, 8, 9, 10, 11, 12, 252, 253]))
        fields.update(changes)
        return SimpleNamespace(**fields)

    def test_padded_rows_preserve_rgb_and_return_independent_copy(self):
        msg = self.message()
        actual = self.script.rgb_array(msg)
        expected = np.arange(1, 13, dtype=np.uint8).reshape(2, 2, 3)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.dtype(np.uint8))
        msg.data[0] = 99
        self.assertEqual(actual[0, 0, 0], 1)
        actual[0, 0, 1] = 88
        self.assertEqual(msg.data[1], 2)

    def test_unpadded_rows(self):
        msg = self.message(step=6, data=bytes(range(12)))
        np.testing.assert_array_equal(self.script.rgb_array(msg), np.arange(12, dtype=np.uint8).reshape(2, 2, 3))

    def test_bad_encoding_endianness_and_short_step_rejected(self):
        for changes in ({"encoding": "bgr8"}, {"encoding": "rgba8"}, {"encoding": "mono8"},
                        {"encoding": "32FC1"}, {"is_bigendian": 1}, {"step": 5}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "Expected little-endian rgb8"):
                self.script.rgb_array(self.message(**changes))

    def test_buffer_length_mismatch_rejected(self):
        for size in (15, 17):
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.script.rgb_array(self.message(data=bytes(size)))


class ModelInputTests(unittest.TestCase):
    def test_bad_rgb_rejected_before_any_model_access(self):
        adapter = TabletopYOLOv9.__new__(TabletopYOLOv9)
        for rgb in (np.zeros((2, 3, 3), dtype=float), np.zeros((2, 3), dtype=np.uint8),
                    np.zeros((2, 3, 4), dtype=np.uint8)):
            with self.subTest(shape=rgb.shape, dtype=rgb.dtype), self.assertRaisesRegex(ValueError, "Expected uint8 RGB image"):
                adapter.detect(rgb)


@unittest.skipUnless(os.environ.get("TABLETOP_TEST_NPZ"), "Optional: set TABLETOP_TEST_NPZ to a trusted local RGB-D recording")
class RecordedModelTests(unittest.TestCase):
    def test_pinned_cpu_model_and_recorded_rgbd_targets(self):
        """Actual inference, not mocked; failure is not converted into a skip."""
        path = Path(os.environ["TABLETOP_TEST_NPZ"])
        self.assertTrue(path.is_absolute() and path.is_file(), "An absolute local NPZ file is required")
        with np.load(path, allow_pickle=False) as frame:
            rgb, depth, k, transform = (frame[key].copy() for key in ("rgb", "depth", "k", "camera_to_world"))
        self.assertEqual(rgb.dtype, np.dtype(np.uint8))
        self.assertEqual(rgb.shape, (*depth.shape, 3))
        config = json.loads((ROOT / "configs/yolov9_tabletop.json").read_text())
        model = TabletopYOLOv9()  # Checks upstream revision, cleanliness, hash, names and dual head.
        self.assertEqual(config["device"], "cpu")
        self.assertTrue(all(p.device.type == "cpu" for p in model.model.parameters()))
        self.assertEqual([model.names[i] for i in range(len(model.names))], config["names"])
        objects = model.detect(rgb)
        unique_target(objects, "Water Bottle", 6)
        unique_target(objects, "Blue Tray", 0)
        estimate = locate_targets(rgb, depth, k, transform, objects)
        self.assertTrue(np.isfinite(estimate["center_xy"]).all())
        self.assertTrue(np.isfinite(estimate["tray_center_xy"]).all())
        self.assertAlmostEqual(estimate["tray_floor_z"], .008, delta=.003)
        self.assertEqual(estimate["detector"], "WongKinYiu/yolov9")


if __name__ == "__main__":
    unittest.main()