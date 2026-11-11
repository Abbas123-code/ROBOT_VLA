"""Pure geometry and real seven-DOF IK tests; never starts Isaac/ROS/physics."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "ros2_ws/src/vla_sim")]
from src.control.bottle_grasp_plan import BottlePosePlanner
from src.control.green_cube_plan import (
    FLOOR, START, TARGET, bounds, nominal_goals, placement_geometry, preflight,
)
from vla_sim.motion import HOME, JOINT_NAMES, LIMITS

BOTTLE_XYZ = [.4466993213, -.2437739223, .0079996884]


class GreenGeometryTests(unittest.TestCase):
    def test_root_is_bottom_not_centre(self):
        lo, hi = bounds(START, [1, 0, 0, 0])
        np.testing.assert_allclose(lo, [.675, .115, 0.])
        np.testing.assert_allclose(hi, [.725, .165, .05])
        self.assertEqual(bounds(TARGET, [1, 0, 0, 0])[0][2], FLOOR)

    def test_nominal_clearances_are_computed_not_claimed_ten_mm(self):
        result = placement_geometry(BOTTLE_XYZ)
        self.assertAlmostEqual(result["rim_clearance_m"], .005)
        self.assertAlmostEqual(result["bottle_clearance_m"], .0083006787)
        self.assertLess(result["bottle_clearance_m"], .010)

    def test_occupied_tray_and_small_rim_clearance_rejected(self):
        for target in ([.49, -.25, FLOOR], [.516, -.284, FLOOR], [.512, -.318, FLOOR]):
            with self.subTest(target=target), self.assertRaises(ValueError):
                placement_geometry(BOTTLE_XYZ, green_xyz=target)

    def test_finite_coords_dimensions_and_quaternions(self):
        for bad in ([np.nan, 0, 0], [0, np.inf, 0], [0, 0], [0, 0, 0, 0]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    placement_geometry(bad)
                with self.assertRaises(ValueError):
                    placement_geometry(BOTTLE_XYZ, green_xyz=bad)
        for q in ([0, 0, 0, 0], [2, 0, 0, 0], [1, np.nan, 0, 0], [1, 0, 0]):
            with self.assertRaises(ValueError):
                placement_geometry(BOTTLE_XYZ, green_quat=q)
            with self.assertRaises(ValueError):
                placement_geometry(BOTTLE_XYZ, bottle_quat=q)

    def test_orientation_projection_catches_rim_before_tilt_limit(self):
        # A .10 rad pitch is nominally within the tilt bound but DOES NOT fit
        # the 5 mm rim margin with the extra positive 2 mm geometry guard.
        q = Rotation.from_euler("y", .10).as_quat()[[3, 0, 1, 2]]
        with self.assertRaises(ValueError):
            placement_geometry(BOTTLE_XYZ, green_quat=q)
        q = Rotation.from_euler("y", .02).as_quat()[[3, 0, 1, 2]]
        self.assertLess(placement_geometry(BOTTLE_XYZ, green_quat=q)["rim_clearance_m"], .005)

    def test_bottle_must_be_upright_on_floor(self):
        with self.assertRaises(ValueError):
            placement_geometry([*BOTTLE_XYZ[:2], .025])
        q = Rotation.from_euler("y", .051).as_quat()[[3, 0, 1, 2]]
        with self.assertRaises(ValueError):
            placement_geometry(BOTTLE_XYZ, bottle_quat=q)


class GreenKinematicTests(unittest.TestCase):
    def test_full_nominal_route_and_exact_home_correction(self):
        urdf = ROOT / "isaac/assets/bullet3/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"
        planner = BottlePosePlanner(urdf, JOINT_NAMES, LIMITS)
        original_segment = planner.segment
        endpoints = []

        def checked_segment(seed, position, orientation):
            samples = original_segment(seed, position, orientation)
            self.assertEqual(samples.shape, (121, 7))
            self.assertTrue(np.all(samples > np.asarray(LIMITS)[:, 0]))
            self.assertTrue(np.all(samples < np.asarray(LIMITS)[:, 1]))
            self.assertLessEqual(np.max(np.abs(np.diff(samples, axis=0))), .12)
            frame = planner.fk(samples[-1])
            np.testing.assert_allclose(frame[:3, 3], position, atol=1e-4)
            self.assertLess(Rotation.from_matrix(orientation.T @ frame[:3, :3]).magnitude(), .0005)
            endpoints.append(samples[-1])
            return samples

        planner.segment = checked_segment
        result = preflight(planner, HOME, BOTTLE_XYZ)
        self.assertEqual(result["segments"], 12)
        self.assertLessEqual(result["home_correction_rad"], .15)
        correction = np.linspace(endpoints[-1], np.asarray(HOME), 121)
        self.assertLessEqual(np.max(np.abs(np.diff(correction, axis=0))), .12)
        self.assertTrue(np.all(correction > np.asarray(LIMITS)[:, 0]))
        self.assertTrue(np.all(correction < np.asarray(LIMITS)[:, 1]))
        goals = nominal_goals(planner, HOME)
        np.testing.assert_allclose(goals[7][1], [.512, -.284, .040])
        np.testing.assert_allclose(goals[-1][1], planner.fk(np.asarray(HOME))[:3, 3])
        print("GREEN_PREFLIGHT", result)

    def test_nominal_failure_propagates_without_retry(self):
        class Unreachable:
            calls = 0

            def fk(self, seed):
                return np.eye(4)

            def segment(self, *args):
                self.calls += 1
                raise RuntimeError("Impossible nominal route")

        planner = Unreachable()
        with self.assertRaisesRegex(RuntimeError, "Impossible nominal"):
            preflight(planner, HOME, BOTTLE_XYZ)
        self.assertEqual(planner.calls, 1)

    def test_large_home_branch_correction_rejected(self):
        class WrongBranch:
            def fk(self, seed):
                return np.eye(4)

            def segment(self, seed, *args):
                return np.array([seed, np.asarray(seed) + .02])

        with self.assertRaisesRegex(RuntimeError, "HOME correction"):
            preflight(WrongBranch(), HOME, BOTTLE_XYZ)


if __name__ == "__main__":
    unittest.main()